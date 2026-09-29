"""Run the three-frame court sampling comparison on selected scenes or whole videos.

Models load once. Each scene's inputs are prepared once and shared; each arm's
detector work is timed on its own, and each arm keeps its own history of courts
for later scenes. README.md describes the manifest and results.
"""

from __future__ import annotations

import os

# Worker processes inherit these settings, as in run_video. Set them before importing NumPy.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import dataclasses
import gzip
import json
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from court_detector import reuse
from court_detector.detect import CourtDetector, CourtFitError, Switches
from court_detector.inputs import FrameReader, PeopleSource
from court_detector.line_sources import LineSource
from court_detector.measurements import jsonable
from court_detector.reuse import KnownCourt
from court_detector.run_video import CourtTools, load_court_tools, validate_scenes
from court_detector.scene_sources import SceneInfo
from court_detector.template_arrays import TEMPLATE_DEVICES
from court_detector.video_inputs import PoseArrays, RtmlibPeople, VideoFrames

from .sampling import (
    ARMS,
    BASELINE,
    CHEAP_FIRST_SCORE_LIMIT,
    FRAME_ROLES,
    KNOWN_COURTS_KEPT,
    KNOWN_COURTS_TRIED,
    METHOD_FUNCTIONS,
    METHODS,
    ManifestScene,
    ManifestVideo,
    Route,
    SceneInputs,
    StoredCourt,
    cheap_first_detector,
    courts_to_try,
    establish,
    method_row,
    prepare_scene,
    read_manifest,
    remember,
    result_fields,
    scene_window,
    timing,
)

RESULTS_SCHEMA = "court-scene-sampling-results/1"


def write_results(path: Path, results: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt") as stream:
        json.dump(jsonable(results), stream, allow_nan=False, separators=(",", ":"))
    temporary.replace(path)


def failure(error: Exception, started: float) -> dict[str, Any]:
    return {"status": "detection_failed", "error": repr(error), "traceback": traceback.format_exc(),
            "seconds": {"detector_walltime_measured": perf_counter() - started}}


def scene_header(scene: ManifestScene) -> dict[str, Any]:
    return {"scene_id": scene.scene_id, "start_frame": scene.start_frame, "end_frame": scene.end_frame,
            "scheduled_midpoint": scene.scheduled_midpoint, "midpoint_frame": scene.midpoint,
            "midpoint_overridden": scene.midpoint_override is not None}


def window_record(inputs: SceneInputs) -> dict[str, Any]:
    """The shared window, and whether the grey check kept each sampled frame in the feet."""
    window = inputs.feet
    differences = window.grey_differences
    record: dict[str, Any] = {"frames": {role: view.frame_index for role, view in inputs.views.items()},
                              "window_first": window.frames[0], "window_last": window.frames[-1],
                              "kept_first": window.kept_frames[0], "kept_last": window.kept_frames[-1]}
    for role, position in (("first", 0), ("last", -1)):
        record[f"{role}_in_kept_feet"] = window.frames[position] in window.kept_frames
        record[f"{role}_grey_difference"] = None if differences is None else differences[position]
    return record


def tried_ids(courts: Sequence[KnownCourt]) -> list[str]:
    return [court.view_id for court in courts]


def baseline_outcome(detector: CourtDetector, inputs: SceneInputs, people: PeopleSource, frames: FrameReader,
                     tried: Sequence[KnownCourt], history: list[StoredCourt]) -> dict[str, Any]:
    """Production detection of the middle frame: run_video's detect() call, with this arm's history.

    detect() repeats the feet step on the frames and people prepare_scene already read.
    That repeat is left out of the detector time; the shared feet measurement is charged
    instead, as for every method.
    """
    view = dataclasses.replace(inputs.views["middle"], alignment_image=inputs.alignment_median)
    outcome: dict[str, Any] = {"known_courts_tried": tried_ids(tried)}
    started = perf_counter()
    try:
        result = detector.detect(view, people, frames, known_courts=tried)
    except CourtFitError as error:
        return {**outcome, **failure(error, started)}
    seconds = perf_counter() - started
    if result.stage_seconds is None:
        raise RuntimeError("the comparison needs Switches(timing=True)")
    outcome.update(result_fields(result), stage_seconds=result.stage_seconds)
    outcome["seconds"] = {**timing(seconds - result.stage_seconds["feet"], inputs.seconds, ["middle"]),
                          "detect_walltime_measured_with_repeat_feet": seconds}
    outcome["established"] = None
    paint = result.paint_score
    # run_video's rule: only a searched court with positive paint joins the history.
    if result.corners_native_px is not None and result.reused_from is None and paint is not None and paint > 0:
        court = reuse.make_known_court(view.view_id, view.frame, result.corners_native_px, paint,
                                       alignment_image=inputs.alignment_median)
        remember(history, court, inputs.scene.histogram)
        outcome["established"] = {"view_id": court.view_id, "role": "middle", "frame_index": view.frame_index,
                                  "route": Route.FULL_SEARCH, "is_chosen": True, "paint_score": paint,
                                  "reference_image": "window_median"}
    return outcome


def method_outcome(method: str, detector: CourtDetector, inputs: SceneInputs, tried: Sequence[KnownCourt],
                   history: list[StoredCourt], baseline_corners: list | None) -> dict[str, Any]:
    """One method on one scene: its frames, its timing and the court it adds to its history.

    A fit failure fails this method on this scene alone, as run_video fails one scene.
    """
    outcome: dict[str, Any] = {"known_courts_tried": tried_ids(tried)}
    started = perf_counter()
    try:
        run = METHOD_FUNCTIONS[method](detector, inputs, tried)
    except CourtFitError as error:
        return {**outcome, **failure(error, started)}
    seconds = perf_counter() - started
    diagnostics_started = perf_counter()
    outcome.update(method_row(detector.live, run, inputs, baseline_corners))
    outcome["established"] = establish(history, run, inputs)
    outcome["seconds"] = timing(seconds, inputs.seconds, run.frames_used)
    outcome["seconds"]["diagnostics_not_charged"] = perf_counter() - diagnostics_started
    return outcome


def scene_row(inputs: SceneInputs, frames: FrameReader, people: PeopleSource, detector: CourtDetector,
              cheap_detector: CourtDetector, histories: dict[str, list[StoredCourt]],
              use_history: bool) -> dict[str, Any]:
    """Every arm on one scene. With use_history, each arm first tries its own earlier courts."""
    scene = inputs.scene
    tried = {arm: courts_to_try(histories[arm], scene.histogram) if use_history else [] for arm in ARMS}
    row: dict[str, Any] = {**scene_header(scene), "status": "analysed", "history_reuse": use_history,
                           "window": window_record(inputs), "input_seconds": inputs.seconds}
    row["baseline"] = baseline_outcome(detector, inputs, people, frames, tried[BASELINE], histories[BASELINE])
    baseline_corners = row["baseline"].get("corners_native_px")
    row["methods"] = {}
    for method in METHODS:
        method_detector = cheap_detector if method == "cheap_first" else detector
        row["methods"][method] = method_outcome(method, method_detector, inputs, tried[method], histories[method],
                                                baseline_corners)
    if not use_history:
        row["baseline_agreement"] = baseline_agreement(row)
    return row


def baseline_agreement(row: dict[str, Any]) -> dict[str, bool | None]:
    """Whether each method's middle-frame detection reproduced the ordinary detect() result exactly.

    cheap_first searches with a score limit, so only the two exhaustive methods must agree.
    Scenes with history reuse have different histories per arm, so they are not compared.
    """
    agreement = {}
    for method in ("full_three", "seed_refit"):
        outcome, baseline = row["methods"][method], row["baseline"]
        if "frames" not in outcome or baseline["status"] == "detection_failed":
            agreement[method] = None
            continue
        middle = outcome["frames"][0]
        fields = ("status", "no_court_reason", "chosen_key", "paint_score", "reused_from")
        same_corners = (middle["corners_native_px"] is None) == (baseline["corners_native_px"] is None) and (
            middle["corners_native_px"] is None
            or np.array_equal(middle["corners_native_px"], baseline["corners_native_px"]))
        agreement[method] = same_corners and all(middle[field] == baseline[field] for field in fields)
    return agreement


def arm_statuses(row: dict[str, Any]) -> dict[str, str | None]:
    """Each arm's status on one scene, for the progress line."""
    if row["status"] != "analysed":
        return {arm: row["status"] for arm in ARMS}
    return {BASELINE: row["baseline"]["status"],
            **{method: outcome["status"] for method, outcome in row["methods"].items()}}


def compare_video(video: ManifestVideo, frames: FrameReader, people: PeopleSource, lines: LineSource,
                  detector: CourtDetector, video_row: dict[str, Any], save: Callable[[], None],
                  scene_limit: int | None = None) -> None:
    """Every scene of one video in order, saving after each. Each arm keeps its own history throughout.

    Reviewed scenes run without history. A full video's scenes and every later scene
    start with each arm's history, as run_video's reuse does.

    :param scene_limit: run only a full video's first scenes, for a smoke check.
    """
    cheap_detector = cheap_first_detector(detector)
    histories: dict[str, list[StoredCourt]] = {arm: [] for arm in ARMS}
    scenes = video.scenes if scene_limit is None else video.scenes[:scene_limit]
    for key, key_scenes, use_history in (("scenes", scenes, video.full_video),
                                         ("later_scenes", video.later_scenes, True)):
        for scene in key_scenes:
            window = scene_window(scene, frames.fps)
            if window is None:
                row = {**scene_header(scene), "status": "scene_too_short_for_feet"}
            else:
                inputs = prepare_scene(scene, window, frames, people, lines, detector.switches)
                row = scene_row(inputs, frames, people, detector, cheap_detector, histories, use_history)
            video_row[key].append(row)
            video_row["histories"] = {arm: [stored.court.view_id for stored in history]
                                      for arm, history in histories.items()}
            save()
            print(json.dumps({"scene_id": scene.scene_id, "statuses": arm_statuses(row)}), flush=True)


def check_scenes(video: ManifestVideo, frame_count: int) -> None:
    """A full video's scenes must partition it, as run_video requires; other scenes must lie inside it."""
    if video.full_video:
        validate_scenes([SceneInfo(scene.start_frame, scene.end_frame) for scene in video.scenes], frame_count)
    for scene in video.scenes + video.later_scenes:
        if scene.end_frame > frame_count:
            raise ValueError(f"{scene.scene_id}: ends at {scene.end_frame}, after the video's {frame_count} frames")


def video_people(video: ManifestVideo, frames: VideoFrames, tools: CourtTools) -> PeopleSource:
    """Saved poses when the manifest names them, else live RTMLib, as run_video chooses."""
    if video.people is not None:
        poses = PoseArrays.from_directory(video.people)
        if poses.frame_count < frames.frame_count:
            raise ValueError(f"{video.video_id}: saved poses do not cover the source video")
        return poses
    if tools.pose_extractor is None:
        raise ValueError(f"{video.video_id}: no saved people and no live pose extractor")
    return RtmlibPeople(frames, tools.pose_extractor)


def run_video_entry(video: ManifestVideo, tools: CourtTools, detector: CourtDetector, results: dict[str, Any],
                    save: Callable[[], None], scene_limit: int | None) -> None:
    """Open one manifest video, add its row to the results and compare every arm on its scenes."""
    started = perf_counter()
    with VideoFrames(video.video) as frames:
        check_scenes(video, frames.frame_count)
        people = video_people(video, frames, tools)
        row = {"video_id": video.video_id, "video": str(video.video), "fps": frames.fps,
               "frame_count": frames.frame_count, "native_size": frames.size,
               "people": "live_rtmlib" if video.people is None else str(video.people),
               "full_video": video.full_video, "manifest_scene_count": len(video.scenes),
               "cut_pass_seconds_shared": video.cut_pass_seconds, "open_seconds": perf_counter() - started,
               "scenes": [], "later_scenes": [], "histories": {arm: [] for arm in ARMS}}
        results["videos"].append(row)
        save()
        compare_video(video, frames, people, tools.lines, detector, row, save,
                      scene_limit if video.full_video else None)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help=".json or .json.gz manifest; see README.md")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for results.json.gz")
    parser.add_argument("--deeplsd-source", type=Path, required=True)
    parser.add_argument("--deeplsd-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="device for DeepLSD and RTMLib")
    parser.add_argument("--template-device", choices=TEMPLATE_DEVICES, default="cuda",
                        help="line-template scoring device (default: cuda, with CuPy)")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--scene-limit", type=int,
                        help="run only each full video's first N scenes, for a smoke check (default: all)")
    args = parser.parse_args()
    if args.scene_limit is not None and args.scene_limit < 1:
        parser.error("--scene-limit must be positive")
    return args


def main() -> int:
    args = parse_arguments()
    videos = read_manifest(args.manifest)
    # Fail before loading any model, rather than mix results with an earlier run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    switches = Switches(workers=args.workers, timing=True, template_device=args.template_device)
    tools = load_court_tools(switches, deeplsd_source=args.deeplsd_source, deeplsd_weights=args.deeplsd_weights,
                             device=args.device, live_pose=any(video.people is None for video in videos))
    results_path = args.output_dir / "results.json.gz"
    results: dict[str, Any] = {
        "schema": RESULTS_SCHEMA, "finished": False, "stopped_by": None, "manifest": str(args.manifest),
        "settings": {"switches": dataclasses.asdict(switches), "cheap_first_full_score_limit": CHEAP_FIRST_SCORE_LIMIT,
                     "arms": ARMS, "frame_tie_order": FRAME_ROLES, "known_courts_kept": KNOWN_COURTS_KEPT,
                     "known_courts_tried": KNOWN_COURTS_TRIED, "scene_limit": args.scene_limit,
                     "cpu_affinity": sorted(os.sched_getaffinity(0)), "device": args.device},
        "cold_setup_seconds": {"load_models": tools.load_seconds}, "videos": [],
    }

    def save() -> None:
        write_results(results_path, results)

    save()
    try:
        with tools.detector as detector:
            for video in videos:
                run_video_entry(video, tools, detector, results, save, args.scene_limit)
    except Exception as error:
        # Only fit failures belong to one scene and arm; anything else leaves the models or inputs in doubt.
        results["stopped_by"] = {"error": repr(error), "traceback": traceback.format_exc()}
        save()
        raise
    results["finished"] = True
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
