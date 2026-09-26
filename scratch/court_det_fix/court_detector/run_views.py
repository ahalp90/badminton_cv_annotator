"""Run the court detector on D17 views and compare each result with the 24 Sept baseline.

Each view's frame, line fragments and person boxes come from the frozen packs, as in the
baseline. People come from the fresh-feet evidence run: PEOPLE_DIR holds one
extract_window_people.py record per view, with its detections, the video path and the
decoded frame count. Each video is decoded once per process, from frame 0, as the evidence
scripts did, and only the window frames are kept.

With --baseline (needs --artefacts), each view is checked against the baseline arm: the
chosen court, the whole refit record, the feet, the grey differences, the G0 and G1 entries,
the line templates' count and metadata, and the W5 record. A view that raises is logged and the run carries on; the exit code is
1 when any view raised or failed a check.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.run_views --people DIR --output DIR
      [--baseline ARM_DIR --feet FEET_FILE] [--artefacts] [--timing] [--no-self-checks] VIEW [VIEW ...]
"""

import os
import sys
from pathlib import Path

# Before numpy loads: one thread per process, as run_d17.py does.
for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_variable] = "1"
# experiments/ imports its court geometry from src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import argparse
import gzip
import hashlib
import json
import math
import resource
import traceback
from collections.abc import Sequence
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
    load_frozen_case_provenance,
)
from scratch.court_det_fix.court_detector import feet
from scratch.court_det_fix.court_detector.detect import (
    ROOT,
    CourtDetector,
    CourtResult,
    Switches,
)
from scratch.court_det_fix.court_detector.inputs import PersonSample, ViewInputs

FRESH_FEET = ROOT / "court_detector_optimisation_handover/claude_evidence/fresh_feet"
VIEWS = FRESH_FEET / "views.json"
SHOT_CHECK = FRESH_FEET / "shot_check.jsonl"
MANIFEST = ROOT / "wider_evaluation/runs/20260922/manifest.json.gz"
CONTROL_PACK = ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz"
LEGACY_ENTRY_FIELDS = ("stripe", "profile")
# refit_selected.refit_selection adds these around the refit itself.
REFIT_WRAPPER_FIELDS = {"case_id", "label", "source_record", "frame_md5"}


def read_json_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class SavedPeople:
    """People from one extract_window_people.py record; asking for an unsaved frame raises."""

    def __init__(self, record: dict) -> None:
        self.by_frame = {sample["frame_index"]: sample for sample in record["samples"]}

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        found = []
        for frame_index in frame_indices:
            sample = self.by_frame[frame_index]
            boxes = np.asarray(sample["bboxes"], dtype=float).reshape(-1, 4)
            keypoints = np.asarray(sample["keypoints"], dtype=float).reshape(-1, 17, 2)
            found.append(PersonSample(frame_index, boxes, keypoints))
        return found


class DecodedVideo:
    """Frames decoded once, in order from frame 0; asking for a frame not kept raises."""

    def __init__(self, path: str, keep: set[int]) -> None:
        # FFmpeg otherwise starts a decoding thread per core, whatever the thread variables say.
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if not capture.isOpened():
            raise FileNotFoundError(path)
        self.fps = capture.get(cv2.CAP_PROP_FPS)
        self.size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames = {}
        for frame_index in range(max(keep) + 1):
            if not capture.grab():
                raise RuntimeError(f"{path}: decoding stopped at frame {frame_index}")
            if frame_index in keep:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError(f"{path}: could not retrieve frame {frame_index}")
                self.frames[frame_index] = frame
        capture.release()

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        return [self.frames[frame_index] for frame_index in frame_indices]


def pack_sources(case_packs: dict[str, str]) -> tuple[dict[str, dict], dict[str, CaseProvenance], dict[str, Path]]:
    """Every pack and control source by view ID, with its provenance and frame path."""
    sources, provenances, frame_paths = {}, {}, {}
    for relative in case_packs.values():
        pack_provenance = load_frozen_case_provenance(ROOT / relative)
        for source in read_json_gz(ROOT / relative)["cases"]:
            sources[source["id"]] = source
            provenances[source["id"]] = pack_provenance[source["id"]]
    for source in read_json_gz(CONTROL_PACK)["cases"]:
        frame_index = source["frame_index"]
        sources[source["id"]] = source
        provenances[source["id"]] = CaseProvenance(source["id"], ImageKind.SOURCE_FRAME, (frame_index,), frame_index)
        frame_paths[source["id"]] = ROOT / source["image"]
    return sources, provenances, frame_paths


def without_keys(value: Any, names: set[str]) -> Any:
    """A copy of a JSON value with every dict key in names removed, at any depth."""
    if isinstance(value, dict):
        return {key: without_keys(item, names) for key, item in value.items() if key not in names}
    if isinstance(value, list):
        return [without_keys(item, names) for item in value]
    return value


def first_difference(left: Any, right: Any, path: str = "") -> str | None:
    """Where two JSON values first differ, or None when they are equal."""
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ by {sorted(left.keys() ^ right.keys())[:5]}"
        for key in left:
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} and {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            found = first_difference(left_item, right_item, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    # Python counts 1 == 1.0 and False == 0, but JSON writes them differently.
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} and {type(right).__name__}"
    # Python also counts -0.0 == 0.0.
    signs_differ = isinstance(left, float) and math.copysign(1.0, left) != math.copysign(1.0, right)
    return None if left == right and not signs_differ else f"{path}: {left!r} != {right!r}"


def baseline_checks(view_id: str, result: CourtResult, artefacts: dict, baseline: Path, feet_by_view: dict,
                    shot_rows: dict) -> dict[str, str | None]:
    """Each check's first difference from the baseline arm; None means equal."""
    summary = read_json_gz(baseline / "d17" / f"{view_id}.json.gz")
    selection = summary["selection"]
    polarity = selection["polarity_refit"]
    refit = artefacts.get("stripe_refit")
    if refit is not None:
        refit = {key: value for key, value in refit.items() if key != "timings_seconds"}
    if polarity is not None:
        polarity = {key: value for key, value in polarity.items()
                    if key != "timings_seconds" and key not in REFIT_WRAPPER_FIELDS}
    checks = {
        "chosen_key": first_difference(result.chosen_key, selection["bounded"]),
        "refit": first_difference(refit, polarity),
        "feet": first_difference(artefacts["feet"]["all_feet_px"], feet_by_view[view_id]),
        "grey_differences": first_difference(artefacts["feet"]["grey_differences"],
                                             shot_rows[view_id]["differences"]),
    }
    for name in ("G0", "G1"):
        saved = read_json_gz(baseline / "populations" / name / f"{view_id}.json.gz")["entries"]
        checks[f"{name}_entries"] = first_difference(artefacts["populations"][name],
                                                     without_keys(saved, set(LEGACY_ENTRY_FIELDS)))
    record = read_json_gz(baseline / "case_records" / f"{view_id}.json.gz")
    checks["line_template_count"] = first_difference(len(artefacts["line_templates"]["entries"]),
                                                     record["population_counts"]["line_template"])
    checks["line_template_metadata"] = first_difference(
        without_keys(artefacts["line_templates"]["metadata"], {"elapsed_seconds"}),
        without_keys(record["population_sources"]["line_template"], {"elapsed_seconds"}),
    )
    ours = artefacts["w5"]["record"]
    for key in ("parents", "valid_children", "fit_attempts"):
        checks[f"w5_{key}"] = first_difference(without_keys(ours[key], {"legacy"}),
                                               without_keys(record[key], {"legacy"}))
    checks["w5_c_ranking"] = first_difference(ours["rankings"]["C"], record["rankings"]["C"])
    checks["w5_identity_resolution"] = first_difference(
        without_keys(artefacts["w5"]["identity_resolution"], {"legacy"}),
        without_keys(record["identity_resolution"], {"legacy"}),
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("views", nargs="+", help="view IDs from views.json")
    parser.add_argument("--people", type=Path, required=True, help="folder of extract_window_people.py records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="baseline arm folder: d17/, populations/, case_records/")
    parser.add_argument("--feet", type=Path, help="the baseline's feet file, {view ID: all_feet_px}")
    parser.add_argument("--artefacts", action="store_true", help="write each view's intermediate results")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-self-checks", action="store_true")
    parser.add_argument("--any-camera-roll", action="store_true",
                        help="keep courts that need a camera rolled past 45 degrees or upside down")
    parser.add_argument("--geometry-weight", type=float, default=0.1,
                        help="share of W5's geometry score in the net choice; 0 is the accepted chain's paint alone")
    parser.add_argument("--line-paint", action="store_true",
                        help="in the net choice, pass or fail each painted line on its average contrast along its length")
    args = parser.parse_args()
    if args.baseline is not None and (args.feet is None or not args.artefacts):
        parser.error("--baseline needs --feet and --artefacts")
    if args.baseline is not None and not args.any_camera_roll:
        parser.error("--baseline compares with a chain that keeps every camera roll; add --any-camera-roll")
    if args.baseline is not None and (args.geometry_weight or args.line_paint):
        parser.error("--baseline compares with the accepted chain, which uses W5's paint alone; "
                     "add --geometry-weight 0 and leave out --line-paint")

    started = perf_counter()
    artefacts_dir = args.output / "artefacts" if args.artefacts else None
    detector = CourtDetector(Switches(self_checks=not args.no_self_checks, timing=args.timing,
                                      artefacts_dir=artefacts_dir, upright_camera=not args.any_camera_roll,
                                      geometry_weight=args.geometry_weight, line_paint=args.line_paint))
    startup_seconds = perf_counter() - started
    verifier = detector.live.verifier
    sources, provenances, frame_paths = pack_sources(verifier.CASE_PACKS)
    manifest = {row["case_id"]: row for row in read_json_gz(MANIFEST)["cases"]}
    views = {view["case_id"]: view for view in json.loads(VIEWS.read_text())}
    shot_rows = {row["case_id"]: row for row in map(json.loads, SHOT_CHECK.read_text().splitlines())}
    feet_by_view = read_json_gz(args.feet) if args.feet is not None else {}

    people_records = {view_id: read_json_gz(args.people / f"{view_id}.json.gz") for view_id in args.views}
    keep_by_video: dict[str, set[int]] = {}
    for view_id, record in people_records.items():
        window = feet.window_frames(views[view_id]["anchor"], record["fps"], 0, record["frame_count"] - 1)
        keep_by_video.setdefault(record["video_path"], set()).update(window)
    videos = {}
    for path, keep in keep_by_video.items():
        decode_started = perf_counter()
        videos[path] = DecodedVideo(path, keep)
        print(f"decoded {path} to frame {max(keep)} in {perf_counter() - decode_started:.0f}s", flush=True)

    (args.output / "results").mkdir(parents=True, exist_ok=True)
    failed = []
    for view_id in args.views:
        record = people_records[view_id]
        video = videos[record["video_path"]]
        row = {"view_id": view_id, "error": None}
        try:
            if video.fps != record["fps"] or list(video.size) != record["video_size"]:
                raise ValueError(f"{view_id}: video fps or size differs from the people record")
            source = sources[view_id]
            frame_path = frame_paths.get(view_id) or verifier.frame_path(ROOT, source, provenances[view_id])
            if not args.no_self_checks:
                frame_md5 = hashlib.md5(frame_path.read_bytes()).hexdigest()
                if frame_md5 != manifest[view_id]["image_md5"]:
                    raise ValueError(f"{view_id}: frame MD5 differs from the manifest")
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(frame_path)
            view = ViewInputs(
                view_id=view_id, frame=frame, frame_index=views[view_id]["anchor"],
                scene_frames=(0, record["frame_count"] - 1),
                segments_px=np.asarray(source["segments_px"], dtype=float).reshape(-1, 4),
                person_boxes_px=np.asarray(source["bbox_px"], dtype=float).reshape(-1, 4),
                provenance=provenances[view_id],
            )
            detect_started = perf_counter()
            result = detector.detect(view, SavedPeople(record), video)
            row.update({"detect_seconds": perf_counter() - detect_started, "chosen_key": result.chosen_key,
                        "no_court_reason": result.no_court_reason, "stage_seconds": result.stage_seconds,
                        "corners_native_px": None if result.corners_native_px is None
                        else result.corners_native_px.tolist()})
            if args.baseline is not None:
                artefacts = read_json_gz(artefacts_dir / f"{view_id}.json.gz")
                checks = baseline_checks(view_id, result, artefacts, args.baseline, feet_by_view, shot_rows)
                row["checks"] = checks
                row["all_checks_equal"] = all(difference is None for difference in checks.values())
                if not row["all_checks_equal"]:
                    failed.append(view_id)
        except Exception as error:  # noqa: BLE001 - log this view and carry on with the rest
            traceback.print_exc()
            row["error"] = repr(error)
            failed.append(view_id)
        (args.output / "results" / f"{view_id}.json").write_text(json.dumps(row, indent=1))
        print(json.dumps({key: row.get(key) for key in ("view_id", "chosen_key", "no_court_reason",
                                                         "all_checks_equal", "error")}), flush=True)
    process = {"views": args.views, "startup_seconds": startup_seconds, "wall_seconds": perf_counter() - started,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "failed": failed}
    print(json.dumps(process), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
