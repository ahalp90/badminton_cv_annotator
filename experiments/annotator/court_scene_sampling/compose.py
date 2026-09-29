"""Compose one court per scene from the best-painted markings across full_three's accepted frames.

Players hide different markings in different frames, so each saved court can miss
paint that another frame shows. This experiment fits one extra court per scene from
the observed line samples of whichever accepted frame shows each marking best. It
then scores the saved courts and the composite on the same frames.

1. Reference. rescore.py's ``ranked`` choice, the net choice's combined score on the
   saved courts, sets the reference frame. It supplies coordinates and the fit's
   starting court, nothing more.
2. Alignment. Each other accepted frame is ECC-aligned to the reference inside its
   own saved court, as ``in_middle_frame`` does, but with both frames' person boxes left
   out. A warp is usable when its correlation reaches the reuse check's level; camera
   movement is allowed. Other frames are skipped.
3. Orientation. A court turned 180 degrees against the reference gets its corners
   rolled by two before measuring, so a marking name means the same painted line in
   every frame.
4. Evidence. Each used court's marking evidence and fragment assignments are measured
   again in its own frame, from the cached PNG, lines and same-frame person boxes.
5. Donors. Each marking comes from the used court with the most q_paint10 on it. Its
   observed fragment samples (stripe_fitting.prepare) outside person boxes are carried
   into reference pixels. Each keeps prepare's weight times the donor's q_paint10.
6. Fit. stripe_fitting.refine fits one court to the donated samples. The stripe refit's
   checks apply, plus the camera and upright checks. Player positions are not checked:
   the recovered people have boxes but no feet.
7. Comparison. The used frames' saved courts and the composite are carried into every
   used frame and scored there with the net choice's formula. The highest mean over
   those same frames wins, and an exact tie keeps a saved court.

Steps 2 to 6 call court_detector.composition, the detector's own composition code.
Scores measure detector evidence, not accuracy. The source results stay untouched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from annotator import court_views
from court_detector import reuse, stripe_fitting
from court_detector.composition import (
    HALF_TURN_ROLL,
    SearchedFrame,
    UsedFrame,
    align,
    corners_between,
    donated_samples,
    fit_in_reference,
    measure,
    use_frame,
)
from court_detector.detect import (
    MAX_HORIZON_TILT_DEG,
    LiveModules,
    freeze_arrays,
    load_live_modules,
)
from court_detector.inputs import same_frame_provenance

from . import rescore
from .render import draw_outline
from .sampling import FRAME_ROLES, read_json

COMPOSE_SCHEMA = "court-scene-compose/2"
PEOPLE_SCHEMA = "court-scene-people/1"
METHOD = "full_three"
COMPOSITE = "composite"
# Saved evidence fields that a fresh measurement of the same court should reproduce.
SCORE_FIELDS = ("q_paint10_span_weighted", "q_geom_span_weighted")
MARKING_COUNT_FIELDS = ("visible_samples", "known_photometry_samples", "exclusive_fragment_count")

COMPARISON_FIELDS = (
    "video_id", "scene_id", "state", "accepted_count", "reference_role", "used_roles", "skipped_frames",
    "half_turned_roles", "reproduction_max_score_difference", "reproduction_mismatches", "donors",
    "fit_status", "fit_valid", "fit_reason", "fit_sample_count", "per_frame_scores", "common_frame_means",
    "paint_choice", "own_frame_choice", "common_frame_original", "composite_beats_originals", "final", "changed",
    "final_corners_reference_native_px",
)


def read_people_cache(path: Path) -> dict[str, Any]:
    cache = read_json(path)
    if cache.get("schema") != PEOPLE_SCHEMA:
        raise ValueError(f"{path}: expected schema {PEOPLE_SCHEMA!r}, found {cache.get('schema')!r}")
    return cache


def candidate_label(role: str) -> str:
    return f"original_{role}"


def load_frame(live: LiveModules, saved: dict[str, Any], lines: dict[str, Any], people: dict[str, Any],
               frames_dir: Path, native_size: list[int]) -> SearchedFrame:
    """Rebuild a saved frame's court and measurement context from its cached PNG, lines and person boxes.

    The boxes were measured on the same image, so photometry hides them as the run did.
    No feet are known, so the player gates stay empty. rescore, not the frame's paint
    score, ranks the frames here.
    """
    view_id = saved["view_id"]
    path = frames_dir / f"{view_id}.png"
    native_frame = cv2.imread(str(path))
    if native_frame is None:
        raise FileNotFoundError(path)
    height, width = native_frame.shape[:2]
    sizes = {"PNG": [width, height], "lines": lines[view_id]["native_size"], "people": people[view_id]["native_size"]}
    for name, size in sizes.items():
        if list(size) != list(native_size):
            raise ValueError(f"{view_id}: {name} size {size} differs from the video's {native_size}")
    source = {"id": view_id, "dimensions": {"width": width, "height": height},
              "segments_px": lines[view_id]["segments_native_px"], "bbox_px": people[view_id]["boxes_native_px"],
              "all_feet_px": [], "provenance": {"people_source": "not_supplied"}}
    native_frame.flags.writeable = False
    provenance = same_frame_provenance(view_id, saved["frame_index"])
    context = live.verifier.view_context(view_id, source, provenance, native_frame, view_id)
    freeze_arrays(context)
    return SearchedFrame(saved["role"], native_frame, context, np.asarray(saved["corners_native_px"], dtype=float),
                         saved.get("paint_score"))


def reproduction(saved: dict[str, Any], measured: dict[str, Any]) -> dict[str, Any]:
    """How far a fresh measurement of a saved court is from its saved evidence.

    Recovered lines and boxes can differ from the run's, so differences are reported, not
    hidden. Score fields report their largest absolute difference. A score present on one
    side only, a differing sample count and differing occlusion awareness are listed by name.
    """
    scores = [(name, saved[name], measured[name]) for name in SCORE_FIELDS]
    mismatches = []
    if saved["photometry_occlusion_aware"] != measured["photometry_occlusion_aware"]:
        mismatches.append("photometry_occlusion_aware")
    for saved_marking, marking in zip(saved["markings"], measured["markings"], strict=True):
        name = marking["marking"]
        if saved_marking["marking"] != name:
            raise ValueError(f"saved marking {saved_marking['marking']!r} is where {name!r} belongs")
        scores.append((f"{name}.q_paint10", saved_marking["q_paint10"], marking["q_paint10"]))
        for count in MARKING_COUNT_FIELDS:
            if saved_marking[count] != marking[count]:
                mismatches.append(f"{name}.{count}")
    largest, largest_at = 0.0, None
    for name, before, after in scores:
        if before is None and after is None:
            continue
        if before is None or after is None:
            mismatches.append(name)
            continue
        if abs(after - before) > largest:
            largest, largest_at = abs(after - before), name
    return {"max_score_difference": largest, "largest_at": largest_at, "mismatches": mismatches}


def fit_composite(live: LiveModules, reference: UsedFrame, constraints: stripe_fitting.Constraints) -> dict[str, Any]:
    """Fit one court to the donated samples in reference px, then check it as reuse does.

    fit_geometry checks the solver, rank, depth, convexity and hard validity. Then the
    camera must be plausible and upright. Player positions are not checked.
    """
    context = reference.frame.context
    geometry = fit_in_reference(live, reference, constraints)
    record = {"status": geometry["status"], "sample_count": len(constraints.points), "fit": geometry["fit"],
              "valid": geometry["valid"], "validity_reason": geometry["validity_reason"],
              "corners_reference_native_px": geometry.get("corners_native_px")}
    if not geometry["valid"]:
        return record
    measurement = geometry["measurement"]
    upright = reuse.upright_court(np.asarray(geometry["homography_working"]),
                                  np.asarray(geometry["corners_working_px"]), context.size, MAX_HORIZON_TILT_DEG)
    record.update(camera_error=measurement["gates"]["camera_error"], upright=upright)
    if not measurement["historical"]["historical_camera"]:
        record.update(valid=False, validity_reason="camera_implausible")
    elif not upright:
        record.update(valid=False, validity_reason="camera_not_upright")
    return record


def compose_court(live: LiveModules, used: list[UsedFrame]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose each marking's donor, gather the donated samples and fit the composite.

    :return: One row per marking, and the fit record.
    """
    rows, constraints = donated_samples(used)
    return rows, fit_composite(live, used[0], constraints)


def score_in(live: LiveModules, target: UsedFrame, corners_native: np.ndarray) -> dict[str, Any]:
    """One court's combined score in one frame, measured on that frame's lines, paint and boxes."""
    evidence = measure(live, target.frame, corners_native)
    net = rescore.net_evidence(corners_native, target.frame.context)
    paint, geometry = evidence["q_paint10_span_weighted"], evidence["q_geom_span_weighted"]
    row: dict[str, Any] = {"frame_role": target.frame.role, "corners_native_px": np.asarray(corners_native).tolist(),
                           "paint_score": paint, "geometry_score": geometry, "net_state": net["net_state"],
                           "net_reward": net["net_reward"], "missing": []}
    if paint is None:
        row["missing"].append("paint_score")
    if geometry is None:
        row["missing"].append("geometry_score")
    if not row["missing"]:
        row.update(rescore.score_parts(paint, geometry, net["net_reward"]))
    return row


def evaluate(live: LiveModules, used: list[UsedFrame], skipped: list[dict[str, Any]],
             composite_corners: list | None) -> list[dict[str, Any]]:
    """Every candidate scored in every used frame, and its mean over exactly those frames.

    A candidate missing a score in any frame, or whose frame did not align, has no mean.
    """
    sources = [(candidate_label(item.frame.role), item, item.corners_native) for item in used]
    if composite_corners is not None:
        sources.append((COMPOSITE, used[0], np.asarray(composite_corners, dtype=float)))
    rows = []
    for label, source, corners in sources:
        scores = [score_in(live, target, corners_between(corners, source, target)) for target in used]
        complete = all("combined_score" in score for score in scores)
        mean = float(np.mean([score["combined_score"] for score in scores])) if complete else None
        role = None if label == COMPOSITE else source.frame.role
        rows.append({"candidate": label, "role": role, "scores": scores, "mean_combined_score": mean})
    for frame_row in skipped:
        rows.append({"candidate": candidate_label(frame_row["role"]), "role": frame_row["role"], "scores": [],
                     "mean_combined_score": None, "unscored_reason": frame_row["alignment"]["skip_reason"]})
    return rows


def decide(evaluated: list[dict[str, Any]], own_frame_label: str) -> dict[str, Any]:
    """The highest common-frame mean wins.

    An exact tie keeps a saved court, and ties between saved courts go middle, first, last.

    If any saved court lacks a mean, the own-frame full-score choice stands: choosing among the
    scored courts alone would compare a different set of saved courts.
    """
    originals = [row for row in evaluated if row["candidate"] != COMPOSITE]
    composite = next((row for row in evaluated if row["candidate"] == COMPOSITE), None)
    if any(row["mean_combined_score"] is None for row in originals):
        return {"state": "original_scores_incomplete", "common_frame_original": None,
                "composite_beats_originals": None, "final": own_frame_label}
    best = min(originals, key=lambda row: (-row["mean_combined_score"], FRAME_ROLES.index(row["role"])))
    beats = None
    if composite is not None and composite["mean_combined_score"] is not None:
        beats = composite["mean_combined_score"] > best["mean_combined_score"]
    return {"state": "compared", "common_frame_original": best["candidate"], "composite_beats_originals": beats,
            "final": COMPOSITE if beats else best["candidate"]}


def prepare_frames(live: LiveModules, video: dict[str, Any], outcome: dict[str, Any], order: list[int],
                   lines: dict[str, Any], people: dict[str, Any],
                   frames_dir: Path) -> tuple[list[dict[str, Any]], list[UsedFrame]]:
    """Load, check and align every accepted frame.

    :param order: Accepted frame positions in own-frame score order; the first is the reference.
    :return: One record per accepted frame, and the frames the composition uses, both in that order.
    """
    frame_rows, used = [], []
    reference: SearchedFrame | None = None
    for position in order:
        saved = outcome["frames"][position]
        frame = load_frame(live, saved, lines, people, frames_dir, video["native_size"])
        row: dict[str, Any] = {"position": position, "role": frame.role, "view_id": saved["view_id"],
                               "frame_index": saved["frame_index"], "is_reference": reference is None,
                               "reproduction": reproduction(saved["evidence"],
                                                            measure(live, frame, frame.corners_native))}
        if reference is None:
            reference = frame
            row["alignment"], to_reference = None, np.eye(3)
        else:
            row["alignment"], to_reference = align(frame, reference)
        row["half_turn_roll"] = None
        if to_reference is not None:
            reference_working = reference.corners_native / reference.native_per_working
            item, row["half_turn_roll"] = use_frame(live, frame, to_reference, reference_working)
            used.append(item)
        frame_rows.append(row)
    return frame_rows, used


def write_outlines(directory: Path, scene_id: str, reference: UsedFrame, evaluated: list[dict[str, Any]]) -> list[str]:
    """Each compared court as a plain 1 px red dashed outline on the reference frame, one image per court."""
    directory.mkdir(parents=True, exist_ok=True)
    names = []
    for row in evaluated:
        if not row["scores"]:
            continue  # an unaligned frame's court cannot be carried into the reference
        image = reference.frame.native_frame.copy()
        draw_outline(image, row["scores"][0]["corners_native_px"])
        name = f"{scene_id}__{row['candidate']}__on_{reference.frame.role}.png"
        if not cv2.imwrite(str(directory / name), image):
            raise OSError(f"Could not write {name}")
        names.append(name)
    return names


def final_fields(choice: dict[str, Any], evaluated: list[dict[str, Any]], outcome: dict[str, Any],
                 reference_view_id: str) -> None:
    """Add the final court's frame and corners, and its corners in the reference frame.

    A saved court keeps its saved corner order in its own frame. The composite's own
    frame is the reference. A court whose frame did not align has no reference corners.
    """
    row = next(item for item in evaluated if item["candidate"] == choice["final"])
    in_reference = row["scores"][0]["corners_native_px"] if row["scores"] else None
    if choice["final"] == COMPOSITE:
        view_id, corners = reference_view_id, in_reference
    else:
        saved = next(frame for frame in outcome["frames"] if candidate_label(frame["role"]) == choice["final"])
        view_id, corners = saved["view_id"], saved["corners_native_px"]
    choice.update(final_view_id=view_id, final_corners_native_px=corners,
                  final_corners_reference_native_px=in_reference)


def compose_scene(live: LiveModules, video: dict[str, Any], scene: dict[str, Any], lines: dict[str, Any],
                  people: dict[str, Any], frames_dir: Path, outlines_dir: Path) -> dict[str, Any]:
    """One scene's composite, its comparison with the saved courts, and the final choice."""
    record: dict[str, Any] = {"video_id": video["video_id"], "scene_id": scene["scene_id"],
                              "scene_status": scene["status"]}
    if scene["status"] != "analysed":
        record["state"] = "no_evaluation"
        return record
    outcome = scene["methods"][METHOD]
    rescore.check_lines_size(video, scene, lines)
    rescored = rescore.rescore_method(outcome, lines)
    record["rescore"] = rescored
    if rescored["state"] != "rescored":
        # no_evaluation, no_court or score_evidence_missing: the saved outcome stands.
        record["state"] = rescored["state"]
        if "selected" in rescored:
            paint_label = candidate_label(outcome["frames"][rescored["selected"]["paint"]]["role"])
            record["choice"] = {"paint": paint_label, "own_frame": None, "final": paint_label, "changed": False}
        return record
    paint_label = candidate_label(outcome["frames"][rescored["selected"]["paint"]]["role"])
    record["choice"] = {"paint": paint_label,
                        "own_frame": candidate_label(outcome["frames"][rescored["selected"]["ranked"]]["role"])}
    record["frames"], used = prepare_frames(live, video, outcome, rescored["ranked_order"], lines, people,
                                            frames_dir)
    record["reference"] = record["frames"][0]["role"]
    if len(used) < 2:
        record["state"] = "too_few_aligned_frames"
        own_frame_label = record["choice"]["own_frame"]
        own_frame = outcome["frames"][rescored["selected"]["ranked"]]
        record["choice"].update(final=own_frame_label, changed=own_frame_label != paint_label,
                                 final_view_id=own_frame["view_id"],
                                 final_corners_native_px=own_frame["corners_native_px"],
                                 final_corners_reference_native_px=own_frame["corners_native_px"])
        return record
    record["state"] = "composed"
    record["markings"], record["fit"] = compose_court(live, used)
    skipped = [row for row in record["frames"] if row["alignment"] is not None and not row["alignment"]["usable"]]
    composite_corners = record["fit"]["corners_reference_native_px"] if record["fit"]["valid"] else None
    evaluated = evaluate(live, used, skipped, composite_corners)
    record["evaluation"] = {"frames": [item.frame.role for item in used], "candidates": evaluated}
    choice = decide(evaluated, record["choice"]["own_frame"])
    final_fields(choice, evaluated, outcome, record["frames"][0]["view_id"])
    record["choice"].update(choice, changed=choice["final"] != paint_label)
    if composite_corners is not None:
        record["outlines"] = write_outlines(outlines_dir, scene["scene_id"], used[0], evaluated)
    return record


def compose(live: LiveModules, results: dict[str, Any], lines: dict[str, Any], people: dict[str, Any],
            frames_dir: Path, outlines_dir: Path) -> list[dict[str, Any]]:
    """One record per scene. Reads the results and caches without changing them."""
    rows = []
    for video in results["videos"]:
        for scene in video["scenes"] + video["later_scenes"]:
            started = perf_counter()
            row = compose_scene(live, video, scene, lines, people, frames_dir, outlines_dir)
            row["seconds"] = perf_counter() - started
            print(f"{scene['scene_id']}: {row['state']}, final {row.get('choice', {}).get('final')}", flush=True)
            rows.append(row)
    return rows


def joined_text(items: list[str]) -> str:
    return ";".join(items)


def comparison_row(scene: dict[str, Any]) -> dict[str, Any]:
    """One compact CSV row per scene."""
    row: dict[str, Any] = {"video_id": scene["video_id"], "scene_id": scene["scene_id"], "state": scene["state"]}
    choice = scene.get("choice")
    if choice is not None:
        row.update(paint_choice=choice["paint"], own_frame_choice=choice["own_frame"], final=choice["final"],
                   changed=choice["changed"])
    if "rescore" in scene and "candidates" in scene["rescore"]:
        row["accepted_count"] = len(scene["rescore"]["candidates"])
    if "frames" not in scene:
        return row
    frames = scene["frames"]
    row.update(
        reference_role=scene["reference"],
        used_roles=joined_text([item["role"] for item in frames if item["half_turn_roll"] is not None]),
        skipped_frames=joined_text([f"{item['role']}:{item['alignment']['skip_reason']}" for item in frames
                                    if item["half_turn_roll"] is None]),
        half_turned_roles=joined_text([item["role"] for item in frames if item["half_turn_roll"] == HALF_TURN_ROLL]),
        reproduction_max_score_difference=max(item["reproduction"]["max_score_difference"] for item in frames),
        reproduction_mismatches=joined_text([f"{item['role']}:{name}" for item in frames
                                             for name in item["reproduction"]["mismatches"]]),
    )
    if scene["state"] != "composed":
        return row
    fit = scene["fit"]
    scores, means = [], []
    for candidate in scene["evaluation"]["candidates"]:
        means.append(f"{candidate['candidate']}={candidate['mean_combined_score']}")
        for score in candidate["scores"]:
            scores.append(f"{candidate['candidate']}@{score['frame_role']}={score.get('combined_score')}")
    row.update(
        donors=joined_text([f"{item['marking']}:{item['donor_role']}:{item['q_paint10']}:{item['samples']}"
                            for item in scene["markings"]]),
        fit_status=fit["status"], fit_valid=fit["valid"], fit_reason=fit["validity_reason"],
        fit_sample_count=fit["sample_count"], per_frame_scores=joined_text(scores),
        common_frame_means=joined_text(means),
        common_frame_original=choice["common_frame_original"],
        composite_beats_originals=choice["composite_beats_originals"],
        final_corners_reference_native_px=json.dumps(choice["final_corners_reference_native_px"]),
    )
    return row


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    with gzip.open(output_dir / "results.json.gz", "wt") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
    with gzip.open(output_dir / "comparison.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_row(scene) for scene in report["scenes"])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="the comparison's results.json.gz")
    parser.add_argument("--lines-cache", type=Path, required=True, help="cached DeepLSD lines by view_id")
    parser.add_argument("--people-cache", type=Path, required=True, help="cached same-frame person boxes by view_id")
    parser.add_argument("--frames-dir", type=Path, required=True, help="cached source PNGs named <view_id>.png")
    parser.add_argument("--output-dir", type=Path, required=True, help="a new directory for the outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    results = read_json(args.results)
    if results["schema"] != rescore.SOURCE_SCHEMA:
        raise ValueError(f"{args.results}: expected schema {rescore.SOURCE_SCHEMA!r}, found {results['schema']!r}")
    lines = rescore.read_lines_cache(args.lines_cache)
    people = read_people_cache(args.people_cache)
    live = load_live_modules()
    args.output_dir.mkdir(parents=True)
    started = perf_counter()
    scenes = compose(live, results, lines["frames"], people["frames"], args.frames_dir, args.output_dir / "outlines")
    report = {
        "schema": COMPOSE_SCHEMA, "source_results": str(args.results), "source_finished": results["finished"],
        "lines_cache": str(args.lines_cache), "people_cache": str(args.people_cache),
        "frames_dir": str(args.frames_dir),
        # Kept apart from the source run's timing: none of these is part of an arm's measured time.
        "line_recovery_seconds": lines["recovery_seconds"], "people_recovery_seconds": people["recovery_seconds"],
        "compose_seconds": perf_counter() - started,
        "rule": {"method": METHOD, "geometry_weight": rescore.GEOMETRY_WEIGHT, "net_weight": rescore.NET_WEIGHT,
                 "net_overrun_working_px": rescore.NET_OVERRUN_WORKING_PX, "frame_tie_order": list(FRAME_ROLES),
                 "min_alignment_correlation": court_views.MIN_ALIGNMENT_CORRELATION,
                 "alignment_mask": "frame's saved court x1.08 without either frame's person boxes",
                 "max_horizon_tilt_deg": MAX_HORIZON_TILT_DEG,
                 "donor_weight": "stripe_fitting.prepare weight x donor q_paint10"},
        "scenes": scenes,
    }
    write_outputs(args.output_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
