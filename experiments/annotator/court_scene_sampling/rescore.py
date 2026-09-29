"""Rescore a finished comparison's saved courts with the net choice's formula, plus a boundary guard.

Each method's accepted frames are ranked three ways:

- ``paint``: the saved choice, the highest final paint score.
- ``ranked``: the net choice's combined score on each saved final court:
  evidence score (paint blended with line support) plus the net weight times the
  post reward. Net posts are measured again from the frame's cached DeepLSD lines.
- ``guarded``: the ranked order, where a court loses to a lower-ranked court that
  scores within GUARD_MAX_GAIN of it but shows much more paint on the same outer
  boundary.

Normal selection applies the formula before the stripe refit; here it applies to
the final courts. This is a comparison within the saved candidates, not a replay
of the detector's choice or of the run's history. The source results stay untouched.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np

from court_detector import measurements, net_choice
from court_detector.detect import NET_OVERRUN_WORKING_PX, NET_WEIGHT, Switches

from .sampling import FRAME_ROLES, METHODS, preference_key, read_json

SOURCE_SCHEMA = "court-scene-sampling-results/1"
LINES_SCHEMA = "court-scene-lines/1"
RESCORE_SCHEMA = "court-scene-rescore/1"
GEOMETRY_WEIGHT = Switches().geometry_weight

# Provisional guard values, fixed in advance; not tuned.
GUARD_MAX_GAIN = 0.01  # combined-score lead small enough to question
GUARD_MIN_DROP = 0.10  # absolute q_paint10 a boundary must lose to veto
GUARD_MIN_KNOWN_SAMPLES = 32  # samples with known photometry, of the 64 per boundary
GUARD_MIN_KNOWN_FRACTION = 0.75  # known over visible samples
GUARD_MIN_SPAN_WORKING_PX = 50.0
GUARD_MAX_CORNER_FRACTION = 0.15  # of the lower-ranked court's median edge length, in middle-frame pixels

# Outer boundaries by their ends in the saved TL TR BR BL corner order (CORNER_COURT_M:
# x = 0 is the left doubles line, y = 0 the far baseline).
OUTER_BOUNDARIES = {"far_baseline": (0, 1), "right_doubles": (1, 2), "near_baseline": (2, 3), "left_doubles": (3, 0)}
BOUNDARY_BY_CORNERS = {frozenset(corners): name for name, corners in OUTER_BOUNDARIES.items()}
# Cyclic corner rolls that keep the same physical court: as saved, or turned 180 degrees.
CORNER_ROLLS = (0, 2)

COMPARISON_FIELDS = (
    "video_id", "scene_id", "method", "state", "accepted_count", "missing_score_evidence",
    "paint_role", "paint_frame", "ranked_role", "ranked_frame", "guarded_role", "guarded_frame",
    "ranked_changed", "guarded_changed", "guard_vetoes", "guard_skipped_pairs",
    "paint_combined", "ranked_combined", "guarded_combined",
    "paint_vs_baseline_px", "ranked_vs_baseline_px", "guarded_vs_baseline_px",
)


def read_lines_cache(path: Path) -> dict[str, Any]:
    cache = read_json(path)
    if cache.get("schema") != LINES_SCHEMA:
        raise ValueError(f"{path}: expected schema {LINES_SCHEMA!r}, found {cache.get('schema')!r}")
    return cache


def net_context(entry: dict[str, Any]) -> SimpleNamespace:
    """The detector context fields net_choice.net_posts reads, prepared as view_context prepares them."""
    width, height = entry["native_size"]
    source = {"dimensions": {"width": width, "height": height}, "segments_px": entry["segments_native_px"]}
    segments, _, size = measurements.prepare_segments(source)
    return SimpleNamespace(segments=segments, size=size, native_size=(width, height))


def boundary_evidence(evidence: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Each outer boundary's paint support and whether it has enough visible, unoccluded evidence for the guard."""
    markings = {} if evidence is None else {marking["marking"]: marking for marking in evidence["markings"]}
    occlusion_aware = evidence is not None and evidence.get("photometry_occlusion_aware") is True
    boundaries = {}
    for name in OUTER_BOUNDARIES:
        marking = markings.get(name)
        if marking is None:
            boundaries[name] = {"eligible": False, "reasons": ["marking_evidence_missing"]}
            continue
        visible, known = marking["visible_samples"], marking["known_photometry_samples"]
        span, q_paint = marking["projected_visible_span_px"], marking["q_paint10"]
        reasons = []
        if not occlusion_aware:
            reasons.append("photometry_not_occlusion_aware")
        if q_paint is None:
            reasons.append("q_paint10_missing")
        if known < GUARD_MIN_KNOWN_SAMPLES:
            reasons.append("too_few_known_samples")
        if not visible or known / visible < GUARD_MIN_KNOWN_FRACTION:
            reasons.append("known_fraction_too_low")
        if span < GUARD_MIN_SPAN_WORKING_PX:
            reasons.append("span_too_short")
        boundaries[name] = {"q_paint10": q_paint, "visible_samples": visible, "known_samples": known,
                            "span_working_px": span, "eligible": not reasons, "reasons": reasons}
    return boundaries


def post_summary(posts: dict[str, dict]) -> dict[str, dict[str, Any]]:
    summary = {}
    for name, feature in posts.items():
        summary[name] = {"visible_count": feature["visible_count"], "covered_count": feature["covered_count"],
                         "lowest_endpoint_offset_working_px": feature["lowest_endpoint_offset_working_px"],
                         "supported": net_choice.supported(feature, NET_OVERRUN_WORKING_PX)}
    return summary


def score_candidate(position: int, frame: dict[str, Any], lines: dict[str, Any]) -> dict[str, Any]:
    """One accepted frame's combined score, or the evidence it lacks. A missing score never counts as zero."""
    evidence = frame.get("evidence")
    geometry = None if evidence is None else evidence.get("q_geom_span_weighted")
    paint = frame["paint_score"]
    placed = frame.get("in_middle_frame") or {}
    candidate: dict[str, Any] = {
        "position": position, "role": frame["role"], "frame_index": frame["frame_index"],
        "view_id": frame["view_id"], "route": frame["route"], "paint_score": paint, "geometry_score": geometry,
        "comparable": placed.get("comparable") is True, "middle_frame_corners_px": placed.get("corners_native_px"),
        "vs_baseline_px": (frame.get("vs_baseline") or {}).get("max_corner_px"),
        "boundaries": boundary_evidence(evidence), "missing": [],
    }
    if paint is None:
        candidate["missing"].append("paint_score")
    if geometry is None:
        candidate["missing"].append("geometry_score")
    entry = lines.get(frame["view_id"])
    if entry is None:
        candidate["missing"].append("lines")
        candidate["net_state"] = "lines_missing"
    else:
        net_state, posts = net_choice.net_posts(frame["corners_native_px"], net_context(entry))
        candidate["net_state"] = net_state  # projection_failed keeps production's zero reward
        candidate["posts"] = post_summary(posts)
        candidate["net_reward"] = net_choice.net_reward(net_state, posts, NET_OVERRUN_WORKING_PX)
    if candidate["missing"]:
        return candidate
    evidence_value = net_choice.evidence_score({"paint_score": paint, "geometry_score": geometry}, GEOMETRY_WEIGHT)
    candidate["evidence_score"] = evidence_value
    candidate["net_bonus"] = NET_WEIGHT * candidate["net_reward"]
    candidate["combined_score"] = evidence_value + candidate["net_bonus"]
    return candidate


def corner_match(upper: np.ndarray, lower: np.ndarray) -> dict[str, Any]:
    """The roll of the lower court's corners closest to the upper court's, and whether it is close enough.

    Rolled corner i is lower corner (i - roll) % 4. An exact distance tie keeps roll 0.
    """
    best_roll, best_distance = 0, np.inf
    for roll in CORNER_ROLLS:
        distance = float(np.linalg.norm(np.roll(lower, roll, axis=0) - upper, axis=1).max())
        if distance < best_distance:
            best_roll, best_distance = roll, distance
    edge_lengths = np.linalg.norm(np.roll(lower, -1, axis=0) - lower, axis=1)  # one per boundary
    limit = GUARD_MAX_CORNER_FRACTION * float(np.median(edge_lengths))
    return {"roll": best_roll, "max_corner_px": best_distance, "limit_px": limit, "close": best_distance <= limit}


def pair_check(upper: dict[str, Any], lower: dict[str, Any]) -> dict[str, Any]:
    """Whether the lower-ranked court vetoes the upper one on a matched outer boundary."""
    gain = upper["combined_score"] - lower["combined_score"]
    check: dict[str, Any] = {"candidate": upper["position"], "against": lower["position"], "gain": gain,
                             "veto": False, "skipped": None}
    if gain > GUARD_MAX_GAIN:
        check["skipped"] = "gain_above_limit"
        return check
    if not (upper["comparable"] and lower["comparable"]):
        check["skipped"] = "not_comparable_in_middle_frame"
        return check
    match = corner_match(np.asarray(upper["middle_frame_corners_px"], dtype=float),
                         np.asarray(lower["middle_frame_corners_px"], dtype=float))
    check["orientation"] = match
    if not match["close"]:
        check["skipped"] = "corners_too_far_apart"
        return check
    check["boundaries"] = []
    for name, (start, end) in OUTER_BOUNDARIES.items():
        lower_name = BOUNDARY_BY_CORNERS[frozenset(((start - match["roll"]) % 4, (end - match["roll"]) % 4))]
        upper_side, lower_side = upper["boundaries"][name], lower["boundaries"][lower_name]
        comparison: dict[str, Any] = {"candidate_boundary": name, "against_boundary": lower_name,
                                      "eligible": upper_side["eligible"] and lower_side["eligible"], "veto": False}
        if comparison["eligible"]:
            drop = lower_side["q_paint10"] - upper_side["q_paint10"]
            comparison["q_paint10_drop"] = drop
            comparison["veto"] = drop >= GUARD_MIN_DROP
        check["boundaries"].append(comparison)
        check["veto"] = check["veto"] or comparison["veto"]
    return check


def ranked_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Highest combined score first; exact ties go middle, then first, then last."""
    return sorted(candidates, key=lambda candidate: (-candidate["combined_score"],
                                                     FRAME_ROLES.index(candidate["role"])))


def guarded_choice(ranked: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The first court in ranked order that no lower-ranked court vetoes, and every pair check made.

    The lowest-ranked court has nothing below it, so a choice always exists.
    """
    checks = []
    for index, upper in enumerate(ranked):
        vetoed = False
        for lower in ranked[index + 1:]:
            check = pair_check(upper, lower)
            checks.append(check)
            vetoed = vetoed or check["veto"]
        if not vetoed:
            return upper, checks
    raise AssertionError("the lowest-ranked court cannot be vetoed")


def paint_position(frames: list[dict[str, Any]]) -> int:
    """The saved choice, checked against the run's own rule so a misread schema fails here."""
    accepted = [position for position, frame in enumerate(frames) if frame["status"] == "court"]
    return max(accepted, key=lambda position: preference_key(frames[position]["paint_score"], frames[position]["role"]))


def rescore_method(outcome: dict[str, Any], lines: dict[str, Any]) -> dict[str, Any]:
    """One method's three choices on one scene. Methods without an accepted court stay as they were."""
    if "frames" not in outcome:
        return {"state": "no_evaluation", "original_status": outcome["status"]}
    if outcome["status"] == "no_court":
        return {"state": "no_court", "original_status": "no_court"}
    frames = outcome["frames"]
    original = outcome["chosen_position"]
    if paint_position(frames) != original:
        raise ValueError(f"saved chosen_position {original} does not match the highest final paint score")
    candidates = []
    for position, frame in enumerate(frames):
        if frame["status"] == "court":
            candidates.append(score_candidate(position, frame, lines))
    record: dict[str, Any] = {"state": "rescored", "original_status": "court", "candidates": candidates,
                              "selected": {"paint": original, "ranked": original, "guarded": original}}
    missing = [{"position": candidate["position"], "view_id": candidate["view_id"], "missing": candidate["missing"]}
               for candidate in candidates if candidate["missing"]]
    record["missing_score_evidence"] = missing
    if missing:
        # Ranking only the scored subset would compare different candidate sets across methods.
        record["state"] = "score_evidence_missing"
        record["guard_checks"] = []
        record["changes"] = {"ranked": False, "guarded": False}
        return record
    ranked = ranked_order(candidates)
    guarded, checks = guarded_choice(ranked)
    record["ranked_order"] = [candidate["position"] for candidate in ranked]
    record["selected"]["ranked"] = ranked[0]["position"]
    record["selected"]["guarded"] = guarded["position"]
    record["guard_checks"] = checks
    record["changes"] = {"ranked": ranked[0]["position"] != original, "guarded": guarded["position"] != original}
    return record


def check_lines_size(video: dict[str, Any], scene: dict[str, Any], lines: dict[str, Any]) -> None:
    """Cached lines must come from this video's frame size, or the net posts would be measured on the wrong scale."""
    for method in METHODS:
        for frame in scene["methods"][method].get("frames", []):
            entry = lines.get(frame["view_id"])
            if entry is not None and list(entry["native_size"]) != list(video["native_size"]):
                raise ValueError(f"{frame['view_id']}: cached size {entry['native_size']} "
                                 f"differs from the video's {video['native_size']}")


def rescore(results: dict[str, Any], cache: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per scene, holding each method's rescoring. Reads the results without changing them."""
    lines = cache["frames"]
    rows = []
    for video in results["videos"]:
        for scene in video["scenes"] + video["later_scenes"]:
            row = {"video_id": video["video_id"], "scene_id": scene["scene_id"], "scene_status": scene["status"]}
            if scene["status"] == "analysed":
                check_lines_size(video, scene, lines)
                row["methods"] = {method: rescore_method(scene["methods"][method], lines) for method in METHODS}
            else:
                row["methods"] = {method: {"state": "no_evaluation", "original_status": scene["status"]}
                                  for method in METHODS}
            rows.append(row)
    return rows


def choice_fields(method: dict[str, Any], choice: str) -> dict[str, Any]:
    position = method["selected"][choice]
    candidate = next(item for item in method["candidates"] if item["position"] == position)
    return {f"{choice}_role": candidate["role"], f"{choice}_frame": candidate["frame_index"],
            f"{choice}_combined": candidate.get("combined_score"), f"{choice}_vs_baseline_px": candidate["vs_baseline_px"]}


def comparison_rows(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One compact row per scene and method, for the comparison table."""
    table = []
    for scene in scenes:
        for method_name, method in scene["methods"].items():
            row: dict[str, Any] = {"video_id": scene["video_id"], "scene_id": scene["scene_id"],
                                   "method": method_name, "state": method["state"]}
            if method["state"] in ("rescored", "score_evidence_missing"):
                missing = [f"{item['view_id']}:{'+'.join(item['missing'])}" for item in method["missing_score_evidence"]]
                vetoes = [f"{check['candidate']}>{check['against']}" for check in method["guard_checks"]
                          if check["veto"]]
                skipped = [f"{check['candidate']}>{check['against']}:{check['skipped']}"
                           for check in method["guard_checks"] if check["skipped"]]
                row.update(accepted_count=len(method["candidates"]), missing_score_evidence=";".join(missing),
                           ranked_changed=method["changes"]["ranked"], guarded_changed=method["changes"]["guarded"],
                           guard_vetoes=";".join(vetoes), guard_skipped_pairs=";".join(skipped))
                for choice in ("paint", "ranked", "guarded"):
                    row.update(choice_fields(method, choice))
            table.append(row)
    return table


def write_outputs(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True)
    with gzip.open(output_dir / "results.json.gz", "wt") as stream:
        json.dump(report, stream, allow_nan=False, separators=(",", ":"))
    with gzip.open(output_dir / "comparison.csv.gz", "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(comparison_rows(report["scenes"]))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="the comparison's results.json.gz")
    parser.add_argument("--lines-cache", type=Path, required=True, help="cached DeepLSD lines by view_id")
    parser.add_argument("--output-dir", type=Path, required=True, help="a new directory for the rescored outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"{args.output_dir} already exists")
    results = read_json(args.results)
    if results["schema"] != SOURCE_SCHEMA:
        raise ValueError(f"{args.results}: expected schema {SOURCE_SCHEMA!r}, found {results['schema']!r}")
    cache = read_lines_cache(args.lines_cache)
    started = perf_counter()
    scenes = rescore(results, cache)
    report = {
        "schema": RESCORE_SCHEMA, "source_results": str(args.results), "source_finished": results["finished"],
        "lines_cache": str(args.lines_cache),
        # Kept apart from the source run's timing: neither is part of an arm's measured time.
        "line_recovery_seconds": cache["recovery_seconds"], "rescore_seconds": perf_counter() - started,
        "rule": {"geometry_weight": GEOMETRY_WEIGHT, "net_weight": NET_WEIGHT,
                 "net_overrun_working_px": NET_OVERRUN_WORKING_PX, "frame_tie_order": list(FRAME_ROLES),
                 "guard": {"max_gain": GUARD_MAX_GAIN, "min_drop": GUARD_MIN_DROP,
                           "min_known_samples": GUARD_MIN_KNOWN_SAMPLES,
                           "min_known_fraction": GUARD_MIN_KNOWN_FRACTION,
                           "min_span_working_px": GUARD_MIN_SPAN_WORKING_PX,
                           "max_corner_fraction": GUARD_MAX_CORNER_FRACTION}},
        "scenes": scenes,
    }
    write_outputs(args.output_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
