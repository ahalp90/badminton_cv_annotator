"""Rescore a finished comparison's saved courts with the net choice's formula.

Each method's accepted frames are ranked two ways:

- ``paint``: the saved choice, the highest final paint score.
- ``ranked``: the net choice's combined score on each saved final court:
  evidence score (paint blended with line support) plus the net weight times the
  post reward. Net posts are measured again from the frame's cached DeepLSD lines.

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

from court_detector import composition, measurements, net_choice
from court_detector.detect import NET_OVERRUN_WORKING_PX, NET_WEIGHT, Switches

from .sampling import FRAME_ROLES, METHODS, preference_key, read_json

SOURCE_SCHEMA = "court-scene-sampling-results/1"
LINES_SCHEMA = "court-scene-lines/1"
RESCORE_SCHEMA = "court-scene-rescore/2"
GEOMETRY_WEIGHT = Switches().geometry_weight

COMPARISON_FIELDS = (
    "video_id", "scene_id", "method", "state", "accepted_count", "missing_score_evidence",
    "paint_role", "paint_frame", "ranked_role", "ranked_frame", "ranked_changed",
    "paint_combined", "ranked_combined", "paint_vs_baseline_px", "ranked_vs_baseline_px",
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


def post_summary(posts: dict[str, dict]) -> dict[str, dict[str, Any]]:
    summary = {}
    for name, feature in posts.items():
        summary[name] = {"visible_count": feature["visible_count"], "covered_count": feature["covered_count"],
                         "lowest_endpoint_offset_working_px": feature["lowest_endpoint_offset_working_px"],
                         "supported": net_choice.supported(feature, NET_OVERRUN_WORKING_PX)}
    return summary


def net_evidence(corners_native: list | np.ndarray, context: Any) -> dict[str, Any]:
    """A court's net posts measured against the context's lines, and production's post reward.

    :param context: Anything with net_choice's segments, size and native_size, such as a ViewContext.
    :return: The projection state, each post's evidence and the reward; projection_failed
        keeps production's zero reward.
    """
    net_state, posts = net_choice.net_posts(corners_native, context)
    return {"net_state": net_state, "posts": post_summary(posts),
            "net_reward": net_choice.net_reward(net_state, posts, NET_OVERRUN_WORKING_PX)}


def score_parts(paint: float, geometry: float, net_reward: float) -> dict[str, float]:
    """The net choice's combined score at the detector's default geometry weight."""
    return composition.score_parts(paint, geometry, net_reward, GEOMETRY_WEIGHT)


def score_candidate(position: int, frame: dict[str, Any], lines: dict[str, Any]) -> dict[str, Any]:
    """One accepted frame's combined score, or the evidence it lacks. A missing score never counts as zero."""
    evidence = frame.get("evidence")
    geometry = None if evidence is None else evidence.get("q_geom_span_weighted")
    paint = frame["paint_score"]
    candidate: dict[str, Any] = {
        "position": position, "role": frame["role"], "frame_index": frame["frame_index"],
        "view_id": frame["view_id"], "route": frame["route"], "paint_score": paint, "geometry_score": geometry,
        "vs_baseline_px": (frame.get("vs_baseline") or {}).get("max_corner_px"), "missing": [],
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
        candidate.update(net_evidence(frame["corners_native_px"], net_context(entry)))
    if candidate["missing"]:
        return candidate
    candidate.update(score_parts(paint, geometry, candidate["net_reward"]))
    return candidate


def ranked_order(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Highest combined score first; exact ties go middle, then first, then last."""
    return sorted(candidates, key=lambda candidate: (-candidate["combined_score"],
                                                     FRAME_ROLES.index(candidate["role"])))


def paint_position(frames: list[dict[str, Any]]) -> int:
    """The saved choice, checked against the run's own rule so a misread schema fails here."""
    accepted = [position for position, frame in enumerate(frames) if frame["status"] == "court"]
    return max(accepted, key=lambda position: preference_key(frames[position]["paint_score"], frames[position]["role"]))


def rescore_method(outcome: dict[str, Any], lines: dict[str, Any]) -> dict[str, Any]:
    """One method's two choices on one scene. Methods without an accepted court stay as they were."""
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
                              "selected": {"paint": original, "ranked": original}}
    missing = [{"position": candidate["position"], "view_id": candidate["view_id"], "missing": candidate["missing"]}
               for candidate in candidates if candidate["missing"]]
    record["missing_score_evidence"] = missing
    if missing:
        # Ranking only the scored subset would compare different candidate sets across methods.
        record["state"] = "score_evidence_missing"
        record["changes"] = {"ranked": False}
        return record
    ranked = ranked_order(candidates)
    record["ranked_order"] = [candidate["position"] for candidate in ranked]
    record["selected"]["ranked"] = ranked[0]["position"]
    record["changes"] = {"ranked": ranked[0]["position"] != original}
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
                row.update(accepted_count=len(method["candidates"]), missing_score_evidence=";".join(missing),
                           ranked_changed=method["changes"]["ranked"])
                for choice in ("paint", "ranked"):
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
                 "net_overrun_working_px": NET_OVERRUN_WORKING_PX, "frame_tie_order": list(FRAME_ROLES)},
        "scenes": scenes,
    }
    write_outputs(args.output_dir, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
