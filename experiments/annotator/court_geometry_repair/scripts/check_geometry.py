"""Check the saved geometry metrics for the two CourtKeyNet controls.

This replays the recorded validation path from saved scene quads, finite
painted-line support summaries and consensus flags. It does not run inference,
read a video or invoke the production annotator adapter.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from courtkeynet.court_corners import (
    CONSENSUS_FLAG_THRESHOLD_PX,
    consensus_repair,
    painted_line_support,
)

CONTROL_VIDEOS = (3, 21)
REFERENCE_SIZE = (1280, 720)
MIN_FAMILY_SUPPORT = 0.5
METRIC_ATOL = 1e-5


def _load(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _points(value: object, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: expected finite corners with shape (4, 2)")
    return points


def _consensus_flags(
    line_support: Mapping[str, Any], accepted: Mapping[int, np.ndarray]
) -> set[int]:
    repair = line_support["consensus_repair"]
    scene_indices = [int(index) for index in repair["scene_indices"]]
    if set(scene_indices) != set(accepted):
        raise ValueError("line support and comparison scenes do not align")
    observed = consensus_repair(np.stack([accepted[index] for index in scene_indices]))
    if not np.allclose(
        observed.consensus_quad, _points(repair["consensus_quad"], "consensus_quad"), atol=METRIC_ATOL, rtol=0.0
    ):
        raise ValueError("saved consensus does not match the accepted scene quads")
    if not np.allclose(observed.distances_px, repair["distances_px"], atol=METRIC_ATOL, rtol=0.0):
        raise ValueError("saved consensus distances do not match the accepted scene quads")
    saved_flags = np.asarray(repair["flagged"], dtype=bool)
    if not np.array_equal(observed.flagged, saved_flags):
        raise ValueError(f"saved consensus flags do not match the {CONSENSUS_FLAG_THRESHOLD_PX:g} px threshold")
    return {index for index, flagged in zip(scene_indices, saved_flags) if flagged}


def _check_line_support(
    line_support: Mapping[str, Any], accepted: Mapping[int, np.ndarray],
    repaired: set[int], donor_consensus: np.ndarray,
) -> None:
    scenes = {int(scene["scene_index"]): scene for scene in line_support["scenes"]}
    if not set(accepted).issubset(scenes):
        raise ValueError("line support is missing an accepted scene")
    for index in repaired:
        segments = tuple(np.asarray(frame, dtype=np.float64).reshape(-1, 4)
                         for frame in scenes[index]["segments_px"])
        if any(not np.isfinite(frame).all() for frame in segments):
            raise ValueError(f"scene {index}: line segments are not finite")
        raw_support = painted_line_support(accepted[index], segments, REFERENCE_SIZE)
        repaired_support = painted_line_support(donor_consensus, segments, REFERENCE_SIZE)
        if min(repaired_support) < MIN_FAMILY_SUPPORT:
            raise ValueError(f"scene {index}: donor support is below 50% in a direction")
        if sum(repaired_support) <= sum(raw_support):
            raise ValueError(f"scene {index}: donor support does not improve the raw court")


def _check_video(evidence_root: Path, video_id: int) -> dict[str, Any]:
    video_dir = evidence_root / f"video_{video_id}"
    comparison = _load(video_dir / "comparison.json.gz")
    line_support = _load(video_dir / "line_support.json.gz")
    expected = _load(video_dir / "expected.json.gz")
    metadata = comparison["metadata"]
    if metadata["video_id"] != video_id or tuple(
        metadata["resize"][key] for key in ("width", "height")
    ) != REFERENCE_SIZE:
        raise ValueError(f"video {video_id}: invalid comparison metadata")
    accepted = {
        int(record["scene_index"]): _points(
            record["current_quad"]["corners_px"], f"scene {record['scene_index']}/current_quad"
        )
        for record in comparison["scene_records"]
        if record.get("current_quad") is not None
    }
    repaired = _consensus_flags(line_support, accepted)
    ground_truth = _points(comparison["ground_truth_corners_px"], "ground_truth_corners_px")
    scene_indices = sorted(accepted)
    raw_stack = np.stack([accepted[index] for index in scene_indices])
    raw_errors = np.linalg.norm(raw_stack - ground_truth, axis=2)
    # Reconstruct the reported control geometry from its unflagged donor group.
    donor_stack = np.stack([accepted[index] for index in scene_indices if index not in repaired])
    donor_consensus = np.median(donor_stack, axis=0)
    _check_line_support(line_support, accepted, repaired, donor_consensus)
    final_stack = np.stack(
        [donor_consensus if index in repaired else accepted[index] for index in scene_indices]
    )
    final_errors = np.linalg.norm(final_stack - ground_truth, axis=2)
    result = {
        "video_id": video_id,
        "accepted_geometry": len(accepted),
        "repaired_scene_indices": sorted(repaired),
        "raw_mean_corner_error_refpx": float(raw_errors.mean()),
        "final_mean_corner_error_refpx": float(final_errors.mean()),
        "final_max_corner_error_refpx": float(final_errors.max()),
    }
    if result["accepted_geometry"] != expected["accepted_geometry"]:
        raise ValueError(f"video {video_id}: accepted geometry does not match expected evidence")
    if result["repaired_scene_indices"] != sorted(map(int, expected["repaired_scene_indices"])):
        raise ValueError(f"video {video_id}: repaired scenes do not match expected evidence")
    for metric, expected_name in (
        ("final_mean_corner_error_refpx", "mean_corner_error_refpx"),
        ("final_max_corner_error_refpx", "max_corner_error_refpx"),
    ):
        if not np.isclose(result[metric], expected[expected_name], atol=METRIC_ATOL, rtol=0.0):
            raise ValueError(f"video {video_id}: {metric} does not match expected evidence")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evidence" / "controls",
    )
    parser.add_argument("--video", type=int, action="append", dest="videos")
    arguments = parser.parse_args(argv)
    try:
        results = [
            _check_video(arguments.evidence_root, video_id)
            for video_id in (tuple(arguments.videos) if arguments.videos else CONTROL_VIDEOS)
        ]
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"geometry check failed: {error}", file=sys.stderr)
        return 1
    for result in results:
        print(
            f"video {result['video_id']}: accepted {result['accepted_geometry']}, "
            f"repaired {len(result['repaired_scene_indices'])}; mean corner error "
            f"{result['raw_mean_corner_error_refpx']:.6f} -> "
            f"{result['final_mean_corner_error_refpx']:.6f} px; "
            f"final max {result['final_max_corner_error_refpx']:.6f} px; PASS"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
