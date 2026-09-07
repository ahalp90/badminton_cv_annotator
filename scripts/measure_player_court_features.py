"""Summarise reusable player court-space features from frozen vision stages.

The command reads only the canonical metadata, shuttle, pose, and court stages.
It derives player positions through the production path, then reports coverage
and descriptive quantiles for metre-valued court speed and half-court-centre
distance. Finite values indicate numerical availability; they do not establish
correct player identity, pose quality, or court geometry.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from annotator.video_metadata import VideoMetadata
from dataset_builder.export_v1 import derive_player_inputs
from dataset_builder.features import (
    InterpolationType,
    PlayerFeatureInputs,
    court_position_speed_mps,
    half_court_centre_distance_m,
)
from dataset_builder.vision import TRACK_FILENAME, load_json_gz, save_json_gz

METADATA_FILENAME = "video_metadata.json.gz"
QUANTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
    ("max", 1.0),
)


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {name: None for name, _probability in QUANTILES}
    return {
        name: float(np.quantile(finite, probability))
        for name, probability in QUANTILES
    }


def _value_summary(values: np.ndarray, unit: str) -> dict[str, object]:
    array = np.asarray(values, dtype=float)
    finite_count = int(np.isfinite(array).sum())
    denominator = int(array.size)
    maximum_sample = None
    if finite_count:
        maximum_index = np.argmax(np.where(np.isfinite(array), array, -np.inf))
        frame, slot = np.unravel_index(maximum_index, array.shape)
        maximum_sample = {"frame": int(frame), "slot": int(slot)}
    return {
        "unit": unit,
        "finite_count": finite_count,
        "denominator": denominator,
        "finite_coverage": finite_count / denominator,
        "quantiles": _quantiles(array),
        "maximum_sample": maximum_sample,
    }


def _summarise_player_inputs(
    metadata: VideoMetadata,
    input_paths: dict[str, str],
    player_inputs: PlayerFeatureInputs,
) -> dict[str, object]:
    positions = np.asarray(player_inputs.court_positions, dtype=float)
    provenance = np.asarray(player_inputs.position_interpolation)
    expected_shape = (metadata.frame_count, 2, 2)
    if positions.shape != expected_shape:
        raise ValueError(
            f"derived court_positions shape {positions.shape} != {expected_shape}"
        )
    if provenance.shape != expected_shape[:2]:
        raise ValueError(
            f"derived position_interpolation shape {provenance.shape} != {expected_shape[:2]}"
        )

    finite_points = np.isfinite(positions).all(axis=2)
    out_of_court = finite_points & ((positions < 0.0) | (positions > 1.0)).any(axis=2)
    observed = finite_points & (provenance == InterpolationType.OBSERVED)
    linear = finite_points & (provenance == InterpolationType.LINEAR)
    speed = court_position_speed_mps(
        positions,
        provenance,
        player_inputs.tracker_segments,
        float(metadata.fps),
    )
    centre_distance = half_court_centre_distance_m(positions)

    return {
        "video_id": input_paths["video_id"],
        "inputs": input_paths,
        "fps": float(metadata.fps),
        "fps_fraction": f"{metadata.fps.numerator}/{metadata.fps.denominator}",
        "frame_count": metadata.frame_count,
        "width": metadata.width,
        "height": metadata.height,
        "segment_count": len(player_inputs.tracker_segments),
        "positions": {
            "finite_count": int(finite_points.sum()),
            "denominator": int(finite_points.size),
            "finite_coverage": float(finite_points.mean()),
            "observed_finite_count": int(observed.sum()),
            "linear_finite_count": int(linear.sum()),
            "out_of_court_finite_count": int(out_of_court.sum()),
        },
        "speed_mps": _value_summary(speed, "m/s"),
        "half_court_centre_distance_m": _value_summary(centre_distance, "m"),
    }


def _summarise_video(stage_root: Path, video_id: str) -> dict[str, object]:
    if Path(video_id).name != video_id or video_id in {".", ".."}:
        raise ValueError(f"video id must be a single path component: {video_id!r}")

    court_dir = stage_root / "court" / video_id
    pose_dir = stage_root / "pose" / video_id
    metadata_path = stage_root / "metadata" / video_id / METADATA_FILENAME
    track_path = stage_root / "shuttle" / video_id / TRACK_FILENAME

    metadata = VideoMetadata.from_dict(load_json_gz(metadata_path))
    player_inputs = derive_player_inputs(
        track_path,
        pose_dir,
        court_dir,
        court_video_id=video_id,
        metadata=metadata,
    )
    return _summarise_player_inputs(
        metadata,
        {
            "video_id": video_id,
            "metadata": str(metadata_path),
            "court": str(court_dir),
            "pose": str(pose_dir),
            "shuttle_track": str(track_path),
        },
        player_inputs,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--video-ids", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(set(args.video_ids)) != len(args.video_ids):
        raise ValueError("--video-ids must not contain repeats")
    stage_root = args.stage_root.resolve(strict=True)
    videos = [_summarise_video(stage_root, video_id) for video_id in args.video_ids]
    summary = {
        "schema": "player-court-features/0.1",
        "stage_root": str(stage_root),
        "video_ids": list(args.video_ids),
        "videos": videos,
        "notes": [
            "Finite positions indicate numerical availability, not correct player identity, pose, or court geometry.",
            "Speed is sensitive to frame-to-frame pose and court-geometry jitter, so high values can expose jitter rather than player motion.",
        ],
    }
    save_json_gz(args.output, summary)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
