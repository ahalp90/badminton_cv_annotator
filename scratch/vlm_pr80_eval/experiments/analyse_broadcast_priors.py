"""Measure simple existing signals on every eligible pure broadcast control."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .build_trials import (
    DEVELOPMENT_VIDEOS,
    _broadcast_records,
    _load_video,
    _write_json,
)


def rule_metrics(
    rows: list[dict[str, Any]],
    signal: str,
    threshold: float,
    *,
    keep_when_high: bool,
) -> dict[str, Any]:
    """Score one threshold as an automatic keep-live rule."""
    predicted_keep = [
        float(row["priors"][signal]) >= threshold
        if keep_when_high
        else float(row["priors"][signal]) <= threshold
        for row in rows
    ]
    expected_keep = [bool(row["truth"]["valid_rally"]) for row in rows]
    true_keep = sum(predicted and expected for predicted, expected in zip(predicted_keep, expected_keep))
    false_keep = sum(predicted and not expected for predicted, expected in zip(predicted_keep, expected_keep))
    true_reject = sum(not predicted and not expected for predicted, expected in zip(predicted_keep, expected_keep))
    live_total = sum(expected_keep)
    nonlive_total = len(expected_keep) - live_total
    return {
        "threshold": threshold,
        "true_keep": true_keep,
        "false_keep": false_keep,
        "true_reject": true_reject,
        "keep_precision": (
            None if true_keep + false_keep == 0 else true_keep / (true_keep + false_keep)
        ),
        "live_recall": true_keep / live_total,
        "nonlive_rejection": true_reject / nonlive_total,
    }


def analyse(
    artifacts_root: Path,
    repo_root: Path,
    scene_labels_dir: Path,
    output_path: Path,
) -> None:
    """Load all pure development controls and write signal threshold sweeps."""
    shots = pd.read_csv(
        repo_root / "training/data/shuttleset/annotations/shots_master.csv"
    )
    videos = [
        _load_video(artifacts_root, scene_labels_dir, shots, name)
        for name in DEVELOPMENT_VIDEOS
    ]
    rows: list[dict[str, Any]] = []
    for video in videos:
        positives, negatives = _broadcast_records(video, dense_target=True)
        rows.extend(positives)
        rows.extend(negatives)
    thresholds = [index / 20 for index in range(21)]
    signal_directions = {
        "track_visible_fraction": True,
        "court_present_fraction": True,
        "raw_mask_fraction": False,
        "definitive_mask_fraction": False,
    }
    sweeps = {
        signal: [
            rule_metrics(
                rows,
                signal,
                threshold,
                keep_when_high=keep_when_high,
            )
            for threshold in thresholds
        ]
        for signal, keep_when_high in signal_directions.items()
    }
    cases = [
        {
            "video_id": row["video_id"],
            "span_index": row["span_index"],
            "valid_rally": row["truth"]["valid_rally"],
            "scene": row["truth"]["dominant_scene_truth"],
            "priors": row["priors"],
        }
        for row in rows
    ]
    _write_json(
        output_path,
        {
            "schema": "vlm-broadcast-prior-analysis/0.1",
            "case_count": len(cases),
            "cases": cases,
            "threshold_sweeps": sweeps,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scene-labels-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analyse(
        args.artifacts_root,
        args.repo_root,
        args.scene_labels_dir,
        args.out,
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
