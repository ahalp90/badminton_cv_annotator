"""Freeze label-blind serve-burst qualification on the pinned remote fixtures.

This script does not import or read ShuttleSet shot labels or the
18 reviewed hallucination spans. Run it on a machine where the pinned pose
fixture is available, before running ``score_serve_protection.py``.
"""

from __future__ import annotations

# This standalone script adds the project source folder before importing it.
# ruff: noqa: E402

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results/serve_qualification.json.gz"
BASELINE_SHA = "3c217d4cc8ab3698e825315a218735322f938a00"
SERVE_THRESHOLD_BH = 0.8
SOURCE_RUN_ROOT = REPO_ROOT / "experiments/annotator/runs/20260730-041328"
FIXTURE_SOURCE = "serve_prepend_lookback_189c5af_20260808/fixtures"

if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from annotator.calibration.fixtures import FIXTURES, FilePin, Fixture, verify_file
from annotator.config import BaseAnnotatorConfig
from annotator.rally.evidence import build_sticky_result, tracker_segments
from annotator.rally.serve import (
    ServeStartMode,
    ServeStartOptions,
    _resolve_serve_gate,
    build_serve_setup_inputs,
)
from annotator.rally.spans import _rally_regions, _rest_mask, find_rally_spans
from annotator.replay_mask import _read_homography_rows
from annotator.resolve import resolve
from annotator.types import compute_speed
from shared.court import load_all_court_info


def write_json_gz(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic UTF-8 JSON in a gzip stream."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        path.open("wb") as raw_target,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_target,
            compresslevel=9,
            mtime=0,
        ) as compressed,
        io.TextIOWrapper(compressed, encoding="utf-8") as text_target,
    ):
        json.dump(payload, text_target, indent=2, sort_keys=True)
        text_target.write("\n")


def array_content_sha256(array: np.ndarray) -> str:
    """Hash an array's shape, dtype, and C-order value bytes."""
    digest = hashlib.sha256()
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
    ).encode()
    digest.update(descriptor)
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def required_fixture_pins(fixture: Fixture) -> tuple[FilePin, ...]:
    """Return track and pose pins shared by current and PR 98 fixtures."""
    required_paths = {
        fixture.track_path,
        fixture.pose_path("bboxes"),
        fixture.pose_path("scores"),
        fixture.pose_path("kps"),
        fixture.pose_path("ndet"),
    }
    return tuple(pin for pin in fixture.files if pin.path in required_paths)


def file_md5(path: Path) -> str:
    """Return a streaming MD5 digest for a manifest-pinned input."""
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pr98_court_inputs(fixture: Fixture, fixture_root: Path) -> list[dict[str, str]]:
    """Verify court inputs against the exact source-run manifest used by PR 98."""
    manifest_path = (
        SOURCE_RUN_ROOT
        / "static_shuttleset_homography"
        / fixture.name
        / "tracknet-stride-8/manifest.json"
    )
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    artifacts = manifest.get("artifacts") if isinstance(manifest, dict) else None
    if not isinstance(artifacts, list):
        raise TypeError(f"{manifest_path}: expected an artifacts list")

    required = {
        "court_present.npy": fixture.court_present_path,
        "scene_rows.csv": fixture.scene_rows_path,
    }
    verified: list[dict[str, str]] = []
    for artifact_name, fixture_path in required.items():
        matches = [
            row
            for row in artifacts
            if isinstance(row, dict) and str(row.get("path", "")).endswith(artifact_name)
        ]
        if len(matches) != 1 or not isinstance(matches[0].get("md5"), str):
            raise ValueError(f"{manifest_path}: expected one pinned {artifact_name}")
        expected_md5 = str(matches[0]["md5"])
        actual_path = fixture_root / fixture_path
        if file_md5(actual_path) != expected_md5:
            raise ValueError(f"{fixture.name}: {artifact_name} differs from its PR 98 manifest")
        verified.append(
            {
                "path": str(fixture_path),
                "md5": expected_md5,
                "pin_source": str(manifest_path.relative_to(REPO_ROOT)),
            }
        )
    return verified


def qualify_burst_rows(
    fast_runs: list[tuple[int, int]],
    regions: list[tuple[int, int]],
    qualifies: Callable[[int], bool],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Qualify every sustained-fast burst and retain per-region counts."""
    rows: list[dict[str, Any]] = []
    qualifying_counts: list[int] = []
    for region_index, (region_start, region_stop) in enumerate(regions):
        region_runs = [
            (start, stop)
            for start, stop in fast_runs
            if region_start <= start < region_stop
        ]
        if not region_runs:
            continue
        qualified_in_region = 0
        for burst_start, burst_stop in region_runs:
            qualified = qualifies(burst_start)
            qualified_in_region += int(qualified)
            rows.append(
                {
                    "region_index": region_index,
                    "region_start": int(region_start),
                    "region_stop_exclusive": int(region_stop),
                    "burst_start": int(burst_start),
                    "burst_stop_exclusive": int(burst_stop),
                    "qualified": qualified,
                }
            )
        qualifying_counts.append(qualified_in_region)
    return rows, qualifying_counts


def load_gate_tables() -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Load tracked court geometry and resolution tables, never shot labels."""
    homography_path = REPO_ROOT / "training/data/shuttleset/annotations/set/homography.csv"
    resolution_path = (
        REPO_ROOT / "training/data/shuttleset/annotations/my_raw_video_resolution.csv"
    )
    courts = {
        str(video_id): info
        for video_id, info in load_all_court_info(homography_path).items()
    }
    resolution = pd.read_csv(resolution_path).set_index("id")
    resolution.index = resolution.index.astype(str)
    return courts, resolution


def freeze_fixture(
    fixture: Fixture,
    fixture_root: Path,
    gate_courts: dict[str, dict[str, Any]],
    gate_resolution: pd.DataFrame,
) -> dict[str, Any]:
    """Build sticky evidence and freeze every burst verdict for one fixture."""
    pins = required_fixture_pins(fixture)
    for pin in pins:
        verify_file(pin)
    court_input_pins = verify_pr98_court_inputs(fixture, fixture_root)

    track = np.load(fixture_root / fixture.track_path, allow_pickle=False)
    bboxes = np.load(fixture_root / fixture.pose_path("bboxes"), allow_pickle=False)
    scores = np.load(fixture_root / fixture.pose_path("scores"), allow_pickle=False)
    keypoints = np.load(fixture_root / fixture.pose_path("kps"), allow_pickle=False)
    detection_counts = np.load(
        fixture_root / fixture.pose_path("ndet"),
        allow_pickle=False,
    )
    court_present = np.load(
        fixture_root / fixture.court_present_path,
        allow_pickle=False,
    )
    frame_count = len(track)
    aligned = {
        "track": track,
        "bboxes": bboxes,
        "scores": scores,
        "keypoints": keypoints,
        "detection_counts": detection_counts,
        "court_present": court_present,
    }
    for name, value in aligned.items():
        if value.ndim == 0 or len(value) != frame_count:
            raise ValueError(f"{fixture.name}: {name} is not aligned to {frame_count} frames")

    scene_rows = _read_homography_rows(
        fixture_root / fixture.scene_rows_path,
        str(fixture.video_id),
    )
    segments = tracker_segments(scene_rows, court_present, frame_count)
    resolved = resolve(BaseAnnotatorConfig(), fixture.fps)
    sticky = build_sticky_result(
        track,
        segments,
        bboxes,
        scores,
        keypoints,
        detection_counts,
        str(fixture.video_id),
        gate_courts,
        gate_resolution,
        fixture.resolution,
        resolved.constants.body_unit_half_window,
    )
    setup = build_serve_setup_inputs(sticky, fixture.resolution)
    production_diagnostics: dict[str, Any] = {}
    options = ServeStartOptions(
        dist=None,
        threshold=SERVE_THRESHOLD_BH,
        mode=ServeStartMode.TRIM,
        close=None,
        diagnostics=production_diagnostics,
        setup=setup,
        stillness_threshold_bh=None,
        lookback_frames=resolved.constants.serve_start_lookback_frames,
        stillness_window_frames=resolved.constants.serve_stillness_window_frames,
    )
    gate = _resolve_serve_gate(options)

    speed = compute_speed(track)
    at_rest = _rest_mask(
        speed,
        track,
        resolved.thresholds,
        constants=resolved.constants,
        gap_state_demotion_bound=resolved.gap_state_demotion_bound,
        reentry_guard_variant=resolved.reentry_guard_variant,
        reentry_guard_buffer=resolved.reentry_guard_buffer,
    )
    fast_runs, _rest_runs, regions = _rally_regions(
        speed,
        at_rest,
        resolved.thresholds,
    )
    burst_rows, qualifying_counts = qualify_burst_rows(
        fast_runs,
        regions,
        gate.qualifies,
    )

    production_spans = find_rally_spans(
        track,
        thresholds=resolved.thresholds,
        serve_start=options,
        span_open=resolved.span_open,
        constants=resolved.constants,
        gap_state_demotion_bound=resolved.gap_state_demotion_bound,
        reentry_guard_variant=resolved.reentry_guard_variant,
        reentry_guard_buffer=resolved.reentry_guard_buffer,
        quiet_start_window=resolved.quiet_start_window,
    )
    if production_diagnostics.get("qualifying_counts") != qualifying_counts:
        raise AssertionError(f"{fixture.name}: direct burst capture differs from production diagnostics")

    return {
        "video_id": fixture.video_id,
        "fps": fixture.fps,
        "frames": frame_count,
        "track_content_sha256": array_content_sha256(track),
        "input_pins": [
            {
                "path": str(pin.path),
                "md5": pin.md5,
                "pin_source": "src/annotator/calibration/fixtures.py",
            }
            for pin in pins
        ]
        + court_input_pins,
        "tracker_segment_count": len(segments),
        "analysed_frames": int(np.count_nonzero(setup.analysed)),
        "regions_with_burst": len(qualifying_counts),
        "burst_count": len(burst_rows),
        "qualified_burst_count": sum(row["qualified"] for row in burst_rows),
        "production_span_count": len(production_spans),
        "production_diagnostics": production_diagnostics,
        "bursts": burst_rows,
    }


def run_freeze(fixture_root: Path) -> dict[str, Any]:
    """Freeze all three fixtures without reading evaluation labels."""
    fixture_root = fixture_root.expanduser().resolve()
    if not fixture_root.is_dir():
        raise FileNotFoundError(fixture_root)
    os.environ["ANNOTATOR_FIXTURES_ROOT"] = str(fixture_root)
    gate_courts, gate_resolution = load_gate_tables()
    fixtures: dict[str, Any] = {}
    for fixture in FIXTURES:
        print(f"qualifying {fixture.name}", flush=True)
        fixtures[fixture.name] = freeze_fixture(
            fixture,
            fixture_root,
            gate_courts,
            gate_resolution,
        )
    return {
        "schema_version": 1,
        "baseline_commit": BASELINE_SHA,
        "status": "label-blind exploratory qualification; threshold is not production-approved",
        "label_firewall": {
            "shots_master_read": False,
            "visual_hallucination_spans_read": False,
        },
        "configuration": {
            "threshold_bh": SERVE_THRESHOLD_BH,
            "mode": ServeStartMode.TRIM.value,
            "close": None,
            "stillness_threshold_bh": None,
            "serve_setup_window": "[burst-lookback, burst)",
            "track_source": "unmasked pinned track",
        },
        "fixture_source": FIXTURE_SOURCE,
        "fixtures": fixtures,
    }


def parse_args() -> argparse.Namespace:
    """Parse the pinned fixture root and output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """Run and retain the label-blind qualification freeze."""
    args = parse_args()
    result = run_freeze(args.fixtures_root)
    write_json_gz(args.output, result)
    for name, fixture in result["fixtures"].items():
        print(
            f"{name}: {fixture['qualified_burst_count']}/"
            f"{fixture['burst_count']} bursts qualify"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
