"""Freeze label-blind per-frame features for the tree contact trial.

The freezer reads standard vision stages and saved annotator spans. It never
loads ShuttleSet tables. The separate scorer verifies this freeze before it
imports ground truth.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import lzma
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_contact_evidence import (
    FIXTURE_SPECS,
    FixtureSpec,
    _load_inputs,
    _sha256,
    _stage_paths,
)

FEATURE_SCHEMA = "tree-contact-features/2"
MANIFEST_SCHEMA = "tree-contact-feature-manifest/2"
FEATURE_FILENAME = "tree_contact_features.npy.xz"
MANIFEST_FILENAME = "tree_contact_features_manifest.json"
MOTION_MODES = ("raw_per_frame", "base30_per_frame")
WINDOW_OFFSETS_BASE30 = (-10, -5, 0, 5, 10)
RELAXED_IMPULSE_MULTIPLE = 1.25
WRIST_LOCAL_MINIMUM_LIMIT = 3.0

MOTION_LINEAR_SIGNALS = (
    "shuttle_vx",
    "shuttle_vy",
    "shuttle_speed",
    "ankle_speed_top",
    "ankle_speed_bot",
)
MOTION_QUADRATIC_SIGNALS = ("shuttle_impulse",)

IDENTITY_FIELDS = ("fixture", "interval_id", "frame", "fps")
REGION_FIELDS = (
    "region_current_raw",
    "region_relaxed_impulse",
    "region_wrist",
    "region_visibility",
    "region_rally_start",
    "region_scene_start",
    "region_serve_lookback",
)
BASE_PHYSICS_SIGNALS = (
    "shuttle_vx",
    "shuttle_vy",
    "shuttle_speed",
    "shuttle_impulse",
    "shuttle_impulse_ratio",
    "wrist_gap_min",
    "wrist_gap_top",
    "wrist_gap_bot",
    "nearest_wrist_dx",
    "nearest_wrist_dy",
    "ankle_speed_top",
    "ankle_speed_bot",
)
BASE_MISSINGNESS_SIGNALS = (
    "shuttle_visible",
    "pose_valid_top",
    "pose_valid_bot",
    "wrist_valid_top",
    "wrist_valid_bot",
)
CONTEXT_FIELDS = (
    "shuttle_x",
    "shuttle_y",
    "ankle_x_top",
    "ankle_y_top",
    "ankle_x_bot",
    "ankle_y_bot",
    "bbox_height_top",
    "bbox_height_bot",
    "standing_count",
    "interval_progress",
    "distance_from_interval_start",
    "distance_to_interval_end",
    "distance_from_scene_start",
) + REGION_FIELDS


def _scaled_frames(base30: int, fps: float) -> int:
    from annotator.fps_constants import ScalingKind

    return int(ScalingKind.FRAME_COUNT.scale(base30, fps))


def _finite_or_nan(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32)


def _motion_scale_factor(fps: float, motion_mode: str) -> float:
    if motion_mode == "raw_per_frame":
        return 1.0
    if motion_mode == "base30_per_frame":
        return fps / 30.0
    raise ValueError(f"unsupported motion mode: {motion_mode!r}")


def _scale_motion_signals(
    signals: Mapping[str, np.ndarray],
    fps: float,
    motion_mode: str,
) -> dict[str, np.ndarray]:
    """Scale frame-rate-dependent signals while preserving region inputs."""
    factor = _motion_scale_factor(fps, motion_mode)
    if motion_mode == "raw_per_frame":
        return dict(signals)

    scaled = dict(signals)
    for name in MOTION_LINEAR_SIGNALS:
        scaled[name] = signals[name] * factor
    for name in MOTION_QUADRATIC_SIGNALS:
        scaled[name] = signals[name] * factor**2
    return scaled


def _difference(values: np.ndarray) -> np.ndarray:
    """Return frame-aligned first differences without bridging missing rows."""
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    valid_frames = np.flatnonzero(valid) + 1
    result[valid_frames] = values[valid_frames] - values[valid_frames - 1]
    return result


def _player_signals(
    track: np.ndarray,
    pose_kps: np.ndarray,
    sticky: Any,
    resolution: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Build frame-aligned player–shuttle geometry from sticky-picked players."""
    from annotator.types import WRIST_L, WRIST_R

    n_frames = len(track)
    width, height = resolution
    scale = np.asarray([width, height], dtype=np.float64)
    wrist_xy = np.full((n_frames, 2, 2), np.nan, dtype=np.float64)
    visible = track[:, 2] == 1
    for slot in range(2):
        valid_frames = np.flatnonzero((sticky.picks[:, slot] >= 0) & visible)
        if not len(valid_frames):
            continue
        raw_slots = sticky.picks[valid_frames, slot].astype(int)
        wrists = pose_kps[valid_frames, raw_slots][:, (WRIST_L, WRIST_R), :] / scale
        shuttle_xy = track[valid_frames, :2]
        gaps = np.linalg.norm(wrists - shuttle_xy[:, None, :], axis=2)
        closest = np.argmin(gaps, axis=1)
        wrist_xy[valid_frames, slot] = wrists[np.arange(len(valid_frames)), closest]

    slot_gaps = _finite_or_nan(np.asarray(sticky.distances_per_slot, dtype=np.float64))
    finite_gaps = np.isfinite(slot_gaps)
    gap_min = np.full(n_frames, np.nan, dtype=np.float32)
    has_gap = finite_gaps.any(axis=1)
    gap_min[has_gap] = np.nanmin(slot_gaps[has_gap], axis=1)

    nearest_wrist = np.full((n_frames, 2), np.nan, dtype=np.float64)
    nearest_slot = np.full(n_frames, -1, dtype=int)
    nearest_slot[has_gap] = np.nanargmin(slot_gaps[has_gap], axis=1)
    valid_nearest = np.flatnonzero(has_gap)
    nearest_wrist[valid_nearest] = wrist_xy[valid_nearest, nearest_slot[valid_nearest]]
    relative = nearest_wrist - track[:, :2]

    ankle = _finite_or_nan(np.asarray(sticky.ankle_pos, dtype=np.float64))
    ankle_dx = np.column_stack([_difference(ankle[:, slot, 0]) for slot in range(2)])
    ankle_dy = np.column_stack([_difference(ankle[:, slot, 1]) for slot in range(2)])
    ankle_speed = np.hypot(ankle_dx, ankle_dy)

    return {
        "wrist_gap_min": gap_min,
        "wrist_gap_top": slot_gaps[:, 0],
        "wrist_gap_bot": slot_gaps[:, 1],
        "nearest_wrist_dx": _finite_or_nan(relative[:, 0]),
        "nearest_wrist_dy": _finite_or_nan(relative[:, 1]),
        "ankle_speed_top": _finite_or_nan(ankle_speed[:, 0]),
        "ankle_speed_bot": _finite_or_nan(ankle_speed[:, 1]),
        "ankle_x_top": ankle[:, 0, 0],
        "ankle_y_top": ankle[:, 0, 1],
        "ankle_x_bot": ankle[:, 1, 0],
        "ankle_y_bot": ankle[:, 1, 1],
        "bbox_height_top": _finite_or_nan(sticky.bbox_height[:, 0] / height),
        "bbox_height_bot": _finite_or_nan(sticky.bbox_height[:, 1] / height),
        "pose_valid_top": (sticky.picks[:, 0] >= 0).astype(np.float32),
        "pose_valid_bot": (sticky.picks[:, 1] >= 0).astype(np.float32),
        "wrist_valid_top": np.isfinite(slot_gaps[:, 0]).astype(np.float32),
        "wrist_valid_bot": np.isfinite(slot_gaps[:, 1]).astype(np.float32),
    }


def _shuttle_signals(
    track: np.ndarray,
    spans: Sequence[tuple[int, int]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Build frame-aligned shuttle kinematics with the production impulse convention."""
    from annotator.config import RallySegmentationThresholds
    from annotator.fps_constants import scale_for_fps
    from annotator.rally.contacts import rolling_floor, span_impulses

    n_frames = len(track)
    visible = track[:, 2] == 1
    x = np.where(visible, track[:, 0], np.nan)
    y = np.where(visible, track[:, 1], np.nan)
    vx = _difference(x)
    vy = _difference(y)
    speed = np.hypot(vx, vy)
    impulse = np.full(n_frames, np.nan, dtype=np.float64)
    impulse_ratio = np.full(n_frames, np.nan, dtype=np.float64)
    values = scale_for_fps(fps)
    thresholds = RallySegmentationThresholds(
        values.rest_speed,
        values.rest_window,
        values.end_rest_frames,
        values.start_speed,
        values.start_min_frames,
        values.smooth_window,
        values.impulse_floor_half_window_frames,
        values.contact_dedup_radius_frames,
        values.contact_suppression_radius_frames,
        RELAXED_IMPULSE_MULTIPLE,
    )
    for start, end in spans:
        span_values = span_impulses(track, start, end, thresholds)
        if span_values is None:
            continue
        span = track[start:end]
        around_visible = (span[:-2, 2] == 1) & (span[1:-1, 2] == 1) & (span[2:, 2] == 1)
        floor = rolling_floor(span_values, around_visible, values.impulse_floor_half_window_frames)
        frames = np.arange(start + 1, start + 1 + len(span_values))
        impulse[frames] = span_values
        impulse_ratio[frames] = span_values / np.maximum(floor, 1e-4)
    return {
        "shuttle_x": _finite_or_nan(x),
        "shuttle_y": _finite_or_nan(y),
        "shuttle_visible": visible.astype(np.float32),
        "shuttle_vx": _finite_or_nan(vx),
        "shuttle_vy": _finite_or_nan(vy),
        "shuttle_speed": _finite_or_nan(speed),
        "shuttle_impulse": _finite_or_nan(impulse),
        "shuttle_impulse_ratio": _finite_or_nan(impulse_ratio),
    }


def _local_minima(values: np.ndarray, limit: float, radius: int) -> np.ndarray:
    finite = np.isfinite(values) & (values <= limit)
    minima = np.zeros(len(values), dtype=bool)
    for frame in np.flatnonzero(finite):
        start = max(0, frame - radius)
        end = min(len(values), frame + radius + 1)
        window = values[start:end]
        if values[frame] == np.nanmin(window):
            minima[frame] = True
    return minima


def build_eligible_intervals(
    tracker_intervals: Sequence[tuple[int, int]],
    exclusion_mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Split court-present tracker intervals around excluded broadcast frames."""
    if exclusion_mask.ndim != 1 or exclusion_mask.dtype != np.bool_:
        raise ValueError("exclusion mask must be a one-dimensional boolean array")

    eligible: list[tuple[int, int]] = []
    for start, end in tracker_intervals:
        if not 0 <= start <= end <= len(exclusion_mask):
            raise ValueError("tracker interval lies outside the exclusion mask")
        run_start: int | None = None
        for frame in range(start, end):
            if not exclusion_mask[frame] and run_start is None:
                run_start = frame
            elif exclusion_mask[frame] and run_start is not None:
                eligible.append((run_start, frame))
                run_start = None
        if run_start is not None:
            eligible.append((run_start, end))
    return eligible


def extend_intervals_with_lookback(
    eligible_intervals: Sequence[tuple[int, int]],
    frame_count: int,
    fps: float,
) -> list[tuple[int, int]]:
    """Add a bounded pre-roll before each eligible court-view interval."""
    lookback = _scaled_frames(45, fps)
    expanded = [(max(0, start - lookback), end) for start, end in eligible_intervals]
    merged: list[tuple[int, int]] = []
    for start, end in expanded:
        if not 0 <= start < end <= frame_count:
            raise ValueError("lookback interval lies outside the source timeline")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _expand_within_span(seed: np.ndarray, start: int, end: int, radius: int) -> np.ndarray:
    expanded = np.zeros(len(seed), dtype=bool)
    local_seed = seed[start:end]
    if local_seed.any():
        kernel = np.ones(2 * radius + 1, dtype=np.int16)
        full = np.convolve(local_seed.astype(np.int16), kernel, mode="full")
        expanded[start:end] = full[radius : radius + len(local_seed)] > 0
    return expanded


def build_region_masks(
    signals: Mapping[str, np.ndarray],
    eligible_intervals: Sequence[tuple[int, int]],
    rally_spans: Sequence[tuple[int, int]],
    raw_contacts: Sequence[Mapping[str, object]],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Build broad search regions without labels or GT-derived boundaries."""
    n_frames = len(signals["shuttle_visible"])
    seeds = {name: np.zeros(n_frames, dtype=bool) for name in REGION_FIELDS}
    for row in raw_contacts:
        frame = int(row["contact_frame"])
        if 0 <= frame < n_frames:
            seeds["region_current_raw"][frame] = True
    seeds["region_relaxed_impulse"] = (
        np.isfinite(signals["shuttle_impulse_ratio"])
        & (signals["shuttle_impulse_ratio"] >= RELAXED_IMPULSE_MULTIPLE)
    )
    wrist_radius = _scaled_frames(3, fps)
    visible = signals["shuttle_visible"].astype(bool)
    for start, end in eligible_intervals:
        seeds["region_wrist"][start:end] = _local_minima(
            signals["wrist_gap_min"][start:end],
            WRIST_LOCAL_MINIMUM_LIMIT,
            radius=wrist_radius,
        )
        if end - start > 1:
            seeds["region_visibility"][start + 1 : end] = visible[start + 1 : end] != visible[start : end - 1]
    for start, _end in rally_spans:
        seeds["region_rally_start"][start] = True
    for start, _end in scene_spans:
        seeds["region_scene_start"][start] = True

    radii_base30 = {
        "region_current_raw": 15,
        "region_relaxed_impulse": 15,
        "region_wrist": 10,
        "region_visibility": 15,
        "region_rally_start": 45,
        "region_scene_start": 15,
    }
    regions = {name: np.zeros(n_frames, dtype=bool) for name in REGION_FIELDS}
    serve_lookback = _scaled_frames(45, fps)
    for start, end in eligible_intervals:
        for name in REGION_FIELDS[:-1]:
            radius = _scaled_frames(radii_base30[name], fps)
            regions[name] |= _expand_within_span(seeds[name], start, end, radius)
        regions["region_serve_lookback"][max(0, start - serve_lookback) : start] = True
    return regions


def _shift_inside_interval(
    values: np.ndarray,
    frames: np.ndarray,
    offset: int,
    start: int,
    end: int,
) -> np.ndarray:
    source = frames + offset
    result = np.full(len(frames), np.nan, dtype=np.float32)
    valid = (source >= start) & (source < end)
    result[valid] = values[source[valid]]
    return result


def _feature_family_names() -> dict[str, list[str]]:
    physics = [f"{signal}_t{offset:+d}" for signal in BASE_PHYSICS_SIGNALS for offset in WINDOW_OFFSETS_BASE30]
    missingness = [
        f"{signal}_t{offset:+d}"
        for signal in BASE_MISSINGNESS_SIGNALS
        for offset in WINDOW_OFFSETS_BASE30
    ]
    return {"physics": physics, "context": list(CONTEXT_FIELDS), "missingness": missingness}


def _record_dtype(feature_families: Mapping[str, Sequence[str]]) -> np.dtype:
    fields: list[tuple[str, str]] = [
        ("fixture", "S7"),
        ("interval_id", "<i2"),
        ("frame", "<i4"),
        ("fps", "<f4"),
    ]
    fields.extend((name, "u1") for name in REGION_FIELDS)
    existing = {name for name, _dtype in fields}
    for family in ("physics", "context", "missingness"):
        for name in feature_families[family]:
            if name not in existing:
                fields.append((name, "<f4"))
                existing.add(name)
    return np.dtype(fields)


def _fixture_rows(
    data_root: Path,
    fixture: FixtureSpec,
    motion_mode: str = "raw_per_frame",
) -> tuple[np.ndarray, dict[str, Any]]:
    from dataset_builder.vision import load_npy_xz

    track, pose, court, tracker_intervals, sticky, annotation = _load_inputs(data_root, fixture)
    exclusion_path = (
        Path(data_root) / "stages" / "annotation" / fixture.name / "definitive_exclusion_mask.npy.xz"
    )
    exclusion_mask = load_npy_xz(exclusion_path)
    if exclusion_mask.shape != (len(track),) or exclusion_mask.dtype != np.bool_:
        raise ValueError(f"{fixture.name}: definitive exclusion mask must match the shuttle timeline")
    eligible_intervals = build_eligible_intervals(tracker_intervals, exclusion_mask)
    search_intervals = extend_intervals_with_lookback(eligible_intervals, len(track), fixture.fps)
    signals = _shuttle_signals(track, search_intervals, fixture.fps)
    signals.update(_player_signals(track, pose.kps, sticky, (fixture.width, fixture.height)))
    signals["standing_count"] = np.asarray(sticky.standing_count, dtype=np.float32)
    signals["sticky_analysed"] = np.asarray(sticky.analysed, dtype=np.float32)
    regions = build_region_masks(
        signals,
        eligible_intervals,
        annotation.spans,
        annotation.contacts,
        court.raw_cuts,
        fixture.fps,
    )
    feature_signals = _scale_motion_signals(signals, fixture.fps, motion_mode)
    union = np.zeros(len(track), dtype=bool)
    for region in regions.values():
        union |= region
    feature_families = _feature_family_names()
    dtype = _record_dtype(feature_families)
    chunks: list[np.ndarray] = []
    scene_starts = np.asarray([start for start, _end in court.raw_cuts], dtype=int)
    for interval_id, (start, end) in enumerate(search_intervals):
        frames = np.arange(start, end, dtype=np.int32)
        rows = np.zeros(len(frames), dtype=dtype)
        rows["fixture"] = fixture.name.encode("ascii")
        rows["interval_id"] = interval_id
        rows["frame"] = frames
        rows["fps"] = fixture.fps
        for name in REGION_FIELDS:
            rows[name] = regions[name][frames]

        for signal in BASE_PHYSICS_SIGNALS:
            for offset_base30 in WINDOW_OFFSETS_BASE30:
                offset = 0 if offset_base30 == 0 else int(math.copysign(_scaled_frames(abs(offset_base30), fixture.fps), offset_base30))
                rows[f"{signal}_t{offset_base30:+d}"] = _shift_inside_interval(
                    feature_signals[signal], frames, offset, start, end
                )
        for signal in BASE_MISSINGNESS_SIGNALS:
            for offset_base30 in WINDOW_OFFSETS_BASE30:
                offset = 0 if offset_base30 == 0 else int(math.copysign(_scaled_frames(abs(offset_base30), fixture.fps), offset_base30))
                rows[f"{signal}_t{offset_base30:+d}"] = _shift_inside_interval(
                    feature_signals[signal], frames, offset, start, end
                )

        for name in ("shuttle_x", "shuttle_y", "ankle_x_top", "ankle_y_top", "ankle_x_bot", "ankle_y_bot", "bbox_height_top", "bbox_height_bot", "standing_count"):
            rows[name] = feature_signals[name][frames]
        rows["interval_progress"] = (frames - start) / max(1, end - start - 1)
        rows["distance_from_interval_start"] = (frames - start) / fixture.fps
        rows["distance_to_interval_end"] = (end - 1 - frames) / fixture.fps
        preceding_scene = np.searchsorted(scene_starts, frames, side="right") - 1
        scene_distance = np.full(len(frames), np.nan, dtype=np.float32)
        has_scene = preceding_scene >= 0
        scene_distance[has_scene] = (frames[has_scene] - scene_starts[preceding_scene[has_scene]]) / fixture.fps
        rows["distance_from_scene_start"] = scene_distance
        for name in REGION_FIELDS:
            rows[name] = regions[name][frames]
        chunks.append(rows)

    fixture_rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
    summary = {
        "fixture": fixture.name,
        "frame_count": len(track),
        "rally_span_count": len(annotation.spans),
        "tracker_interval_count": len(tracker_intervals),
        "tracker_intervals": [list(interval) for interval in tracker_intervals],
        "eligible_interval_count": len(eligible_intervals),
        "eligible_intervals": [list(interval) for interval in eligible_intervals],
        "eligible_frame_count": int(sum(end - start for start, end in eligible_intervals)),
        "search_interval_count": len(search_intervals),
        "search_intervals": [list(interval) for interval in search_intervals],
        "search_frame_count": int(sum(end - start for start, end in search_intervals)),
        "seeded_frame_count": int(union.sum()),
        "row_count": len(fixture_rows),
        "region_frame_counts": {name: int(regions[name].sum()) for name in REGION_FIELDS},
    }
    return fixture_rows, summary


def _write_npy_xz(path: Path, values: np.ndarray) -> None:
    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=9) as destination:
        np.save(destination, values, allow_pickle=False)


def freeze(
    data_root: Path,
    output_dir: Path,
    source_commit: str,
    motion_mode: str = "raw_per_frame",
) -> tuple[Path, Path]:
    """Freeze all three fixtures and return feature and manifest paths."""
    if not source_commit.strip():
        raise ValueError("source_commit must be non-empty")
    _motion_scale_factor(30.0, motion_mode)
    feature_families = _feature_family_names()
    fixture_chunks: list[np.ndarray] = []
    fixture_summaries: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for name, (video_id, fps) in FIXTURE_SPECS.items():
        fixture = FixtureSpec(name, video_id, fps)
        rows, summary = _fixture_rows(data_root, fixture, motion_mode)
        fixture_chunks.append(rows)
        fixture_summaries.append(summary)
        files = []
        for role, path in _stage_paths(data_root, fixture).items():
            files.append({
                "role": role,
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
        exclusion_path = (
            Path(data_root) / "stages" / "annotation" / fixture.name / "definitive_exclusion_mask.npy.xz"
        )
        files.append({
            "role": "definitive_exclusion_mask",
            "filename": exclusion_path.name,
            "size_bytes": exclusion_path.stat().st_size,
            "sha256": _sha256(exclusion_path),
        })
        input_rows.append({"fixture": name, "files": files})
    rows = np.concatenate(fixture_chunks)
    order = np.lexsort((rows["frame"], rows["interval_id"], rows["fixture"]))
    rows = rows[order]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / FEATURE_FILENAME
    _write_npy_xz(feature_path, rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "labels_read": False,
        "row_domain": "eligible tracker intervals plus 45-base-30 serve pre-roll",
        "model_search_surface": "seeded region union",
        "source_commit": source_commit,
        "fixture_set": list(FIXTURE_SPECS),
        "feature_file": FEATURE_FILENAME,
        "feature_sha256": _sha256(feature_path),
        "row_count": len(rows),
        "feature_families": feature_families,
        "identity_fields": list(IDENTITY_FIELDS),
        "region_fields": list(REGION_FIELDS),
        "window_offsets_base30": list(WINDOW_OFFSETS_BASE30),
        "seed_parameters": {
            "relaxed_impulse_multiple": RELAXED_IMPULSE_MULTIPLE,
            "wrist_local_minimum_limit": WRIST_LOCAL_MINIMUM_LIMIT,
        },
        "fixtures": fixture_summaries,
        "inputs": input_rows,
    }
    if motion_mode == "base30_per_frame":
        manifest["motion_mode"] = motion_mode
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return feature_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--motion-mode", choices=MOTION_MODES, default="raw_per_frame")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    feature_path, manifest_path = freeze(
        arguments.data_root,
        arguments.output_dir,
        arguments.source_commit,
        arguments.motion_mode,
    )
    print(f"wrote {feature_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
