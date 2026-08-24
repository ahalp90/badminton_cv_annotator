"""Freeze label-blind per-frame features for the tree contact trial.

The freezer reads standard vision stages and saved annotator spans. It never
loads ShuttleSet tables. The separate scorer verifies this freeze before it
imports ground truth.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
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

FEATURE_SCHEMA = "tree-contact-features/1"
MANIFEST_SCHEMA = "tree-contact-feature-manifest/1"
FEATURE_FILENAME = "tree_contact_features.npy.xz"
MANIFEST_FILENAME = "tree_contact_features_manifest.json"
WINDOW_OFFSETS_BASE30 = (-10, -5, 0, 5, 10)
RELAXED_IMPULSE_MULTIPLE = 1.25
WRIST_LOCAL_MINIMUM_LIMIT = 3.0

IDENTITY_FIELDS = ("fixture", "span_id", "frame", "fps")
REGION_FIELDS = (
    "region_current_raw",
    "region_relaxed_impulse",
    "region_wrist",
    "region_visibility",
    "region_rally_start",
    "region_scene_start",
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
    "span_progress",
    "distance_from_span_start",
    "distance_to_span_end",
    "distance_from_scene_start",
) + REGION_FIELDS


def _scaled_frames(base30: int, fps: float) -> int:
    from annotator.fps_constants import ScalingKind

    return int(ScalingKind.FRAME_COUNT.scale(base30, fps))


def _finite_or_nan(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32)


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
    spans: Sequence[tuple[int, int]],
    raw_contacts: Sequence[Mapping[str, object]],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Build broad search regions without labels or GT-derived boundaries."""
    n_frames = len(signals["shuttle_visible"])
    seeds = {name: np.zeros(n_frames, dtype=bool) for name in REGION_FIELDS}
    for row in raw_contacts:
        seeds["region_current_raw"][int(row["contact_frame"])] = True
    seeds["region_relaxed_impulse"] = (
        np.isfinite(signals["shuttle_impulse_ratio"])
        & (signals["shuttle_impulse_ratio"] >= RELAXED_IMPULSE_MULTIPLE)
    )
    seeds["region_wrist"] = _local_minima(
        signals["wrist_gap_min"],
        WRIST_LOCAL_MINIMUM_LIMIT,
        radius=_scaled_frames(3, fps),
    )
    visible = signals["shuttle_visible"].astype(bool)
    seeds["region_visibility"][1:] = visible[1:] != visible[:-1]
    for start, _end in spans:
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
    for start, end in spans:
        for name in REGION_FIELDS:
            radius = _scaled_frames(radii_base30[name], fps)
            regions[name] |= _expand_within_span(seeds[name], start, end, radius)
    return regions


def _shift_inside_span(values: np.ndarray, frames: np.ndarray, offset: int, start: int, end: int) -> np.ndarray:
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
    fields: list[tuple[str, str]] = [("fixture", "S7"), ("span_id", "<i2"), ("frame", "<i4"), ("fps", "<f4")]
    fields.extend((name, "u1") for name in REGION_FIELDS)
    existing = {name for name, _dtype in fields}
    for family in ("physics", "context", "missingness"):
        for name in feature_families[family]:
            if name not in existing:
                fields.append((name, "<f4"))
                existing.add(name)
    return np.dtype(fields)


def _fixture_rows(data_root: Path, fixture: FixtureSpec) -> tuple[np.ndarray, dict[str, Any]]:
    track, pose, court, _segments, sticky, annotation = _load_inputs(data_root, fixture)
    spans = annotation.spans
    signals = _shuttle_signals(track, spans, fixture.fps)
    signals.update(_player_signals(track, pose.kps, sticky, (fixture.width, fixture.height)))
    signals["standing_count"] = np.asarray(sticky.standing_count, dtype=np.float32)
    signals["sticky_analysed"] = np.asarray(sticky.analysed, dtype=np.float32)
    regions = build_region_masks(signals, spans, annotation.contacts, court.raw_cuts, fixture.fps)
    union = np.zeros(len(track), dtype=bool)
    for region in regions.values():
        union |= region
    feature_families = _feature_family_names()
    dtype = _record_dtype(feature_families)
    chunks: list[np.ndarray] = []
    scene_starts = np.asarray([start for start, _end in court.raw_cuts], dtype=int)
    for span_id, (start, end) in enumerate(spans):
        frames = np.flatnonzero(union[start:end]) + start
        if not len(frames):
            continue
        rows = np.zeros(len(frames), dtype=dtype)
        rows["fixture"] = fixture.name.encode("ascii")
        rows["span_id"] = span_id
        rows["frame"] = frames
        rows["fps"] = fixture.fps
        for name in REGION_FIELDS:
            rows[name] = regions[name][frames]

        for signal in BASE_PHYSICS_SIGNALS:
            for offset_base30 in WINDOW_OFFSETS_BASE30:
                offset = 0 if offset_base30 == 0 else int(math.copysign(_scaled_frames(abs(offset_base30), fixture.fps), offset_base30))
                rows[f"{signal}_t{offset_base30:+d}"] = _shift_inside_span(
                    signals[signal], frames, offset, start, end
                )
        for signal in BASE_MISSINGNESS_SIGNALS:
            for offset_base30 in WINDOW_OFFSETS_BASE30:
                offset = 0 if offset_base30 == 0 else int(math.copysign(_scaled_frames(abs(offset_base30), fixture.fps), offset_base30))
                rows[f"{signal}_t{offset_base30:+d}"] = _shift_inside_span(
                    signals[signal], frames, offset, start, end
                )

        for name in ("shuttle_x", "shuttle_y", "ankle_x_top", "ankle_y_top", "ankle_x_bot", "ankle_y_bot", "bbox_height_top", "bbox_height_bot", "standing_count"):
            rows[name] = signals[name][frames]
        rows["span_progress"] = (frames - start) / max(1, end - start - 1)
        rows["distance_from_span_start"] = (frames - start) / fixture.fps
        rows["distance_to_span_end"] = (end - 1 - frames) / fixture.fps
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
        "span_count": len(spans),
        "row_count": len(fixture_rows),
        "region_frame_counts": {name: int(regions[name].sum()) for name in REGION_FIELDS},
    }
    return fixture_rows, summary


def _write_npy_xz(path: Path, values: np.ndarray) -> None:
    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=9) as destination:
        np.save(destination, values, allow_pickle=False)


def freeze(data_root: Path, output_dir: Path, source_commit: str) -> tuple[Path, Path]:
    """Freeze all three fixtures and return feature and manifest paths."""
    if not source_commit.strip():
        raise ValueError("source_commit must be non-empty")
    feature_families = _feature_family_names()
    fixture_chunks: list[np.ndarray] = []
    fixture_summaries: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    for name, (video_id, fps) in FIXTURE_SPECS.items():
        fixture = FixtureSpec(name, video_id, fps)
        rows, summary = _fixture_rows(data_root, fixture)
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
        input_rows.append({"fixture": name, "files": files})
    rows = np.concatenate(fixture_chunks)
    order = np.lexsort((rows["frame"], rows["span_id"], rows["fixture"]))
    rows = rows[order]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    feature_path = output_root / FEATURE_FILENAME
    _write_npy_xz(feature_path, rows)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "labels_read": False,
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
    manifest_path = output_root / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return feature_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    feature_path, manifest_path = freeze(arguments.data_root, arguments.output_dir, arguments.source_commit)
    print(f"wrote {feature_path}")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
