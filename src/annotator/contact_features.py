"""Build label-free contact features from in-memory vision and heuristic evidence.

Motion stays in raw per-frame units, matching the trained contact tree. Feature
windows and candidate regions scale with FPS. Windows stop at search-interval
boundaries, including the serve lookback before each eligible interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from annotator.types import SmoothingMode, StickyResult

WINDOW_OFFSETS_BASE30 = (-10, -5, 0, 5, 10)
RELAXED_IMPULSE_MULTIPLE = 1.25
WRIST_LOCAL_MINIMUM_LIMIT = 3.0
WRIST_MINIMUM_RADIUS_BASE30 = 3
SERVE_LOOKBACK_BASE30 = 45
REGION_RADII_BASE30 = {
    "region_current_raw": 15,
    "region_relaxed_impulse": 15,
    "region_wrist": 10,
    "region_visibility": 15,
    "region_rally_start": 45,
    "region_scene_start": 15,
}

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


def frame_differences(values: np.ndarray) -> np.ndarray:
    """Return frame-aligned first differences without bridging missing rows."""
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    valid_frames = np.flatnonzero(valid) + 1
    result[valid_frames] = values[valid_frames] - values[valid_frames - 1]
    return result


def build_player_signals(
    track: np.ndarray,
    pose_kps: np.ndarray,
    sticky: StickyResult,
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
    ankle_dx = np.column_stack([frame_differences(ankle[:, slot, 0]) for slot in range(2)])
    ankle_dy = np.column_stack([frame_differences(ankle[:, slot, 1]) for slot in range(2)])
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


def build_shuttle_signals(
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
    vx = frame_differences(x)
    vy = frame_differences(y)
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
        # The tree was trained on zero-filled impulse smoothing, including invisible
        # shuttle rows. Changing it to ignore missing rows changes model inputs.
        span_values = span_impulses(track, start, end, thresholds, smoothing_mode=SmoothingMode.ZERO_FILL)
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


def find_local_minima(values: np.ndarray, limit: float, radius: int) -> np.ndarray:
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
    lookback = _scaled_frames(SERVE_LOOKBACK_BASE30, fps)
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


def expand_seed_region(seed: np.ndarray, start: int, end: int, radius: int) -> np.ndarray:
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
    raw_contact_frames: Sequence[int],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Build broad search regions without labels or GT-derived boundaries."""
    n_frames = len(signals["shuttle_visible"])
    seeds = {name: np.zeros(n_frames, dtype=bool) for name in REGION_FIELDS}
    for frame in raw_contact_frames:
        if 0 <= frame < n_frames:
            seeds["region_current_raw"][frame] = True
    seeds["region_relaxed_impulse"] = (
        np.isfinite(signals["shuttle_impulse_ratio"])
        & (signals["shuttle_impulse_ratio"] >= RELAXED_IMPULSE_MULTIPLE)
    )
    wrist_radius = _scaled_frames(WRIST_MINIMUM_RADIUS_BASE30, fps)
    visible = signals["shuttle_visible"].astype(bool)
    for start, end in eligible_intervals:
        seeds["region_wrist"][start:end] = find_local_minima(
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

    regions = {name: np.zeros(n_frames, dtype=bool) for name in REGION_FIELDS}
    serve_lookback = _scaled_frames(SERVE_LOOKBACK_BASE30, fps)
    for start, end in eligible_intervals:
        for name in REGION_FIELDS[:-1]:
            radius = _scaled_frames(REGION_RADII_BASE30[name], fps)
            regions[name] |= expand_seed_region(seeds[name], start, end, radius)
        regions["region_serve_lookback"][max(0, start - serve_lookback) : start] = True
    return regions


def shift_within_interval(
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


def contact_feature_families() -> dict[str, list[str]]:
    physics = [f"{signal}_t{offset:+d}" for signal in BASE_PHYSICS_SIGNALS for offset in WINDOW_OFFSETS_BASE30]
    missingness = [
        f"{signal}_t{offset:+d}"
        for signal in BASE_MISSINGNESS_SIGNALS
        for offset in WINDOW_OFFSETS_BASE30
    ]
    return {"physics": physics, "context": list(CONTEXT_FIELDS), "missingness": missingness}


def contact_feature_dtype(feature_families: Mapping[str, Sequence[str]], identity_bytes: int) -> np.dtype:
    fields: list[tuple[str, str]] = [
        ("fixture", f"S{identity_bytes}"),
        ("interval_id", "<i4"),
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


class ContactFeatures(NamedTuple):
    """Feature rows and the interval boundaries already used to build them."""

    rows: np.ndarray
    eligible_intervals: list[tuple[int, int]]
    search_intervals: list[tuple[int, int]]


def build_contact_features(
    track: np.ndarray,
    pose_kps: np.ndarray,
    sticky: StickyResult,
    tracker_intervals: Sequence[tuple[int, int]],
    exclusion_mask: np.ndarray,
    heuristic_spans: Sequence[tuple[int, int]],
    raw_contact_frames: Sequence[int],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
    resolution: tuple[float, float],
    video_identity: str,
) -> ContactFeatures:
    """Build one video's feature rows without reading files or contact labels.

    :param track: Frame-aligned normalised shuttle ``[x, y, visible]``.
    :param pose_kps: Pixel keypoints indexed by frame, raw player slot and joint.
    :param sticky: Cached player selections and body-height wrist distances.
    :param tracker_intervals: Court-present half-open frame intervals.
    :param exclusion_mask: Boolean replay/broadcast exclusion for every frame.
    :param heuristic_spans: Half-open rally spans from heuristic segmentation.
    :param raw_contact_frames: Heuristic contact seeds before filtering.
    :param scene_spans: Half-open scene spans in chronological order.
    :param fps: Source video frame rate.
    :param resolution: Source width and height in pixels.
    :param video_identity: UTF-8 video identity stored without truncation.
    :return: All search rows, including rows outside candidate regions, and intervals.
    """
    if exclusion_mask.shape != (len(track),) or exclusion_mask.dtype != np.bool_:
        raise ValueError("exclusion mask must match the shuttle timeline")
    encoded_identity = video_identity.encode("utf-8")
    eligible_intervals = build_eligible_intervals(tracker_intervals, exclusion_mask)
    search_intervals = extend_intervals_with_lookback(eligible_intervals, len(track), fps)
    signals = build_shuttle_signals(track, search_intervals, fps)
    signals.update(build_player_signals(track, pose_kps, sticky, resolution))
    signals["standing_count"] = np.asarray(sticky.standing_count, dtype=np.float32)
    regions = build_region_masks(
        signals,
        eligible_intervals,
        heuristic_spans,
        raw_contact_frames,
        scene_spans,
        fps,
    )
    feature_families = contact_feature_families()
    dtype = contact_feature_dtype(feature_families, max(1, len(encoded_identity)))
    chunks: list[np.ndarray] = []
    scene_starts = np.asarray([start for start, _end in scene_spans], dtype=int)
    for interval_id, (start, end) in enumerate(search_intervals):
        frames = np.arange(start, end, dtype=np.int32)
        rows = np.zeros(len(frames), dtype=dtype)
        rows["fixture"] = encoded_identity
        rows["interval_id"] = interval_id
        rows["frame"] = frames
        rows["fps"] = fps
        for name in REGION_FIELDS:
            rows[name] = regions[name][frames]

        for signal in BASE_PHYSICS_SIGNALS + BASE_MISSINGNESS_SIGNALS:
            for offset_base30 in WINDOW_OFFSETS_BASE30:
                offset = 0
                if offset_base30:
                    offset = int(math.copysign(_scaled_frames(abs(offset_base30), fps), offset_base30))
                rows[f"{signal}_t{offset_base30:+d}"] = shift_within_interval(
                    signals[signal], frames, offset, start, end
                )

        for name in (
            "shuttle_x", "shuttle_y", "ankle_x_top", "ankle_y_top", "ankle_x_bot", "ankle_y_bot",
            "bbox_height_top", "bbox_height_bot", "standing_count",
        ):
            rows[name] = signals[name][frames]
        rows["interval_progress"] = (frames - start) / max(1, end - start - 1)
        rows["distance_from_interval_start"] = (frames - start) / fps
        rows["distance_to_interval_end"] = (end - 1 - frames) / fps
        preceding_scene = np.searchsorted(scene_starts, frames, side="right") - 1
        scene_distance = np.full(len(frames), np.nan, dtype=np.float32)
        has_scene = preceding_scene >= 0
        scene_distance[has_scene] = (frames[has_scene] - scene_starts[preceding_scene[has_scene]]) / fps
        rows["distance_from_scene_start"] = scene_distance
        chunks.append(rows)

    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
    return ContactFeatures(rows, eligible_intervals, search_intervals)
