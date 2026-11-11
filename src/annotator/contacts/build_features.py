"""Measure candidate contacts for model training and prediction.

Each row combines shuttle motion, tracked-player positions, standing counts and visibility
near a candidate frame. The heuristic route does not need these features.
Motion uses per-frame units. Time offsets and search windows scale with
video frame rate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from annotator.array_calc_helpers import true_runs
from annotator.contacts.heuristics import span_impulses
from annotator.types import WRIST_L, WRIST_R, SmoothingMode, TrackedPlayers, WristGaps
from shared.frame_timing import frames_at_fps
from shared.progress import progress_iter

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

_FEATURE_COLUMNS = tuple(
    (signal, offset, f"{signal}_t{offset:+d}")
    for signal in BASE_PHYSICS_SIGNALS + BASE_MISSINGNESS_SIGNALS + ("standing_count",)
    for offset in WINDOW_OFFSETS_BASE30
)
CONTACT_FEATURE_NAMES = tuple(name for _signal, _offset, name in _FEATURE_COLUMNS)


class ContactFeatures(NamedTuple):
    """Candidate measurements and the frame ranges used to build them.

    ``rows`` has one structured row per candidate frame, sorted by frame.
    It includes frame/search-interval IDs and the named model features.
    ``eligible_intervals`` are tracked court ranges with excluded footage
    removed. ``search_intervals`` extend them backwards to find earlier serves
    and merge overlaps. All ranges include start, exclude end.
    """

    rows: np.ndarray
    eligible_intervals: list[tuple[int, int]]
    search_intervals: list[tuple[int, int]]


def build_contact_features(
    track: np.ndarray,
    pose_kps: np.ndarray,
    players: TrackedPlayers,
    wrist_gaps: WristGaps,
    tracker_intervals: Sequence[tuple[int, int]],
    exclusion_mask: np.ndarray,
    heuristic_spans: Sequence[tuple[int, int]],
    raw_contact_frames: Sequence[int],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
    resolution: tuple[float, float],
) -> ContactFeatures:
    """Measure candidate frames from shuttle motion, players and initial rallies.

    :param track: (frames, 3) normalised shuttle [x, y, visible].
    :param pose_kps: Pixel keypoints indexed by frame, pose detection and joint.
    :param players: Selected detections, body positions and standing counts.
    :param wrist_gaps: Shuttle-to-wrist distances in body heights.
    :param tracker_intervals: Court-view ranges, incl start, excl end.
    :param exclusion_mask: True on footage excluded from annotation.
    :param heuristic_spans: Initial rally ranges, incl start, excl end.
    :param raw_contact_frames: All heuristic candidate frames, before filtering.
    :param scene_spans: Camera-scene ranges in time order, incl start, excl end.
    :param fps: Video frame rate.
    :param resolution: Video width and height in pixels.
    :return: Candidate rows sorted by frame and the ranges used to measure them.
    """
    if exclusion_mask.shape != (len(track),) or exclusion_mask.dtype != np.bool_:
        raise ValueError("exclusion mask must match the shuttle timeline")
    eligible_intervals = build_eligible_intervals(tracker_intervals, exclusion_mask)
    search_intervals = extend_intervals_with_lookback(eligible_intervals, len(track), fps)
    signals = build_shuttle_signals(track, search_intervals, fps)
    signals.update(build_player_signals(track, pose_kps, players, wrist_gaps, resolution))
    candidates = build_candidate_mask(
        signals,
        eligible_intervals,
        heuristic_spans,
        raw_contact_frames,
        scene_spans,
        fps,
    )
    dtype = contact_feature_dtype()
    # frames_at_fps returns at least one frame, so set the centre offset directly.
    scaled_offsets = {0: 0}
    for offset in WINDOW_OFFSETS_BASE30:
        if offset:
            scaled_offsets[offset] = int(math.copysign(frames_at_fps(abs(offset), fps), offset))
    chunks: list[np.ndarray] = []
    for interval_id, (start, end) in enumerate(progress_iter(search_intervals, "building contact features", unit="intervals")):
        frames = np.arange(start, end, dtype=np.int32)
        frames = frames[candidates[start:end]]
        rows = np.zeros(len(frames), dtype=dtype)
        rows["interval_id"] = interval_id
        rows["frame"] = frames
        # Offsets refer to video frames, including frames we did not nominate.
        for signal, offset_base30, name in _FEATURE_COLUMNS:
            rows[name] = shift_within_interval(signals[signal], frames, scaled_offsets[offset_base30], start, end)

        chunks.append(rows)

    rows = np.concatenate(chunks) if chunks else np.empty(0, dtype=dtype)
    return ContactFeatures(rows, eligible_intervals, search_intervals)


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
        for run_start, run_end in true_runs(~exclusion_mask[start:end]):
            eligible.append((start + run_start, start + run_end))
    return eligible


def extend_intervals_with_lookback(
    eligible_intervals: Sequence[tuple[int, int]],
    frame_count: int,
    fps: float,
) -> list[tuple[int, int]]:
    """Extend court-view intervals backwards to find earlier serves; merge overlaps."""
    lookback = frames_at_fps(SERVE_LOOKBACK_BASE30, fps)
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


def build_shuttle_signals(
    track: np.ndarray,
    spans: Sequence[tuple[int, int]],
    fps: float,
) -> dict[str, np.ndarray]:
    """Measure shuttle movement at each frame and impulses within search ranges.

    Motion features use the smoothing settings used to train the contact model,
    independently of the current heuristic settings.
    """
    n_frames = len(track)
    visible = track[:, 2] == 1
    shuttle_xy = np.where(visible[:, None], track[:, :2], np.nan)
    vx, vy = frame_differences(shuttle_xy).T
    speed = np.hypot(vx, vy)
    impulse = np.full(n_frames, np.nan, dtype=np.float64)
    impulse_ratio = np.full(n_frames, np.nan, dtype=np.float64)
    smooth_window = frames_at_fps(3, fps)
    impulse_floor_half_window = frames_at_fps(12, fps)
    for start, end in spans:
        measured = span_impulses(
            track, start, end, smooth_window=smooth_window,
            impulse_floor_half_window=impulse_floor_half_window,
            smoothing_mode=SmoothingMode.INCLUDE_INVISIBLE,
        )
        if measured is None:
            continue
        magnitudes, ratios, _visible = measured
        frames = slice(start + 1, end - 1)
        impulse[frames] = magnitudes
        impulse_ratio[frames] = ratios
    return {
        "shuttle_visible": visible.astype(np.float32),
        "shuttle_vx": finite_to_fp32_inf_as_nan(vx),
        "shuttle_vy": finite_to_fp32_inf_as_nan(vy),
        "shuttle_speed": finite_to_fp32_inf_as_nan(speed),
        "shuttle_impulse": finite_to_fp32_inf_as_nan(impulse),
        "shuttle_impulse_ratio": finite_to_fp32_inf_as_nan(impulse_ratio),
    }


def build_player_signals(
    track: np.ndarray,
    pose_kps: np.ndarray,
    players: TrackedPlayers,
    wrist_gaps: WristGaps,
    resolution: tuple[float, float],
) -> dict[str, np.ndarray]:
    """Measure player movement and shuttle-to-wrist positions at each video frame."""
    n_frames = len(track)
    width, height = resolution
    scale = np.asarray([width, height], dtype=np.float64)
    wrist_xy = np.full((n_frames, 2, 2), np.nan, dtype=np.float64)
    visible = track[:, 2] == 1
    for player_row in range(2):
        valid_frames = np.flatnonzero((players.detection_indices[:, player_row] >= 0) & visible)
        if not len(valid_frames):
            continue
        detection_indices = players.detection_indices[valid_frames, player_row].astype(int)
        wrists = pose_kps[valid_frames, detection_indices][:, (WRIST_L, WRIST_R), :] / scale
        shuttle_xy = track[valid_frames, :2]
        gaps = np.linalg.norm(wrists - shuttle_xy[:, None, :], axis=2)
        closest = np.argmin(gaps, axis=1)
        wrist_xy[valid_frames, player_row] = wrists[np.arange(len(valid_frames)), closest]

    slot_gaps = finite_to_fp32_inf_as_nan(np.asarray(wrist_gaps.per_player, dtype=np.float64))
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

    ankle = finite_to_fp32_inf_as_nan(np.asarray(players.ankle_xy, dtype=np.float64))
    ankle_steps = frame_differences(ankle)  # (frame, court half, xy).
    ankle_speed = np.hypot(ankle_steps[:, :, 0], ankle_steps[:, :, 1])

    return {
        "wrist_gap_min": gap_min,
        "wrist_gap_top": slot_gaps[:, 0],
        "wrist_gap_bot": slot_gaps[:, 1],
        "nearest_wrist_dx": finite_to_fp32_inf_as_nan(relative[:, 0]),
        "nearest_wrist_dy": finite_to_fp32_inf_as_nan(relative[:, 1]),
        "ankle_speed_top": finite_to_fp32_inf_as_nan(ankle_speed[:, 0]),
        "ankle_speed_bot": finite_to_fp32_inf_as_nan(ankle_speed[:, 1]),
        "pose_valid_top": (players.detection_indices[:, 0] >= 0).astype(np.float32),
        "pose_valid_bot": (players.detection_indices[:, 1] >= 0).astype(np.float32),
        "wrist_valid_top": np.isfinite(slot_gaps[:, 0]).astype(np.float32),
        "wrist_valid_bot": np.isfinite(slot_gaps[:, 1]).astype(np.float32),
        "standing_count": players.standing_count,
    }


def frame_differences(values: np.ndarray) -> np.ndarray:
    """Frame-to-frame changes; a coordinate stays NaN if either value is missing."""
    result = np.full_like(values, np.nan, dtype=np.float64)
    valid = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    np.subtract(values[1:], values[:-1], out=result[1:], where=valid)
    return result


def finite_to_fp32_inf_as_nan(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan).astype(np.float32)


def build_candidate_mask(
    signals: Mapping[str, np.ndarray],
    eligible_intervals: Sequence[tuple[int, int]],
    rally_spans: Sequence[tuple[int, int]],
    raw_contact_frames: Sequence[int],
    scene_spans: Sequence[tuple[int, int]],
    fps: float,
) -> np.ndarray:
    """Choose frames to score from motion, wrist proximity and rally/scene starts.

    Expand each clue into a nearby region within an eligible court interval.
    Also include every frame in the earlier-serve search before each interval.
    """
    n_frames = len(signals["shuttle_visible"])
    seeds = {name: np.zeros(n_frames, dtype=bool) for name in REGION_RADII_BASE30}
    for frame in raw_contact_frames:
        if 0 <= frame < n_frames:
            seeds["region_current_raw"][frame] = True
    seeds["region_relaxed_impulse"] = (
        np.isfinite(signals["shuttle_impulse_ratio"])
        & (signals["shuttle_impulse_ratio"] >= RELAXED_IMPULSE_MULTIPLE)
    )
    wrist_radius = frames_at_fps(WRIST_MINIMUM_RADIUS_BASE30, fps)
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

    candidates = np.zeros(n_frames, dtype=bool)
    serve_lookback = frames_at_fps(SERVE_LOOKBACK_BASE30, fps)
    for start, end in eligible_intervals:
        for name, radius_at_30_fps in REGION_RADII_BASE30.items():
            radius = frames_at_fps(radius_at_30_fps, fps)
            candidates[start:end] |= expand_seed_region(seeds[name][start:end], radius)
        candidates[max(0, start - serve_lookback) : start] = True
    return candidates


def find_local_minima(values: np.ndarray, limit: float, radius: int) -> np.ndarray:
    """Mark finite values no greater than the limit or any neighbour within the radius."""
    finite = np.isfinite(values) & (values <= limit)
    minima = np.zeros(len(values), dtype=bool)
    for frame in np.flatnonzero(finite):
        start = max(0, frame - radius)
        end = min(len(values), frame + radius + 1)
        window = values[start:end]
        if values[frame] == np.nanmin(window):
            minima[frame] = True
    return minima


def expand_seed_region(seed: np.ndarray, radius: int) -> np.ndarray:
    """Mark frames within radius of any True sample; keep the input length."""
    if not seed.any():
        return np.zeros(len(seed), dtype=bool)
    kernel = np.ones(2 * radius + 1, dtype=np.int16)
    full = np.convolve(seed.astype(np.int16), kernel, mode="full")
    return full[radius : radius + len(seed)] > 0


def contact_feature_dtype() -> np.dtype:
    """Structured row format: search-interval ID, video frame and named float features."""
    fields = [("interval_id", "<i4"), ("frame", "<i4")]
    fields.extend((name, "<f4") for name in CONTACT_FEATURE_NAMES)
    return np.dtype(fields)


def shift_within_interval(
    values: np.ndarray,
    frames: np.ndarray,
    offset: int,
    start: int,
    end: int,
) -> np.ndarray:
    """Read values at frames + offset; NaN outside the range (incl start, excl end)."""
    source = frames + offset
    result = np.full(len(frames), np.nan, dtype=np.float32)
    valid = (source >= start) & (source < end)
    result[valid] = values[source[valid]]
    return result
