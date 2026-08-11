"""Small, explicit measurements for the corrected serve-trajectory EDA."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal, NamedTuple

import numpy as np

from annotator.fps_constants import ScalingKind
from annotator.point_winner import Half, _phase_assignment
from annotator.types import true_runs

MIN_PATH_FRAMES = 5
MIN_TOTAL_MOVEMENT_BH = 0.25
MAX_LARGEST_STEP_RATIO = 4.0
PRIMARY_MIN_NET_CLOSURE_BH = 0.25
HISTORICAL_MIN_CLOSING_FRACTION = 0.55
ROBUST_TREND_MIN_DECREASE_BH = 0.05


class PreContactRun(NamedTuple):
    """A maximal usable shuttle run before an anchor contact.

    ``end`` is one past the final usable frame. ``gap_frames`` is the frame-index
    distance from the final usable sample to the contact, matching the EDA's
    maximum-gap threshold. A run ending at ``contact_frame - 1`` has a gap of 1.
    """

    start: int
    end: int
    frames_to_contact: int


class IncomingMotion(NamedTuple):
    """Aligned shuttle and anchor-player motion measurements.

    Distances and shuttle movement are in player body heights. A stationary
    path has ``largest_step_ratio == 0`` because it has no non-zero step from
    which to form a jump ratio.
    """

    n_frames: int
    start_distance_bh: float
    end_distance_bh: float
    net_closure_bh: float
    closing_fraction: float
    total_movement_bh: float
    largest_step_ratio: float


class CurveFit(NamedTuple):
    """Linear and quadratic shuttle-path residual diagnostics."""

    linear_rmse: float
    quadratic_rmse: float
    quadratic_improvement: float


class AnchorAlignment(NamedTuple):
    """Nearest GT stroke and inclusive tolerance membership for one anchor."""

    nearest_gt_ordinal: int
    signed_offset_base30: float
    absolute_offset_base30: float
    in_window_count: int
    multiple_within_tolerance: bool
    label: Literal["unmatched", "contact_1", "contact_2", "later"]


class AcceptedSequenceSummary(NamedTuple):
    """Later GT matches after an anchor that is unmatched at the chosen tolerance."""

    later_contacts_checked: int
    later_serve_within_tolerance: bool
    later_first_return_within_tolerance: bool
    first_gt_match_rank: int | None
    first_gt_match_ordinal: int | None
    first_gt_match_multiple: bool
    reused_gt_ordinal: bool


class RobustDistanceTrend(NamedTuple):
    """Robust trend and residual diagnostics for shuttle-to-player distance."""

    slope_bh_per_path: float
    intercept_bh: float
    fitted_decrease_bh: float
    residual_rms_bh: float
    trend_to_jitter: float


class FixedRuleDecisions(NamedTuple):
    """Eligibility and calls for the two predeclared incoming-motion rules."""

    common_path_eligible: bool
    historical_path_eligible: bool
    historical_incoming: bool
    robust_trend_incoming: bool


AnchorCategory = Literal["unmatched", "ambiguous", "contact_1", "contact_2", "later"]


def _integer(value: object, name: str) -> int:
    """Return an integer argument while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _ordered_frames(values: Sequence[int] | np.ndarray, name: str) -> tuple[int, ...]:
    """Return a non-empty, strictly increasing sequence of integer frames."""
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    frames = tuple(_integer(value, name) for value in array)
    if not frames:
        raise ValueError(f"{name} must not be empty")
    if any(current <= previous for previous, current in pairwise(frames)):
        raise ValueError(f"{name} must be strictly increasing")
    return frames


def _scaled_tolerance(tolerance_base30: int, fps: float) -> tuple[int, float]:
    """Return the inclusive source-frame tolerance and checked frame rate."""
    tolerance = _integer(tolerance_base30, "tolerance_base30")
    if tolerance <= 0:
        raise ValueError("tolerance_base30 must be positive")
    if isinstance(fps, bool) or not isinstance(fps, (int, float, np.number)):
        raise TypeError("fps must be numeric")
    fps_value = float(fps)
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be positive and finite")
    source_tolerance = int(ScalingKind.FRAME_COUNT.scale(tolerance, fps_value))
    return source_tolerance, fps_value


def _alignment_label(nearest_ordinal: int, in_window_count: int) -> Literal[
    "unmatched", "contact_1", "contact_2", "later"
]:
    """Name the nearest stroke when at least one GT stroke is in tolerance."""
    if in_window_count == 0:
        return "unmatched"
    if nearest_ordinal == 1:
        return "contact_1"
    if nearest_ordinal == 2:
        return "contact_2"
    return "later"


def align_anchor_to_gt(
    anchor_frame: int,
    gt_stroke_frames: Sequence[int] | np.ndarray,
    fps: float,
    tolerance_base30: int,
) -> AnchorAlignment:
    """Retain nearest-stroke details separately from tolerance ambiguity."""
    anchor = _integer(anchor_frame, "anchor_frame")
    gt_frames = np.asarray(_ordered_frames(gt_stroke_frames, "gt_stroke_frames"), dtype=np.int64)
    source_tolerance, fps_value = _scaled_tolerance(tolerance_base30, fps)

    signed_source_offsets = anchor - gt_frames
    absolute_source_offsets = np.abs(signed_source_offsets)
    nearest_index = int(np.argmin(absolute_source_offsets))
    in_window_count = int(np.count_nonzero(absolute_source_offsets <= source_tolerance))
    signed_offset_base30 = float(signed_source_offsets[nearest_index] * 30.0 / fps_value)
    nearest_ordinal = nearest_index + 1
    return AnchorAlignment(
        nearest_ordinal,
        signed_offset_base30,
        abs(signed_offset_base30),
        in_window_count,
        in_window_count > 1,
        _alignment_label(nearest_ordinal, in_window_count),
    )


def summarise_unmatched_anchor_sequence(
    accepted_contact_frames: Sequence[int] | np.ndarray,
    gt_stroke_frames: Sequence[int] | np.ndarray,
    fps: float,
    tolerance_base30: int,
) -> AcceptedSequenceSummary:
    """Summarise independent later-contact matches after an unmatched anchor."""
    accepted = _ordered_frames(accepted_contact_frames, "accepted_contact_frames")
    gt_frames = np.asarray(_ordered_frames(gt_stroke_frames, "gt_stroke_frames"), dtype=np.int64)
    source_tolerance, _fps_value = _scaled_tolerance(tolerance_base30, fps)

    anchor_offsets = np.abs(gt_frames - accepted[0])
    if np.any(anchor_offsets <= source_tolerance):
        raise ValueError("the first accepted contact must be unmatched at the chosen tolerance")

    later_serve_match = False
    later_first_return_match = False
    first_match_rank: int | None = None
    first_match_ordinal: int | None = None
    first_match_multiple = False
    matches_per_ordinal = np.zeros(len(gt_frames), dtype=np.int64)

    for accepted_rank, contact_frame in enumerate(accepted[1:], start=2):
        absolute_offsets = np.abs(gt_frames - contact_frame)
        matching_ordinals = np.flatnonzero(absolute_offsets <= source_tolerance)
        if len(matching_ordinals) == 0:
            continue
        matches_per_ordinal[matching_ordinals] += 1
        later_serve_match |= bool(np.any(matching_ordinals == 0))
        later_first_return_match |= bool(np.any(matching_ordinals == 1))
        if first_match_rank is None:
            first_match_rank = accepted_rank
            first_match_ordinal = int(np.argmin(absolute_offsets)) + 1
            first_match_multiple = len(matching_ordinals) > 1

    return AcceptedSequenceSummary(
        len(accepted) - 1,
        later_serve_match,
        later_first_return_match,
        first_match_rank,
        first_match_ordinal,
        first_match_multiple,
        bool(np.any(matches_per_ordinal > 1)),
    )


def fit_robust_distance_trend(
    distances_bh: Sequence[float] | np.ndarray,
) -> RobustDistanceTrend:
    """Fit the predeclared pairwise-median trend over normalised sample time."""
    distances = np.asarray(distances_bh, dtype=float)
    if distances.ndim != 1:
        raise ValueError("distances_bh must be one-dimensional")
    if len(distances) < 2:
        raise ValueError("a distance trend requires at least two samples")
    if not np.isfinite(distances).all():
        raise ValueError("distances_bh must contain only finite values")

    sample_time = np.linspace(0.0, 1.0, len(distances))
    slopes = [
        (distances[end] - distances[start]) / (sample_time[end] - sample_time[start])
        for start in range(len(distances) - 1)
        for end in range(start + 1, len(distances))
    ]
    slope = float(np.median(slopes))
    intercept = float(np.median(distances - slope * sample_time))
    residuals = distances - (intercept + slope * sample_time)
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    fitted_decrease = -slope
    if residual_rms == 0.0:
        trend_to_jitter = math.copysign(math.inf, fitted_decrease) if fitted_decrease else 0.0
    else:
        trend_to_jitter = fitted_decrease / residual_rms
    return RobustDistanceTrend(
        slope,
        intercept,
        fitted_decrease,
        residual_rms,
        trend_to_jitter,
    )


def decide_fixed_motion_rules(
    motion: IncomingMotion,
    trend: RobustDistanceTrend,
    frames_to_contact: int,
    maximum_frames_to_contact: int,
) -> FixedRuleDecisions:
    """Apply the historical and 0.05-BH rules without score-driven selection."""
    frames_to_contact_value = _integer(frames_to_contact, "frames_to_contact")
    maximum_gap = _integer(maximum_frames_to_contact, "maximum_frames_to_contact")
    if frames_to_contact_value < 1 or maximum_gap < 1:
        raise ValueError("contact gaps must be positive")

    common_path_eligible = (
        motion.n_frames >= MIN_PATH_FRAMES
        and frames_to_contact_value <= maximum_gap
        and motion.largest_step_ratio <= MAX_LARGEST_STEP_RATIO
    )
    historical_path_eligible = (
        common_path_eligible and motion.total_movement_bh >= MIN_TOTAL_MOVEMENT_BH
    )
    historical_incoming = (
        historical_path_eligible
        and motion.net_closure_bh >= PRIMARY_MIN_NET_CLOSURE_BH
        and motion.closing_fraction >= HISTORICAL_MIN_CLOSING_FRACTION
    )
    robust_trend_incoming = (
        common_path_eligible and trend.fitted_decrease_bh >= ROBUST_TREND_MIN_DECREASE_BH
    )
    return FixedRuleDecisions(
        common_path_eligible,
        historical_path_eligible,
        historical_incoming,
        robust_trend_incoming,
    )


def closest_pre_contact_run(
    usable: np.ndarray,
    contact_frame: int,
    lookback_frames: int,
    same_scene_mask: np.ndarray | None = None,
) -> PreContactRun | None:
    """Select the latest maximal usable run in a strict pre-contact window.

    The search is limited to ``[contact_frame - lookback_frames, contact_frame)``.
    If ``same_scene_mask`` is supplied, a frame is usable only when both masks
    are true. The contact frame itself is never considered.
    """
    contact = _integer(contact_frame, "contact_frame")
    lookback = _integer(lookback_frames, "lookback_frames")
    usable_array = np.asarray(usable)
    if usable_array.ndim != 1:
        raise ValueError("usable must be a one-dimensional mask")
    if not 0 <= contact <= len(usable_array):
        raise ValueError("contact_frame must be within the usable mask")
    if lookback < 0:
        raise ValueError("lookback_frames must be non-negative")

    if same_scene_mask is not None:
        scene_array = np.asarray(same_scene_mask)
        if scene_array.shape != usable_array.shape:
            raise ValueError("same_scene_mask must have the same shape as usable")
        usable_array = usable_array.astype(bool) & scene_array.astype(bool)
    else:
        usable_array = usable_array.astype(bool)

    window_start = max(0, contact - lookback)
    window = usable_array[window_start:contact]
    runs = true_runs(window)
    if not runs:
        return None

    relative_start, relative_end = runs[-1]
    start = window_start + relative_start
    end = window_start + relative_end
    return PreContactRun(start, end, contact - (end - 1))


def measure_incoming_motion(
    distances_bh: np.ndarray,
    shuttle_xy: np.ndarray,
    bbox_heights_px: np.ndarray,
    resolution: tuple[float, float],
) -> IncomingMotion:
    """Measure whether a visible path closes on its anchor player.

    ``shuttle_xy`` is normalised ``(x, y)`` image position and ``resolution``
    is ``(width, height)`` in pixels. Each step is divided by the destination
    frame's anchor-player bbox height. All four inputs must be finite and
    frame-aligned. At least two frames are required.
    """
    distances = np.asarray(distances_bh, dtype=float)
    shuttle = np.asarray(shuttle_xy, dtype=float)
    heights = np.asarray(bbox_heights_px, dtype=float)
    if distances.ndim != 1 or heights.ndim != 1 or shuttle.ndim != 2 or shuttle.shape[1:] != (2,):
        raise ValueError("motion arrays must have shapes (frames,), (frames, 2), and (frames,)")
    n_frames = len(distances)
    if n_frames < 2:
        raise ValueError("motion measurements require at least two frames")
    if len(shuttle) != n_frames or len(heights) != n_frames:
        raise ValueError("motion arrays must have the same frame count")
    if not np.isfinite(distances).all() or not np.isfinite(shuttle).all() or not np.isfinite(heights).all():
        raise ValueError("motion arrays must contain only finite values")
    if np.any(heights <= 0):
        raise ValueError("bbox_heights_px must be positive")

    resolution_array = np.asarray(resolution, dtype=float)
    if resolution_array.shape != (2,) or not np.isfinite(resolution_array).all():
        raise ValueError("resolution must contain two finite values")
    if np.any(resolution_array <= 0):
        raise ValueError("resolution must contain two positive values")

    distance_changes = np.diff(distances)
    step_pixels = np.linalg.norm(np.diff(shuttle, axis=0) * resolution_array, axis=1)
    step_bh = step_pixels / heights[1:]
    non_zero_steps = step_bh[step_bh > 0]
    if len(non_zero_steps) == 0:
        largest_step_ratio = 0.0
    else:
        largest_step_ratio = float(np.max(step_bh) / np.median(non_zero_steps))

    return IncomingMotion(
        n_frames,
        float(distances[0]),
        float(distances[-1]),
        float(distances[0] - distances[-1]),
        float(np.mean(distance_changes < 0)),
        float(np.sum(step_bh)),
        largest_step_ratio,
    )


def fit_path(points: np.ndarray) -> CurveFit:
    """Fit x/y against frame time and compare linear and quadratic residuals."""
    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[1:] != (2,) or len(path) < 2:
        raise ValueError("points must have shape (frames, 2) with at least two frames")
    if not np.isfinite(path).all():
        raise ValueError("points must contain only finite values")

    frame_numbers = np.arange(len(path), dtype=float)
    linear_design = np.column_stack((frame_numbers, np.ones(len(path))))
    linear_coefficients, *_ = np.linalg.lstsq(linear_design, path, rcond=None)
    linear_residual = path - linear_design @ linear_coefficients
    linear_rmse = float(np.sqrt(np.mean(np.sum(linear_residual**2, axis=1))))

    if len(path) < 5:
        return CurveFit(linear_rmse, float("nan"), float("nan"))

    quadratic_design = np.column_stack((frame_numbers**2, frame_numbers, np.ones(len(path))))
    quadratic_coefficients, *_ = np.linalg.lstsq(quadratic_design, path, rcond=None)
    quadratic_residual = path - quadratic_design @ quadratic_coefficients
    quadratic_rmse = float(np.sqrt(np.mean(np.sum(quadratic_residual**2, axis=1))))
    improvement = 0.0 if linear_rmse == 0 else 1.0 - quadratic_rmse / linear_rmse
    return CurveFit(linear_rmse, quadratic_rmse, improvement)


def classify_anchor_frame(
    anchor_frame: int,
    gt_stroke_frames: Sequence[int] | np.ndarray,
    tolerance_frames: int,
) -> AnchorCategory:
    """Classify an anchor against ordered ground-truth stroke frames.

    Exactly one frame within the inclusive tolerance is labelled ``contact_1``,
    ``contact_2`` or ``later``. Zero matches is ``unmatched`` and multiple
    matches is ``ambiguous``.
    """
    anchor = _integer(anchor_frame, "anchor_frame")
    tolerance = _integer(tolerance_frames, "tolerance_frames")
    if tolerance < 0:
        raise ValueError("tolerance_frames must be non-negative")

    frames = np.asarray(gt_stroke_frames)
    if frames.ndim != 1:
        raise ValueError("gt_stroke_frames must be one-dimensional")
    matches = np.flatnonzero(np.abs(frames.astype(np.int64) - anchor) <= tolerance)
    if len(matches) == 0:
        return "unmatched"
    if len(matches) > 1:
        return "ambiguous"
    ordinal = int(matches[0])
    if ordinal == 0:
        return "contact_1"
    if ordinal == 1:
        return "contact_2"
    return "later"


def first_player_from_final_half(final_half: Half | None, contact_count: int) -> Half | None:
    """Return the first player implied by a fitted final half and contact count."""
    count = _integer(contact_count, "contact_count")
    if count < 1:
        raise ValueError("contact_count must be positive when final_half is fitted")
    if final_half is None:
        return None
    return _phase_assignment(final_half, count)[0]
