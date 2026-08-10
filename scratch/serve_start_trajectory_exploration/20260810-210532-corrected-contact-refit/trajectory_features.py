"""Small, explicit measurements for the corrected serve-trajectory EDA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np

from annotator.point_winner import Half, _phase_assignment
from annotator.types import true_runs


class PreContactRun(NamedTuple):
    """A maximal usable shuttle run before an anchor contact.

    ``end`` is one past the final usable frame. ``gap_frames`` is the frame-index
    distance from the final usable sample to the contact, matching the EDA's
    maximum-gap threshold. A run ending at ``contact_frame - 1`` has a gap of 1.
    """

    start: int
    end: int
    frames_to_contact: int

    @property
    def gap_frames(self) -> int:
        """Compatibility name for the requested contact-gap measurement."""
        return self.frames_to_contact


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

    @property
    def frame_count(self) -> int:
        """Alias for the number of aligned frames."""
        return self.n_frames

class CurveFit(NamedTuple):
    """Linear and quadratic shuttle-path residual diagnostics."""

    linear_rmse: float
    quadratic_rmse: float
    quadratic_improvement: float


AnchorCategory = Literal["unmatched", "ambiguous", "contact_1", "contact_2", "later"]


def _integer(value: object, name: str) -> int:
    """Return an integer argument while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


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
