"""Refine one court while keeping raw-fragment and finite marking identities fixed."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from .assignment import (
    MARKING_INTERVALS,
    SUPPORT_DISTANCE_PX,
    Observations,
    distances_to_segments,
)
from .detector import CORNER_COURT_M, SEGMENTS_M, UNIT_CORNERS, project
from .junction_observations import PRESENT_SUPPORT
from .paint_geometry import positioned_segments

MAX_EVALUATIONS = 100


@dataclass(frozen=True)
class Constraints:
    points: np.ndarray  # One observed xy per retained fragment sample, in working pixels.
    intervals: np.ndarray  # Finite template interval ID for each point.
    positions: np.ndarray
    weights: np.ndarray
    fragment_ids: np.ndarray
    sample_ids: np.ndarray


def shifted_intervals(
    intervals: np.ndarray, positions: np.ndarray, centres: np.ndarray = SEGMENTS_M,
) -> np.ndarray:
    """Keep the metric stripe positions consistent with the observation experiment."""
    return positioned_segments(centres, intervals, positions)


def prepare(
    homography: np.ndarray, observations: Observations, assignments: dict, fragment_weights: np.ndarray,
    centres: np.ndarray = SEGMENTS_M,
) -> Constraints:
    """Freeze the local fitting subset and each sample's finite interval identity."""
    points, intervals, positions, weights, fragment_ids, sample_ids = [], [], [], [], [], []
    for fragment, strength in enumerate(assignments["strength"]):
        if strength < PRESENT_SUPPORT:
            continue
        marking, position = assignments["marking"][fragment], assignments["position"][fragment]
        choices = np.asarray(MARKING_INTERVALS[marking])
        segments_m = shifted_intervals(choices, np.full(len(choices), position), centres)
        projected, _ = project(homography[None], segments_m)
        distances = distances_to_segments(observations.samples[fragment], projected.reshape(-1, 2, 2))
        nearest = distances.argmin(axis=1)
        selected = np.flatnonzero(distances[np.arange(len(nearest)), nearest] <= SUPPORT_DISTANCE_PX)
        for sample in selected:
            points.append(observations.samples[fragment, sample])
            intervals.append(choices[nearest[sample]])
            positions.append(position)
            weights.append(fragment_weights[fragment] / len(selected))
            fragment_ids.append(observations.fragment_ids[fragment])
            sample_ids.append(sample)
    return Constraints(np.asarray(points).reshape(-1, 2), np.asarray(intervals, dtype=int),
                       np.asarray(positions, dtype=int), np.asarray(weights), np.asarray(fragment_ids, dtype=int),
                       np.asarray(sample_ids, dtype=int))


def residual(parameters: np.ndarray, points: np.ndarray, segments: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return weighted xy residuals to the assigned finite segments, in working pixels."""
    homography = np.append(parameters, 1.0).reshape(3, 3)
    projected, _ = project(homography[None], segments)
    projected = projected.reshape(-1, 2, 2)
    vectors = projected[:, 1] - projected[:, 0]
    fraction = np.einsum("pd,pd->p", points - projected[:, 0], vectors) / np.square(vectors).sum(axis=1)
    closest = projected[:, 0] + np.clip(fraction, 0, 1)[:, None] * vectors
    return ((points - closest) * weights[:, None]).ravel()


def initial_parameters(corners: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Expose the exact normalised state received by the optimiser for cache identity."""
    transform = cv2.getPerspectiveTransform(UNIT_CORNERS, (corners / max(size)).astype(np.float32))
    return (transform / transform[2, 2]).ravel()[:8]


def refine(
    corners: np.ndarray, constraints: Constraints, size: tuple[int, int], use_positions: bool,
    initial: np.ndarray | None = None,
    centres: np.ndarray = SEGMENTS_M,
) -> dict:
    """Fit one fixed interpretation; diagnostic failures never become accepted courts."""
    if len(constraints.points) < 4:
        return {"status": "insufficient_samples", "successful": False, "corners_px": None, "nfev": 0}
    image_scale = max(size)
    court_scale = CORNER_COURT_M.max(axis=0)
    positions = constraints.positions if use_positions else np.zeros_like(constraints.positions)
    segments = shifted_intervals(constraints.intervals, positions, centres) / court_scale
    points = constraints.points / image_scale
    weights = np.sqrt(constraints.weights / constraints.weights.sum()) * image_scale
    if initial is None:
        initial = initial_parameters(corners, size)
    arguments = (points, segments, weights)
    before = residual(initial, *arguments)
    fit = least_squares(residual, initial, args=arguments, jac="3-point", max_nfev=MAX_EVALUATIONS)
    final = np.append(fit.x, 1.0).reshape(3, 3)
    projected, denominator = project(final[None], UNIT_CORNERS)
    attempted = projected[0] * image_scale
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    # A finite-difference Jacobian needs a noise floor above exact-arithmetic rank.
    rank = int((singular_values > np.sqrt(np.finfo(float).eps) * singular_values[0]).sum())
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None
    valid = bool(np.isfinite(attempted).all() and (denominator > 0).all())
    if not fit.success:
        status = "budget_exhausted" if fit.status == 0 else "solver_failed"
    elif not valid:
        status = "invalid_projection"
    elif rank < 8:
        status = "rank_deficient"
    else:
        status = "converged"
    return {"status": status, "successful": status == "converged", "nfev": fit.nfev,
            "corners_px": attempted.tolist() if np.isfinite(attempted).all() else None,
            "objective_before": float(before @ before), "objective_after": float(fit.fun @ fit.fun),
            "jacobian_rank": int(rank), "jacobian_condition": condition, "valid_projection": valid,
            "solver_status": fit.status, "minimum_corner_denominator": float(denominator.min())}
