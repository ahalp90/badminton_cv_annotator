"""Focused checks for fixed finite-stripe court refits."""

from __future__ import annotations

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as refit
from experiments.annotator.independent_court.assignment import Observations
from experiments.annotator.independent_court.detector import CORNER_COURT_M, project

IMAGE_SIZE = (960, 720)
KNOWN_CORNERS = np.array(
    [[180.0, 110.0], [810.0, 145.0], [760.0, 620.0], [205.0, 580.0]],
)


def _known_homography() -> np.ndarray:
    return cv2.getPerspectiveTransform(CORNER_COURT_M, KNOWN_CORNERS.astype(np.float32))


def _project_interval_samples(
    homography: np.ndarray,
    intervals: np.ndarray,
    positions: np.ndarray,
    fractions: np.ndarray,
) -> np.ndarray:
    metric_points = []
    for interval, position in zip(intervals, positions):
        segment = refit.shifted_intervals(
            np.asarray([interval]), np.asarray([position]),
        )[0]
        metric_points.extend(segment[0] + fractions[:, None] * (segment[1] - segment[0]))
    projected, _ = project(homography[None], np.asarray(metric_points))
    return projected[0]


def _constraints(
    intervals: tuple[int, ...],
    positions: tuple[int, ...],
    samples_per_interval: int = 12,
) -> refit.Constraints:
    interval_ids = np.asarray(intervals, dtype=int)
    position_ids = np.asarray(positions, dtype=int)
    fractions = np.linspace(0.12, 0.88, samples_per_interval)
    points = _project_interval_samples(_known_homography(), interval_ids, position_ids, fractions)
    return refit.Constraints(
        points=points,
        intervals=np.repeat(interval_ids, samples_per_interval),
        positions=np.repeat(position_ids, samples_per_interval),
        weights=np.ones(len(points)),
        fragment_ids=np.arange(len(points)),
        sample_ids=np.arange(len(points)),
    )


def _observations_for_prepare() -> tuple[np.ndarray, Observations]:
    intervals = np.array([2, 3, 0], dtype=int)
    positions = np.array([1, 2, 0], dtype=int)
    fractions = np.linspace(0.20, 0.80, 16)
    samples = np.stack([
        _project_interval_samples(_known_homography(), np.asarray([interval]), np.asarray([position]), fractions)
        for interval, position in zip(intervals, positions)
    ])
    segments = samples[:, [0, -1]]
    vectors = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    directions = vectors / lengths[:, None]
    groups = tuple(np.asarray([index]) for index in range(len(samples)))
    observations = Observations(
        segments=segments,
        fragment_ids=np.array([101, 202, 303]),
        groups=groups,
        group_lengths=lengths,
        directions=directions,
        lengths=lengths,
        samples=samples,
    )
    return positions, observations


def test_position_aware_refine_recovers_known_geometry_from_edge_samples() -> None:
    constraints = _constraints((0, 1, 4, 6, 7, 10, 11), (1, 2, 1, 2, 1, 2, 1))
    assert set(constraints.intervals) & set(range(6))
    assert set(constraints.intervals) & set(range(6, 12))
    assert set(constraints.positions) == {1, 2}

    perturbed_corners = KNOWN_CORNERS + np.array(
        [[15.0, -10.0], [-12.0, 18.0], [20.0, 15.0], [-18.0, -14.0]],
    )
    fitted = refit.refine(perturbed_corners, constraints, IMAGE_SIZE, use_positions=True)

    assert fitted["status"] == "converged"
    assert fitted["successful"] is True
    assert fitted["jacobian_rank"] == 8
    assert fitted["objective_after"] < fitted["objective_before"]
    fitted_corners = np.asarray(fitted["corners_px"])
    assert np.linalg.norm(fitted_corners - KNOWN_CORNERS, axis=1).max() < 0.05

    nominal_control = refit.refine(perturbed_corners, constraints, IMAGE_SIZE, use_positions=False)
    control_corners = np.asarray(nominal_control["corners_px"])
    assert np.linalg.norm(control_corners - KNOWN_CORNERS, axis=1).max() > 0.05
    assert np.linalg.norm(control_corners - fitted_corners, axis=1).max() > 0.05


def test_refine_rejects_fewer_than_four_samples_without_geometry() -> None:
    constraints = _constraints((0,), (1,), samples_per_interval=3)

    result = refit.refine(KNOWN_CORNERS, constraints, IMAGE_SIZE, use_positions=True)

    assert result["status"] == "insufficient_samples"
    assert result["successful"] is False
    assert result["corners_px"] is None


def test_refine_rejects_single_line_family_as_rank_deficient() -> None:
    constraints = _constraints((0, 1, 4), (1, 2, 1))
    perturbed_corners = KNOWN_CORNERS + np.array(
        [[15.0, -10.0], [-12.0, 18.0], [20.0, 15.0], [-18.0, -14.0]],
    )

    result = refit.refine(perturbed_corners, constraints, IMAGE_SIZE, use_positions=True)

    assert result["status"] == "rank_deficient"
    assert result["successful"] is False
    assert result["jacobian_rank"] < 8


def test_prepare_keeps_centre_pieces_and_drops_weak_fragments() -> None:
    _, observations = _observations_for_prepare()
    assignments = {
        "strength": np.array([0.90, 0.80, 0.54]),
        "marking": np.array([2, 2, 0]),
        "position": np.array([1, 2, 0]),
    }
    fragment_weights = np.array([0.25, 0.65, 0.10])

    constraints = refit.prepare(_known_homography(), observations, assignments, fragment_weights)

    assert len(constraints.points) == 32
    assert np.array_equal(constraints.intervals, np.r_[np.full(16, 2), np.full(16, 3)])
    assert np.array_equal(constraints.positions, np.r_[np.full(16, 1), np.full(16, 2)])
    assert np.array_equal(constraints.fragment_ids, np.r_[np.full(16, 101), np.full(16, 202)])
    assert np.array_equal(constraints.sample_ids, np.r_[np.arange(16), np.arange(16)])
    assert np.allclose(constraints.weights, np.r_[np.full(16, 0.25 / 16), np.full(16, 0.65 / 16)])
