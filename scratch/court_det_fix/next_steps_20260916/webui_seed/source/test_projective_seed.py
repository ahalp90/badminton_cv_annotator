"""The generator's fast forms must reproduce the slower forms they replace."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[3]
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic")]
from run_w5 import add_helper_paths  # pyrefly: ignore[missing-import]

add_helper_paths(COURT_ROOT)
import projective_seed  # pyrefly: ignore[missing-import]
import run_given  # pyrefly: ignore[missing-import]
import zone_net  # pyrefly: ignore[missing-import]

COURT_WIDTH_M, COURT_LENGTH_M = zone_net.COURT_SIZE_M


def axis_matches(parameters: np.ndarray) -> projective_seed.AxisMatches:
    """Axis hypotheses that are all retained; only parameters and retained matter to the player test."""
    count = len(parameters)
    every = np.arange(count)
    return projective_seed.AxisMatches(parameters, np.zeros(count), np.full((count, 1), -1),
                                       np.zeros((count, 4), dtype=int), np.zeros(count, dtype=int),
                                       np.ones(count, dtype=bool), every, every, {})


def axis_parameters(rng: np.random.Generator, count: int, extent: float) -> np.ndarray:
    """Scale/offset pairs near the true axis, half of them reversed so canonicalise must relabel."""
    scale = rng.uniform(.8, 1.2, count) * np.where(np.arange(count) % 2, -1., 1.)
    shift = np.where(scale < 0, extent, 0.) + rng.uniform(-1.5, 1.5, count)
    return np.column_stack((scale, shift))


def test_joint_player_fractions_match_zone_net_on_every_combined_court() -> None:
    rng = np.random.default_rng(7)
    court_corners = np.array([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
                             dtype=np.float32)
    image_corners = np.array([[380, 150], [580, 150], [760, 470], [200, 470]], dtype=np.float32)
    basis = cv2.getPerspectiveTransform(court_corners, image_corners).astype(float)
    horizontal = axis_matches(axis_parameters(rng, 40, COURT_WIDTH_M))
    vertical = axis_matches(axis_parameters(rng, 40, COURT_LENGTH_M))

    # Feet spread over and around the court, as (sampled frames, foot samples, image xy), with missing slots.
    feet_m = rng.uniform((-2., -3.), (COURT_WIDTH_M + 2., COURT_LENGTH_M + 3.), size=(12, 6, 2))
    feet_px = cv2.perspectiveTransform(feet_m.reshape(1, -1, 2), basis).reshape(feet_m.shape)
    feet_px[rng.random(feet_px.shape[:2]) < .2] = np.nan

    transforms, _ = run_given.combine(basis, horizontal, vertical)
    transforms, rotated = run_given.canonicalise(transforms)
    expected_one, expected_two = zone_net.player_fractions(transforms, feet_px)
    one, two = projective_seed.joint_player_fractions(basis, horizontal, vertical, feet_px)

    assert rotated.any() and not rotated.all()
    usable = (expected_one == 1) & (expected_two >= .5)
    assert usable.any() and not usable.all()
    np.testing.assert_array_equal(one, expected_one)
    np.testing.assert_array_equal(two, expected_two)


def test_axis_scores_match_the_einsum_form_exactly() -> None:
    """score_axes' per-endpoint distances must reproduce the einsum form bit for bit."""
    from experiments.annotator.independent_court import assignment, detector

    rng = np.random.default_rng(20260924)
    for trial in range(40):
        axis = trial % 2
        coordinates = detector.X_COORDS if axis == 0 else detector.Y_COORDS
        basis = np.array([[300., 5., 480.], [10., -200., 270.], [.01, .2, 1.]])
        basis[:2] += rng.normal(0, 20, (2, 3))
        parameters = np.column_stack((rng.uniform(.1, 2, 64) * rng.choice([-1, 1], 64), rng.uniform(-3, 3, 64)))
        endpoints = rng.uniform(-100, 1100, (int(rng.integers(3, 129)), 2, 2))

        inverse = np.linalg.inv(basis)
        predicted = parameters[:, :1] * coordinates + parameters[:, 1:]
        lines = inverse[axis][None, None] - predicted[..., None] * inverse[2]
        norms = np.linalg.norm(lines[..., :2], axis=2)
        distances = np.abs(np.einsum('hmd,ged->hmge', lines[..., :2], endpoints) + lines[..., 2, None, None])
        distances = distances.max(axis=3) / np.maximum(norms[..., None], 1e-15)
        nearest = distances.argmin(axis=2)
        residual = np.take_along_axis(distances, nearest[..., None], axis=2)[..., 0]
        response = np.exp(-.5 * np.square(residual / assignment.DISTANCE_SIGMA_PX))
        same_group = nearest[:, :, None] == nearest[:, None, :]
        stronger = response[:, None, :] > response[:, :, None]
        order = np.arange(len(coordinates))
        tied_before = (response[:, None, :] == response[:, :, None]) & (order[None, None, :] < order[None, :, None])
        exclusive = ~np.any(same_group & (stronger | tied_before), axis=2)
        supported = exclusive & (residual <= assignment.SUPPORT_DISTANCE_PX)
        expected = ((response * exclusive).mean(axis=1), np.where(supported, nearest, -1), supported.sum(axis=1))

        actual = projective_seed.score_axes(parameters, coordinates, endpoints, basis, axis)
        # Bytes rather than values, so signed zeros and NaN bit patterns must match too.
        for actual_part, expected_part in zip(actual, expected, strict=True):
            assert (actual_part.dtype, actual_part.shape, actual_part.tobytes()) == (
                expected_part.dtype, expected_part.shape, expected_part.tobytes())
