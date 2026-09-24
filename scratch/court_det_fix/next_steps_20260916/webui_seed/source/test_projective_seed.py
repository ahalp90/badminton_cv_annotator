"""The generator's joint player test must match zone_net's per-court test on every combined court."""

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
