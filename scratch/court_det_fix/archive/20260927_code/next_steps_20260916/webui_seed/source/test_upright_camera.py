"""The horizon tests must keep courts an upright camera can see and reject rolled or upside-down ones."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[3]
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic")]
from run_w5 import add_helper_paths  # pyrefly: ignore[missing-import]

add_helper_paths(COURT_ROOT)
import run_automatic  # pyrefly: ignore[missing-import]
import run_given  # pyrefly: ignore[missing-import]

SIZE = (960, 540)
CAMERA_MATRIX = np.array([[800., 0., 480.], [0., 800., 270.], [0., 0., 1.]])
# Court on the floor (z = 0): x across the 6.1 m width, y along the 13.4 m length.
COURT_CORNERS_M = np.array([[0., 0., 0.], [6.1, 0., 0.], [6.1, 13.4, 0.], [0., 13.4, 0.]])
CAMERA_POSITION_M = np.array([3.05, -6., 5.])


def camera_rows(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera x (right), y (down) and z (forward) axes in floor coordinates, one per row.

    The camera stands behind the near baseline facing along the court, pitched down, then
    rolled about its forward axis.
    """
    pitch, roll = np.radians(pitch_deg), np.radians(roll_deg)
    right = np.array([1., 0., 0.])
    forward = np.array([0., np.cos(pitch), -np.sin(pitch)])
    down = np.cross(forward, right)
    return np.array([np.cos(roll) * right + np.sin(roll) * down,
                     -np.sin(roll) * right + np.cos(roll) * down, forward])


def view(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The court's two vanishing points (homogeneous px) and its corners (1, 4, 2) in px."""
    vanishing_points = (CAMERA_MATRIX @ rows @ np.eye(3)[:, :2]).T
    projected = (CAMERA_MATRIX @ rows @ (COURT_CORNERS_M - CAMERA_POSITION_M).T).T
    return vanishing_points, (projected[:, :2] / projected[:, 2:])[None]


def test_upright_and_mildly_rolled_cameras_pass_both_tests() -> None:
    for roll in (0., 10., -30.):
        points, corners = view(camera_rows(30., roll))
        assert abs(run_automatic.horizon_tilt_deg(points, SIZE) - abs(roll)) < 1e-6
        assert run_given.below_horizon(points, corners, SIZE).tolist() == [True]


def test_a_camera_on_its_side_has_a_steep_horizon() -> None:
    points, _ = view(camera_rows(30., 90.))
    assert abs(run_automatic.horizon_tilt_deg(points, SIZE) - 90.) < 1e-6


def test_an_upside_down_camera_has_a_level_horizon_but_the_court_above_it() -> None:
    points, corners = view(camera_rows(30., 180.))
    assert run_automatic.horizon_tilt_deg(points, SIZE) < 1e-6
    assert run_given.below_horizon(points, corners, SIZE).tolist() == [False]


def test_cameras_looking_straight_or_nearly_straight_down_pass() -> None:
    overhead = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    for rows in (overhead, camera_rows(89.9, 170.)):
        points, corners = view(rows)
        assert run_given.horizon(points, SIZE) is None
        assert run_automatic.horizon_tilt_deg(points, SIZE) is None
        assert run_given.below_horizon(points, corners, SIZE).tolist() == [True]
