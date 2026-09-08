"""Tests for the independent-court net projection helper."""

from __future__ import annotations

import numpy as np
import pytest

from courtkeynet.court_corners import CORNER_COURT_M
from experiments.annotator.independent_court.net_geometry import project_net

FRAME_SIZE = (1600.0, 900.0)


def _camera() -> tuple[np.ndarray, np.ndarray]:
    focal = 800.0
    width, height = FRAME_SIZE
    intrinsic = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    )
    tilt = np.deg2rad(20.0)
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, np.sin(tilt), -np.cos(tilt)])
    # Image coordinates are y-down: world z-up is opposite cross(x, y).
    world_up = -np.cross(x_axis, y_axis)
    camera_basis = np.column_stack((x_axis, y_axis, world_up))
    translation = np.array([-3.05, -3.0, 20.0])
    return intrinsic, np.column_stack((camera_basis, translation))


def _project(points_xyz: np.ndarray, intrinsic: np.ndarray, camera: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points_xyz, np.ones(len(points_xyz))))
    camera_points = homogeneous @ camera.T
    pixels_h = camera_points @ intrinsic.T
    return pixels_h[:, :2] / pixels_h[:, 2, None]


def _reference_net(intrinsic: np.ndarray, camera: np.ndarray) -> np.ndarray:
    ground = np.array([[0.0, 6.7, 0.0], [3.05, 6.7, 0.0], [6.1, 6.7, 0.0]])
    top = np.array([[0.0, 6.7, 1.55], [3.05, 6.7, 1.524], [6.1, 6.7, 1.55]])
    ground_px = _project(ground, intrinsic, camera)
    top_px = _project(top, intrinsic, camera)
    return np.array(
        [[top_px[0], top_px[1]], [top_px[1], top_px[2]], [ground_px[0], top_px[0]], [ground_px[2], top_px[2]]]
    )


def test_project_net_matches_independent_pinhole_projection() -> None:
    intrinsic, camera = _camera()
    corners = _project(np.column_stack((CORNER_COURT_M, np.zeros(4))), intrinsic, camera)

    result = project_net(corners, FRAME_SIZE)

    np.testing.assert_allclose(result.segments_px, _reference_net(intrinsic, camera), atol=2.0)
    assert np.isfinite(result.camera_error)
    assert 0.4 < result.focal_widths < 0.6


def test_project_net_scales_to_native_resolution() -> None:
    intrinsic, camera = _camera()
    corners = _project(np.column_stack((CORNER_COURT_M, np.zeros(4))), intrinsic, camera)

    result = project_net(corners, FRAME_SIZE)
    scaled = project_net(corners * 0.5, (800.0, 450.0))

    np.testing.assert_allclose(scaled.segments_px, result.segments_px * 0.5, atol=1.0)


@pytest.mark.parametrize(
    "corners, frame_size",
    [
        (np.full((4, 2), np.nan), FRAME_SIZE),
        (np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]), FRAME_SIZE),
        (np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]), (0.0, 900.0)),
    ],
)
def test_project_net_rejects_invalid_inputs(corners: np.ndarray, frame_size: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        project_net(corners, frame_size)
