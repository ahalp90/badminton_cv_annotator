"""Tests for scene-local court geometry lookup and projection."""

import numpy as np
import pytest

from annotator.point_winner import project_pixels_to_court
from annotator.scene_courts import (
    SceneCourt,
    build_scene_courts,
    court_at_frame,
    scene_ref_corners,
)


def _empty_scene(start_frame: int, end_frame: int) -> SceneCourt:
    return SceneCourt(start_frame, end_frame, {}, (0.0, 0.0), 0.0)


def _row(start_frame: int, end_frame: int) -> dict[str, float | int]:
    return {
        'start_frame': start_frame,
        'end_frame': end_frame,
        'upleft_x': 0.0,
        'upleft_y': 0.0,
        'upright_x': 640.0,
        'upright_y': 0.0,
        'downright_x': 640.0,
        'downright_y': 360.0,
        'downleft_x': 0.0,
        'downleft_y': 360.0,
    }


def test_court_at_frame_uses_half_open_bounds_and_rejects_gaps():
    first = _empty_scene(10, 20)
    second = _empty_scene(25, 30)
    scenes = (first, second)

    assert court_at_frame(scenes, 10) is first
    assert court_at_frame(scenes, 19) is first
    assert court_at_frame(scenes, 25) is second
    assert court_at_frame(scenes, 29) is second

    for frame in (9, 20, 24, 30):
        with pytest.raises(ValueError, match='no accepted court geometry'):
            court_at_frame(scenes, frame)


def test_scene_row_resolution_scales_native_corners_before_projection():
    row = _row(4, 8)
    corners = scene_ref_corners(row, (640.0, 360.0))
    np.testing.assert_allclose(
        corners,
        [[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]],
    )

    scene = build_scene_courts([row], (640.0, 360.0))[0]
    native_points = np.array([[160.0, 480.0], [90.0, 270.0]])
    projected = project_pixels_to_court(native_points, (640.0, 360.0), scene.court_info)
    np.testing.assert_allclose(projected, [[0.25, 0.75], [0.25, 0.75]])
