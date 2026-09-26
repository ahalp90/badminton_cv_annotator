"""The court detector's line-averaged paint test."""

from __future__ import annotations

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import line_paint

# 50 native px per floor metre, with the court's corner at (100, 20).
HOMOGRAPHY_NATIVE = np.array([[50.0, 0.0, 100.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]])
NO_BOXES = np.empty((0, 4))


def test_side_points_stay_off_the_neighbouring_line() -> None:
    sidelines = [0, 1, 4, 5]  # 0.46 m from their neighbours
    for segment, side_m in enumerate(line_paint.SIDE_DISTANCES_M):
        expected = 0.4 * 0.46 if segment in sidelines else line_paint.MAX_SIDE_M
        assert side_m == pytest.approx(expected)


def test_a_faint_line_passes_on_average_and_bare_floor_does_not() -> None:
    # One painted line 3 grey levels brighter than the floor, under noise of 5 grey levels a pixel:
    # single samples could not tell them apart.
    rng = np.random.default_rng(0)
    grey = (100 + rng.normal(0, 5, (720, 1280))).astype(np.float32)
    line_m = np.array([[0.48, 0.0], [0.48, 13.4]])
    grey[:, 124] += 3  # 0.48 m across the floor
    painted = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, line_m, 0.184, NO_BOXES))
    floor_m = np.array([[2.0, 0.0], [2.0, 13.4]])
    floor = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, floor_m, 0.184, NO_BOXES))
    assert painted is not None and floor is not None
    assert painted >= line_paint.PASS_BAR_GREY > floor


def test_a_line_under_a_person_has_no_samples() -> None:
    grey = np.full((720, 1280), 100, dtype=np.float32)
    line_m = np.array([[0.48, 0.0], [0.48, 13.4]])
    covering_box = np.array([[0.0, 0.0, 1279.0, 719.0]])
    samples = line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, line_m, 0.184, covering_box)
    assert line_paint.line_contrast(samples) is None
