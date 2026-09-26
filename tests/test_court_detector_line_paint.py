"""The court detector's line-averaged paint test."""

from __future__ import annotations

import numpy as np

from scratch.court_det_fix.court_detector import line_paint

# 50 native px per floor metre, with the court's corner at (100, 20).
HOMOGRAPHY_NATIVE = np.array([[50.0, 0.0, 100.0], [0.0, 50.0, 20.0], [0.0, 0.0, 1.0]])
NO_BOXES = np.empty((0, 4))
SIDELINE = np.array([[0.48, 0.0], [0.48, 13.4]])
SIDELINE_STEPS = line_paint.SEGMENT_STEPS[1]


def test_side_points_and_reach_stay_off_the_neighbouring_line() -> None:
    sidelines = [0, 1, 4, 5]  # 0.46 m from their neighbours
    for segment, steps in enumerate(line_paint.SEGMENT_STEPS):
        assert steps == ((9, 5) if segment in sidelines else (15, 7))


def faint_line_frame(column: int) -> np.ndarray:
    """Floor with noise of 5 grey levels a pixel, and one painted column 6 grey levels brighter."""
    rng = np.random.default_rng(0)
    grey = (100 + rng.normal(0, 5, (720, 1280))).astype(np.float32)
    grey[:, column] += 6
    return grey


def test_a_faint_line_passes_on_average_and_bare_floor_does_not() -> None:
    grey = faint_line_frame(124)  # 0.48 m across the floor
    painted = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, SIDELINE, SIDELINE_STEPS,
                                                               NO_BOXES), SIDELINE_STEPS)
    floor_m = np.array([[2.0, 0.0], [2.0, 13.4]])
    floor = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, floor_m, SIDELINE_STEPS,
                                                             NO_BOXES), SIDELINE_STEPS)
    assert painted is not None and floor is not None
    assert painted >= line_paint.PASS_BAR_GREY > floor


def test_a_line_a_few_centimetres_off_its_prediction_still_passes() -> None:
    grey = faint_line_frame(127)  # 6 cm from the predicted 0.48 m
    samples = line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, SIDELINE, SIDELINE_STEPS, NO_BOXES)
    contrast = line_paint.line_contrast(samples, SIDELINE_STEPS)
    assert contrast is not None and contrast >= line_paint.PASS_BAR_GREY


def test_a_line_under_a_person_has_no_samples() -> None:
    grey = np.full((720, 1280), 100, dtype=np.float32)
    covering_box = np.array([[0.0, 0.0, 1279.0, 719.0]])
    samples = line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, SIDELINE, SIDELINE_STEPS, covering_box)
    assert line_paint.line_contrast(samples, SIDELINE_STEPS) is None
