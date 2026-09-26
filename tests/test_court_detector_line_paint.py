"""The court detector's line-averaged paint test."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_a_faint_line_stands_out_from_bare_floor_on_average() -> None:
    grey = faint_line_frame(124)  # 0.48 m across the floor
    painted = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, SIDELINE, SIDELINE_STEPS,
                                                               NO_BOXES), SIDELINE_STEPS)
    floor_m = np.array([[2.0, 0.0], [2.0, 13.4]])
    floor = line_paint.line_contrast(line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, floor_m, SIDELINE_STEPS,
                                                             NO_BOXES), SIDELINE_STEPS)
    assert painted is not None and floor is not None
    assert painted - floor >= 4


def test_a_line_a_few_centimetres_off_its_prediction_keeps_its_contrast() -> None:
    contrasts = [line_paint.line_contrast(line_paint.line_samples(faint_line_frame(column), HOMOGRAPHY_NATIVE,
                                                                  SIDELINE, SIDELINE_STEPS, NO_BOXES), SIDELINE_STEPS)
                 for column in (124, 127)]  # on the predicted 0.48 m, and 6 cm from it
    on_line, off_line = contrasts
    assert on_line is not None and off_line is not None
    assert off_line >= 0.9 * on_line


def test_a_line_under_a_person_has_no_samples() -> None:
    grey = np.full((720, 1280), 100, dtype=np.float32)
    covering_box = np.array([[0.0, 0.0, 1279.0, 719.0]])
    samples = line_paint.line_samples(grey, HOMOGRAPHY_NATIVE, SIDELINE, SIDELINE_STEPS, covering_box)
    assert line_paint.line_contrast(samples, SIDELINE_STEPS) is None


def test_a_court_scores_its_lines_contrast_as_a_share_of_the_views_strongest_line() -> None:
    evidence = {"markings": [{"q_geom": 1.0, "projected_visible_span_px": 100.0} for _ in range(11)]}
    contrasts: list[float | None] = [20.0] * 11
    contrasts[0] = -5.0  # darker than its sides: counts as 0
    contrasts[5] = None  # no usable row: left out
    # Lengthwise lines: 0, then four at 20 of 40; transverse lines: five at 20 of 40. The weaker direction counts.
    assert line_paint.court_paint(evidence, contrasts, reference=40.0) == pytest.approx(0.4)
