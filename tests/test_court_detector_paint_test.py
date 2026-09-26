"""The gap-bounded paint test finds a court's own far lines but not a neighbouring line's paint."""

from __future__ import annotations

import cv2
import numpy as np

from experiments.annotator.independent_court.paint_geometry import (
    CENTRE_SEGMENTS_M,
    STRIPE_WIDTH_M,
)
from scratch.court_det_fix.w5_holistic.verifier import (
    TRANSVERSE_CENTRES_M,
    gap_bounded_photometry,
    parallel_neighbours_m,
    photometric_samples,
)

NATIVE_SIZE = (1920, 1080)
NATIVE_PER_WORKING = 2.0
SUPERSAMPLE = 4
FAR_LONG_SERVICE = 7
# A steep broadcast-like view: the far end is narrow and its lines only a few pixels apart.
COURT_CORNERS_M = np.float32([[0, 0], [6.1, 0], [0, 13.4], [6.1, 13.4]])
IMAGE_CORNERS_PX = np.float32([[860, 300], [1060, 300], [300, 1050], [1620, 1050]])
TRUE_COURT = cv2.getPerspectiveTransform(COURT_CORNERS_M, IMAGE_CORNERS_PX).astype(float)
# The same camera with the court squeezed one line in at the far end: its far baseline sits on the
# painted far long-service line, and its far long-service line on bare floor beyond it.
SLIP_M = TRANSVERSE_CENTRES_M[1] - TRANSVERSE_CENTRES_M[0]
SQUEEZE = np.array([[1.0, 0.0, 0.0], [0.0, (13.4 - SLIP_M) / 13.4, SLIP_M], [0.0, 0.0, 1.0]])
SLIPPED_COURT = TRUE_COURT @ SQUEEZE


def project(homography: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    points = np.column_stack((points_m, np.ones(len(points_m)))) @ homography.T
    return points[:, :2] / points[:, 2:]


def painted_grey() -> np.ndarray:
    """Bright 4 cm stripes on a dark floor, drawn supersampled and averaged down like a camera would."""
    canvas = np.full((NATIVE_SIZE[1] * SUPERSAMPLE, NATIVE_SIZE[0] * SUPERSAMPLE), 60, dtype=np.uint8)
    to_canvas = np.diag([SUPERSAMPLE, SUPERSAMPLE, 1.0]) @ TRUE_COURT
    for start_m, end_m in CENTRE_SEGMENTS_M:
        across_m = np.array([1.0, 0.0]) if start_m[0] == end_m[0] else np.array([0.0, 1.0])
        half_m = STRIPE_WIDTH_M / 2 * across_m
        corners_m = np.array([start_m - half_m, end_m - half_m, end_m + half_m, start_m + half_m])
        cv2.fillPoly(canvas, [np.round(project(to_canvas, corners_m)).astype(np.int32)], 200)
    grey = cv2.resize(canvas, NATIVE_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    return grey + np.random.default_rng(0).normal(0, 2, grey.shape).astype(np.float32)


def line_samples(homography: np.ndarray, segment: int) -> np.ndarray:
    start_m, end_m = CENTRE_SEGMENTS_M[segment]
    fractions = np.linspace(0.05, 0.95, 64)[:, None]
    return project(homography, start_m + fractions * (end_m - start_m))


def gap_bounded_pass_rate(grey: np.ndarray, homography: np.ndarray, boxes: np.ndarray) -> float:
    line_m = CENTRE_SEGMENTS_M[FAR_LONG_SERVICE]
    _, passed = gap_bounded_photometry(grey, homography, line_samples(homography, FAR_LONG_SERVICE), line_m,
                                       parallel_neighbours_m(line_m), boxes, NATIVE_PER_WORKING)
    return float(passed.mean())


def test_fixed_test_finds_paint_for_a_court_slipped_one_line() -> None:
    """The failure the gap-bounded test fixes, reproduced on the half-size working image."""
    working = cv2.resize(painted_grey(), (960, 540), interpolation=cv2.INTER_AREA)
    working_bgr = cv2.cvtColor(np.clip(working, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    samples = line_samples(SLIPPED_COURT, FAR_LONG_SERVICE) / NATIVE_PER_WORKING
    direction = samples[-1] - samples[0]
    _, passed = photometric_samples(working_bgr, samples, direction, np.empty((0, 4)))
    assert passed.mean() > 0.5


def test_gap_bounded_test_passes_the_right_court_and_fails_the_slipped_one() -> None:
    grey = painted_grey()
    no_boxes = np.empty((0, 4))
    assert gap_bounded_pass_rate(grey, TRUE_COURT, no_boxes) > 0.9
    assert gap_bounded_pass_rate(grey, SLIPPED_COURT, no_boxes) < 0.1


def test_gap_bounded_test_treats_masked_samples_as_unknown() -> None:
    line_m = CENTRE_SEGMENTS_M[FAR_LONG_SERVICE]
    whole_frame = np.array([[0.0, 0.0, NATIVE_SIZE[0], NATIVE_SIZE[1]]])
    contrast, passed = gap_bounded_photometry(painted_grey(), TRUE_COURT, line_samples(TRUE_COURT, FAR_LONG_SERVICE),
                                              line_m, parallel_neighbours_m(line_m), whole_frame, NATIVE_PER_WORKING)
    assert np.isnan(contrast).all()
    assert not passed.any()
