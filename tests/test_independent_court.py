"""Synthetic boundary tests for the independent court proposal detector.

The detector is an isolated experiment, so this module loads it directly from
its file path. Fixtures draw the repository's finite badminton markings under a
known homography and keep detector acceptance separate from proposal accuracy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (480, 360)


def _load_detector() -> ModuleType:
    """Load the experiment without making it part of the production import graph."""
    spec = importlib.util.spec_from_file_location(
        "independent_court_detector_under_test",
        REPO_ROOT / "experiments/annotator/independent_court/detector.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detector = _load_detector()
TEST_SETTINGS = detector.Settings(max_family_lines=8)


def _render_template(
    corners_px: np.ndarray,
    *,
    line_indices: set[int] | None = None,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Render selected finite template segments into a black BGR frame."""
    if image is None:
        width, height = FRAME_SIZE
        image = np.zeros((height, width, 3), dtype=np.uint8)
    homography = cv2.getPerspectiveTransform(
        detector.CORNER_COURT_M.astype(np.float32), corners_px.astype(np.float32),
    )
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 2, 2)
    for index, (start, end) in enumerate(endpoints):
        if line_indices is not None and index not in line_indices:
            continue
        cv2.line(
            image,
            tuple(np.rint(start).astype(int)),
            tuple(np.rint(end).astype(int)),
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return image


FULL_CORNERS = np.array(
    [[130.0, 70.0], [350.0, 70.0], [410.0, 330.0], [70.0, 330.0]],
    dtype=np.float32,
)
OFFSCREEN_NEAR_CORNERS = np.array(
    [[130.0, 70.0], [350.0, 70.0], [440.0, 450.0], [40.0, 450.0]],
    dtype=np.float32,
)


def test_full_template_keeps_an_accurate_proposal_separate_from_acceptance() -> None:
    """A complete court yields a close proposal even when repeated lines are ambiguous."""
    image = _render_template(FULL_CORNERS)
    detection = detector.detect(image, TEST_SETTINGS)

    assert detection.candidates
    proposal_errors = [
        np.linalg.norm(candidate.corners_px - FULL_CORNERS, axis=1).max()
        for candidate in detection.candidates
    ]
    assert min(proposal_errors) <= 15.0
    if detection.accepted:
        assert proposal_errors[0] <= 15.0
    else:
        assert detection.reason == "ambiguous"


def test_repeated_court_patterns_are_rejected_as_ambiguous() -> None:
    """Two complete marking patterns must not silently choose one court."""
    image = _render_template(
        np.array([[20.0, 50.0], [220.0, 50.0], [230.0, 350.0], [10.0, 350.0]], dtype=np.float32),
    )
    _render_template(
        np.array([[260.0, 50.0], [460.0, 50.0], [470.0, 350.0], [250.0, 350.0]], dtype=np.float32),
        image=image,
    )
    # Two complete courts contribute ten lengthwise lines before clutter.
    detection = detector.detect(image, detector.Settings(max_family_lines=12))

    assert len(detection.candidates) >= 2
    assert not detection.accepted
    assert detection.reason == "ambiguous"
    assert detection.score_gap is not None
    assert detection.score_gap > TEST_SETTINGS.ambiguity_gap


def test_offscreen_near_baseline_is_not_silently_accepted() -> None:
    """A court whose near baseline is outside the frame remains an explicit rejection."""
    image = _render_template(OFFSCREEN_NEAR_CORNERS)
    detection = detector.detect(image, TEST_SETTINGS)

    if detection.accepted:
        error = np.linalg.norm(detection.candidates[0].corners_px - OFFSCREEN_NEAR_CORNERS, axis=1)
        assert error.max() <= 15.0
    else:
        assert detection.reason in {"ambiguous", "unsupported"}
    assert all(np.isfinite(candidate.corners_px).all() for candidate in detection.candidates)


def test_finite_projected_segments_clip_before_sampling() -> None:
    """Only the in-frame portion of a finite segment is sampleable."""
    endpoints = np.array(
        [
            [
                [[-40.0, 45.0], [60.0, 45.0]],
                [[120.0, 45.0], [180.0, 45.0]],
                [[20.0, -80.0], [20.0, -20.0]],
            ],
        ],
    )
    samples, visible = detector._visible_samples(endpoints, (100, 80), count=8)

    np.testing.assert_array_equal(visible, [[True, False, False]])
    assert np.all(samples[0, 0, :, 0] >= 0)
    assert np.all(samples[0, 0, :, 0] < 100)
    assert np.all(samples[0, 0, :, 1] >= 0)
    assert np.all(samples[0, 0, :, 1] < 80)


def test_offscreen_template_lines_are_excluded_after_projection() -> None:
    """Projected near service and baseline lines outside the frame do not count."""
    homography = cv2.getPerspectiveTransform(
        detector.CORNER_COURT_M.astype(np.float32), OFFSCREEN_NEAR_CORNERS,
    )
    projected, denominator = detector.project(homography[None], detector.SEGMENTS_M)
    endpoints = projected.reshape(1, -1, 2, 2)
    _, visible = detector._visible_samples(endpoints, FRAME_SIZE, count=12)

    assert np.all(denominator > 0)
    np.testing.assert_array_equal(visible[0, 10:], [False, False])


def test_blank_frame_is_rejected_without_hypotheses() -> None:
    """No edge evidence must produce no court proposal."""
    image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    detection = detector.detect(image, TEST_SETTINGS)

    assert not detection.accepted
    assert detection.reason == "insufficient_lines"
    assert detection.candidates == ()
    assert detection.hypotheses_scored == 0


def test_single_outer_rectangle_lacks_enough_internal_markings() -> None:
    """A rectangle with no service lines is insufficient evidence for badminton."""
    image = _render_template(FULL_CORNERS, line_indices={0, 5, 6, 11})
    detection = detector.detect(image, TEST_SETTINGS)

    assert not detection.accepted
    assert detection.reason == "unsupported"
    assert detection.candidates == ()


def test_two_edges_of_one_stripe_cannot_supply_distinct_court_markings() -> None:
    image = _render_template(FULL_CORNERS, line_indices={0, 3, 5, 6, 9, 11})
    detection = detector.detect(image, TEST_SETTINGS)

    assert not detection.accepted
    assert detection.reason in {"unsupported", "ambiguous"}


def test_candidate_limit_must_preserve_an_ambiguity_comparison() -> None:
    with pytest.raises(ValueError, match="at least two"):
        detector.Settings(keep_candidates=1)


def test_broad_candidate_cannot_hide_two_separate_supported_courts() -> None:
    candidates = []
    for corners in (
        [[0, 0], [480, 0], [480, 360], [0, 360]],
        [[20, 50], [220, 50], [230, 350], [10, 350]],
        [[260, 50], [460, 50], [470, 350], [250, 350]],
    ):
        candidates.append(detector.Candidate(np.asarray(corners, dtype=float), 0.9, (0.9, 0.9), (5, 6)))
    assert detector._separate_court(candidates, FRAME_SIZE)
