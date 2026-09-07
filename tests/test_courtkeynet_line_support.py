"""Contracts for painted-line corroboration of fallback courts."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from courtkeynet.court_corners import (
    CORNER_COURT_M,
    PAINTED_SEGMENTS_M,
    painted_line_support,
)

FRAME_WH = (1280.0, 720.0)
COURT_CORNERS = np.array(
    [[100.0, 90.0], [1180.0, 105.0], [1110.0, 650.0], [160.0, 635.0]],
    dtype=np.float32,
)


def _projected_segments(corners: np.ndarray) -> np.ndarray:
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, corners)
    projected = []
    for endpoint_a, endpoint_b in PAINTED_SEGMENTS_M:
        points_m = np.stack((endpoint_a, endpoint_b)).reshape(-1, 1, 2)
        points_px = cv2.perspectiveTransform(points_m, homography).reshape(2, 2)
        projected.append(np.concatenate((points_px[0], points_px[1])))
    return np.asarray(projected, dtype=np.float32)


def test_full_pattern_supports_both_court_line_families() -> None:
    support = painted_line_support(
        COURT_CORNERS,
        (_projected_segments(COURT_CORNERS),),
        FRAME_WH,
    )

    np.testing.assert_allclose(support, (1.0, 1.0), atol=0.02)


def test_finite_fragment_extent_limits_coverage() -> None:
    segments = _projected_segments(COURT_CORNERS)
    endpoint_a, endpoint_b = segments[0, :2], segments[0, 2:]
    segments[0, :2] = endpoint_a + 0.45 * (endpoint_b - endpoint_a)
    segments[0, 2:] = endpoint_a + 0.55 * (endpoint_b - endpoint_a)

    lengthwise, crosscourt = painted_line_support(COURT_CORNERS, (segments,), FRAME_WH)

    assert 0.0 < lengthwise < 0.95
    assert crosscourt > 0.98


def test_perpendicular_fragment_does_not_support_a_painted_line() -> None:
    segments = _projected_segments(COURT_CORNERS)
    midpoint = segments[0, :2] + 0.5 * (segments[0, 2:] - segments[0, :2])
    half_length = 0.25 * np.linalg.norm(segments[0, 2:] - segments[0, :2])
    segments[0] = [midpoint[0] - half_length, midpoint[1], midpoint[0] + half_length, midpoint[1]]

    lengthwise, crosscourt = painted_line_support(COURT_CORNERS, (segments,), FRAME_WH)

    assert lengthwise < 0.95
    assert crosscourt > 0.98


def test_one_frame_noise_does_not_accumulate_across_samples() -> None:
    noise = np.array([[20.0, 20.0, 500.0, 500.0]], dtype=np.float32)
    empty = np.empty((0, 4), dtype=np.float32)

    support = painted_line_support(COURT_CORNERS, (noise, empty, empty), FRAME_WH)

    np.testing.assert_array_equal(support, (0.0, 0.0))


def test_offscreen_partial_pattern_uses_visible_line_extents() -> None:
    partially_offscreen = np.array(
        [[-180.0, -70.0], [1080.0, -50.0], [1130.0, 820.0], [-210.0, 800.0]],
        dtype=np.float32,
    )

    support = painted_line_support(
        partially_offscreen,
        (_projected_segments(partially_offscreen),),
        FRAME_WH,
    )

    np.testing.assert_allclose(support, (1.0, 1.0), atol=0.03)


@pytest.mark.parametrize('scale', [(1.5, 1.5), (2.0, 1.0)])
def test_support_is_resolution_scaled(scale: tuple[float, float]) -> None:
    native_scale = np.asarray(scale, dtype=np.float32)
    scaled_corners = COURT_CORNERS * native_scale
    segments = _projected_segments(COURT_CORNERS)
    # Offset fragments slightly so the tolerance is exercised under x-only scaling.
    segments += np.array([6.0, 0.0, 6.0, 0.0], dtype=np.float32)

    original = painted_line_support(
        COURT_CORNERS,
        (segments,),
        FRAME_WH,
    )
    scaled = painted_line_support(
        scaled_corners,
        (segments * np.tile(native_scale, 2),),
        tuple(np.asarray(FRAME_WH) * native_scale),
    )

    np.testing.assert_allclose(scaled, original, atol=0.02)


def test_missing_line_evidence_raises() -> None:
    with pytest.raises(ValueError, match='missing its sampled line evidence'):
        painted_line_support(COURT_CORNERS, (), FRAME_WH)
