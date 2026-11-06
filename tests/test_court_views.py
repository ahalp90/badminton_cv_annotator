"""Camera identity checks that protect shared court calibration."""

import cv2
import numpy as np

from court_detector import view_matching as court_views


def _court_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.integers(30, 100, (540, 960, 3), dtype=np.uint8)
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]], np.int32)
    cv2.polylines(image, [corners], True, (240, 240, 240), 3)
    for position in range(240, 700, 70):
        cv2.circle(image, (position, 300), 12, (170, 200, 190), -1)
    return image


def test_view_change_with_two_fixed_corners_stays_separate() -> None:
    image = _court_image()
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]], dtype=np.float32)
    changed_corners = corners.copy()
    changed_corners[:2, 1] += 10
    warp = cv2.getPerspectiveTransform(corners, changed_corners)
    changed_image = cv2.warpPerspective(image, warp, (960, 540))
    view = court_views.describe_court_view([image] * 3)
    changed = court_views.describe_court_view([changed_image] * 3)
    alignment = court_views.measure_view_alignment(view.image, changed.image, corners * (4 / 3))
    assert alignment is not None and not alignment.matches
    shifted_image = cv2.warpAffine(image, np.array([[1., 0., 5.], [0., 1., 0.]], dtype=np.float32), (960, 540))
    shifted = court_views.describe_court_view([shifted_image] * 3)
    alignment = court_views.measure_view_alignment(view.image, shifted.image, corners * (4 / 3))
    assert alignment is not None and not alignment.matches


def test_identical_views_align_and_textureless_views_cannot_be_measured() -> None:
    view = court_views.describe_court_view([_court_image()])
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]]) * (4 / 3)
    alignment = court_views.measure_view_alignment(view.image, view.image, corners)
    assert alignment is not None and alignment.matches
    blank = np.zeros_like(view.image)
    assert court_views.measure_view_alignment(blank, blank, corners) is None
