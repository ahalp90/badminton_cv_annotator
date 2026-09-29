"""Camera identity checks that protect shared court calibration."""

import cv2
import numpy as np

from annotator import court_views


def _court_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    image = rng.integers(30, 100, (540, 960, 3), dtype=np.uint8)
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]], np.int32)
    cv2.polylines(image, [corners], True, (240, 240, 240), 3)
    for position in range(240, 700, 70):
        cv2.circle(image, (position, 300), 12, (170, 200, 190), -1)
    return image


def test_same_camera_groups_despite_corner_noise_but_zoom_remains_separate() -> None:
    image = _court_image()
    view = court_views.describe_court_view([image, image, image])
    warp = cv2.getRotationMatrix2D((480, 270), 0, 1.02)
    zoomed_image = cv2.warpAffine(image, warp, (960, 540))
    zoomed = court_views.describe_court_view([zoomed_image] * 3)
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]]) * (4 / 3)
    groups = court_views.matching_view_groups(
        [view, view, view, zoomed, None], [corners, corners + 5, corners - 3, corners, corners], [True] * 5,
    )
    assert groups == [[0, 1, 2]]


def test_hash_groups_do_not_merge_by_a_chain_of_similar_scenes(monkeypatch) -> None:
    hashes = np.zeros((1, 16, 16), dtype=bool)
    middle = hashes.copy()
    middle.reshape(-1)[:60] = True
    last = hashes.copy()
    last.reshape(-1)[:120] = True
    image = np.zeros((540, 960), np.uint8)
    views = [court_views.CourtView(value, image) for value in (hashes, hashes, hashes, middle, last)]
    monkeypatch.setattr(court_views, '_view_alignment', lambda *_args: True)
    assert court_views.matching_view_groups(views, [np.ones((4, 2))] * 5, [True] * 5) == [[0, 1, 2, 3]]


def test_aligned_views_can_share_a_representative_despite_pairwise_hash_variation() -> None:
    hashes = np.zeros((1, 16, 16), dtype=bool)
    first = hashes.copy()
    first.reshape(-1)[:60] = True
    last = hashes.copy()
    last.reshape(-1)[-60:] = True
    image = cv2.cvtColor(_court_image(), cv2.COLOR_BGR2GRAY)
    views = [court_views.CourtView(value, image) for value in (first, hashes, hashes, last)]
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]]) * (4 / 3)
    assert court_views.matching_view_groups(views, [corners] * 4, [True] * 4) == [[0, 1, 2, 3]]


def test_failed_representative_can_still_join_another_view_group(monkeypatch) -> None:
    views = [court_views.CourtView(np.zeros((1, 16, 16), dtype=bool), np.zeros((540, 960), np.uint8))
             for _ in range(3)]
    # Image alignment is directional: an unsuitable template can still align as a sample.
    monkeypatch.setattr(court_views, '_view_alignment',
                        lambda template, sample, _corners: template is not views[0] or sample is template)
    assert court_views.matching_view_groups(views, [np.ones((4, 2))] * 3, [True] * 3) == [[0, 1, 2]]


def test_textureless_images_and_small_groups_keep_existing_courts() -> None:
    image = np.zeros((540, 960, 3), np.uint8)
    view = court_views.describe_court_view([image])
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]]) * (4 / 3)
    assert court_views.matching_view_groups([view] * 3, [corners] * 3, [True] * 3) == []
    assert court_views.matching_view_groups([view] * 2, [corners] * 2, [True] * 2) == []


def test_view_change_with_two_fixed_corners_stays_separate() -> None:
    image = _court_image()
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]], dtype=np.float32)
    changed_corners = corners.copy()
    changed_corners[:2, 1] += 10
    warp = cv2.getPerspectiveTransform(corners, changed_corners)
    changed_image = cv2.warpPerspective(image, warp, (960, 540))
    view = court_views.describe_court_view([image] * 3)
    changed = court_views.describe_court_view([changed_image] * 3)
    assert not court_views._view_alignment(view, changed, corners * (4 / 3))
    shifted_image = cv2.warpAffine(image, np.array([[1., 0., 5.], [0., 1., 0.]], dtype=np.float32), (960, 540))
    shifted = court_views.describe_court_view([shifted_image] * 3)
    assert not court_views._view_alignment(view, shifted, corners * (4 / 3))


def test_repeated_zoomed_view_can_form_its_own_group() -> None:
    image = _court_image()
    warp = cv2.getRotationMatrix2D((480, 270), 0, 1.02)
    zoomed_image = cv2.warpAffine(image, warp, (960, 540))
    original = court_views.describe_court_view([image] * 3)
    zoomed = court_views.describe_court_view([zoomed_image] * 3)
    corners = np.array([[280, 170], [680, 170], [800, 490], [160, 490]], dtype=np.float32)
    zoomed_corners = cv2.transform(corners[None], warp)[0]
    groups = court_views.matching_view_groups(
        [original] * 3 + [zoomed] * 3,
        [corners * (4 / 3)] * 3 + [zoomed_corners * (4 / 3)] * 3,
        [True] * 6,
    )
    assert sorted(groups) == [[0, 1, 2], [3, 4, 5]]
