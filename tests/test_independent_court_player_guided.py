"""Check player guidance at the complete-court boundary."""

import cv2
import numpy as np

from experiments.annotator.independent_court import detector, player_guided


def test_presence_uses_all_frames_and_distinguishes_missing_players() -> None:
    feet = np.array([[[2.0, 2.0], [4.0, 11.0]], [[2.0, 2.0], [np.nan, np.nan]]])
    one, two = player_guided.player_fractions(np.eye(3)[None], feet)
    np.testing.assert_array_equal(one, [1.0])
    np.testing.assert_array_equal(two, [0.5])
    feet[1, 0] = np.nan
    one, two = player_guided.player_fractions(np.eye(3)[None], feet)
    np.testing.assert_array_equal(one, [0.5])
    np.testing.assert_array_equal(two, [0.5])


def test_two_people_in_same_half_do_not_supply_opposing_players() -> None:
    feet = np.array([[[2.0, 2.0], [4.0, 3.0]]])
    one, two = player_guided.player_fractions(np.eye(3)[None], feet)
    np.testing.assert_array_equal(one, [1.0])
    np.testing.assert_array_equal(two, [0.0])


def test_internal_rectangle_can_propose_a_court_containing_both_players(monkeypatch) -> None:
    corners = np.array([[220, 80], [420, 80], [540, 500], [100, 500]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners)
    internal = np.array([[0.46, 4.72], [5.64, 4.72], [5.64, 8.68], [0.46, 8.68]])
    projected_internal, _ = detector.project(homography[None], internal)
    seed = projected_internal[0].astype(np.float32)
    seed_transform = cv2.getPerspectiveTransform(detector.UNIT_CORNERS, seed)
    projected_feet, _ = detector.project(homography[None], np.array([[2.0, 2.0], [4.0, 11.0]]))
    for point in projected_feet[0]:
        assert cv2.pointPolygonTest(seed, tuple(point), False) < 0
    monkeypatch.setattr(detector, "_image_rectangles", lambda *_args: seed_transform[None])
    segments, _ = detector.project(homography[None], detector.SEGMENTS_M)
    result = player_guided.detect(
        np.zeros((560, 640, 3), dtype=np.uint8),
        np.repeat(projected_feet, 4, axis=0),
        detector.Settings(wide_families=True, min_supported_lines=3),
        segments_px=segments.reshape(-1, 4),
    )
    assert result.hypotheses_with_players > 0
    errors = [np.linalg.norm(candidate.corners_px - corners, axis=1).max()
              for candidate in result.detection.candidates]
    assert min(errors) < 0.01
