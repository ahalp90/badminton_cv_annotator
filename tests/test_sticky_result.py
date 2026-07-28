"""Direct contract tests for the cached StickyResult evidence arrays."""

import numpy as np
import pandas as pd
import pytest

from annotator.rally_segmentation import WRIST_L, WRIST_R, build_sticky_result, segment_video


def _bbox(x: float, foot_y: float, height: float = 120.0) -> np.ndarray:
    return np.array([x - 30.0, foot_y - height, x + 30.0, foot_y], dtype=np.float32)


def _standing_pose(box: np.ndarray, ankle_left: tuple[float, float], ankle_right: tuple[float, float]):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    h = y2 - y1
    pose = np.zeros((17, 2), dtype=np.float32)
    pose[5] = (cx - 15, y1 + 0.25 * h)
    pose[6] = (cx + 15, y1 + 0.25 * h)
    pose[11] = (cx - 12, y1 + 0.55 * h)
    pose[12] = (cx + 12, y1 + 0.55 * h)
    pose[13] = (cx - 14, y1 + 0.80 * h)
    pose[14] = (cx + 14, y1 + 0.80 * h)
    pose[15] = ankle_left
    pose[16] = ankle_right
    pose[9] = pose[15]
    pose[10] = pose[16]
    return pose


def _sticky_inputs():
    n_frames, n_slots = 5, 3
    bboxes = np.full((n_frames, n_slots, 4), np.nan, dtype=np.float32)
    scores = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    kps = np.full((n_frames, n_slots, 17, 2), np.nan, dtype=np.float32)
    ndet = np.zeros(n_frames, dtype=np.int64)

    # Frame 1 has one counted, standing detection but the two selected detections
    # are both outside the generous court, forcing a picker failure.
    frame_one = [
        (_bbox(640, -144), 0.9),
        (_bbox(640, 864), 0.9),
        (_bbox(0, 360), 0.9),
    ]
    # Frame 2 makes the raw-slot mapping observable: slot 0 is filtered by score.
    frame_two = [
        (_bbox(640, 360), 0.1),
        (_bbox(640, 180), 0.9),
        (_bbox(640, 540), 0.9),
    ]
    # Frame 3 has a single top-half detection: exactly one sticky slot picks,
    # making the per-slot distance mapping observable.
    frame_three = [
        (_bbox(640, 180), 0.9),
    ]
    for frame, detections in ((1, frame_one), (2, frame_two), (3, frame_three)):
        ndet[frame] = len(detections)
        for slot, (box, score) in enumerate(detections):
            bboxes[frame, slot] = box
            scores[frame, slot] = score
            kps[frame, slot] = _standing_pose(box, (600, box[3]), (620, box[3]))

    track = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [600 / 1280, 170 / 720, 1.0],
        [600 / 1280, 170 / 720, 1.0],
        [0.5, 0.5, 0.0],
    ])
    court_info = {
        'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0,
        'border_U': 0.0, 'border_D': 720.0,
    }
    resolution_table = pd.DataFrame({'width': [1280.0], 'height': [720.0]}, index=['1'])
    return (track, [(1, 4)], bboxes, scores, kps, ndet, {'1': court_info}, resolution_table)


def test_build_sticky_result_pins_failure_defaults_and_success_contract():
    (track, segments, bboxes, scores, kps, ndet, court_info, resolution_table) = _sticky_inputs()
    result = build_sticky_result(
        track, segments, bboxes, scores, kps, ndet, '1', court_info,
        resolution_table, (1280.0, 720.0), half_window=1,
    )

    assert np.isposinf(result.distances[0])
    assert result.picks[0].tolist() == [-1, -1]
    assert result.standing_count[0] == 0
    assert np.isnan(result.ankle_pos[0]).all()
    assert np.isnan(result.bbox_height[0]).all()

    assert np.isnan(result.distances[1])
    assert result.standing_count[1] == 1
    assert result.picks[2].tolist() == [1, 2]
    assert result.bbox_height[2].tolist() == [120.0, 120.0]
    np.testing.assert_allclose(result.ankle_pos[2, 0], [610 / 1280, 180 / 720])
    np.testing.assert_allclose(result.ankle_pos[2, 1], [610 / 1280, 540 / 720])

    # Additive Stage 5 fields: analysed marks exactly the segment frames; per-slot
    # distances are +inf outside the segments, NaN on a visited frame with no
    # finite gap, and consistent with the min-collapsed series where finite.
    assert result.analysed.tolist() == [False, True, True, True, False]
    assert np.isposinf(result.distances_per_slot[0]).all()
    assert np.isposinf(result.distances_per_slot[4]).all()
    assert np.isnan(result.distances_per_slot[1]).all()
    per_slot = result.distances_per_slot[2]
    finite = np.isfinite(per_slot)
    assert np.isfinite(result.distances[2]) == finite.any()
    if finite.any():
        assert result.distances[2] == per_slot[finite].min()

    # Frame 3's single top-half detection: the unpicked bottom slot stays NaN
    # while the picked top slot owns whatever gap the collapse saw.
    assert result.picks[3].tolist()[1] == -1
    assert result.picks[3][0] >= 0
    assert np.isnan(result.distances_per_slot[3, 1])
    top_gap = result.distances_per_slot[3, 0]
    assert np.isfinite(result.distances[3]) == bool(np.isfinite(top_gap))
    if np.isfinite(top_gap):
        assert result.distances[3] == top_gap


def test_build_sticky_result_wrist_distance_visibility_and_sentinels():
    (track, segments, bboxes, scores, kps, ndet, court_info, resolution_table) = _sticky_inputs()
    track = track.copy()
    track[2, :2] = 0.0
    track[2, 2] = 0.0
    kps[3, 0, WRIST_L] = (640.0, 300.0)
    kps[3, 0, WRIST_R] = (600.0, 180.0)

    result = build_sticky_result(
        track, segments, bboxes, scores, kps, ndet, '1', court_info,
        resolution_table, (1280.0, 720.0), half_window=1,
    )

    assert result.wrist_dist_px.shape == (len(track), 2)
    assert result.wrist_dist_px.dtype == np.float64
    assert np.isposinf(result.wrist_dist_px[0]).all()
    assert np.isposinf(result.wrist_dist_px[4]).all()
    assert np.isnan(result.wrist_dist_px[2]).all()
    assert np.isnan(result.wrist_dist_px[3, 1])
    assert result.wrist_dist_px[3, 0] == pytest.approx(10.0)

    # The invisible frame keeps the old corner-derived contact-distance cache and collapse,
    # while the serve-only raw wrist cache refuses to measure it.
    assert np.isfinite(result.distances_per_slot[2]).all()
    assert np.isfinite(result.distances[2])
    assert result.distances[2] == pytest.approx(np.min(result.distances_per_slot[2]))


def test_segment_video_sticky_distance_shape_is_checked():
    track = np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match='shape'):
        segment_video(track, sticky_distances=np.zeros(2))


def test_build_sticky_result_clips_height_windows_at_touching_segment_bounds():
    n_frames = 6
    bboxes = np.full((n_frames, 1, 4), np.nan, dtype=np.float32)
    scores = np.ones((n_frames, 1), dtype=np.float32)
    kps = np.full((n_frames, 1, 17, 2), np.nan, dtype=np.float32)
    for frame in range(n_frames):
        height = 100.0 if frame < 3 else 300.0
        box = _bbox(640.0, 360.0, height)
        bboxes[frame, 0] = box
        kps[frame, 0] = _standing_pose(box, (600.0, 360.0), (620.0, 360.0))
    track = np.tile(np.array([600.0 / 1280.0, 170.0 / 720.0, 1.0]), (n_frames, 1))
    court_info = {'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0, 'border_U': 0.0, 'border_D': 720.0}
    resolution_table = pd.DataFrame({'width': [1280.0], 'height': [720.0]}, index=['1'])
    result = build_sticky_result(
        track, [(0, 3), (3, 6)], bboxes, scores, kps, np.ones(n_frames, dtype=np.int64), '1',
        {'1': court_info}, resolution_table, (1280.0, 720.0), half_window=1,
    )

    numerator = np.hypot(600.0 - 600.0, 360.0 - 170.0)
    picked_half = int(np.flatnonzero(result.picks[2] >= 0)[0])
    assert result.distances_per_slot[2, picked_half] == pytest.approx(numerator / 100.0)


def test_build_sticky_result_fails_when_candidate_has_no_finite_height() -> None:
    (track, segments, bboxes, scores, kps, ndet, court_info, resolution_table) = _sticky_inputs()
    bboxes[1, 1, 1] = np.nan
    bboxes[2, 1, 1] = np.nan
    bboxes[3, 0, 1] = np.nan

    with pytest.raises(ValueError, match='no accepted finite height'):
        build_sticky_result(
            track, segments, bboxes, scores, kps, ndet, '1', court_info,
            resolution_table, (1280.0, 720.0), half_window=1,
        )


def test_build_sticky_result_fails_on_non_positive_height_denominator() -> None:
    (track, segments, bboxes, scores, kps, ndet, court_info, resolution_table) = _sticky_inputs()
    bboxes[1, 1, 1] = bboxes[1, 1, 3]
    bboxes[2, 1, 1] = bboxes[2, 1, 3]
    bboxes[3, 0, 1] = bboxes[3, 0, 3]

    with pytest.raises(ValueError, match='non-finite or non-positive body-scale denominator'):
        build_sticky_result(
            track, segments, bboxes, scores, kps, ndet, '1', court_info,
            resolution_table, (1280.0, 720.0), half_window=1,
        )
