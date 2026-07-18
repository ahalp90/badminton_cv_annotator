"""Direct contract tests for the cached StickyResult evidence arrays."""

import numpy as np
import pandas as pd
import pytest

from annotator.rally_segmentation import CourtBox, build_sticky_result, segment_video


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
    n_frames, n_slots = 4, 3
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
    for frame, detections in ((1, frame_one), (2, frame_two)):
        ndet[frame] = len(detections)
        for slot, (box, score) in enumerate(detections):
            bboxes[frame, slot] = box
            scores[frame, slot] = score
            kps[frame, slot] = _standing_pose(box, (600, box[3]), (620, box[3]))

    track = np.array([
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 1.0],
        [600 / 1280, 170 / 720, 1.0],
        [0.5, 0.5, 0.0],
    ])
    court_box = CourtBox(
        x_range=(0.0, 1280.0), y_range=(0.0, 720.0),
        height_band=(1.0, 1000.0), mid_band=(640.0, 640.0),
    )
    court_info = {
        'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0,
        'border_U': 0.0, 'border_D': 720.0,
    }
    resolution_table = pd.DataFrame({'width': [1280.0], 'height': [720.0]}, index=['1'])
    return (track, [(1, 3)], bboxes, scores, kps, ndet, court_box,
            {'1': court_info}, resolution_table)


def test_build_sticky_result_pins_failure_defaults_and_success_contract():
    (track, spans, bboxes, scores, kps, ndet, court_box,
     court_info, resolution_table) = _sticky_inputs()
    result = build_sticky_result(
        track, spans, bboxes, scores, kps, ndet, '1', court_info,
        resolution_table, court_box, (1280.0, 720.0), half_window=1,
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


def test_segment_video_sticky_distance_exclusions_are_mutual():
    track = np.zeros((3, 3), dtype=np.float64)
    distances = np.zeros(3)

    with pytest.raises(ValueError, match='shape'):
        segment_video(track, sticky_distances=np.zeros(2))
    with pytest.raises(ValueError, match='combined'):
        segment_video(track, sticky_distances=distances, body_unit_dist=np.zeros(3))
    with pytest.raises(ValueError, match='combined'):
        segment_video(track, sticky_distances=distances, pose_bboxes=np.zeros((3, 1, 4)))
