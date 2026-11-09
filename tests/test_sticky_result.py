"""Tracking intervals, missing measurements and wrist-gap normalisation."""

import numpy as np
import pytest

from annotator.courts.frame_geometry import SceneCourt, build_scene_courts
from annotator.inference.prepare_annotation_inputs import build_player_evidence
from annotator.rallies.find_rallies import segment_video
from annotator.types import WRIST_L, WRIST_R
from tests.annotator_config import trajectory_config


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
    return (track, [(1, 4)], bboxes, scores, kps, ndet, {'1': court_info})


def _scene_sticky_inputs():
    n_frames, n_slots = 4, 2
    bboxes = np.full((n_frames, n_slots, 4), np.nan, dtype=np.float32)
    scores = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    kps = np.full((n_frames, n_slots, 17, 2), np.nan, dtype=np.float32)
    ndet = np.full(n_frames, n_slots, dtype=np.int64)
    for frame in range(n_frames):
        x_positions = (320.0, 960.0) if frame < 2 else (890.0, 890.0)
        for slot, (x_position, foot_y) in enumerate(zip(x_positions, (180.0, 540.0))):
            box = _bbox(x_position, foot_y)
            bboxes[frame, slot] = box
            scores[frame, slot] = 0.9
            kps[frame, slot] = _standing_pose(box, (x_position - 14, foot_y), (x_position + 14, foot_y))

    def row(start_frame: int, end_frame: int, left: float, right: float):
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'upleft_x': left,
            'upleft_y': 0.0,
            'upright_x': right,
            'upright_y': 0.0,
            'downright_x': right,
            'downright_y': 720.0,
            'downleft_x': left,
            'downleft_y': 720.0,
        }

    scene_courts = build_scene_courts(
        [row(0, 2, 0.0, 1280.0), row(2, 4, 250.0, 1530.0)],
        (1280.0, 720.0),
    )
    track = np.tile(np.array([0.5, 0.5, 1.0]), (n_frames, 1))
    return track, bboxes, scores, kps, ndet, scene_courts, row


def test_sticky_picker_uses_each_scene_geometry_for_consecutive_segments():
    (track, bboxes, scores, kps, ndet, scene_courts, row) = (
        _scene_sticky_inputs()
    )
    assert not np.allclose(scene_courts[0].court_info['H'], scene_courts[1].court_info['H'])

    local_players, _ = build_player_evidence(
        track,
        [(0, 2), (2, 4)],
        bboxes,
        scores,
        kps,
        ndet,
        (1280.0, 720.0),
        scene_courts=scene_courts,
        half_window=12,
    )

    # A displaced global outline rejects the top player in scene 0, while the
    # scene-local rows retain both picks in both camera views.
    global_scene = build_scene_courts([row(0, 4, 500.0, 1780.0)], (1280.0, 720.0))[0]
    global_players, _ = build_player_evidence(
        track,
        [(0, 4)],
        bboxes,
        scores,
        kps,
        ndet,
        (1280.0, 720.0),
        half_window=12,
        scene_courts=(global_scene,),
    )

    np.testing.assert_array_equal(local_players.detection_indices, np.tile([0, 1], (4, 1)))
    assert local_players.detection_indices[0, 0] == 0
    assert global_players.detection_indices[0].tolist() == [-1, 1]


def test_build_player_evidence_pins_failure_defaults_and_success_contract():
    (track, segments, bboxes, scores, kps, ndet, court_info) = _sticky_inputs()
    players, gaps = build_player_evidence(
        track,
        segments,
        bboxes,
        scores,
        kps,
        ndet,
        (1280.0, 720.0),
        half_window=1,
        scene_courts=(SceneCourt(0, len(track), court_info['1'], (0.0, 0.0), 0.0),),
    )

    assert np.isposinf(gaps.nearest[0])
    assert players.detection_indices[0].tolist() == [-1, -1]
    assert np.isnan(players.ankle_xy[0]).all()
    assert np.isnan(players.box_heights_px[0]).all()

    assert np.isnan(gaps.nearest[1])
    assert players.detection_indices[2].tolist() == [1, 2]
    assert players.box_heights_px[2].tolist() == [120.0, 120.0]
    np.testing.assert_allclose(players.ankle_xy[2, 0], [610 / 1280, 180 / 720])
    np.testing.assert_allclose(players.ankle_xy[2, 1], [610 / 1280, 540 / 720])

    # Gaps distinguish untracked frames from missing measurements inside an interval.
    assert np.isposinf(gaps.per_player[0]).all()
    assert np.isposinf(gaps.per_player[4]).all()
    assert np.isnan(gaps.per_player[1]).all()
    per_slot = gaps.per_player[2]
    finite = np.isfinite(per_slot)
    assert np.isfinite(gaps.nearest[2]) == finite.any()
    if finite.any():
        assert gaps.nearest[2] == per_slot[finite].min()

    # Frame 3's single top-half detection: the unpicked bottom slot stays NaN
    # while the picked top slot owns whatever gap the collapse saw.
    assert players.detection_indices[3].tolist()[1] == -1
    assert players.detection_indices[3][0] >= 0
    assert np.isnan(gaps.per_player[3, 1])
    top_gap = gaps.per_player[3, 0]
    assert np.isfinite(gaps.nearest[3]) == bool(np.isfinite(top_gap))
    if np.isfinite(top_gap):
        assert gaps.nearest[3] == top_gap


def test_build_player_evidence_wrist_distance_visibility_and_sentinels():
    (track, segments, bboxes, scores, kps, ndet, court_info) = _sticky_inputs()
    track = track.copy()
    track[2, :2] = 0.0
    track[2, 2] = 0.0
    kps[3, 0, WRIST_L] = (640.0, 300.0)
    kps[3, 0, WRIST_R] = (600.0, 180.0)

    _, gaps = build_player_evidence(
        track,
        segments,
        bboxes,
        scores,
        kps,
        ndet,
        (1280.0, 720.0),
        half_window=1,
        scene_courts=(SceneCourt(0, len(track), court_info['1'], (0.0, 0.0), 0.0),),
    )


    # Invisible coordinates still contribute to contact-distance measurements.
    assert np.isfinite(gaps.per_player[2]).all()
    assert np.isfinite(gaps.nearest[2])
    assert gaps.nearest[2] == pytest.approx(np.min(gaps.per_player[2]))


def test_segment_video_sticky_distance_shape_is_checked():
    track = np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match='shape'):
        segment_video(track, nearest_wrist_distances=np.zeros(2), video_config=trajectory_config())


def test_build_player_evidence_clips_height_windows_at_touching_segment_bounds():
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
    players, gaps = build_player_evidence(
        track,
        [(0, 3), (3, 6)],
        bboxes,
        scores,
        kps,
        np.ones(n_frames, dtype=np.int64),
        (1280.0, 720.0),
        half_window=1,
        scene_courts=(SceneCourt(0, len(track), court_info, (0.0, 0.0), 0.0),),
    )

    numerator = np.hypot(600.0 - 600.0, 360.0 - 170.0)
    picked_half = int(np.flatnonzero(players.detection_indices[2] >= 0)[0])
    assert gaps.per_player[2, picked_half] == pytest.approx(numerator / 100.0)


def test_build_player_evidence_fails_when_candidate_has_no_finite_height() -> None:
    (track, segments, bboxes, scores, kps, ndet, court_info) = _sticky_inputs()
    bboxes[1, 1, 1] = np.nan
    bboxes[2, 1, 1] = np.nan
    bboxes[3, 0, 1] = np.nan

    with pytest.raises(ValueError, match='no accepted finite height'):
        build_player_evidence(
            track,
            segments,
            bboxes,
            scores,
            kps,
            ndet,
            (1280.0, 720.0),
            half_window=1,
            scene_courts=(SceneCourt(0, len(track), court_info['1'], (0.0, 0.0), 0.0),),
        )


def test_build_player_evidence_fails_on_non_positive_height_denominator() -> None:
    (track, segments, bboxes, scores, kps, ndet, court_info) = _sticky_inputs()
    bboxes[1, 1, 1] = bboxes[1, 1, 3]
    bboxes[2, 1, 1] = bboxes[2, 1, 3]
    bboxes[3, 0, 1] = bboxes[3, 0, 3]

    with pytest.raises(ValueError, match='non-finite or non-positive body-scale denominator'):
        build_player_evidence(
            track,
            segments,
            bboxes,
            scores,
            kps,
            ndet,
            (1280.0, 720.0),
            half_window=1,
            scene_courts=(SceneCourt(0, len(track), court_info['1'], (0.0, 0.0), 0.0),),
        )
