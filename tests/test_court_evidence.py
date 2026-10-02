"""Contracts for court evidence from the static table and court_detector results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import annotator.court_evidence as evidence
import annotator.point_winner as point_winner
from annotator.calibration.fixtures import FIXTURES
from annotator.config import COMPOSITION_CONTENT_THRESHOLD
from annotator.point_winner import (
    corner_error_band_from_corners,
    project_pixels_to_court,
)

def _identity_info() -> dict[str, object]:
    return {
        'H': np.eye(3),
        'border_L': 0.0,
        'border_R': 1.0,
        'border_U': 0.0,
        'border_D': 1.0,
    }


def _pose_inputs(n_frames: int, n_slots: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bboxes = np.zeros((n_frames, n_slots, 4), dtype=float)
    scores = np.full((n_frames, n_slots), np.nan, dtype=float)
    ndet = np.full(n_frames, 2, dtype=int)
    for frame in range(n_frames):
        bboxes[frame, 0] = (100.0, 100.0, 200.0, 200.0)
        bboxes[frame, 1] = (900.0, 300.0, 1000.0, 400.0)
        scores[frame, :2] = 0.9
    return bboxes, scores, ndet


FULL_FRAME_PX = [[0.0, 0.0], [512.0, 0.0], [512.0, 288.0], [0.0, 288.0]]  # TL, TR, BR, BL at 512x288


def _scene(start: int, end: int, corners: object = None, status: str | None = None, **fields: str) -> evidence.DetectorScene:
    """One typed detector scene; a scene with corners defaults to ``court``."""
    if status is None:
        status = 'court' if corners is not None else 'no_court'
    return evidence.DetectorScene(
        start, end, (start + end) // 2, evidence.SceneStatus(status),
        None if corners is None else np.asarray(corners, dtype=float),
        fields.get('no_court_reason'), fields.get('reused_from'), fields.get('error'),
    )


def _row(start: int, end: int, status: str = 'court', **fields: object) -> dict[str, object]:
    """One detector scene row, shaped as ``court_detector.run_video.scene_courts`` writes it."""
    row: dict[str, object] = {
        'view_id': f'video_scene_{start}', 'start_frame': start, 'end_frame': end,
        'frame_index': (start + end) // 2, 'status': status,
        'corners_native_px': FULL_FRAME_PX if status == 'court' else None,
    }
    if status in {'court', 'no_court'}:
        row.update(chosen_key=None, no_court_reason=None, reused_from=None, stage_seconds={})
    if status == 'detection_failed':
        row.update(error="CourtFitError('no fit')", traceback='Traceback ...')
    row.update(fields)
    return row


def _result(rows: list[dict[str, object]], frame_count: int = 10) -> dict[str, object]:
    return {
        'schema': evidence.DETECTOR_RESULT_SCHEMA, 'video_id': 'video', 'frame_count': frame_count,
        'native_size': [512, 288], 'scenes': rows,
    }


def test_static_corner_order_matches_pose_columns_and_landing_band(monkeypatch) -> None:
    camera_order = np.array([
        [11.0, 12.0],  # top-left
        [21.0, 22.0],  # top-right
        [41.0, 42.0],  # bottom-left
        [31.0, 32.0],  # bottom-right
    ])
    clockwise_order = np.array([
        [11.0, 12.0],
        [21.0, 22.0],
        [31.0, 32.0],
        [41.0, 42.0],
    ])
    monkeypatch.setattr(evidence, 'get_corner_camera', lambda _row: camera_order.T)
    static_corners = evidence._static_corners_refpx(pd.Series(dtype=float))
    np.testing.assert_array_equal(static_corners, clockwise_order)

    rows = evidence.build_scene_rows(
        7, [(2, 6)], [static_corners], (1280.0, 720.0),
    )
    pose_columns = rows.loc[0, [
        'upleft_x', 'upleft_y', 'upright_x', 'upright_y',
        'downleft_x', 'downleft_y', 'downright_x', 'downright_y',
    ]].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        pose_columns,
        clockwise_order[[0, 1, 3, 2]].reshape(-1),
    )

    received = []

    def fake_error_band(corners, _court_info, _err_px):
        received.append(corners.copy())
        return 1.25

    monkeypatch.setattr(point_winner, 'get_corner_camera', lambda _row: camera_order.T)
    monkeypatch.setattr(point_winner, 'corner_error_band_from_corners', fake_error_band)
    result = point_winner.corner_error_band_m(
        7, pd.DataFrame(index=[7]), _identity_info(), 3.5,
    )
    assert result == 1.25
    np.testing.assert_array_equal(received[0], clockwise_order)


@pytest.mark.parametrize(('fps', 'minimum'), [(25.0, 13), (30.0, 15)])
def test_raw_cut_wrapper_uses_exact_arguments_and_partitions(monkeypatch, fps, minimum) -> None:
    calls = []

    def fake_detect(video_path, **kwargs):
        calls.append((video_path, kwargs))
        return np.array([13, 28], dtype=int)

    monkeypatch.setattr(evidence, 'detect_cuts', fake_detect)
    intervals = evidence.build_raw_cut_intervals('video.mp4', 40, fps)

    assert intervals == [(0, 13), (13, 28), (28, 40)]
    assert calls == [(
        'video.mp4',
        {
            'expected_frames': 40,
            'threshold': COMPOSITION_CONTENT_THRESHOLD,
            'min_scene_len': minimum,
        },
    )]



def test_detector_schema_matches_the_runner() -> None:
    """court_evidence keeps its own copy so it need not import run_video's thread limits."""
    from court_detector.run_video import VIDEO_RESULT_SCHEMA

    assert evidence.DETECTOR_RESULT_SCHEMA == VIDEO_RESULT_SCHEMA


def test_read_detector_scenes_types_every_status() -> None:
    rows = [
        _row(0, 3),
        _row(3, 5, 'scene_too_short_for_feet'),
        _row(5, 8, 'no_court', no_court_reason='no_candidates'),
        _row(8, 10, 'detection_failed'),
    ]
    scenes = evidence.read_detector_scenes(_result(rows), frame_count=10, native_size=(512.0, 288.0))
    assert [scene.status.value for scene in scenes] == [
        'court', 'scene_too_short_for_feet', 'no_court', 'detection_failed',
    ]
    assert [(scene.start_frame, scene.end_frame, scene.analysed_frame) for scene in scenes] == [
        (0, 3, 1), (3, 5, 4), (5, 8, 6), (8, 10, 9),
    ]
    np.testing.assert_array_equal(scenes[0].corners_native_px, FULL_FRAME_PX)
    assert scenes[1].corners_native_px is None and scenes[1].no_court_reason is None
    assert scenes[2].no_court_reason == 'no_candidates'
    assert scenes[3].error == "CourtFitError('no fit')"


def test_read_detector_scenes_names_an_unsupported_schema_first() -> None:
    stale = {'schema': 'court-detector-video/0', 'frame_count': 1, 'native_size': [1, 1], 'scenes': []}
    with pytest.raises(ValueError, match="unsupported court detector schema 'court-detector-video/0'"):
        evidence.read_detector_scenes(stale, frame_count=10, native_size=(512.0, 288.0))


@pytest.mark.parametrize(('frame_count', 'native_size', 'message'), [
    (11, (512.0, 288.0), 'frame count 10 differs from 11'),
    (10, (1280.0, 720.0), r'native size \[512, 288\] differs'),
])
def test_read_detector_scenes_rejects_a_different_video(frame_count, native_size, message) -> None:
    with pytest.raises(ValueError, match=message):
        evidence.read_detector_scenes(_result([_row(0, 10)]), frame_count=frame_count, native_size=native_size)


@pytest.mark.parametrize(('rows', 'message'), [
    ([], 'no scene rows'),
    ([_row(0, 4), _row(5, 10)], 'must start at frame 4'),
    ([_row(0, 4), _row(4, 4), _row(4, 10)], 'end after it'),
    ([_row(0, 4), _row(4, 9)], 'end at frame 9, expected 10'),
    ([_row(0, 10, 'no_court', corners_native_px=FULL_FRAME_PX)], "only court scenes carry corners"),
    ([_row(0, 10, corners_native_px=None)], "only court scenes carry corners"),
    ([_row(0, 10, corners_native_px=[[0.0, 0.0]] * 3)], 'four finite points'),
    ([_row(0, 10, 'detection_failed', error=None)], 'failed without an error'),
    ([_row(0, 10, 'not_a_status')], 'not_a_status'),
])
def test_read_detector_scenes_rejects_bad_rows(rows, message) -> None:
    with pytest.raises(ValueError, match=message):
        evidence.read_detector_scenes(_result(rows), frame_count=10, native_size=(512.0, 288.0))


@pytest.mark.parametrize('n_people, expected', [(0, False), (1, False), (2, True), (3, False)])
def test_keep_vote_requires_exactly_two_in_margin_people(n_people: int, expected: bool) -> None:
    bboxes = np.zeros((1, 3, 4), dtype=float)
    scores = np.full((1, 3), np.nan, dtype=float)
    ndet = np.array([n_people], dtype=int)
    centres = [(0.2, 0.2), (0.8, 0.8), (0.5, 0.5)]
    for slot, (x, y) in enumerate(centres[:n_people]):
        pixel_x, pixel_y = x * 1280.0, y * 720.0
        bboxes[0, slot] = (pixel_x - 1.0, pixel_y - 2.0, pixel_x + 1.0, pixel_y)
        scores[0, slot] = 0.9

    vote = evidence.build_keep_vote(
        bboxes,
        scores,
        ndet,
        (1280.0, 720.0),
        [(0, 1)],
        [evidence.detected_court_info(np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]]))],
    )
    assert vote.tolist() == [expected]


def test_keep_vote_ignores_outside_people_and_includes_margin_boundaries() -> None:
    bboxes = np.zeros((1, 4, 4), dtype=float)
    scores = np.full((1, 4), 0.9, dtype=float)
    ndet = np.array([4], dtype=int)
    margin = evidence.PERSON_COURT_MARGIN
    centres = [(-margin, 0.5), (1.0 + margin, 1.0 + margin), (0.5, 0.5), (-margin - 0.1, 0.5)]
    for slot, (x, y) in enumerate(centres):
        pixel_x, pixel_y = x * 1280.0, y * 720.0
        bboxes[0, slot] = (pixel_x - 1.0, pixel_y - 2.0, pixel_x + 1.0, pixel_y)

    vote = evidence.build_keep_vote(
        bboxes,
        scores,
        ndet,
        (1280.0, 720.0),
        [(0, 1)],
        [evidence.detected_court_info(np.array([[0, 0], [1280, 0], [1280, 720], [0, 720]]))],
    )
    assert vote.tolist() == [True]



def test_detected_parent_applies_inclusive_scene_majority() -> None:
    bboxes, scores, ndet = _pose_inputs(7)
    ndet[2:4] = 0
    ndet[5:] = 0
    result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (512.0, 288.0),
        [_scene(0, 4, FULL_FRAME_PX), _scene(4, 7, FULL_FRAME_PX)],
        bboxes, scores, ndet,
    )
    first_record, second_record = result.scene_records
    assert first_record.parent == evidence.DETECTED_PARENT
    assert first_record.exactly_two_fraction == pytest.approx(evidence.SCENE_VALID_MIN_FRACTION)
    assert first_record.scene_valid is True
    assert second_record.exactly_two_fraction == pytest.approx(1 / 3)
    assert second_record.scene_valid is False
    assert result.court_present.tolist() == [True] * 4 + [False] * 3


def test_detected_inputs_use_only_accepted_scene_courts() -> None:
    bboxes, scores, ndet = _pose_inputs(20)
    ndet[10:] = 0
    result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (512.0, 288.0),
        [_scene(0, 10, FULL_FRAME_PX), _scene(10, 20, FULL_FRAME_PX)],
        bboxes, scores, ndet,
    )
    inputs = result.inputs
    assert inputs is not None
    assert inputs.homography_rows[['start_frame', 'end_frame']].values.tolist() == [[0, 10]]
    projected = project_pixels_to_court(
        np.array([[0.0, 1280.0, 1280.0, 0.0], [0.0, 0.0, 720.0, 720.0]]),
        (1280.0, 720.0),
        inputs.court_info,
    )
    np.testing.assert_allclose(projected.T, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_scenes_without_a_court_keep_their_status_as_the_absence_reason() -> None:
    bboxes, scores, ndet = _pose_inputs(20)
    scenes = [
        _scene(0, 8, FULL_FRAME_PX),
        _scene(8, 10, status='scene_too_short_for_feet'),
        _scene(10, 15, no_court_reason='no_candidates'),
        _scene(15, 20, status='detection_failed', error='CourtFitError()'),
    ]
    result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (512.0, 288.0), scenes, bboxes, scores, ndet,
    )
    assert result.court_present.tolist() == [True] * 8 + [False] * 12
    assert result.keep_vote.tolist() == [True] * 8 + [False] * 12
    statuses = [(record.status, record.scene_valid, record.corners_native_px is None) for record in result.scene_records]
    assert statuses == [
        (evidence.SceneStatus.COURT, True, False),
        (evidence.SceneStatus.SCENE_TOO_SHORT_FOR_FEET, False, True),
        (evidence.SceneStatus.NO_COURT, False, True),
        (evidence.SceneStatus.DETECTION_FAILED, False, True),
    ]
    assert result.scene_records[2].no_court_reason == 'no_candidates'
    assert result.scene_records[3].error == 'CourtFitError()'
    assert result.inputs is not None
    assert result.inputs.homography_rows[['start_frame', 'end_frame']].values.tolist() == [[0, 8]]


def test_distinct_camera_views_keep_their_own_courts() -> None:
    """Two accepted scenes with different courts each keep their geometry; nothing is shared."""
    wide = FULL_FRAME_PX
    shifted = [[80.0, 0.0], [592.0, 0.0], [592.0, 288.0], [80.0, 288.0]]
    bboxes, scores, ndet = _pose_inputs(20)
    bboxes[:, 0] = (650.0, 100.0, 750.0, 200.0)
    bboxes[:, 1] = (950.0, 300.0, 1050.0, 400.0)
    result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (512.0, 288.0),
        [_scene(0, 10, wide), _scene(10, 20, shifted, reused_from='video_scene_0')],
        bboxes, scores, ndet,
    )
    assert result.court_present.all()
    for record, corners in zip(result.scene_records, (wide, shifted)):
        np.testing.assert_array_equal(record.corners_native_px, corners)
    assert result.scene_records[1].reused_from == 'video_scene_0'
    assert result.inputs is not None
    rows = result.inputs.homography_rows
    assert rows[['start_frame', 'end_frame']].values.tolist() == [[0, 10], [10, 20]]
    # 80 source pixels at 512 wide are 200 reference pixels at 1280 wide.
    assert rows['upleft_x'].tolist() == pytest.approx([0.0, 200.0])


def test_detector_corners_convert_from_native_to_pose_pixels() -> None:
    """A 1920x1080 video's court maps onto 1280x720 pose boxes through reference pixels."""
    bboxes, scores, ndet = _pose_inputs(4)
    native = np.array(FULL_FRAME_PX) * np.array([1920.0 / 512.0, 1080.0 / 288.0])
    result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (1920.0, 1080.0), [_scene(0, 4, native)], bboxes, scores, ndet,
    )
    assert result.court_present.all()
    assert result.inputs is not None
    np.testing.assert_allclose(
        result.inputs.active_corners_refpx, [[0.0, 0.0], [1280.0, 0.0], [1280.0, 720.0], [0.0, 720.0]],
    )
    np.testing.assert_array_equal(result.scene_records[0].corners_native_px, native)


def test_no_accepted_court_hands_back_the_scene_evidence() -> None:
    bboxes, scores, ndet = _pose_inputs(10)
    ndet[:] = 0
    scenes = [_scene(0, 6, FULL_FRAME_PX), _scene(6, 10, status='scene_too_short_for_feet')]
    with pytest.raises(evidence.NoAcceptedCourtError, match='scene_too_short_for_feet') as failure:
        evidence.build_court_detector_evidence(
            'case-a', 1, (1280.0, 720.0), (512.0, 288.0), scenes, bboxes, scores, ndet,
        )
    handoff = failure.value.result
    assert handoff.inputs is None
    assert not handoff.keep_vote.any()
    assert not handoff.court_present.any()
    assert [record.status for record in handoff.scene_records] == [
        evidence.SceneStatus.COURT, evidence.SceneStatus.SCENE_TOO_SHORT_FOR_FEET,
    ]
    np.testing.assert_array_equal(handoff.scene_records[0].corners_native_px, FULL_FRAME_PX)


@pytest.mark.parametrize('video_id', [1, 15, 21])
def test_static_parent_net_band_matches_fixture(video_id: int) -> None:
    fixture = next(item for item in FIXTURES if item.video_id == video_id)
    homography = pd.read_csv(
        'training/data/shuttleset/annotations/set/homography.csv',
    ).set_index('id')
    inputs = evidence.build_static_court_inputs(
        video_id, homography, fixture.resolution, [(0, 10)],
    )
    assert inputs.net_band == fixture.net_band



def test_static_records_and_detected_parent_are_isolated(monkeypatch) -> None:
    homography = pd.read_csv(
        'training/data/shuttleset/annotations/set/homography.csv',
    ).set_index('id')
    fixture = next(item for item in FIXTURES if item.video_id == 1)
    static_bboxes = np.zeros((4, 1, 4), dtype=float)
    static_scores = np.full((4, 1), np.nan, dtype=float)
    static_ndet = np.zeros(4, dtype=int)
    static_result = evidence.build_static_court_evidence(
        'case-a',
        'static_shuttleset_homography',
        1,
        homography,
        fixture.resolution,
        [(0, 4)],
        static_bboxes,
        static_scores,
        static_ndet,
    )
    static_record = static_result.scene_records[0]
    assert static_result.inputs is not None
    assert static_record.video_id == 1
    assert static_record.scene_index == 0
    assert static_record.status is evidence.SceneStatus.COURT
    assert static_record.analysed_frame is None
    assert static_record.no_court_reason is None and static_record.reused_from is None
    np.testing.assert_allclose(
        static_record.corners_native_px,
        static_result.inputs.active_corners_refpx * np.array([512.0, 288.0]) / np.array([1280.0, 720.0]),
    )

    def fail_static_lookup(*_args, **_kwargs):
        raise AssertionError('detected parent must not read static homography')

    monkeypatch.setattr(evidence, 'get_court_info', fail_static_lookup)
    bboxes, scores, ndet = _pose_inputs(4)
    detected_result = evidence.build_court_detector_evidence(
        'case-a', 1, (1280.0, 720.0), (512.0, 288.0), [_scene(0, 4, FULL_FRAME_PX)], bboxes, scores, ndet,
    )
    assert detected_result.keep_vote.any()
    assert detected_result.court_present.any()
    assert detected_result.inputs is not None
    assert detected_result.inputs.net_band != static_result.inputs.net_band
    assert detected_result.inputs.landing_error_band_m != static_result.inputs.landing_error_band_m
    assert not np.shares_memory(
        detected_result.inputs.active_corners_refpx,
        static_result.inputs.active_corners_refpx,
    )


def test_court_inputs_copy_mutable_values() -> None:
    court_info = _identity_info()
    table = pd.DataFrame({'width': [10.0], 'height': [20.0]}, index=['1'])
    rows = pd.DataFrame([{'video_id': 1, 'start_frame': 0, 'end_frame': 1}])
    corners = np.zeros((4, 2), dtype=float)
    inputs = evidence.CourtInputs(
        court_info, {'1': court_info}, (1.0, 2.0), (10.0, 20.0), table, rows, 0.1, corners,
    )
    court_info['H'][0, 0] = 9.0
    table.loc['1', 'width'] = 99.0
    rows.loc[0, 'start_frame'] = 99
    corners[0, 0] = 9.0
    assert inputs.court_info['H'][0, 0] == 1.0
    assert inputs.gate_resolution_table.loc['1', 'width'] == 10.0
    assert inputs.homography_rows.loc[0, 'start_frame'] == 0
    assert inputs.active_corners_refpx[0, 0] == 0.0


def test_static_error_band_wrapper_matches_pure_helper() -> None:
    homography = pd.read_csv(
        'training/data/shuttleset/annotations/set/homography.csv',
    ).set_index('id')
    fixture = next(item for item in FIXTURES if item.video_id == 1)
    inputs = evidence.build_static_court_inputs(1, homography, fixture.resolution, [(0, 1)])
    assert inputs.landing_error_band_m == pytest.approx(
        corner_error_band_from_corners(inputs.active_corners_refpx, inputs.court_info, 3.5),
    )
