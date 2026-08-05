"""Focused contracts for the Packet 2 court-evidence adapter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import annotator.court_evidence as evidence
import annotator.point_winner as point_winner
from annotator.calibration.fixtures import FIXTURES
from annotator.config import COMPOSITION_CONTENT_THRESHOLD
from annotator.point_winner import corner_error_band_from_corners, project_pixels_to_court
from courtkeynet.court_corners import CourtQuad, FallbackDiagnostics


def _quad(
    corners: np.ndarray,
    source: str = 'model',
    diagnostics: FallbackDiagnostics | None = None,
) -> CourtQuad:
    return CourtQuad(
        corners_px=corners.astype(np.float32),
        peak=np.full(4, 0.9, dtype=np.float32),
        source=source,
        corner_source=(source, source, source, source),
        diagnostics=diagnostics,
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


def test_static_corner_order_matches_pose_columns_and_landing_band(monkeypatch) -> None:
    camera_order = np.array([
        [11.0, 12.0],  # top-left
        [21.0, 22.0],  # top-right
        [41.0, 42.0],  # bottom-left
        [31.0, 32.0],  # bottom-right
    ])
    courtkeynet_order = np.array([
        [11.0, 12.0],
        [21.0, 22.0],
        [31.0, 32.0],
        [41.0, 42.0],
    ])
    monkeypatch.setattr(evidence, 'get_corner_camera', lambda _row: camera_order.T)
    static_corners = evidence._static_corners_refpx(pd.Series(dtype=float))
    np.testing.assert_array_equal(static_corners, courtkeynet_order)

    rows = evidence.build_scene_rows(
        7, [(2, 6)], [static_corners], (1280.0, 720.0),
    )
    pose_columns = rows.loc[0, [
        'upleft_x', 'upleft_y', 'upright_x', 'upright_y',
        'downleft_x', 'downleft_y', 'downright_x', 'downright_y',
    ]].to_numpy(dtype=float)
    np.testing.assert_array_equal(
        pose_columns,
        courtkeynet_order[[0, 1, 3, 2]].reshape(-1),
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
    np.testing.assert_array_equal(received[0], courtkeynet_order)


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


@pytest.mark.parametrize(
    ('start', 'end', 'expected'),
    [(4, 9, [4, 5, 6, 7, 8]), (0, 40, [2, 6, 10, 14, 18, 22, 26, 30, 34, 38])],
)
def test_centred_bin_samples(start, end, expected) -> None:
    assert evidence.scene_sample_indices(start, end) == expected


def test_scene_sample_limit_is_the_executable_cap() -> None:
    samples = evidence.scene_sample_indices(0, evidence.COURT_SCENE_SAMPLE_LIMIT + 5)
    assert len(samples) == evidence.COURT_SCENE_SAMPLE_LIMIT


def test_detect_scene_evidence_decodes_once_in_scene_order(monkeypatch) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.read_indices = []
            self.released = False
            self.next_frame = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            frame_index = self.next_frame
            self.next_frame += 1
            self.read_indices.append(frame_index)
            return True, np.full((1, 1, 3), frame_index, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    class FakeDetector:
        corner_min_peak_conf = 0.75

        def __init__(self) -> None:
            self.calls = []

        def detect_batch(self, frames):
            frame_indices = [int(frame[0, 0, 0]) for frame in frames]
            self.calls.append(frame_indices)
            return [f'detection-{len(self.calls)}']

    capture = FakeCapture()
    detector = FakeDetector()
    model = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]), 'model')
    fallback = _quad(np.array([[1, 2], [511, 1], [510, 287], [0, 286]]), 'fallback')
    picked = [model, fallback, None]
    pick_calls = []

    def fake_pick(frames, detections, *, corner_min_peak_conf):
        pick_calls.append((
            [int(frame[0, 0, 0]) for frame in frames], detections, corner_min_peak_conf,
        ))
        return picked[len(pick_calls) - 1]

    monkeypatch.setattr(evidence.cv2, 'VideoCapture', lambda _path: capture)
    monkeypatch.setattr(evidence, 'pick_scene_corners', fake_pick)

    result = evidence.detect_scene_evidence(
        'video.mp4', [(0, 3), (3, 8), (8, 10)], detector,
    )

    assert capture.read_indices == list(range(10))
    assert detector.calls == [[0, 1, 2], [3, 4, 5, 6, 7], [8, 9]]
    assert [call[0] for call in pick_calls] == detector.calls
    assert [call[1] for call in pick_calls] == [
        ['detection-1'], ['detection-2'], ['detection-3'],
    ]
    assert result[0].quad is model
    assert result[1].quad is fallback
    assert result[2].quad is None
    assert [scene.sampled_frame_indices for scene in result] == [(0, 1, 2), (3, 4, 5, 6, 7), (8, 9)]
    assert capture.released


def test_court_quad_provenance_is_preserved() -> None:
    model = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]), 'model')
    fallback = _quad(np.array([[1, 2], [511, 1], [510, 287], [0, 286]]), 'fallback')
    assert model.source == 'model'
    assert fallback.source == 'fallback'
    assert evidence.SceneEvidence(0, 5, (2,), None).quad is None


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


def test_parent_evidence_applies_inclusive_scene_majority() -> None:
    bboxes, scores, ndet = _pose_inputs(7)
    ndet[2:4] = 0
    ndet[5:] = 0
    quad = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]))
    result = evidence.build_detected_court_evidence(
        'case-a',
        'detected_ckn_opencv_consensus',
        1,
        (1280.0, 720.0),
        [(0, 4), (4, 7)],
        [evidence.SceneEvidence(0, 4, (1, 3), quad), evidence.SceneEvidence(4, 7, (5, 6), quad)],
        bboxes,
        scores,
        ndet,
    )
    first_record, second_record = result.scene_records
    assert first_record.exactly_two_fraction == pytest.approx(evidence.SCENE_VALID_MIN_FRACTION)
    assert first_record.scene_valid is True
    assert second_record.exactly_two_fraction == pytest.approx(1 / 3)
    assert second_record.scene_valid is False
    assert result.court_present.tolist() == [True] * 4 + [False] * 3


def test_detected_inputs_use_only_accepted_scene_quads() -> None:
    cuts = [(0, 10), (10, 20)]
    quad = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]))
    bboxes, scores, ndet = _pose_inputs(20)
    ndet[10:] = 0
    result = evidence.build_detected_court_evidence(
        '', '', 1,
        (1280.0, 720.0),
        cuts,
        [evidence.SceneEvidence(0, 10, (2,), quad), evidence.SceneEvidence(10, 20, (12,), None)],
        bboxes,
        scores,
        ndet,
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


def test_detected_consensus_failure_handoff_preserves_raw_values() -> None:
    bboxes, scores, ndet = _pose_inputs(10)
    null_scene = [evidence.SceneEvidence(0, 10, (), None)]
    with pytest.raises(evidence.CourtConsensusError) as null_failure:
        evidence.build_detected_court_evidence(
            'case-a', 'detected', 1, (1280.0, 720.0), [(0, 10)], null_scene,
            bboxes, scores, ndet,
        )
    null_handoff = null_failure.value.result
    assert null_failure.value.original_error.args == (
        'detected court consensus requires at least one accepted scene quad',
    )
    assert null_handoff.inputs is None
    assert not null_handoff.keep_vote.any()
    assert not null_handoff.court_present.any()
    assert null_handoff.scene_records[0].raw_corners_px is None
    assert null_handoff.scene_records[0].active_corners_native_px is None

    quad_a = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]))
    quad_b = _quad(np.array([[200, 0], [712, 0], [712, 288], [200, 288]]))
    bboxes, scores, ndet = _pose_inputs(20)
    bboxes[:, 0] = (650.0, 100.0, 750.0, 200.0)
    bboxes[:, 1] = (950.0, 300.0, 1050.0, 400.0)
    with pytest.raises(evidence.CourtConsensusError) as consensus_failure:
        evidence.build_detected_court_evidence(
            'case-a', 'detected', 1,
            (1280.0, 720.0),
            [(0, 10), (10, 20)],
            [
                evidence.SceneEvidence(0, 10, (), quad_a),
                evidence.SceneEvidence(10, 20, (), quad_b),
            ],
            bboxes,
            scores,
            ndet,
        )
    handoff = consensus_failure.value.result
    assert 'no trustworthy majority' in str(consensus_failure.value.original_error)
    assert handoff.inputs is None
    assert handoff.keep_vote.all()
    assert handoff.court_present.all()
    assert handoff.consensus is None
    for record, quad in zip(handoff.scene_records, (quad_a, quad_b)):
        np.testing.assert_array_equal(record.raw_corners_px, quad.corners_px)
        assert record.raw_source == 'model'
        assert record.consensus_distance_px is None
        assert record.consensus_flag is None
        assert record.active_corners_native_px is None


def test_consensus_outlier_uses_matching_repaired_native_active_quad() -> None:
    base_corners = np.array([[0, 0], [512, 0], [512, 288], [0, 288]])
    outlier_corners = base_corners.copy()
    outlier_corners[:, 0] += 100
    bboxes, scores, ndet = _pose_inputs(12)
    bboxes[:, 0] = (300.0, 100.0, 400.0, 200.0)
    bboxes[:, 1] = (600.0, 300.0, 700.0, 400.0)
    quads = [_quad(base_corners), _quad(base_corners), _quad(outlier_corners)]
    result = evidence.build_detected_court_evidence(
        'case-a', 'detected_ckn_opencv_consensus', 1, (1280.0, 720.0),
        [(0, 4), (4, 8), (8, 12)],
        [evidence.SceneEvidence(start, end, (), quad)
         for (start, end), quad in zip([(0, 4), (4, 8), (8, 12)], quads)],
        bboxes,
        scores,
        ndet,
    )
    outlier_record = result.scene_records[2]
    assert outlier_record.video_id == 1
    assert outlier_record.scene_index == 2
    assert outlier_record.consensus_flag is True
    assert outlier_record.consensus_distance_px is not None
    assert outlier_record.consensus_distance_px > 55.0
    np.testing.assert_array_equal(outlier_record.raw_corners_px, outlier_corners)
    np.testing.assert_allclose(outlier_record.active_corners_native_px, base_corners)


def test_typed_detected_records_preserve_provenance_vote_and_consensus_fields() -> None:
    diagnostics = FallbackDiagnostics(1.0, 2.0, 0.1, 0.2, 3, 4, 5.0)
    model = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]), 'model')
    fallback = _quad(
        np.array([[1, 2], [511, 1], [510, 287], [0, 286]]),
        'fallback',
        diagnostics,
    )
    bboxes, scores, ndet = _pose_inputs(10)
    ndet[8:] = 0
    result = evidence.build_detected_court_evidence(
        'case-a',
        'detected_ckn_opencv_consensus',
        1,
        (1280.0, 720.0),
        [(0, 4), (4, 8), (8, 10)],
        [
            evidence.SceneEvidence(0, 4, (1, 3), model),
            evidence.SceneEvidence(4, 8, (5, 7), fallback),
            evidence.SceneEvidence(8, 10, (9,), None),
        ],
        bboxes,
        scores,
        ndet,
    )

    model_record, fallback_record, null_record = result.scene_records
    assert model_record.case_id == 'case-a'
    assert model_record.parent == 'detected_ckn_opencv_consensus'
    assert model_record.raw_source == 'model'
    assert model_record.raw_peaks is not None
    assert model_record.raw_corner_source == ('model',) * 4
    assert model_record.consensus_distance_px is not None
    assert model_record.consensus_flag is False
    assert model_record.active_corners_native_px is not None
    assert fallback_record.raw_source == 'fallback'
    assert fallback_record.fallback_diagnostics == diagnostics
    assert fallback_record.consensus_distance_px is not None
    assert fallback_record.consensus_flag is False
    assert null_record.raw_corners_px is None
    assert null_record.raw_source is None
    assert null_record.raw_peaks is None
    assert null_record.raw_corner_source is None
    assert null_record.fallback_diagnostics is None
    assert null_record.exactly_two_count == 0
    assert null_record.exactly_two_fraction == 0.0
    assert null_record.scene_valid is False
    assert null_record.consensus_distance_px is None
    assert null_record.consensus_flag is None
    assert null_record.active_corners_native_px is None
    assert result.keep_vote[:8].all()
    assert result.court_present[:8].all()
    assert not result.court_present[8:].any()


def test_detected_consensus_receives_native_quads_before_scaling(monkeypatch) -> None:
    quad = _quad(np.array([[10, 20], [500, 20], [500, 280], [10, 280]]))
    bboxes, scores, ndet = _pose_inputs(4)
    received = []
    import courtkeynet.court_corners as corners_module

    real_consensus = corners_module.consensus_repair

    def spy(quads):
        received.append(quads.copy())
        return real_consensus(quads)

    monkeypatch.setattr(corners_module, 'consensus_repair', spy)
    evidence.build_detected_court_evidence(
        'case-a', 'detected', 1, (1280.0, 720.0), [(0, 4)],
        [evidence.SceneEvidence(0, 4, (1, 3), quad)], bboxes, scores, ndet,
    )
    assert len(received) == 1
    np.testing.assert_array_equal(received[0], quad.corners_px[None, ...])


def test_rejected_detected_quad_keeps_raw_evidence_without_active_geometry() -> None:
    accepted = _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]]))
    rejected = _quad(np.array([[2, 2], [510, 2], [510, 286], [2, 286]]), 'fallback')
    bboxes, scores, ndet = _pose_inputs(8)
    ndet[4:] = 0
    result = evidence.build_detected_court_evidence(
        'case-a', 'detected', 1, (1280.0, 720.0), [(0, 4), (4, 8)],
        [
            evidence.SceneEvidence(0, 4, (1, 3), accepted),
            evidence.SceneEvidence(4, 8, (5, 7), rejected),
        ],
        bboxes,
        scores,
        ndet,
    )
    record = result.scene_records[1]
    assert record.raw_source == 'fallback'
    assert record.raw_corners_px is not None
    assert record.exactly_two_count == 0
    assert record.exactly_two_fraction == 0.0
    assert record.scene_valid is False
    assert record.consensus_distance_px is None
    assert record.consensus_flag is None
    assert record.active_corners_native_px is None


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
    assert static_record.raw_source is None
    assert static_record.raw_peaks is None
    assert static_record.raw_corner_source is None
    assert static_record.fallback_diagnostics is None
    assert static_record.consensus_distance_px is None
    assert static_record.consensus_flag is None
    np.testing.assert_allclose(
        static_record.raw_corners_px,
        static_result.inputs.active_corners_refpx * np.array([512.0, 288.0]) / np.array([1280.0, 720.0]),
    )
    np.testing.assert_array_equal(static_record.raw_corners_px, static_record.active_corners_native_px)
    assert static_record.sampled_frame_indices == ()

    def fail_static_lookup(*_args, **_kwargs):
        raise AssertionError('detected parent must not read static homography')

    monkeypatch.setattr(evidence, 'get_court_info', fail_static_lookup)
    bboxes, scores, ndet = _pose_inputs(4)
    detected_result = evidence.build_detected_court_evidence(
        'case-a', 'detected_ckn_opencv_consensus', 1, (1280.0, 720.0), [(0, 4)],
        [evidence.SceneEvidence(
            0, 4, tuple(evidence.scene_sample_indices(0, 4)),
            _quad(np.array([[0, 0], [512, 0], [512, 288], [0, 288]])),
        )],
        bboxes,
        scores,
        ndet,
    )
    assert detected_result.keep_vote.any()
    assert detected_result.court_present.any()
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
