"""Smoke coverage for the public annotator video composition."""
import numpy as np
import pandas as pd
import pytest

import annotator.run_video as run_video_module
import annotator.rally_segmentation as stage8_seg
from annotator.calibration.gt_scoring import write_geometric_verdicts_csv
from annotator.config import BaseAnnotatorConfig
from annotator.point_winner import GeometricVerdictRow, Half, Landing, LandingFilterOptions, Verdict
from annotator.fps_constants import scale_for_fps
from annotator.rally_segmentation import CourtBox, ServeStartClose, ServeStartMode, StickyResult
from annotator.run_video import AnnotatorResult, build_serve_options, run_video, scoring_filter
from annotator.types import ContactCandidate, ServeStartConfig


def test_run_video_no_play_returns_empty_result():
    video_id = 1
    resolution = (1920.0, 1080.0)
    track = np.zeros((300, 3), dtype=np.float64)
    bboxes = np.zeros((300, 1, 4), dtype=np.float32)
    scores = np.zeros((300, 1), dtype=np.float32)
    kps = np.zeros((300, 1, 17, 2), dtype=np.float32)
    ndet = np.zeros(300, dtype=np.int64)
    dead = np.zeros(300, dtype=bool)
    court_info = {
        "H": np.eye(3),
        "border_L": 0.0,
        "border_R": 1280.0,
        "border_U": 0.0,
        "border_D": 720.0,
    }
    homo_df = pd.DataFrame(
        {
            "upleft_x": [0.0], "upright_x": [1280.0],
            "downleft_x": [0.0], "downright_x": [1280.0],
            "upleft_y": [0.0], "upright_y": [0.0],
            "downleft_y": [720.0], "downright_y": [720.0],
        },
        index=[video_id],
    )
    gate_resolution_table = pd.DataFrame(
        {"width": [1920.0], "height": [1080.0]}, index=[str(video_id)],
    )

    result = run_video(
        track, bboxes, scores, kps, ndet,
        fps=25.0,
        landing_options=LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        court_box=CourtBox(
            x_range=(635.0, 1316.0), y_range=(254.0, 1030.0),
            height_band=(84.0, 336.0), mid_band=(642.0, 642.0),
        ),
        net_band=(664.6, 703.7),
        resolution=resolution,
        video_id=video_id,
        court_info=court_info,
        homo_df=homo_df,
        gate_court_info={str(video_id): court_info},
        gate_resolution_table=gate_resolution_table,
        dead_mask=dead,
        **_default_scene_inputs(len(track)),
    )

    assert result == AnnotatorResult([], [], [], {}, [], [], [], [], {}, {}, {}, {}, [])


def test_scoring_filter_keeps_only_unfailed_unsuppressed_rows():
    rows = [
        ContactCandidate(0, 1, None, True, False),
        ContactCandidate(0, 2, None, True, True),
        ContactCandidate(0, 3, None, False, False),
        ContactCandidate(0, 4, None, None, None),
    ]

    assert scoring_filter(rows) == [rows[0], rows[3]]


def test_build_serve_options_wires_sticky_setup() -> None:
    n_frames = 40
    sticky = StickyResult(
        distances=np.full(n_frames, np.nan), picks=np.full((n_frames, 2), -1),
        standing_count=np.full(n_frames, 2),
        ankle_pos=np.full((n_frames, 2, 2), (0.2, 0.3)),
        bbox_height=np.full((n_frames, 2), 100.0),
        distances_per_slot=np.full((n_frames, 2), 0.1),
        wrist_dist_px=np.full((n_frames, 2), 100.0), analysed=np.ones(n_frames, dtype=bool),
    )
    resolution = (1920.0, 1080.0)
    options = build_serve_options(
        ServeStartConfig(threshold_bh=0.75, mode=ServeStartMode.TRIM, stillness_threshold_bh=0.2),
        sticky, scale_for_fps(30.0), resolution,
    )
    assert options.dist is None
    assert options.setup is not None
    assert options.stillness_threshold_bh == 0.2
    assert options.lookback_frames == 25
    assert options.stillness_window_frames == 15
    assert options.setup.top_height[0] == pytest.approx(100.0 / 1080.0)

    with pytest.raises(ValueError, match='serve_start.close is unsupported with BACK_FILL'):
        build_serve_options(
            ServeStartConfig(threshold_bh=0.1, mode=ServeStartMode.TRIM, close=ServeStartClose.BURST),
            sticky, scale_for_fps(30.0), resolution,
        )


def test_run_video_injected_spans_bypass_natural_span_finding(monkeypatch):
    inputs = _synthetic_inputs()
    injected = [(10, 20)]
    monkeypatch.setattr(
        stage8_seg, 'find_rally_spans',
        lambda *args, **kwargs: pytest.fail('natural span finding was not bypassed'),
    )

    result = run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=injected)

    assert result.spans == injected


@pytest.mark.parametrize('kwargs', [{}, {'spans': [(10, 20)]}, {'spans': [(10, 20)], 'contacts': {0: [14]}}])
def test_run_video_requires_scene_inputs_for_every_sticky_consumer(kwargs):
    with pytest.raises(ValueError, match='^scene-gated sticky needs homography_rows and court_present$'):
        run_video(**_synthetic_inputs(), **kwargs)


def test_run_video_hands_tracker_segments_output_to_sticky_builder(monkeypatch):
    """The sticky builder must receive exactly the list tracker_segments produced.

    The handoff is internal to run_video, so a wrong list (rally spans, say) would
    only show up as plausible end-number movement. The spy records the argument and
    calls the real builder through, so the real path still runs.
    """
    inputs = _synthetic_inputs()
    n_frames = len(inputs['track'])
    court_present = np.ones(n_frames, dtype=bool)
    court_present[40:60] = False
    scene_row = _default_scene_inputs(n_frames)['homography_rows'][0]
    homography_rows = [
        {**scene_row, 'start_frame': '0', 'end_frame': '150'},
        {**scene_row, 'start_frame': '150', 'end_frame': str(n_frames)},
    ]
    expected = stage8_seg.tracker_segments(homography_rows, court_present, n_frames)
    # Dropout splits the first scene row: the fixture must stay non-trivial.
    assert expected == [(0, 40), (60, 150), (150, 300)]

    real_builder = stage8_seg.build_sticky_result
    received = []

    def spy(track, segments, *args, **kwargs):
        received.append(segments)
        return real_builder(track, segments, *args, **kwargs)

    monkeypatch.setattr(stage8_seg, 'build_sticky_result', spy)

    run_video(**inputs, court_present=court_present, homography_rows=homography_rows)

    assert received == [expected]


def test_run_video_builds_serve_sticky_from_original_track_before_replay_mask(monkeypatch):
    inputs = _synthetic_inputs()
    original_track = inputs['track'].copy()
    del inputs['dead_mask']
    real_build_sticky = stage8_seg.build_sticky_result
    real_build_options = run_video_module.build_serve_options
    sticky_tracks = []
    option_stickies = []
    segment_tracks = []

    def spy_build_sticky(track, *args, **kwargs):
        sticky_tracks.append(track.copy())
        return real_build_sticky(track, *args, **kwargs)

    def spy_build_options(*args, **kwargs):
        option_stickies.append(args[1])
        return real_build_options(*args, **kwargs)

    def fake_dead_mask(*args, **kwargs):
        mask = np.zeros(len(original_track), dtype=bool)
        mask[0] = True
        return mask

    real_segment_video = stage8_seg.segment_video

    def spy_segment_video(track, *args, **kwargs):
        segment_tracks.append((track.copy(), kwargs['replay_mask'].copy()))
        return real_segment_video(track, *args, **kwargs)

    monkeypatch.setattr(stage8_seg, 'build_sticky_result', spy_build_sticky)
    monkeypatch.setattr(run_video_module, 'build_serve_options', spy_build_options)
    monkeypatch.setattr(run_video_module, 'build_dead_mask', fake_dead_mask)
    monkeypatch.setattr(stage8_seg, 'segment_video', spy_segment_video)

    run_video(
        **inputs, **_default_scene_inputs(len(original_track)),
        serve_start=ServeStartConfig(threshold_bh=0.8, mode=ServeStartMode.TRIM),
    )

    assert len(sticky_tracks) == 1
    np.testing.assert_array_equal(sticky_tracks[0], original_track)
    assert len(option_stickies) == 1
    assert len(segment_tracks) == 1
    np.testing.assert_array_equal(segment_tracks[0][0], original_track)
    assert segment_tracks[0][1][0]


def test_run_video_rejects_serve_start_with_injected_spans() -> None:
    inputs = _synthetic_inputs()
    with pytest.raises(ValueError, match='serve_start cannot be combined with injected spans'):
        run_video(
            **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
            serve_start=ServeStartConfig(threshold_bh=0.5, mode=ServeStartMode.TRIM),
        )


def test_run_video_injected_contacts_are_unmeasured_and_scored():
    inputs = _synthetic_inputs()
    spans = [(10, 20)]
    frames = [14, 16]
    expected = [ContactCandidate(0, frame, None, None, None) for frame in frames]

    result = run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=spans, contacts={0: frames})

    assert result.contacts == expected
    assert result.filtered_contacts == expected
    assert result.filtered_by_rally == {0: frames}


def test_run_video_injected_contacts_without_mask_completes(monkeypatch):
    inputs = _synthetic_inputs()
    del inputs['dead_mask']
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.striker_halves == [Half.TOP]
    assert 0 in result.verdict_rows
    assert 0 in result.geometric_verdict_rows


def test_run_video_uses_latest_unmasked_contact_for_landing(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[16] = 1
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1})), inpaint_codes=codes,
    )
    called_frames = []
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half', lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda final_contact, *args, **kwargs: called_frames.append(final_contact) or None,
    )

    run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]},
    )

    assert called_frames == [14]


def test_run_video_exhausts_masked_contacts_without_calling_landing(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[12:17] = 1
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1})), inpaint_codes=codes,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half', lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda *args, **kwargs: pytest.fail('landing must not run without an unmasked contact'),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]},
    )

    assert result.verdict_rows[0].verdict is None
    assert result.landings[0] is None


def test_run_video_rejection_diagnostic_uses_earliest_masked_code(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[14] = 3
    codes[16] = 3
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1, 2, 3})), inpaint_codes=codes,
    )
    rows = []
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half', lambda *args, **kwargs: Half.TOP,
    )

    run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]}, rejection_diagnostics=rows,
    )

    assert rows == [{
        'rule': 'final_contact', 'rally_id': 0, 'start_frame': 14, 'end_frame': 17,
        'trigger_frame': 14, 'trigger_code': 3,
    }]


def test_run_video_does_not_record_an_unaffected_mid_rally_mask(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[14] = 3
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1, 2, 3})), inpaint_codes=codes,
    )
    rows = []
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half', lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing', lambda *args, **kwargs: None,
    )

    run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]}, rejection_diagnostics=rows,
    )

    assert rows == []


def test_run_video_keeps_next_server_verdict_and_masked_contact_measurements(monkeypatch):
    inputs = _synthetic_inputs()
    contacts = {0: [12, 14, 16], 1: [30, 32, 34, 36]}
    for frame in sum(contacts.values(), []):
        inputs['track'][frame] = (0.5, 0.4, 1.0)
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[contacts[0]] = 3
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1, 2, 3})), inpaint_codes=codes,
    )

    def attribute(_frame, *_args, **_kwargs):
        return Half.TOP if _frame in {12, 14, 16, 30, 34} else Half.BOT

    monkeypatch.setattr(run_video_module.point_winner, 'attribute_half', attribute)
    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing', lambda *args, **kwargs: None)

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20), (25, 45)],
        contacts=contacts,
    )

    assert result.verdict_rows[0].verdict is Verdict.WON
    assert result.verdict_rows[0].verdict_source.value == 'next_server'
    assert result.filtered_by_rally[0] == contacts[0]
    assert set(result.hit_height_by_frame) == set(sum(contacts.values(), []))
    assert result.hit_height_failures == []


def test_run_video_code_three_rejects_each_diagnostic_rule(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[[16, 25, 26, 27, 40]] = 3
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1, 2, 3})), inpaint_codes=codes,
    )
    rows = []
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half', lambda *args, **kwargs: Half.TOP,
    )

    def fake_pick(_final_contact, _next_start, _track, _dead, _kin, _opts, _striker, _net_band,
                  _resolution, _court_info, _constants, _fps, *, event_non_evidence_mask,
                  rejected_intervals):
        assert event_non_evidence_mask[40]
        rejected_intervals.append((39, 41))
        return None

    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing', fake_pick)
    track = inputs['track']
    track[15:27, 2] = 1
    track[15:27, 1] = 0.5
    track[27:, 2] = 0

    run_video(
        **inputs, **_default_scene_inputs(len(track)), spans=[(10, 20), (22, 45)],
        contacts={0: [12, 14, 16], 1: [24]}, rejection_diagnostics=rows,
    )

    assert {row['rule'] for row in rows} == {
        'final_contact', 'lost_shuttle_guard', 'landing_descent',
    }
    assert all(row['trigger_code'] == 3 for row in rows)


def test_run_video_geometric_diagnostic_has_nullable_agreement(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict is None
    assert diagnostic.geometric_winner is None
    assert diagnostic.agreement is None


def test_run_video_geometric_diagnostic_records_a_resolved_winner(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing',
        lambda *args, **kwargs: Landing(15, (0.5, 0.75), Half.BOT, False, False, False),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict.value == 'won'
    assert diagnostic.geometric_winner is Half.TOP
    assert diagnostic.agreement is True


def test_run_video_has_no_geometric_diagnostic_without_resolved_striker(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.verdict_rows == {}
    assert result.geometric_verdict_rows == {}


def test_run_video_injected_contacts_build_shared_sticky_once(monkeypatch):
    inputs = _synthetic_inputs()
    real_build_sticky_result = stage8_seg.build_sticky_result
    build_calls = 0

    def count_builds(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return real_build_sticky_result(*args, **kwargs)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('dead-mask builder must be bypassed')

    monkeypatch.setattr(
        stage8_seg, 'build_sticky_result', count_builds,
    )
    monkeypatch.setattr(
        run_video_module, 'build_dead_mask', fail_if_called,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.spans == [(10, 20)]
    assert result.contacts == [ContactCandidate(0, 14, None, None, None)]
    assert build_calls == 1


def _synthetic_inputs():
    video_id = 1
    n_frames = 300
    resolution = (1920.0, 1080.0)
    court_info = {
        'H': np.eye(3), 'border_L': 0.0, 'border_R': 1280.0,
        'border_U': 0.0, 'border_D': 720.0,
    }
    return {
        'track': np.zeros((n_frames, 3), dtype=np.float64),
        'bboxes': np.zeros((n_frames, 1, 4), dtype=np.float32),
        'scores': np.zeros((n_frames, 1), dtype=np.float32),
        'kps': np.zeros((n_frames, 1, 17, 2), dtype=np.float32),
        'ndet': np.zeros(n_frames, dtype=np.int64),
        'fps': 25.0,
        'landing_options': LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        'court_box': CourtBox(
            x_range=(635.0, 1316.0), y_range=(254.0, 1030.0),
            height_band=(84.0, 336.0), mid_band=(642.0, 642.0),
        ),
        'net_band': (664.6, 703.7), 'resolution': resolution,
        'video_id': video_id, 'court_info': court_info,
        'homo_df': pd.DataFrame({
            'upleft_x': [0.0], 'upright_x': [1280.0],
            'downleft_x': [0.0], 'downright_x': [1280.0],
            'upleft_y': [0.0], 'upright_y': [0.0],
            'downleft_y': [720.0], 'downright_y': [720.0],
        }, index=[video_id]),
        'gate_court_info': {str(video_id): court_info},
        'gate_resolution_table': pd.DataFrame(
            {'width': [1920.0], 'height': [1080.0]}, index=[str(video_id)],
        ),
        'dead_mask': np.zeros(n_frames, dtype=bool),
    }


def _default_scene_inputs(n_frames: int):
    return {
        'court_present': np.ones(n_frames, dtype=bool),
        'homography_rows': [{
            'start_frame': '0', 'end_frame': str(n_frames),
            'upleft_x': 0.0, 'upright_x': 1280.0,
            'downleft_x': 0.0, 'downright_x': 1280.0,
            'upleft_y': 0.0, 'upright_y': 0.0,
            'downleft_y': 720.0, 'downright_y': 720.0,
        }],
    }


def test_write_geometric_verdicts_csv_serialises_nulls_blank(tmp_path) -> None:
    rows = [
        GeometricVerdictRow(0, Verdict.WON, Half.TOP, True),
        GeometricVerdictRow(2, None, None, None),
    ]
    path = tmp_path / 'pilot_geometric_verdicts.csv'
    write_geometric_verdicts_csv(rows, path)
    assert path.read_text(encoding='utf-8').splitlines() == [
        'rally_id,geometric_verdict,geometric_winner,agreement',
        '0,won,Top,True',
        '2,,,',
    ]


@pytest.mark.parametrize('contacts_mode', ['injected', 'natural'])
@pytest.mark.parametrize('mask_source', ['inpaint_codes', 'event_non_evidence_mask'])
def test_run_video_threads_event_mask_to_dead_mask_builder(
    monkeypatch, contacts_mode, mask_source,
) -> None:
    inputs = _synthetic_inputs()
    del inputs['dead_mask']
    n_frames = len(inputs['track'])
    if mask_source == 'inpaint_codes':
        codes = np.zeros(n_frames, dtype=np.uint8)
        codes[[40, 80]] = 3
        inputs['inpaint_codes'] = codes
        expected_mask = codes == 3
    else:
        expected_mask = np.zeros(n_frames, dtype=bool)
        expected_mask[[40, 80]] = True
        inputs['event_non_evidence_mask'] = expected_mask

    received = []

    def fake_dead_mask(*_args, **kwargs):
        received.append(kwargs['non_evidence'].copy())
        return np.zeros(n_frames, dtype=bool)

    monkeypatch.setattr(run_video_module, 'build_dead_mask', fake_dead_mask)
    kwargs = {'spans': [(10, 20)], 'contacts': {0: [14]}} if contacts_mode == 'injected' else {}

    run_video(**inputs, **_default_scene_inputs(n_frames), **kwargs)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected_mask)
