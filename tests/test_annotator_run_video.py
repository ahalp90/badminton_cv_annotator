"""Smoke coverage for the public annotator video composition."""
import numpy as np
import pandas as pd
import pytest

import annotator.run_video as run_video_module
import annotator.rally_segmentation as rally_segmentation
from annotator.calibration.gt_scoring import write_geometric_verdicts_csv
from annotator.config import BaseAnnotatorConfig
from annotator.point_winner import GeometricVerdictRow, Half, Landing, LandingFilterOptions, Verdict
from annotator.fps_constants import scale_for_fps
from annotator.rally_segmentation import ServeStartClose, ServeStartMode, StickyResult
from annotator.run_video import AnnotatorResult, RunCapture, build_serve_options, run_video, scoring_filter
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
        net_band=(664.6, 703.7),
        resolution=resolution,
        video_id=video_id,
        court_info=court_info,
        homo_df=homo_df,
        gate_court_info={str(video_id): court_info},
        gate_resolution_table=gate_resolution_table,
        raw_exclusion_mask=dead,
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
        rally_segmentation, 'find_rally_spans',
        lambda *args, **kwargs: pytest.fail('natural span finding was not bypassed'),
    )

    result = run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=injected)

    assert result.spans == injected


def test_run_video_court_optional_stop_early_preserves_positions_and_raw_contacts(monkeypatch):
    track = np.zeros((20, 3), dtype=np.float64)
    positions = np.zeros((20, 2, 2), dtype=np.float64)
    expected_contacts = [ContactCandidate(0, 7, True, None, None)]
    received = {}

    def fake_segment_video(track_arg, **kwargs):
        received.update(kwargs)
        return [(3, 12)], expected_contacts

    monkeypatch.setattr(rally_segmentation, 'segment_video', fake_segment_video)
    monkeypatch.setattr(
        rally_segmentation, 'tracker_segments',
        lambda *args, **kwargs: pytest.fail('court-optional mode must not build tracker segments'),
    )
    monkeypatch.setattr(
        rally_segmentation, 'build_sticky_result',
        lambda *args, **kwargs: pytest.fail('court-optional mode must not build sticky'),
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: pytest.fail('stop-early mode must not enter landing'),
    )

    result = run_video(
        track, fps=25.0, positions=positions,
        raw_exclusion_mask=np.zeros(len(track), dtype=bool),
        court_optional=True, stop_after_segmentation=True,
    )

    assert received['positions'] is positions
    assert received['sticky_distances'] is None
    assert result.spans == [(3, 12)]
    assert result.contacts == expected_contacts
    assert result.filtered_contacts == []
    assert result.filtered_by_rally == {}
    assert result.striker_halves == []
    assert result.verdict_rows == {}
    assert result.landings == {}
    assert result.hit_height_by_frame == {}


def test_run_video_court_optional_ignores_hard_court_union_flag() -> None:
    n_frames = 20
    capture = RunCapture()
    result = run_video(
        np.zeros((n_frames, 3)),
        fps=25.0,
        positions=np.zeros((n_frames, 2, 2)),
        raw_exclusion_mask=np.zeros(n_frames, dtype=bool),
        court_optional=True,
        stop_after_segmentation=True,
        court_invalid_is_excluded=True,
        capture=capture,
    )
    assert result.verdict_rows == {}
    assert capture.definitive_exclusion_mask is not None
    assert not capture.definitive_exclusion_mask.any()


def test_run_video_court_optional_rejects_contradictory_court_evidence():
    with pytest.raises(ValueError, match='court_optional rejects supplied inputs: homography_rows'):
        run_video(
            np.zeros((10, 3)), fps=25.0, homography_rows=[],
            court_optional=True, stop_after_segmentation=True,
        )


def test_run_video_court_optional_requires_stop_early():
    with pytest.raises(ValueError, match='court_optional requires stop_after_segmentation'):
        run_video(np.zeros((10, 3)), fps=25.0, court_optional=True)


@pytest.mark.parametrize('field', ['bboxes', 'scores', 'kps', 'ndet', 'resolution', 'video_id',
                                   'gate_court_info', 'gate_resolution_table'])
def test_run_video_normal_mode_requires_sticky_inputs(field):
    inputs = _synthetic_inputs()
    del inputs['raw_exclusion_mask']
    inputs[field] = None
    with pytest.raises(ValueError, match=rf'normal mode requires .*\b{field}\b'):
        run_video(**inputs, **_default_scene_inputs(len(inputs['track'])), stop_after_segmentation=True)


@pytest.mark.parametrize('field', ['landing_options', 'net_band', 'court_info', 'homo_df'])
def test_run_video_full_chain_requires_downstream_inputs(field):
    inputs = _synthetic_inputs()
    del inputs['raw_exclusion_mask']
    inputs[field] = None
    with pytest.raises(ValueError, match=rf'full-chain mode requires .*\b{field}\b'):
        run_video(**inputs, **_default_scene_inputs(len(inputs['track'])))


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
    expected = rally_segmentation.tracker_segments(homography_rows, court_present, n_frames)
    # Dropout splits the first scene row: the fixture must stay non-trivial.
    assert expected == [(0, 40), (60, 150), (150, 300)]

    real_builder = rally_segmentation.build_sticky_result
    received = []

    def spy(track, segments, *args, **kwargs):
        received.append(segments)
        return real_builder(track, segments, *args, **kwargs)

    monkeypatch.setattr(rally_segmentation, 'build_sticky_result', spy)

    run_video(**inputs, court_present=court_present, homography_rows=homography_rows)

    assert received == [expected]


def test_run_video_builds_serve_sticky_from_original_track_before_replay_mask(monkeypatch):
    inputs = _synthetic_inputs()
    original_track = inputs['track'].copy()
    del inputs['raw_exclusion_mask']
    real_build_sticky = rally_segmentation.build_sticky_result
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

    real_segment_video = rally_segmentation.segment_video

    def spy_segment_video(track, *args, **kwargs):
        segment_tracks.append((track.copy(), kwargs['exclusion_mask'].copy()))
        return real_segment_video(track, *args, **kwargs)

    monkeypatch.setattr(rally_segmentation, 'build_sticky_result', spy_build_sticky)
    monkeypatch.setattr(run_video_module, 'build_serve_options', spy_build_options)
    monkeypatch.setattr(run_video_module, 'build_dead_mask', fake_dead_mask)
    monkeypatch.setattr(rally_segmentation, 'segment_video', spy_segment_video)

    run_video(
        **inputs, **_default_scene_inputs(len(original_track)),
        serve_start=ServeStartConfig(threshold_bh=0.8, mode=ServeStartMode.TRIM),
    )

    assert len(sticky_tracks) == 1
    np.testing.assert_array_equal(sticky_tracks[0], original_track)
    assert len(option_stickies) == 1
    assert len(segment_tracks) == 1
    np.testing.assert_array_equal(segment_tracks[0][0], original_track)
    assert not segment_tracks[0][1][0]  # the one-frame raw flag is cleared by duration filtering


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
    del inputs['raw_exclusion_mask']
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
        run_video_module.point_winner, 'pick_landing_to_end',
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
        run_video_module.point_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: pytest.fail('landing must not run without an unmasked contact'),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]},
    )

    assert result.verdict_rows[0].verdict is None
    assert result.landings[0] is None


def test_run_video_drops_trusted_dead_contacts_and_records_the_rejection(monkeypatch):
    inputs = _synthetic_inputs()
    raw_exclusion_mask = np.zeros(len(inputs['track']), dtype=bool)
    raw_exclusion_mask[:23] = True  # contacts 12, 14, and 16 are all past the filter threshold at 25 fps
    inputs['raw_exclusion_mask'] = raw_exclusion_mask
    rows = []
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: pytest.fail('trusted-dead contacts must not be attributed'),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]}, rejection_diagnostics=rows,
    )

    assert result.filtered_contacts == []
    assert result.striker_halves == [None]
    assert result.geometric_verdict_rows == {}
    assert rows == [{
        'rule': 'all_contacts_on_believed_mask', 'rally_id': 0,
        'start_frame': 10, 'end_frame': 20, 'trigger_frame': 12, 'trigger_code': '',
    }]


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
        run_video_module.point_winner, 'pick_landing_to_end', lambda *args, **kwargs: None,
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
    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing_to_end', lambda *args, **kwargs: None)

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

    def fake_pick(_final_contact, _end_frame, _track, _kin, _opts, _striker, _net_band,
                  _resolution, _court_info, _constants, _fps, *, shuttle_hallucination_mask,
                  rejected_intervals):
        assert shuttle_hallucination_mask[40]
        rejected_intervals.append((39, 41))
        return None

    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing_to_end', fake_pick)
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
        run_video_module.point_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict is None
    assert diagnostic.geometric_winner is None
    assert diagnostic.agreement is None
    assert diagnostic.window_closed_by_mask is False


def test_run_video_geometric_diagnostic_records_a_resolved_winner(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: Landing(15, (0.5, 0.75), Half.BOT, False, False),
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    diagnostic = result.geometric_verdict_rows[0]
    assert diagnostic.geometric_verdict.value == 'won'
    assert diagnostic.geometric_winner is Half.TOP
    assert diagnostic.agreement is True
    assert diagnostic.window_closed_by_mask is False


def test_run_video_geometric_diagnostic_marks_a_trusted_mask_window_close(monkeypatch):
    inputs = _synthetic_inputs()
    raw_exclusion_mask = np.zeros(len(inputs['track']), dtype=bool)
    raw_exclusion_mask[15:28] = True  # filtering starts at frame 27 at 25 fps
    inputs['raw_exclusion_mask'] = raw_exclusion_mask
    inputs['track'][14:, 2] = 1.0
    monkeypatch.setattr(
        run_video_module.point_winner, 'attribute_half',
        lambda *args, **kwargs: Half.TOP,
    )
    monkeypatch.setattr(
        run_video_module.point_winner, 'pick_landing_to_end', lambda *args, **kwargs: None,
    )

    result = run_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]},
    )

    assert result.geometric_verdict_rows[0].window_closed_by_mask is True


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
    real_build_sticky_result = rally_segmentation.build_sticky_result
    build_calls = 0

    def count_builds(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return real_build_sticky_result(*args, **kwargs)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('dead-mask builder must be bypassed')

    monkeypatch.setattr(
        rally_segmentation, 'build_sticky_result', count_builds,
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


def test_run_video_capture_resets_and_copies_masks() -> None:
    inputs = _synthetic_inputs()
    raw_mask = inputs['raw_exclusion_mask']
    capture = RunCapture(
        raw_exclusion_mask=np.ones(len(raw_mask), dtype=bool),
        definitive_exclusion_mask=np.ones(len(raw_mask), dtype=bool),
    )

    run_video(
        **inputs,
        **_default_scene_inputs(len(inputs['track'])),
        capture=capture,
    )

    np.testing.assert_array_equal(capture.raw_exclusion_mask, raw_mask)
    np.testing.assert_array_equal(capture.definitive_exclusion_mask, raw_mask)
    assert capture.raw_exclusion_mask is not raw_mask
    assert capture.definitive_exclusion_mask is not raw_mask
    capture.raw_exclusion_mask[0] = True
    capture.definitive_exclusion_mask[1] = True
    assert not raw_mask[0]
    assert not raw_mask[1]


def test_run_video_rejects_invalid_horizon_configuration():
    with pytest.raises(ValueError, match='requires capture'):
        run_video(np.zeros((10, 3)), fps=25.0, landing_horizons_s=(1.0,))

    with pytest.raises(ValueError, match='strictly increasing'):
        run_video(
            np.zeros((10, 3)), fps=25.0, capture=RunCapture(),
            landing_horizons_s=(1.0, 1.0),
        )


def test_run_video_captures_three_horizons_without_extending_safe_end(monkeypatch):
    inputs = _synthetic_inputs()
    capture = RunCapture()
    monkeypatch.setattr(run_video_module.point_winner, 'attribute_half', lambda *args: Half.TOP)
    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing_to_end', lambda *args, **kwargs: None)

    run_video(
        **inputs,
        **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, capture=capture,
        landing_horizons_s=(1.0, 2.0, 3.0),
    )

    assert [row.horizon_seconds for row in capture.landing_horizon_rows] == [1.0, 2.0, 3.0]
    assert all(row.effective_end_frame <= row.safe_end_frame for row in capture.landing_horizon_rows)
    assert all(row.strict_landing is None and row.capped_landing is None for row in capture.landing_horizon_rows)


def test_run_video_captures_horizon_landing_and_winner_changes(monkeypatch):
    inputs = _synthetic_inputs()
    inputs['track'][14:, 2] = 1.0
    capture = RunCapture()
    monkeypatch.setattr(run_video_module.point_winner, 'attribute_half', lambda *args: Half.TOP)

    def fake_pick(_final_contact, end_frame, *_args, **_kwargs):
        if end_frame < len(inputs['track']):
            return Landing(end_frame - 1, (0.5, 0.25), Half.TOP, False, False)
        return Landing(end_frame - 1, (0.5, 0.75), Half.BOT, False, False)

    monkeypatch.setattr(run_video_module.point_winner, 'pick_landing_to_end', fake_pick)

    run_video(
        **inputs,
        **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, capture=capture,
        landing_horizons_s=(1.0, 11.44),
    )

    short, tied = capture.landing_horizon_rows
    assert short.landing_changed is True
    assert short.winner_changed is True
    assert short.strict_verdict.verdict is Verdict.WON
    assert short.capped_verdict.verdict is Verdict.LOST
    assert short.strict_verdict.verdict_source.value == 'landing_geometry'
    assert short.capped_verdict.verdict_source.value == 'landing_geometry'
    assert tied.effective_end_frame == tied.safe_end_frame == len(inputs['track'])
    assert tied.closure_reasons == ('horizon_cap', 'video_end')


def test_run_video_default_empty_horizon_capture_stays_empty():
    capture = RunCapture(landing_horizon_rows=[object()])

    run_video(
        **_synthetic_inputs(),
        **_default_scene_inputs(300),
        capture=capture,
    )

    assert capture.landing_horizon_rows == []


def test_run_video_court_invalid_union_is_full_chain_only() -> None:
    inputs = _synthetic_inputs()
    court_present = np.ones(len(inputs['track']), dtype=bool)
    court_present[10] = False
    capture = RunCapture()
    scene_inputs = _default_scene_inputs(len(inputs['track']))
    scene_inputs['court_present'] = court_present

    run_video(
        **inputs,
        **scene_inputs,
        stop_after_segmentation=True,
        capture=capture,
        court_invalid_is_excluded=True,
    )
    assert not capture.definitive_exclusion_mask[10]

    run_video(
        **inputs,
        **scene_inputs,
        capture=capture,
        court_invalid_is_excluded=True,
    )
    assert capture.definitive_exclusion_mask[10]
    assert capture.definitive_exclusion_mask[~court_present].all()


def test_run_video_fails_after_hard_court_union_becomes_all_true() -> None:
    inputs = _synthetic_inputs()
    capture = RunCapture()
    scene_inputs = _default_scene_inputs(len(inputs['track']))
    scene_inputs['court_present'] = np.zeros(len(inputs['track']), dtype=bool)
    with pytest.raises(ValueError, match='mask is all True'):
        run_video(
            **inputs,
            **scene_inputs,
            capture=capture,
            court_invalid_is_excluded=True,
        )
    assert capture.raw_exclusion_mask is not None
    assert capture.definitive_exclusion_mask is not None
    assert capture.definitive_exclusion_mask.all()


def test_run_video_uses_supplied_landing_error_band_without_static_homography(monkeypatch) -> None:
    inputs = _synthetic_inputs()
    inputs['homo_df'] = None
    monkeypatch.setattr(
        run_video_module.point_winner,
        'corner_error_band_m',
        lambda *args, **kwargs: pytest.fail('static homography should not be read'),
    )
    result = run_video(
        **inputs,
        **_default_scene_inputs(len(inputs['track'])),
        landing_error_band_m=0.12,
    )
    assert result.verdict_rows == {}


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
        'raw_exclusion_mask': np.zeros(n_frames, dtype=bool),
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
        GeometricVerdictRow(0, Verdict.WON, Half.TOP, True, False),
        GeometricVerdictRow(2, None, None, None, False),
    ]
    path = tmp_path / 'pilot_geometric_verdicts.csv'
    write_geometric_verdicts_csv(rows, path)
    assert path.read_text(encoding='utf-8').splitlines() == [
        'rally_id,geometric_verdict,geometric_winner,agreement,window_closed_by_mask',
        '0,won,Top,True,False',
        '2,,,,False',
    ]


@pytest.mark.parametrize('contacts_mode', ['injected', 'natural'])
@pytest.mark.parametrize('mask_source', ['inpaint_codes', 'shuttle_hallucination_mask'])
def test_run_video_threads_event_mask_to_dead_mask_builder(
    monkeypatch, contacts_mode, mask_source,
) -> None:
    inputs = _synthetic_inputs()
    del inputs['raw_exclusion_mask']
    n_frames = len(inputs['track'])
    if mask_source == 'inpaint_codes':
        codes = np.zeros(n_frames, dtype=np.uint8)
        codes[[40, 80]] = 3
        inputs['inpaint_codes'] = codes
        expected_mask = codes == 3
    else:
        expected_mask = np.zeros(n_frames, dtype=bool)
        expected_mask[[40, 80]] = True
        inputs['shuttle_hallucination_mask'] = expected_mask

    received = []

    def fake_dead_mask(*_args, **kwargs):
        received.append(kwargs['shuttle_hallucination_mask'].copy())
        return np.zeros(n_frames, dtype=bool)

    monkeypatch.setattr(run_video_module, 'build_dead_mask', fake_dead_mask)
    kwargs = {'spans': [(10, 20)], 'contacts': {0: [14]}} if contacts_mode == 'injected' else {}

    run_video(**inputs, **_default_scene_inputs(n_frames), **kwargs)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected_mask)
