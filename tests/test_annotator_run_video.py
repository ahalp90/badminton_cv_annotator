"""Smoke coverage for the public annotator video composition."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import annotator.annotate_video as run_video_module
from annotator import player_tracking
from annotator.annotate_video import (
    AnnotatorResult,
    annotate_video,
)
from annotator.config import BaseAnnotatorConfig
from annotator.courts.frame_geometry import build_scene_courts
from annotator.inference import heuristic as heuristic_inference
from annotator.inference import prepare_annotation_inputs as player_evidence
from annotator.inference import prepare_annotation_inputs as preparation
from annotator.inference.heuristic import accepted_heuristic_contacts
from annotator.rally_results.landing_and_winner import (
    Landing,
    LandingFilterOptions,
    convert_landing_options,
)
from annotator.types import (
    ContactCandidate,
    ContactEvent,
    ContactSequence,
    CourtHalf,
    FinishedRally,
)
from experiments.annotator.landing_search import (
    compare_landing_searches,
    validate_search_limits,
)
from experiments.annotator.legacy_evaluation.gt_scoring import (
    write_geometric_verdicts_csv,
)


def test_model_contacts_and_fitted_sides_survive_the_full_chain(monkeypatch, annotator_models):
    inputs = _synthetic_inputs()
    inputs['raw_exclusion_mask'][1:30] = True
    final_events = (ContactEvent(3, .7, None), ContactEvent(9, .9, CourtHalf.TOP), ContactEvent(17, .8, CourtHalf.BOT))
    sequence = ContactSequence(0, 8, 23, final_events[1:])
    prediction = SimpleNamespace(
        rally_prediction=SimpleNamespace(rallies=(FinishedRally(sequence, CourtHalf.BOT),), events=final_events),
        rally_review=SimpleNamespace(scores=np.array([.73])),
    )

    def predict(evidence, models):
        assert evidence.heuristic_spans == [(10, 20)]
        assert evidence.raw_contact_frames == [14]
        assert evidence.exclusion_mask[9]
        assert models is annotator_models
        return prediction

    def unexpected_reattribution(*args, **kwargs):
        raise AssertionError('model-selected sides must not be re-attributed')

    monkeypatch.setattr(run_video_module, 'predict_rallies_and_contacts', predict)
    monkeypatch.setattr(run_video_module, 'select_heuristic_contacts', unexpected_reattribution)
    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, models=annotator_models, keep_working_data=True,
    )
    result = annotation.result
    assert result.contact_events == final_events
    assert result.rallies[0].final_half is CourtHalf.BOT
    assert result.rally_confidence == (.73,)
    assert annotation.working_data is not None
    assert annotation.working_data.learned_model_prediction is prediction
    assert result.rallies is prediction.rally_prediction.rallies
    assert result.heuristic_candidates == []


def test_run_video_no_play_returns_empty_result():
    resolution = (1920.0, 1080.0)
    track = np.zeros((300, 3), dtype=np.float64)
    bboxes = np.zeros((300, 1, 4), dtype=np.float32)
    scores = np.zeros((300, 1), dtype=np.float32)
    kps = np.zeros((300, 1, 17, 2), dtype=np.float32)
    ndet = np.zeros(300, dtype=np.int64)
    dead = np.zeros(300, dtype=bool)

    annotation = annotate_video(
        track, bboxes, scores, kps, ndet,
        fps=25.0,
        landing_options=LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        resolution=resolution,
        raw_exclusion_mask=dead,
        **_default_scene_inputs(len(track)), heuristic_only=True,
    )
    result = annotation.result

    assert result == AnnotatorResult((), {}, {}, {}, (), [], ())


def test_accepted_heuristic_contacts_keeps_only_unfailed_unsuppressed_rows():
    rows = [
        ContactCandidate(0, 1, True, False),
        ContactCandidate(0, 2, True, True),
        ContactCandidate(0, 3, False, False),
        ContactCandidate(0, 4, None, None),
    ]

    assert accepted_heuristic_contacts(rows) == [rows[0], rows[3]]


def test_run_video_injected_spans_bypass_natural_span_finding(monkeypatch):
    inputs = _synthetic_inputs()
    injected = [(10, 20)]
    monkeypatch.setattr(
        preparation, 'find_rally_spans',
        lambda *args, **kwargs: pytest.fail('natural span finding was not bypassed'),
    )

    annotation = annotate_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=injected, heuristic_only=True)
    result = annotation.result

    assert [(rally.sequence.start_frame, rally.sequence.end_frame) for rally in result.rallies] == injected


@pytest.mark.parametrize('field', [
    'bboxes', 'scores', 'kps', 'ndet', 'resolution', 'landing_options',
    'court_present', 'homography_rows',
])
def test_run_video_requires_player_and_scene_inputs(field):
    inputs = {**_synthetic_inputs(), **_default_scene_inputs(300)}
    del inputs[field]
    with pytest.raises(TypeError, match=field):
        annotate_video(**inputs, heuristic_only=True)


def test_run_video_hands_tracker_segments_output_to_sticky_builder(monkeypatch):
    """The sticky builder must receive exactly the list tracker_segments produced.

    The handoff is internal to annotate_video, so a wrong list (rally spans, say) would
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
    scene_courts = build_scene_courts(homography_rows, inputs["resolution"])
    expected = player_tracking.tracker_segments(scene_courts, court_present, n_frames)
    # Dropout splits the first scene row: the fixture must stay non-trivial.
    assert expected == [(0, 40), (60, 150), (150, 300)]

    real_builder = player_evidence.build_player_evidence
    received = []

    def spy(track, segments, *args, **kwargs):
        received.append(segments)
        return real_builder(track, segments, *args, **kwargs)

    monkeypatch.setattr(player_evidence, 'build_player_evidence', spy)

    annotate_video(**inputs, court_present=court_present, homography_rows=homography_rows, heuristic_only=True)

    assert received == [expected]


def test_run_video_injected_contacts_are_unmeasured_and_scored():
    inputs = _synthetic_inputs()
    spans = [(10, 20)]
    frames = [14, 16]
    expected = [ContactCandidate(0, frame, None, None) for frame in frames]

    annotation = annotate_video(**inputs, **_default_scene_inputs(len(inputs['track'])), spans=spans, contacts={0: frames}, heuristic_only=True)
    result = annotation.result

    assert result.heuristic_candidates == expected
    assert [event.frame for rally in result.rallies for event in rally.sequence.events] == frames
    assert [event.frame for event in result.rallies[0].sequence.events] == frames


def test_heuristic_only_keeps_unknown_probabilities_without_using_models(monkeypatch):
    from dataset_builder._runtime_support import _annotation_result
    from dataset_builder.vision import annotation_result_payload

    def unexpected(*args, **kwargs):
        pytest.fail('heuristic-only annotation must not load or use fitted models')

    monkeypatch.setattr(run_video_module, 'load_models', unexpected)
    monkeypatch.setattr(run_video_module, 'predict_rallies_and_contacts', unexpected)
    inputs = _synthetic_inputs()
    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result
    assert result.heuristic_candidates == [ContactCandidate(0, 14, None, None)]
    assert np.isnan(result.rallies[0].sequence.events[0].probability)
    payload = annotation_result_payload('heuristic', result)['result']
    assert payload['rallies'][0]['sequence']['events'][0]['probability'] is None
    restored = _annotation_result(payload)
    assert np.isnan(restored.rallies[0].sequence.events[0].probability)
    assert restored.heuristic_candidates == result.heuristic_candidates

    invalid = result._replace(contact_events=(ContactEvent(2, np.inf, None),))
    with pytest.raises(ValueError, match='must be finite'):
        annotation_result_payload('heuristic', invalid)


def test_run_video_injected_contacts_without_mask_completes(monkeypatch):
    inputs = _synthetic_inputs()
    del inputs['raw_exclusion_mask']
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: (CourtHalf.TOP, None),
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    assert [rally.final_half for rally in result.rallies] == [CourtHalf.TOP]
    assert 0 in result.outcomes


def test_run_video_does_not_use_earlier_contact_when_final_is_rejected(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[16] = 1
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1})), shuttle_grades=codes,
    )
    called_frames = []
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence', lambda *args, **kwargs: (CourtHalf.TOP, None),
    )
    monkeypatch.setattr(
        run_video_module.landing_and_winner, 'pick_landing_to_end',
        lambda final_contact, *args, **kwargs: called_frames.append(final_contact) or None,
    )

    annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]}, heuristic_only=True,
    )

    assert called_frames == []


def test_run_video_exhausts_masked_contacts_without_calling_landing(monkeypatch):
    inputs = _synthetic_inputs()
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[12:17] = 1
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1})), shuttle_grades=codes,
    )
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence', lambda *args, **kwargs: (CourtHalf.TOP, None),
    )
    monkeypatch.setattr(
        run_video_module.landing_and_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: pytest.fail('landing must not run without an unmasked contact'),
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]}, heuristic_only=True,
    )
    result = annotation.result

    assert result.outcomes[0].winner is None
    assert result.outcomes[0].landing is None


def test_run_video_drops_trusted_dead_contacts(monkeypatch):
    inputs = _synthetic_inputs()
    raw_exclusion_mask = np.zeros(len(inputs['track']), dtype=bool)
    raw_exclusion_mask[:23] = True  # contacts 12, 14, and 16 are all past the filter threshold at 25 fps
    inputs['raw_exclusion_mask'] = raw_exclusion_mask
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: pytest.fail('trusted-dead contacts must not be attributed'),
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)],
        contacts={0: [12, 14, 16]},  heuristic_only=True,
    )
    result = annotation.result

    assert all(not rally.sequence.events for rally in result.rallies)
    assert result.rallies[0].final_half is None
    assert result.outcomes == {}



def test_run_video_keeps_next_server_verdict_and_masked_contact_measurements(monkeypatch):
    inputs = _synthetic_inputs()
    contacts = {0: [12, 14, 16], 1: [30, 32, 34, 36]}
    for frame in [frame for frames in contacts.values() for frame in frames]:
        inputs['track'][frame] = (0.5, 0.4, 1.0)
    codes = np.zeros(len(inputs['track']), dtype=np.uint8)
    codes[contacts[0]] = 3
    inputs.update(
        base=BaseAnnotatorConfig(rejected_grades=frozenset({1, 2, 3})), shuttle_grades=codes,
    )

    def attribute(_frame, *_args, **_kwargs):
        side = CourtHalf.TOP if _frame in {12, 14, 16, 30, 34} else CourtHalf.BOT
        return side, None

    monkeypatch.setattr(heuristic_inference, 'contact_player_evidence', attribute)
    monkeypatch.setattr(run_video_module.landing_and_winner, 'pick_landing_to_end', lambda *args, **kwargs: None)

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20), (25, 45)],
        contacts=contacts, heuristic_only=True,
    )
    result = annotation.result

    assert result.outcomes[0].winner is CourtHalf.TOP
    assert result.outcomes[0].winner_source.value == 'next_server'
    assert [event.frame for event in result.rallies[0].sequence.events] == contacts[0]
    assert set(result.hit_heights) == {frame for frames in contacts.values() for frame in frames}
    assert result.unavailable_hit_height_reasons == {}


def test_run_video_geometric_diagnostic_has_nullable_agreement(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: (CourtHalf.TOP, None),
    )
    monkeypatch.setattr(
        run_video_module.landing_and_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: None,
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    diagnostic = result.outcomes[0]
    assert diagnostic.conservative_landing_winner is None
    assert diagnostic.winner is None
    assert diagnostic.search_shortened_by_exclusion is False


def test_run_video_geometric_diagnostic_records_a_resolved_winner(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: (CourtHalf.TOP, None),
    )
    monkeypatch.setattr(
        run_video_module.landing_and_winner, 'pick_landing_to_end',
        lambda *args, **kwargs: Landing(15, (0.5, 0.75), CourtHalf.BOT, False, False),
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    diagnostic = result.outcomes[0]
    assert diagnostic.conservative_landing_winner is CourtHalf.TOP
    assert diagnostic.winner is CourtHalf.TOP
    assert diagnostic.search_shortened_by_exclusion is False


def test_run_video_geometric_diagnostic_marks_a_trusted_mask_window_close(monkeypatch):
    inputs = _synthetic_inputs()
    raw_exclusion_mask = np.zeros(len(inputs['track']), dtype=bool)
    raw_exclusion_mask[15:28] = True  # filtering starts at frame 27 at 25 fps
    inputs['raw_exclusion_mask'] = raw_exclusion_mask
    inputs['track'][14:, 2] = 1.0
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: (CourtHalf.TOP, None),
    )
    monkeypatch.setattr(
        run_video_module.landing_and_winner, 'pick_landing_to_end', lambda *args, **kwargs: None,
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    assert result.outcomes[0].search_shortened_by_exclusion is True


def test_run_video_tied_mask_and_shuttle_loss_is_not_mask_closed(monkeypatch):
    inputs = _synthetic_inputs()
    inputs['raw_exclusion_mask'][15:28] = True  # 13-frame run survives filtering at 25 fps
    inputs['track'][14, 1] = 0.5
    inputs['track'][14, 2] = 1.0
    monkeypatch.setattr(heuristic_inference, 'contact_player_evidence', lambda *args, **kwargs: (CourtHalf.TOP, None))
    monkeypatch.setattr(run_video_module.landing_and_winner, 'pick_landing_to_end', lambda *args, **kwargs: None)

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    assert result.outcomes[0].search_shortened_by_exclusion is False


def test_run_video_has_no_geometric_diagnostic_without_resolved_striker(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(
        heuristic_inference, 'contact_player_evidence',
        lambda *args, **kwargs: (None, None),
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    assert result.outcomes == {}


def test_run_video_injected_contacts_build_shared_sticky_once(monkeypatch):
    inputs = _synthetic_inputs()
    real_build_player_evidence = player_evidence.build_player_evidence
    build_calls = 0

    def count_builds(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return real_build_player_evidence(*args, **kwargs)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('dead-mask builder must be bypassed')

    monkeypatch.setattr(
        player_evidence, 'build_player_evidence', count_builds,
    )
    monkeypatch.setattr(
        preparation, 'combine_mask', fail_if_called,
    )

    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])), spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    result = annotation.result

    assert [(rally.sequence.start_frame, rally.sequence.end_frame) for rally in result.rallies] == [(10, 20)]
    assert result.heuristic_candidates == [ContactCandidate(0, 14, None, None)]
    assert build_calls == 1


def test_run_video_retains_prepared_evidence_when_requested() -> None:
    inputs = _synthetic_inputs()
    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        keep_working_data=True, heuristic_only=True,
    )
    assert annotation.working_data is not None
    np.testing.assert_array_equal(annotation.working_data.prepared_inputs.raw_exclusion_mask, inputs['raw_exclusion_mask'])
    np.testing.assert_array_equal(annotation.working_data.prepared_inputs.exclusion_mask, inputs['raw_exclusion_mask'])


@pytest.mark.parametrize('limits', [(0.0,), (float('nan'),), (1.0, 1.0)])
def test_landing_search_rejects_invalid_time_limits(limits):
    with pytest.raises(ValueError, match='landing search seconds'):
        validate_search_limits(limits)



def test_run_video_scales_landing_options_before_search(monkeypatch):
    inputs = _synthetic_inputs()
    seen_options = []
    monkeypatch.setattr(heuristic_inference, 'contact_player_evidence', lambda *args: (CourtHalf.TOP, None))

    def record_options(_start, _end, _track, _kinematics, options, *_args, **_kwargs):
        seen_options.append(options)

    monkeypatch.setattr(run_video_module.landing_and_winner, 'pick_landing_to_end', record_options)
    annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, heuristic_only=True,
    )
    assert inputs['fps'] == 25.0
    assert seen_options == [convert_landing_options(inputs['landing_options'], 25.0)]
    assert seen_options[0] != inputs['landing_options']


def test_landing_trials_do_not_extend_the_normal_search(monkeypatch):
    inputs = _synthetic_inputs()
    monkeypatch.setattr(heuristic_inference, 'contact_player_evidence', lambda *args: (CourtHalf.TOP, None))
    monkeypatch.setattr(run_video_module.landing_and_winner, 'pick_landing_to_end', lambda *args, **kwargs: None)
    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, keep_working_data=True, heuristic_only=True,
    )
    result = annotation.result
    comparisons = compare_landing_searches(
        annotation.working_data.prepared_inputs, result.rallies, result.outcomes, (1.0, 2.0, 3.0), inputs["landing_options"],
    )
    assert [row.search_seconds for row in comparisons] == [1.0, 2.0, 3.0]
    assert all(row.trial_end_frame <= row.normal_end_frame for row in comparisons)
    assert all(row.normal_outcome.landing is None and row.trial_outcome.landing is None for row in comparisons)


def test_landing_trials_record_landing_and_winner_changes(monkeypatch):
    inputs = _synthetic_inputs()
    inputs['track'][14:, 2] = 1.0
    monkeypatch.setattr(heuristic_inference, 'contact_player_evidence', lambda *args: (CourtHalf.TOP, None))

    def fake_pick(_final_contact, end_frame, *_args, **_kwargs):
        position = (0.5, 0.25) if end_frame < len(inputs['track']) else (0.5, 0.75)
        half = CourtHalf.TOP if position[1] < 0.5 else CourtHalf.BOT
        return Landing(end_frame - 1, position, half, False, False)

    monkeypatch.setattr(run_video_module.landing_and_winner, 'pick_landing_to_end', fake_pick)
    annotation = annotate_video(
        **inputs, **_default_scene_inputs(len(inputs['track'])),
        spans=[(10, 20)], contacts={0: [14]}, keep_working_data=True, heuristic_only=True,
    )
    result = annotation.result
    short, tied = compare_landing_searches(
        annotation.working_data.prepared_inputs, result.rallies, result.outcomes, (1.0, 11.44), inputs["landing_options"],
    )
    assert short.normal_outcome.landing != short.trial_outcome.landing
    assert short.normal_outcome.winner is CourtHalf.TOP
    assert short.trial_outcome.winner is CourtHalf.BOT
    assert short.normal_outcome.winner_source.value == 'landing_geometry'
    assert short.trial_outcome.winner_source.value == 'landing_geometry'
    assert tied.trial_end_frame == tied.normal_end_frame == len(inputs['track'])
    assert tied.search_stop_reasons == ('time_limit', 'video_end')


def test_run_video_excludes_frames_without_court_presence() -> None:
    inputs = _synthetic_inputs()
    court_present = np.ones(len(inputs['track']), dtype=bool)
    court_present[10] = False
    scene_inputs = _default_scene_inputs(len(inputs['track']))
    scene_inputs['court_present'] = court_present

    annotation = annotate_video(
        **inputs,
        **scene_inputs,
        keep_working_data=True,
        heuristic_only=True,
    )
    assert annotation.working_data.prepared_inputs.exclusion_mask[10]
    assert annotation.working_data.prepared_inputs.exclusion_mask[~court_present].all()


@pytest.mark.parametrize('gap', [(0, 10), (150, 160), (290, 300)])
def test_run_video_excludes_scene_gaps(
    gap: tuple[int, int],
) -> None:
    inputs = _synthetic_inputs()
    n_frames = len(inputs['track'])
    scene_inputs = _default_scene_inputs(n_frames)
    row = scene_inputs['homography_rows'][0]
    gap_start, gap_end = gap
    scene_inputs['homography_rows'] = [
        dict(row, start_frame=start, end_frame=end)
        for start, end in [(0, gap_start), (gap_end, n_frames)]
        if start < end
    ]
    scene_inputs['court_present'][gap_start:gap_end] = False
    # Frame 120 has geometry but fails the separate court-presence check.
    scene_inputs['court_present'][120] = False
    gap_contact = (gap_start + gap_end) // 2
    contact_frames = sorted([120, gap_contact, 180])

    annotation = annotate_video(
        **inputs, **scene_inputs,
        spans=[(0, n_frames)], contacts={0: contact_frames}, keep_working_data=True,
        heuristic_only=True,
    )
    result = annotation.result

    assert not annotation.working_data.prepared_inputs.raw_exclusion_mask.any()
    assert [contact.contact_frame for contact in result.heuristic_candidates] == contact_frames
    expected_mask = np.zeros(n_frames, dtype=bool)
    expected_mask[gap_start:gap_end] = True
    expected_mask[120] = True
    np.testing.assert_array_equal(annotation.working_data.prepared_inputs.exclusion_mask, expected_mask)
    assert [event.frame for rally in result.rallies for event in rally.sequence.events] == [180]


def test_run_video_fails_after_hard_court_union_becomes_all_true() -> None:
    inputs = _synthetic_inputs()
    scene_inputs = _default_scene_inputs(len(inputs['track']))
    scene_inputs['court_present'] = np.zeros(len(inputs['track']), dtype=bool)
    with pytest.raises(ValueError, match='mask is all True'):
        annotate_video(
            **inputs,
            **scene_inputs,
            keep_working_data=True,
            heuristic_only=True,
        )


def _synthetic_inputs():
    n_frames = 300
    resolution = (1920.0, 1080.0)
    return {
        'track': np.zeros((n_frames, 3), dtype=np.float64),
        'bboxes': np.zeros((n_frames, 1, 4), dtype=np.float32),
        'scores': np.zeros((n_frames, 1), dtype=np.float32),
        'kps': np.zeros((n_frames, 1, 17, 2), dtype=np.float32),
        'ndet': np.zeros(n_frames, dtype=np.int64),
        'fps': 25.0,
        'landing_options': LandingFilterOptions(7, 0.004, 5, 7, 0.75),
        'resolution': resolution,
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
        (0, "won", "Top", True, False),
        (2, "", "", None, False),
    ]
    path = tmp_path / 'pilot_geometric_verdicts.csv'
    write_geometric_verdicts_csv(rows, path)
    assert path.read_text(encoding='utf-8').splitlines() == [
        'rally_id,geometric_verdict,geometric_winner,agreement,search_shortened_by_exclusion',
        '0,won,Top,True,False',
        '2,,,,False',
    ]


@pytest.mark.parametrize('contacts_mode', ['injected', 'natural'])
def test_run_video_threads_event_mask_to_dead_mask_builder(
    monkeypatch, contacts_mode,
) -> None:
    inputs = _synthetic_inputs()
    del inputs['raw_exclusion_mask']
    n_frames = len(inputs['track'])
    codes = np.zeros(n_frames, dtype=np.uint8)
    codes[[40, 80]] = 3
    inputs['shuttle_grades'] = codes
    expected_mask = codes == 3

    received = []

    def fake_dead_mask(*_args, **kwargs):
        received.append(kwargs['rejected_shuttle_frames'].copy())
        return np.zeros(n_frames, dtype=bool)

    monkeypatch.setattr(preparation, 'combine_mask', fake_dead_mask)
    kwargs = {'spans': [(10, 20)], 'contacts': {0: [14]}} if contacts_mode == 'injected' else {}

    annotate_video(**inputs, **_default_scene_inputs(n_frames), **kwargs, heuristic_only=True)

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected_mask)


@pytest.mark.parametrize('rejected_grades', [frozenset({1, 2, 3, 4}), frozenset({3}), frozenset({4})])
def test_contact_evidence_reuses_mask_from_fitted_shuttle_grades(monkeypatch, annotator_models, rejected_grades):
    inputs = _synthetic_inputs()
    frame_count = len(inputs['track'])
    codes = np.zeros(frame_count, dtype=np.uint8)
    codes[10:15] = [0, 1, 2, 3, 4]
    mask_calls = []
    original = preparation.build_rejected_shuttle_mask

    def build_mask(*args):
        result = original(*args)
        mask_calls.append(result)
        return result

    def predict(evidence, models):
        assert evidence.rejected_shuttle_frames is mask_calls[0]
        return SimpleNamespace(
            rally_prediction=SimpleNamespace(rallies=(), events=()),
            rally_review=SimpleNamespace(scores=np.empty(0)),
        )

    models = replace(annotator_models, preprocessing=BaseAnnotatorConfig(rejected_grades=rejected_grades))
    monkeypatch.setattr(preparation, 'build_rejected_shuttle_mask', build_mask)
    monkeypatch.setattr(run_video_module, 'predict_rallies_and_contacts', predict)
    annotate_video(
        **inputs, **_default_scene_inputs(frame_count), spans=[(10, 20)], contacts={0: [14]},
        models=models, shuttle_grades=codes, keep_working_data=True,
    )
    assert len(mask_calls) == 1
    np.testing.assert_array_equal(mask_calls[0], np.isin(codes, tuple(rejected_grades)))
