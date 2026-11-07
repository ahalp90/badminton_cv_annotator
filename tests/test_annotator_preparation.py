"""Shared preparation preserves original evidence without calculating outcomes."""

from functools import wraps
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import annotator.annotate_video as orchestration
from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.inference import prepare_annotation_inputs as player_evidence
from annotator.inference.prepare_annotation_inputs import (
    PreparedAnnotationInputs,
    prepare_annotation_inputs,
)
from annotator.types import ContactCandidate
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import SearchVideo
from experiments.annotator.tree_simplification import prepare as experiment
from tests.test_annotator_run_video import _default_scene_inputs, _synthetic_inputs


@pytest.fixture
def preparation_inputs():
    inputs = _synthetic_inputs()
    return {
        'track': inputs['track'], 'bboxes': inputs['bboxes'], 'scores': inputs['scores'],
        'kps': inputs['kps'], 'ndet': inputs['ndet'], 'video_config': scale_config_to_fps(BaseAnnotatorConfig(), inputs['fps']),
        'resolution': inputs['resolution'],
        'raw_exclusion_mask': inputs['raw_exclusion_mask'], **_default_scene_inputs(len(inputs['track'])),
    }


def test_preparation_keeps_original_arrays_and_injected_contact_order(preparation_inputs, tmp_path):
    spans = [(0, 100), (150, 250)]
    evidence = prepare_annotation_inputs(**preparation_inputs, spans=spans, contacts={1: [190, 160], 0: [80, 20]})
    assert evidence.track is preparation_inputs['track']
    assert evidence.bboxes is preparation_inputs['bboxes']
    assert evidence.pose_kps is preparation_inputs['kps']
    assert evidence.heuristic_spans is spans
    assert evidence.heuristic_contacts == [
        ContactCandidate(1, 190, None, None), ContactCandidate(1, 160, None, None),
        ContactCandidate(0, 80, None, None), ContactCandidate(0, 20, None, None),
    ]
    assert evidence.raw_contact_frames == [190, 160, 80, 20]
    path = tmp_path / 'prepared.joblib'
    joblib.dump(evidence, path)
    restored = joblib.load(path)
    assert isinstance(restored, PreparedAnnotationInputs)
    assert restored.video_config == evidence.video_config
    assert restored.fps == preparation_inputs['video_config'].fps
    assert restored.heuristic_contacts == evidence.heuristic_contacts
    assert restored.raw_contact_frames == evidence.raw_contact_frames
    np.testing.assert_array_equal(restored.track, evidence.track)
    np.testing.assert_array_equal(restored.exclusion_mask, evidence.exclusion_mask)


def test_injected_contacts_find_missing_spans_on_original_track(preparation_inputs, monkeypatch):
    from annotator.inference import prepare_annotation_inputs as preparation

    track = preparation_inputs['track']
    preparation_inputs['raw_exclusion_mask'][10:100] = True
    original = track.copy()

    def find(track_arg, video_config):
        assert track_arg is track
        np.testing.assert_array_equal(track_arg, original)
        return [(0, 200)]

    monkeypatch.setattr(preparation, 'find_rally_spans', find)
    monkeypatch.setattr('annotator.rallies.find_rallies.find_rally_spans', find)
    evidence = prepare_annotation_inputs(**preparation_inputs, contacts={0: [20, 150]})
    assert evidence.heuristic_spans == [(0, 200)]
    assert evidence.raw_contact_frames == [20, 150]
    np.testing.assert_array_equal(evidence.track, original)


@pytest.mark.parametrize('supplied_spans', [False, True])
@pytest.mark.parametrize(('supplied_mask', 'supplied_contacts', 'unmasked_count', 'masked_count'), [
    (False, False, 1, 1),
    (True, False, 0, 1),
    (False, True, 1, 0),
    (True, True, 1, 0),
])
def test_preparation_uses_original_or_masked_spans_for_each_input_path(
    preparation_inputs, monkeypatch, supplied_contacts, supplied_mask, supplied_spans, unmasked_count, masked_count,
):
    from annotator.inference import prepare_annotation_inputs as preparation

    preparation_inputs.pop('raw_exclusion_mask')
    track = preparation_inputs['track']
    track[:, 0] = np.linspace(0.1, 0.9, len(track))
    track[:, 1:] = 1.0
    original = track.copy()
    provided_spans = [(10, 180)] if supplied_spans else None
    unmasked_spans = [(0, len(track))]
    masked_spans = [(150, 250)]
    raw_mask = np.zeros(len(track), dtype=bool)
    raw_mask[10:100] = True
    unmasked_calls = []
    masked_calls = []

    def find(track_arg, video_config):
        if track_arg is track:
            np.testing.assert_array_equal(track_arg, original)
            unmasked_calls.append(track_arg)
            return unmasked_spans
        np.testing.assert_array_equal(track_arg[10:100, :2], np.tile(track[9, :2], (90, 1)))
        masked_calls.append(track_arg)
        return masked_spans

    def replay_mask(court_present, track_arg, spans, n_frames, fps, *, rejected_shuttle_frames):
        assert not supplied_mask
        assert track_arg is track
        assert spans is (provided_spans if supplied_spans else unmasked_spans)
        return raw_mask

    if supplied_mask:
        preparation_inputs['raw_exclusion_mask'] = raw_mask
    monkeypatch.setattr(preparation, 'find_rally_spans', find)
    monkeypatch.setattr('annotator.rallies.find_rallies.find_rally_spans', find)
    monkeypatch.setattr(preparation, 'combine_mask', replay_mask)
    contacts = {0: [20, 150]} if supplied_contacts else None
    evidence = prepare_annotation_inputs(**preparation_inputs, contacts=contacts, spans=provided_spans)

    assert len(unmasked_calls) == (0 if supplied_spans else unmasked_count)
    assert len(masked_calls) == (0 if supplied_spans else masked_count)
    expected_spans = provided_spans if supplied_spans else (unmasked_spans if supplied_contacts else masked_spans)
    assert evidence.heuristic_spans is expected_spans
    assert evidence.bounds_supplied is supplied_spans
    if supplied_contacts:
        assert evidence.raw_contact_frames == [20, 150]
    np.testing.assert_array_equal(track, original)


def test_preparation_filters_short_runs_and_excludes_unusable_court_frames(preparation_inputs):
    raw_mask = preparation_inputs['raw_exclusion_mask']
    raw_mask[1:3] = True
    raw_mask[40:120] = True
    row = preparation_inputs['homography_rows'][0]
    preparation_inputs['homography_rows'] = [dict(row, end_frame=150), dict(row, start_frame=160)]
    preparation_inputs['court_present'][200] = False
    evidence = prepare_annotation_inputs(**preparation_inputs, spans=[], contacts={})
    expected = np.zeros(len(raw_mask), dtype=bool)
    expected[40:120] = True
    expected[150:160] = True
    expected[200] = True
    assert evidence.raw_exclusion_mask is raw_mask
    np.testing.assert_array_equal(evidence.exclusion_mask, expected)
    assert raw_mask[1:3].all()


def test_direct_preparation_reuses_actual_sticky_producer(preparation_inputs, tmp_path, monkeypatch):
    original = player_evidence.build_player_evidence
    calls = []

    @wraps(original)
    def build(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(player_evidence, 'build_player_evidence', build)
    with cache_sticky_result(tmp_path, 'video'):
        first = prepare_annotation_inputs(**preparation_inputs, spans=[], contacts={})
        second = prepare_annotation_inputs(**preparation_inputs, spans=[], contacts={})
    assert len(calls) == 1
    for first_array, second_array in zip((*first.players, *first.wrist_gaps), (*second.players, *second.wrist_gaps), strict=True):
        np.testing.assert_array_equal(first_array, second_array)
        assert not first_array.flags.writeable
        assert not second_array.flags.writeable


def test_experiment_prepares_and_saves_features_without_outcomes(preparation_inputs, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('feature preparation must not select contacts or calculate outcomes')

    monkeypatch.setattr(orchestration, 'select_heuristic_contacts', unexpected)
    monkeypatch.setattr(orchestration, 'build_outcomes', unexpected)
    monkeypatch.setattr(orchestration, 'build_hit_heights', unexpected)
    inputs = preparation_inputs
    source = SimpleNamespace(
        identity='video', fps=inputs['video_config'].fps, track=inputs['track'],
        pose=SimpleNamespace(bboxes=inputs['bboxes'], scores=inputs['scores'], kps=inputs['kps'], ndet=inputs['ndet']),
        court=SimpleNamespace(resolution=inputs['resolution'],
                              homography_rows=inputs['homography_rows']),
        court_present=inputs['court_present'], raw_cuts=((0, len(inputs['track'])),),
        keep_vote=np.ones(len(inputs['track']), dtype=bool), shuttle_grades=np.zeros(len(inputs['track']), dtype=np.uint8),
    )
    monkeypatch.setattr(experiment, 'load_original_inputs', lambda *_args: source)
    video = SearchVideo(source.identity, 'test', 'train', source.fps, inputs['resolution'], ())
    record = experiment.prepare_video(video, 'group', {'stages': tmp_path, 'development_courts': tmp_path}, tmp_path)
    prepared = joblib.load(tmp_path / 'videos' / f'{source.identity}.joblib')
    assert isinstance(prepared.evidence, PreparedAnnotationInputs)
    assert prepared.evidence.video_config == inputs['video_config']
    assert len(prepared.features.rows) == record['feature_rows']
    np.testing.assert_array_equal(prepared.evidence.track, inputs['track'])
