"""Shared preparation preserves original evidence without calculating outcomes."""

from functools import wraps
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

import annotator.run_video as orchestration
from annotator.config import BaseAnnotatorConfig, resolve
from annotator.preparation import AnnotationEvidence, prepare_annotation
from annotator.rally import evidence as rally_evidence
from annotator.rally_segmentation import segment_shuttle_track
from annotator.types import ContactCandidate
from experiments.annotator.heuristic_only_tune.cache import cache_sticky_result
from experiments.annotator.heuristic_only_tune.data import SearchVideo
from experiments.annotator.tree_simplification import prepare as experiment
from tests.annotator_config import burst_config
from tests.test_annotator_rally_segmentation import _build_rally_track
from tests.test_annotator_run_video import _default_scene_inputs, _synthetic_inputs


@pytest.fixture
def preparation_inputs():
    inputs = _synthetic_inputs()
    return {
        'track': inputs['track'], 'bboxes': inputs['bboxes'], 'scores': inputs['scores'],
        'kps': inputs['kps'], 'ndet': inputs['ndet'], 'resolved': resolve(BaseAnnotatorConfig(), inputs['fps']),
        'resolution': inputs['resolution'], 'video_id': inputs['video_id'], 'net_band': inputs['net_band'],
        'gate_court_info': inputs['gate_court_info'], 'gate_resolution_table': inputs['gate_resolution_table'],
        'raw_exclusion_mask': inputs['raw_exclusion_mask'], **_default_scene_inputs(len(inputs['track'])),
    }


def test_preparation_keeps_original_arrays_and_injected_contact_order(preparation_inputs, tmp_path):
    spans = [(0, 100), (150, 250)]
    evidence = prepare_annotation(**preparation_inputs, spans=spans, contacts={1: [190, 160], 0: [80, 20]})
    assert evidence.track is preparation_inputs['track']
    assert evidence.bboxes is preparation_inputs['bboxes']
    assert evidence.pose_kps is preparation_inputs['kps']
    assert evidence.heuristic_spans is spans
    assert evidence.heuristic_contacts == [
        ContactCandidate(1, 190, None, None, None), ContactCandidate(1, 160, None, None, None),
        ContactCandidate(0, 80, None, None, None), ContactCandidate(0, 20, None, None, None),
    ]
    assert evidence.raw_contact_frames == [190, 160, 80, 20]
    path = tmp_path / 'prepared.joblib'
    joblib.dump(evidence, path)
    restored = joblib.load(path)
    assert isinstance(restored, AnnotationEvidence)
    assert restored.resolved == evidence.resolved
    assert restored.fps == preparation_inputs['resolved'].fps
    assert restored.heuristic_contacts == evidence.heuristic_contacts
    assert restored.raw_contact_frames == evidence.raw_contact_frames
    np.testing.assert_array_equal(restored.track, evidence.track)
    np.testing.assert_array_equal(restored.exclusion_mask, evidence.exclusion_mask)


def test_injected_contacts_find_missing_spans_on_original_track(preparation_inputs, monkeypatch):
    from annotator import preparation

    track = preparation_inputs['track']
    preparation_inputs['raw_exclusion_mask'][10:100] = True
    original = track.copy()

    def find(track_arg, resolved):
        assert track_arg is track
        np.testing.assert_array_equal(track_arg, original)
        return [(0, 200)]

    monkeypatch.setattr(preparation, 'find_rally_spans', find)
    evidence = prepare_annotation(**preparation_inputs, contacts={0: [20, 150]})
    assert evidence.heuristic_spans == [(0, 200)]
    assert evidence.raw_contact_frames == [20, 150]
    np.testing.assert_array_equal(evidence.track, original)


def test_preparation_filters_short_runs_and_excludes_unusable_court_frames(preparation_inputs):
    raw_mask = preparation_inputs['raw_exclusion_mask']
    raw_mask[1:3] = True
    raw_mask[40:120] = True
    row = preparation_inputs['homography_rows'][0]
    preparation_inputs['homography_rows'] = [dict(row, end_frame=150), dict(row, start_frame=160)]
    preparation_inputs['court_present'][200] = False
    evidence = prepare_annotation(**preparation_inputs, spans=[], contacts={}, court_invalid_is_excluded=True)
    expected = np.zeros(len(raw_mask), dtype=bool)
    expected[40:120] = True
    expected[150:160] = True
    expected[200] = True
    assert evidence.raw_exclusion_mask is raw_mask
    np.testing.assert_array_equal(evidence.exclusion_mask, expected)
    assert raw_mask[1:3].all()


def test_direct_preparation_reuses_actual_sticky_producer(preparation_inputs, tmp_path, monkeypatch):
    original = rally_evidence.build_sticky_result
    calls = []

    @wraps(original)
    def build(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(rally_evidence, 'build_sticky_result', build)
    with cache_sticky_result(tmp_path, str(preparation_inputs['video_id'])):
        first = prepare_annotation(**preparation_inputs, spans=[], contacts={})
        second = prepare_annotation(**preparation_inputs, spans=[], contacts={})
    assert len(calls) == 1
    for first_array, second_array in zip(first.sticky, second.sticky, strict=True):
        np.testing.assert_array_equal(first_array, second_array)
        assert not first_array.flags.writeable
        assert not second_array.flags.writeable


def test_experiment_prepares_and_saves_features_without_outcomes(preparation_inputs, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('feature preparation must not select contacts or calculate outcomes')

    monkeypatch.setattr(orchestration, 'select_heuristic_contacts', unexpected)
    monkeypatch.setattr(orchestration, 'build_verdict_data', unexpected)
    monkeypatch.setattr(orchestration, 'build_hit_heights', unexpected)
    inputs = preparation_inputs
    source = SimpleNamespace(
        identity=str(inputs['video_id']), fps=inputs['resolved'].fps, track=inputs['track'],
        pose=SimpleNamespace(bboxes=inputs['bboxes'], scores=inputs['scores'], kps=inputs['kps'], ndet=inputs['ndet']),
        court=SimpleNamespace(resolution=inputs['resolution'], net_band=inputs['net_band'],
                              gate_court_info=inputs['gate_court_info'], gate_resolution_table=inputs['gate_resolution_table'],
                              homography_rows=inputs['homography_rows']),
        court_present=inputs['court_present'], raw_cuts=((0, len(inputs['track'])),),
        keep_vote=np.ones(len(inputs['track']), dtype=bool), guard_codes=np.zeros(len(inputs['track']), dtype=np.uint8),
    )
    monkeypatch.setattr(experiment, 'load_original_inputs', lambda *_args: source)
    video = SearchVideo(source.identity, 'test', 'train', source.fps, inputs['resolution'], ())
    record = experiment.prepare_video(video, 'group', {'stages': tmp_path, 'development_courts': tmp_path}, tmp_path)
    prepared = joblib.load(tmp_path / 'videos' / f'{source.identity}.joblib')
    assert isinstance(prepared.evidence, AnnotationEvidence)
    assert prepared.evidence.resolved == inputs['resolved']
    assert len(prepared.features.rows) == record['feature_rows']
    np.testing.assert_array_equal(prepared.evidence.track, inputs['track'])


def test_track_only_segmentation_retains_numeric_contacts_and_position_verdicts():
    track, rally_start, rally_end, _truth = _build_rally_track()
    positions = np.repeat(track[:, None, :2], 2, axis=1)
    original = track.copy()
    spans, contacts = segment_shuttle_track(track, burst_config(), positions=positions)
    assert len(spans) == 1
    assert abs(spans[0][0] - rally_start) <= 4
    assert abs(spans[0][1] - rally_end) <= 4
    assert [contact.contact_frame for contact in contacts] == [46, 50, 57, 64, 71]
    assert all(contact.proximity_ok is True for contact in contacts)
    assert all(contact.wrist_near is None and contact.suppressed is None for contact in contacts)
    dead_mask = np.zeros(len(track), dtype=bool)
    dead_mask[rally_start:rally_end] = True
    assert segment_shuttle_track(track, burst_config(), positions=positions, raw_exclusion_mask=dead_mask) == ([], [])
    np.testing.assert_array_equal(track, original)


@pytest.mark.parametrize('mask', [np.zeros(2, dtype=bool), np.ones(20, dtype=bool)])
def test_track_only_segmentation_rejects_invalid_masks(mask):
    with pytest.raises(ValueError, match='mask length|mask is all True'):
        segment_shuttle_track(np.zeros((20, 3)), resolve(BaseAnnotatorConfig(), 30.0), raw_exclusion_mask=mask)
