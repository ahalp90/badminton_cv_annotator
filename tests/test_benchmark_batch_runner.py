"""Resume predictions without changing them and share only compatible rally pools."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from annotator.contacts.build_features import (
    CONTACT_FEATURE_NAMES,
    contact_feature_dtype,
)
from annotator.contacts.score_candidates import ScoredContacts
from annotator.inference.learned import (
    LearnedModelPrediction,
    predict_from_rally_options,
)
from annotator.models import AnnotatorModels
from annotator.rallies.build_contact_proposals import RallyContext, build_rally_options
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReview,
    RallyReviewFeatures,
    RallySelector,
)
from annotator.types import ContactEvent, ContactSequence, CourtHalf, FinishedRally
from experiments.annotator.monotrack_benchmark.writeup_scripts import (
    annotator_batch_runs as batch,
)
from experiments.annotator.monotrack_benchmark.writeup_scripts.annotator_runs import (
    Run,
    planned_runs,
)


class ContactTree:
    def __init__(self, identity=b'contact'):
        self.identity = identity

    def get_booster(self):
        return self

    def save_raw(self):
        return bytearray(self.identity)


def models(identity=b'contact', selector='selector'):
    return AnnotatorModels(ContactTree(identity), RallySelector(selector), RallySelector(selector), f'review-{selector}')


def prediction(probability=0.8, pool=None):
    event = ContactEvent(100, 0.9, CourtHalf.BOT)
    sequence = ContactSequence(0, 90, 110, (event,))
    rallies = (FinishedRally(sequence, CourtHalf.BOT),)
    refined = RallyPrediction(rallies, (event,), pool, np.array([0.9]), {0: 0})
    review = RallyReview(RallyReviewFeatures((0,), np.empty((1, 0))), np.array([probability]))
    return LearnedModelPrediction(None, None, refined, review, rallies, (event,), (0,))


def paths(tmp_path):
    return batch.BatchPaths(tmp_path / 'training', tmp_path / '40', tmp_path / 'outputs', {})


def test_grouping_reuses_preparation_across_seeds_and_contact_sizes(tmp_path):
    groups = batch.grouped_jobs(planned_runs())
    assert len(groups) == 192
    assert sum(len(group.runs) * len(paths(tmp_path).policies) for group in groups) == 1650
    raw = next(group for group in groups if group.split == 'test' and group.preprocessing == 'raw_baseline')
    assert len(raw.runs) == 10
    assert {run.family for run in raw.runs} == {'32_videos', '40_videos'}
    assert len({run.seed for run in raw.runs}) == 5
    body = next(group for group in groups if group.preprocessing == 'body_window_36')
    assert len(body.runs) == 5
    assert all(run.family == 'body_window_36' for run in body.runs)


def test_pool_compatibility_accepts_distinct_selectors_but_rejects_contact_changes():
    first = models(selector='first')
    other = models(selector='second')
    batch.require_shared_pool((first, other))
    with pytest.raises(ValueError, match='same fitted contact tree'):
        batch.require_shared_pool((first, models(b'different-seed')))
    changed_budget = replace(other, rally_selector=replace(other.rally_selector, max_edits=1))
    with pytest.raises(ValueError, match='proposal budgets'):
        batch.require_shared_pool((first, changed_budget))
    changed_cutoff = replace(other, contact_settings=replace(other.contact_settings, score_cutoff=0.8))
    with pytest.raises(ValueError, match='contact settings'):
        batch.require_shared_pool((first, changed_cutoff))


def test_shared_prediction_runs_contact_pipeline_once_and_each_own_selector_review(monkeypatch):
    events = tuple(ContactEvent(frame, 0.9, CourtHalf.TOP) for frame in (40, 60, 200, 220))
    sequence = ContactSequence(7, 10, 250, events)
    physical = {event.frame: np.zeros(len(CONTACT_FEATURE_NAMES)) for event in events}
    context = RallyContext(30.0, (sequence,), events, {}, physical, frame_count=300)
    pool = build_rally_options(context, max_edits=0)
    calls = []

    def tree(name, probability):
        def predict(matrix):
            calls.append(name)
            return np.tile([1 - probability, probability], (len(matrix), 1))
        return SimpleNamespace(predict_proba=predict)

    fitted = tuple(
        AnnotatorModels(ContactTree(), RallySelector(tree(f'selector-{index}', 0.9), max_edits=0),
                        RallySelector(tree(f'followup-{index}', 0.9), max_edits=0),
                        tree(f'review-{index}', score))
        for index, score in enumerate((0.5, 0.6, 0.7))
    )

    rows = np.zeros(len(events), dtype=contact_feature_dtype())
    rows['frame'] = [event.frame for event in events]
    scored = ScoredContacts(rows, np.zeros((len(rows), len(CONTACT_FEATURE_NAMES))),
                            np.full(len(rows), .9), np.arange(len(rows)))
    evidence = SimpleNamespace(fps=30.0, track=np.zeros((300, 3)), bounds_supplied=False)

    def contact_pipeline(evidence, features, model):
        calls.append('contact')
        return predict_from_rally_options(evidence, features, scored, pool, model)

    monkeypatch.setattr(batch, 'predict_from_contact_features', contact_pipeline)
    predictions = batch.predict_shared_pool(evidence, None, fitted)
    assert calls == ['contact', 'selector-0', 'followup-0', 'review-0', 'selector-1', 'followup-1',
                     'review-1', 'selector-2', 'followup-2', 'review-2']
    for predicted, score in zip(predictions, (0.5, 0.6, 0.7), strict=True):
        assert predicted.refinement.pool is pool
        assert [[event.frame for event in sequence.events] for sequence in predicted.sequences] == [[40, 60], [200, 220]]
        assert predicted.source_span_ids == (7, 7)
        assert predicted.rally_confidence == (score, score)
        record = batch.prediction_record(Run('test', 'discovery', '32_videos', 0), '8', predicted)
        assert record['confidence'] == {'0': score, '1': score}


@pytest.mark.parametrize('condition', ['discovery', 'our_courts'])
def test_resume_retains_saved_bytes_and_rejects_changed_predictions(tmp_path, condition):
    run = Run('test', condition, '32_videos', 0)
    destination = tmp_path / '8.json.gz'
    record = batch.prediction_record(run, '8', prediction())
    assert batch.save_or_compare(destination, run, '8', record)
    original = destination.read_bytes()
    assert not batch.save_or_compare(destination, run, '8', record)
    assert destination.read_bytes() == original
    changed = batch.prediction_record(run, '8', prediction(0.7))
    if condition != 'discovery':
        changed['stream']['events'][0][0] += 1
    with pytest.raises(ValueError, match='differs from the saved prediction'):
        batch.save_or_compare(destination, run, '8', changed)
    assert destination.read_bytes() == original


def test_truncated_saved_prediction_is_not_overwritten(tmp_path):
    run = Run('test', 'discovery', '32_videos', 0)
    destination = tmp_path / '8.json.gz'
    destination.write_bytes(b'not gzip')
    with pytest.raises(OSError):
        batch.save_or_compare(destination, run, '8', batch.prediction_record(run, '8', prediction()))
    assert destination.read_bytes() == b'not gzip'


def test_failed_atomic_write_leaves_no_prediction_or_temporary_file(tmp_path):
    run = Run('test', 'discovery', '32_videos', 0)
    destination = tmp_path / '8.json.gz'
    record = batch.prediction_record(run, '8', prediction())
    record['confidence']['0'] = float('nan')
    with pytest.raises(ValueError):
        batch.save_or_compare(destination, run, '8', record)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('policies', [('2_3', '3_3', '5_5'), ('no_max',)])
def test_complete_group_skips_preparation_and_models(tmp_path, monkeypatch, policies):
    run = Run('test', 'discovery', '32_videos', 0)
    group = batch.VideoGroup('8', 'test', 'discovery', 'raw_baseline', (run,))
    configured = replace(paths(tmp_path), policies=policies)
    for policy in configured.policies:
        batch.save_or_compare(batch.prediction_path(configured, policy, run, '8'), run, '8',
                              batch.prediction_record(run, '8', prediction()))

    def unexpected(*_args):
        raise AssertionError('completed group must not load models or prepare features')

    monkeypatch.setattr(batch, 'cached_models', unexpected)
    monkeypatch.setattr(batch, 'prepare_group', unexpected)
    result = batch.run_group(group, configured)
    assert result['created'] == 0
    assert result['reused'] == len(policies)
    assert result['compared'] == 0


def test_partial_group_checks_existing_policy_and_uses_separate_pools_for_contact_sizes(tmp_path, monkeypatch):
    runs = (Run('test', 'discovery', '32_videos', 0), Run('test', 'discovery', '40_videos', 0))
    group = batch.VideoGroup('8', 'test', 'discovery', 'raw_baseline', runs)
    configured = replace(paths(tmp_path), policies=('2_3', '3_3', '5_5'))
    existing = batch.prediction_path(configured, '2_3', runs[0], '8')
    batch.save_or_compare(existing, runs[0], '8', batch.prediction_record(runs[0], '8', prediction()))
    before = existing.read_bytes()
    prepared, predicted = [], []
    baseline = models()
    monkeypatch.setattr(batch, 'cached_models', lambda _path: baseline)
    monkeypatch.setattr(batch, 'policy_models', lambda bundle_paths: (baseline,) * 3)

    def prepare(*args):
        prepared.append(args)
        return 'evidence', 'features'

    def predict(evidence, features, fitted):
        predicted.append((evidence, features, fitted))
        return tuple(prediction() for _ in fitted)

    monkeypatch.setattr(batch, 'prepare_group', prepare)
    monkeypatch.setattr(batch, 'predict_shared_pool', predict)
    result = batch.run_group(group, configured)
    assert len(prepared) == 1
    assert len(predicted) == 2
    assert result['created'] == 5
    assert result['reused'] == result['compared'] == 1
    assert existing.read_bytes() == before


def test_run_receipt_rejects_different_models_before_resume(tmp_path):
    configured = paths(tmp_path)
    batch.prepare_run_policy(configured, '2_3')
    batch.prepare_run_policy(configured, '2_3')
    with pytest.raises(ValueError, match='model paths or evaluation policy differ'):
        batch.prepare_run_policy(replace(configured, training=tmp_path / 'other'), '2_3')
