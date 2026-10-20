"""Bounded checks for the refit runner's population and selection rules."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.annotator.tree_hp_tune import run as grid


def original_videos() -> tuple:
    groups = json.loads(grid.GROUPS_FILE.read_text())
    validation = set(groups['fixed_validation_videos'])
    identities = list(validation)
    for group in groups['groups']:
        identities.extend(video['fixture'] for video in group['videos'])
    return tuple(
        grid.SearchVideo(identity, 'ShuttleSet', 'validation' if identity in validation else 'train', 25, (1280, 720), ())
        for identity in identities
    )


def result(identity: str, score: float, label: str = 'grid', correct: int = 3) -> dict:
    return {
        'id': identity, 'label': label,
        'totals': {'10': {'attributed_f1': score, 'fully_correct_rallies': correct}, '5': {'attributed_f1': score}},
    }


def test_population_and_fit_orders() -> None:
    development = grid.development_for(original_videos())
    plans = grid.retrain.contact_fit_videos(development)
    assert len(plans['v_32']) == 32
    assert plans['v_32'] == development.id_order(plans['v_32'])
    validation = set(development.videos(('V',)))
    for group in grid.retrain.TRAINING_GROUPS:
        held = plans[f'held_{group}']
        assert len(held) == 24
        assert not validation.intersection(held)
        assert all(development.group_by_video[identity] != group for identity in held)
        expected = tuple(identity for identity in development.videos(grid.retrain.TRAINING_GROUPS)
                         if development.group_by_video[identity] != group)
        assert held == expected


def test_population_rejects_changed_membership() -> None:
    videos = list(original_videos())
    from dataclasses import replace
    videos[0] = replace(videos[0], split='train')
    with pytest.raises(ValueError, match='membership differs'):
        grid.development_for(tuple(videos))


def test_pool_counts_before_rates() -> None:
    first = dict(zip(grid.COUNT_FIELDS, (2, 10, 20, 8, 6, 1), strict=True))
    second = dict(zip(grid.COUNT_FIELDS, (3, 20, 10, 9, 7, 2), strict=True))
    pooled = grid.pooled_scores({'one': {'5': first, '10': first}, 'two': {'5': second, '10': second}})
    assert pooled['10']['attributed_f1'] == 26 / 60
    assert pooled['10']['fully_correct_rallies'] == 3
    assert pooled['5']['labelled_contacts'] == 30


def test_rank_breaks_f1_ties_on_correct_rallies() -> None:
    assert grid.rank_key(result('a', 0.4, correct=4)) > grid.rank_key(result('b', 0.4, correct=3))


def test_candidate_preprocessing_and_serve_rejection() -> None:
    baseline = grid.preprocessing(grid.CandidateSpec('shipped', {}, {}))
    candidate = grid.preprocessing(grid.CandidateSpec('window', {'smooth_window': 12}, {}))
    assert baseline != candidate
    with pytest.raises(ValueError, match='serve configurations'):
        grid.preprocessing(grid.CandidateSpec('serve', {'threshold_bh': 0.5}, {'mode': 'TRIM'}))


def test_resume_keeps_full_grid_and_rejects_snapshot_changes(tmp_path: Path) -> None:
    baseline = grid.CandidateSpec('shipped', {}, {})
    other = grid.CandidateSpec('other', {'smooth_window': 12}, {})
    specs = [baseline, other]
    baseline_id = grid.candidate_id(baseline)
    (tmp_path / 'trials' / baseline_id).mkdir(parents=True)
    grid.write_json(tmp_path / 'trials' / baseline_id / 'result.json.gz', result(baseline_id, 0.5, 'shipped'))
    completed = grid.completed(tmp_path, specs, 'result.json.gz')
    assert len(completed) == 1
    snapshot_path = tmp_path / 'snapshot.json.gz'
    snapshot = {'source': {'revision': 'one', 'files_md5': {'code': 'abc'}}, 'candidates': ['both']}
    grid.keep_run_identity(snapshot_path, snapshot)
    grid.keep_run_identity(snapshot_path, {**snapshot, 'source': {**snapshot['source'], 'revision': 'two'}})
    with pytest.raises(ValueError, match='changed'):
        grid.keep_run_identity(snapshot_path, {**snapshot, 'candidates': ['pilot-only']})


def test_trial_chain_preserves_candidate_and_fit_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    videos = original_videos()
    development = grid.development_for(videos)
    fit_calls, saved = [], []
    pose = SimpleNamespace(bboxes=None, scores=None, kps=None, ndet=None)
    inputs = SimpleNamespace(track=None, pose=pose)
    features = SimpleNamespace(rows=grid.np.zeros((1, 1)))
    monkeypatch.setattr(grid, 'load_original_videos', lambda _path: videos)
    monkeypatch.setattr(grid, 'load_original_inputs', lambda *_args: inputs)
    monkeypatch.setattr(grid, 'cache_sticky_result', lambda *_args: nullcontext())
    monkeypatch.setattr(grid, 'run_video_kwargs', lambda _inputs: {})
    monkeypatch.setattr(grid, 'features_from_evidence', lambda _evidence: features)

    def run_video(*_args: object, **kwargs: object) -> None:
        kwargs['capture'].contact_evidence = SimpleNamespace()

    def contact_fit(training: list, _config: object, *, fit_video_order: tuple) -> SimpleNamespace:
        sampling_order = tuple(video.identity for video in training)
        assert fit_video_order == development.id_order(sampling_order)
        fit_calls.append(sampling_order)
        return SimpleNamespace(model='model', selection=SimpleNamespace(labels=grid.np.asarray([1])))

    def pool(entry: object, *_args: object) -> SimpleNamespace:
        return SimpleNamespace(identity=entry.identity)

    def downstream(pools: list) -> tuple:
        assert tuple(pool.identity for pool in pools) == development.videos(grid.retrain.TRAINING_GROUPS)
        predictions = {pool.identity: SimpleNamespace(sequences=(), events=()) for pool in pools}
        return SimpleNamespace(models='sequences', cross_fitted=predictions), 'confidence', grid.np.zeros(1)

    monkeypatch.setattr(grid, 'run_video', run_video)
    monkeypatch.setattr(grid, 'fit_contact_model', contact_fit)
    monkeypatch.setattr(grid, 'sequence_training_video', pool)
    monkeypatch.setattr(grid.retrain, 'fit_downstream', downstream)
    monkeypatch.setattr(grid, 'save_models', lambda models, _path: saved.append(models))
    monkeypatch.setattr(grid, 'progress', lambda *_args, **_kwargs: None)
    spec = grid.CandidateSpec('window', {'smooth_window': 12}, {})
    paths = {'shots_master': tmp_path, 'stages': tmp_path, 'development_courts': tmp_path}
    result = grid.fit_trial(spec, paths, tmp_path, 1)
    plans = grid.retrain.contact_fit_videos(development)
    assert fit_calls == [order for name, order in plans.items() if name != 'final_40']
    assert saved[0].preprocessing == grid.preprocessing(spec)
    assert len(result['per_video']) == 32
    assert len(list((tmp_path / 'trials' / result['id'] / 'training_streams').glob('*.json.gz'))) == 32


@pytest.mark.parametrize('field,value', [('score_cutoff', 0.5), ('reject_masked_without_player', True)])
def test_prediction_settings_are_checked_for_new_and_legacy_grids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object,
) -> None:
    settings = grid.DEFAULT_TRAINING_SETTINGS
    changed = replace(settings, contact_prediction=replace(settings.contact_prediction, **{field: value}))
    assert grid.settings_record(settings) != grid.settings_record(changed)
    output = tmp_path / 'grid48'
    bundle = output / 'trials' / 'candidate' / 'bundle_v32'
    bundle.mkdir(parents=True)
    (tmp_path / 'repeats').mkdir()
    legacy_settings = grid.settings_record(settings)
    legacy_settings.pop('contact_prediction')
    grid.write_json(output / 'snapshot.json.gz', {'inputs': 'fixed'})
    grid.write_json(tmp_path / 'repeats/plan.json.gz', {'seed_settings': {'0': legacy_settings}})
    monkeypatch.setattr(grid, 'load_models', lambda _path: SimpleNamespace(contact_settings=settings.contact_prediction))
    grid.keep_grid_identity(output, {'inputs': 'fixed'})
    monkeypatch.setattr(grid, 'DEFAULT_TRAINING_SETTINGS', changed)
    with pytest.raises(ValueError, match='training settings changed'):
        grid.keep_grid_identity(output, {'inputs': 'fixed'})
