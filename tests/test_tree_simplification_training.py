"""Held-group selector fitting, fit reuse and validation isolation."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from annotator.contacts.score_candidates import positive_probabilities
from annotator.rallies.build_contact_proposals import RallyContext
from annotator.training import rally as production_training
from annotator.training.sequences import (
    SequenceTrainingVideo,
)
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from experiments.annotator.tree_simplification import train
from experiments.annotator.tree_simplification.options import LabelledOptions


def make_pool(identity: str, group: str) -> LabelledOptions:
    event = ContactEvent(10, 0.8, CourtHalf.TOP)
    initial = ContactSequence(0, 0, 20, (event,))
    deleted = ContactSequence(0, 0, 20, ())
    context = RallyContext(
        30.0, (initial,), (event,), {}, {}, frame_count=30,
    )
    original = SequenceTrainingVideo(identity, group, context, ())
    return LabelledOptions(
        (initial, deleted), np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int8), {0: np.asarray([0, 1])}, original,
    )


def training_pools(videos_per_group: int = 2) -> dict[str, LabelledOptions]:
    pools = {}
    for group in 'ABCD':
        for index in range(videos_per_group):
            identity = f'train-{group}-{index}'
            pools[identity] = make_pool(identity, group)
    return pools



def test_selector_fit_reuses_group_fit_and_keeps_weight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pools = training_pools()
    calls = []

    by_matrix = {id(pool.matrix): pool.video for pool in pools.values()}

    def fit(selected, targets, config, *, span_weight):
        videos = [by_matrix[id(pool.matrix)] for pool in selected]
        calls.append(([video.video_id for video in videos], span_weight))
        return {'groups': sorted({video.group for video in videos})}

    monkeypatch.setattr(train.rally_training, 'fit_selector', fit)
    monkeypatch.setattr(train, 'progress', lambda *_args, **_kwargs: None)
    trainer = train.SelectorTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0)
    assert trainer.fit(tuple('ABC')) == {'groups': ['A', 'B', 'C']}
    assert trainer.fit(tuple('ABC')) == {'groups': ['A', 'B', 'C']}
    assert len(calls) == 1
    assert all('-D-' not in identity for identity in calls[0][0])
    assert calls[0][1] == 2.0
    restarted = train.SelectorTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0)
    assert restarted.fit(tuple('ABC')) == {'groups': ['A', 'B', 'C']}
    assert len(calls) == 1


def test_real_selector_uses_only_option_columns(tmp_path: Path) -> None:
    pools = training_pools(videos_per_group=1)
    trainer = train.SelectorTrainer(pools, TreeFitConfig(max_iter=3, min_child_weight=0), tmp_path, span_weight=2.0)
    with threadpool_limits(limits=1):
        model = trainer.fit(tuple('ABCD'))
        scores = positive_probabilities(model, make_pool('validation', 'A').matrix)
    assert model.n_features_in_ == 2
    assert scores.shape == (2,) and np.isfinite(scores).all()


def test_production_selector_fits_each_held_group_once(monkeypatch: pytest.MonkeyPatch) -> None:
    pools = {}
    groups_by_pool = {}
    for identity, pool in training_pools().items():
        runtime = pool.runtime
        groups_by_pool[id(runtime)] = pool.video.group
        pools[identity] = production_training.RallyTrainingPool(identity, pool.video.group, runtime, pool.targets)
    videos = [pool.video for pool in training_pools().values()]
    fits = []
    predictions = []

    def fit(options, _targets, _config, *, span_weight):
        assert span_weight == 2.0
        groups = frozenset(groups_by_pool[id(pool)] for pool in options)
        fits.append(groups)
        return groups

    def predict(model, pool):
        group = groups_by_pool[id(pool)]
        assert group not in model.selector
        predictions.append(group)
        return model

    monkeypatch.setattr(production_training, 'prepare_training_pools', lambda *_args: pools)
    monkeypatch.setattr(production_training, 'fit_selector', fit)
    monkeypatch.setattr(production_training, 'predict_rally_options', predict)
    result = production_training.fit_rally_models(videos, production_training.RallyFitConfig(span_weight=2.0))
    assert fits[0] == frozenset('ABCD')
    assert set(fits[1:]) == {frozenset('ABCD') - {group} for group in 'ABCD'}
    assert len(fits) == 5
    assert len(predictions) == len(videos)
    assert set(result.held_group_predictions) == set(pools)


@pytest.mark.parametrize('span_weight', [None, 2.5])
def test_cli_fits_training_pools_and_evaluates_validation_only(tmp_path, monkeypatch, span_weight):
    training = training_pools(videos_per_group=8)
    validation = {f'validation-{index}': make_pool(f'validation-{index}', 'ABCD'[index % 4]) for index in range(8)}
    pools = {**training, **validation}
    pool_dir, prepared_dir, output = tmp_path / 'pools', tmp_path / 'prepared', tmp_path / 'output'
    videos = [SimpleNamespace(identity=identity, split='train' if identity in training else 'validation')
              for identity in pools]
    original_read, original_load = train.read_json, train.joblib.load
    fitted = []

    def read(path):
        if path == pool_dir / 'pools.json.gz':
            return {'schema': train.POOL_SCHEMA, 'prepared': str(prepared_dir)}
        if path == prepared_dir / 'preparation.json.gz':
            return {'inputs': {'paths': {'shots_master': str(tmp_path / 'shots.csv')},
                               'groups': {identity: pool.video.group for identity, pool in pools.items()}}}
        if path.parent == pool_dir / 'videos':
            identity = path.name.removesuffix('.json.gz')
            return {'split': 'train' if identity in training else 'validation'}
        return original_read(path)

    def load(path, *args, **kwargs):
        if path.parent == pool_dir / 'videos':
            return pools[path.stem]
        return original_load(path, *args, **kwargs)

    by_matrix = {id(pool.matrix): pool.video.video_id for pool in pools.values()}

    def fit(selected, targets, _config, *, span_weight):
        fitted.append(({by_matrix[id(pool.matrix)] for pool in selected}, span_weight))
        return {'selector': True}

    def evaluate(model, selected, _videos, destination):
        assert model == {'selector': True}
        assert set(selected) == set(validation)
        destination.mkdir(parents=True, exist_ok=True)
        return {'totals': {'fixture': True}}

    monkeypatch.setattr(train, 'read_json', read)
    monkeypatch.setattr(train.joblib, 'load', load)
    monkeypatch.setattr(train, 'load_original_videos', lambda _path: videos)
    monkeypatch.setattr(train, 'runtime_versions', lambda: {'python': '3.12', 'packages': {'numpy': '2.0'}})
    monkeypatch.setattr(train.rally_training, 'fit_selector', fit)
    monkeypatch.setattr(train, 'evaluate', evaluate)
    monkeypatch.setattr(train, 'progress', lambda *_args, **_kwargs: None)
    arguments = ['train', '--pools', str(pool_dir), '--output', str(output), '--device', 'cpu']
    if span_weight is not None:
        arguments.extend(('--span-weight', str(span_weight)))
    monkeypatch.setattr(train.sys, 'argv', arguments)
    train.main()
    assert fitted == [(set(training), 372.0 if span_weight is None else span_weight)]
    assert (output / 'selector.joblib').is_file()


@pytest.mark.parametrize('span_weight', ['0', '-1', 'nan', 'inf'])
def test_cli_rejects_invalid_span_weight(monkeypatch: pytest.MonkeyPatch, span_weight: str) -> None:
    monkeypatch.setattr(train.sys, 'argv', [
        'train', '--pools', 'unused', '--output', 'unused', '--span-weight', span_weight,
    ])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
