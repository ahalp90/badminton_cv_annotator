"""Recursive rally-group exclusion, stack reuse and validation isolation."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import ServeEditInputs
from annotator.sequence.refine import OptionPool
from annotator.training.sequences import SequenceTrainingVideo, TreeFitConfig
from experiments.annotator.tree_simplification import train
from experiments.annotator.tree_simplification.options import ExpandedPool
from experiments.annotator.tree_simplification.specialists import SpecialistModels


def make_pool(identity: str, group: str) -> ExpandedPool:
    event = ContactEvent(10, 0.8, Half.TOP)
    initial = ContactSequence(0, 0, 20, (event,))
    deleted = ContactSequence(0, 0, 20, ())
    legacy = OptionPool((), {}, ServeEditInputs({}, np.empty((0, 0))), np.empty((0, 0)), np.empty((0, 0)), {})
    original = SequenceTrainingVideo(identity, group, 30.0, (initial,), (event,), legacy, (), 30)
    return ExpandedPool(
        (initial, deleted), np.asarray([[0, 1], [1, 0]], dtype=np.float32),
        np.asarray([0, 1], dtype=np.int8), {0: np.asarray([0, 1])}, original,
    )


def training_pools(videos_per_group: int = 2) -> dict[str, ExpandedPool]:
    pools = {}
    for group in 'ABCD':
        for index in range(videos_per_group):
            identity = f'train-{group}-{index}'
            pools[identity] = make_pool(identity, group)
    return pools


@dataclass(frozen=True)
class RecordedModel:
    groups: frozenset[str]
    ancestors: frozenset[str]
    depth: int
    serial: int


class RecordingFits:
    """Carry training ancestry through score vectors without fitting real trees."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.models: list[RecordedModel] = []
        self.predictions: list[tuple[int, str]] = []
        monkeypatch.setattr(train, 'fit_layer', self.fit)
        monkeypatch.setattr(train, 'predict_layer', self.predict)
        monkeypatch.setattr(train, 'progress', lambda *_args, **_kwargs: None)

    def model_for(self, scores: np.ndarray) -> RecordedModel:
        return self.models[int(scores[0]) - 1]

    def fit(
        self, pools: Sequence[ExpandedPool], histories: Sequence[Sequence[np.ndarray]], config: TreeFitConfig,
        *, span_weight: float,
    ) -> RecordedModel:
        assert span_weight > 0
        assert all(pool.original.identity.startswith('train-') for pool in pools)
        groups = frozenset(pool.original.group for pool in pools)
        depth = len(histories[0]) + 1
        ancestors = set(groups)
        for pool, history in zip(pools, histories, strict=True):
            assert len(history) == depth - 1
            for previous_depth, scores in enumerate(history, 1):
                producer = self.model_for(scores)
                assert producer.depth == previous_depth
                assert producer.groups == groups - {pool.original.group}
                assert pool.original.group not in producer.ancestors
                assert producer.ancestors <= groups - {pool.original.group}
                ancestors.update(producer.ancestors)
        model = RecordedModel(groups, frozenset(ancestors), depth, len(self.models) + 1)
        self.models.append(model)
        return model

    def predict(self, model: RecordedModel, pool: ExpandedPool, history: Sequence[np.ndarray]) -> np.ndarray:
        assert len(history) == model.depth - 1
        for previous_depth, scores in enumerate(history, 1):
            producer = self.model_for(scores)
            assert producer.depth == previous_depth
            assert producer.groups == model.groups
        self.predictions.append((model.serial, pool.original.identity))
        return np.full(len(pool.sequences), model.serial, dtype=float)


def test_depth_three_excludes_held_groups_through_all_ancestors_and_reuses_subsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = RecordingFits(monkeypatch)
    trainer = train.LayerTrainer(training_pools(), TreeFitConfig(), tmp_path, span_weight=2.0)
    groups = tuple('ABCD')
    models = trainer.fit(groups, 3)
    assert len(recorded.models) == 17
    expected = {(frozenset(groups), depth) for depth in (1, 2, 3)}
    expected.update((frozenset(subset), depth) for subset in combinations(groups, 3) for depth in (1, 2))
    expected.update((frozenset(subset), 1) for subset in combinations(groups, 2))
    assert {(model.groups, model.depth) for model in recorded.models} == expected
    assert trainer.fit(groups, 2) == models[:2]
    assert trainer.fit(groups, 1) == models[:1]
    assert trainer.fit(groups, 3) is models
    cached = trainer.held_scores(tuple('ABC'), 2, 'train-D-0')
    prediction_count = len(recorded.predictions)
    assert trainer.held_scores(tuple('ABC'), 2, 'train-D-0') is cached
    assert len(recorded.predictions) == prediction_count
    assert len(recorded.models) == 17
    for (subset, depth, identity), scores in trainer.scores.items():
        assert len(scores) == depth
        assert trainer.pools[identity].original.group not in subset
        assert all(trainer.pools[identity].original.group not in recorded.model_for(row).ancestors for row in scores)


def test_resume_loads_complete_models_and_refits_only_an_incomplete_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = RecordingFits(monkeypatch)
    pools, groups = training_pools(), tuple('ABCD')
    original = train.LayerTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0).fit(groups, 3)
    resumed = train.LayerTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0)
    assert resumed.fit(groups, 3) == original
    assert len(recorded.models) == 17
    resumed.held_scores(tuple('ABC'), 2, 'train-D-0')
    assert len(recorded.models) == 17
    # A model without its completion record must be fitted again.
    (tmp_path / 'fits/ABCD/layer-3.json.gz').unlink()
    repaired = train.LayerTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0).fit(groups, 3)
    assert repaired[:2] == original[:2]
    assert repaired[2].serial == 18
    assert repaired[2].groups == frozenset(groups)
    assert repaired[2].depth == 3
    assert len(recorded.models) == 18


def test_real_hgb_prefixes_receive_one_context_block_per_previous_layer(tmp_path: Path) -> None:
    pools = training_pools(videos_per_group=1)
    trainer = train.LayerTrainer(pools, TreeFitConfig(max_iter=3, min_samples_leaf=1), tmp_path, span_weight=2.0)
    with threadpool_limits(limits=1):
        models = trainer.fit(tuple('ABCD'), 3)
        scores = train.score_layers(models, make_pool('validation', 'A'))
    assert [model.n_features_in_ for model in models] == [2, 10, 18]
    assert len(scores) == 3
    assert all(row.shape == (2,) and np.isfinite(row).all() for row in scores)


def test_specialists_exclude_the_predicted_group_at_every_rally_depth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded = RecordingFits(monkeypatch)
    specialist_fits = []

    def group_code(groups: frozenset[str]) -> int:
        return sum(1 << 'ABCD'.index(group) for group in groups)

    def fit_specialists(
        videos: Sequence[SequenceTrainingVideo], _config: Any, roles: tuple[str, ...], *, targets: Any,
    ) -> SpecialistModels:
        assert roles == ('serve',)
        assert all(video.identity.startswith('train-') for video in videos)
        assert all(targets[video.identity] == video.identity for video in videos)
        groups = frozenset(video.group for video in videos)
        specialist_fits.append(groups)
        return SpecialistModels(serve=groups, insertion=None)

    def augment(pool: ExpandedPool, models: SpecialistModels) -> ExpandedPool:
        code = np.full(len(pool.targets), group_code(models.serve))
        return replace(pool, matrix=np.column_stack((pool.matrix, code)))

    def fit(pools: Sequence[ExpandedPool], histories: Any, config: TreeFitConfig, *, span_weight: float) -> RecordedModel:
        groups = frozenset(pool.original.group for pool in pools)
        for pool in pools:
            assert np.all(pool.matrix[:, -1] == group_code(groups - {pool.original.group}))
        return recorded.fit(pools, histories, config, span_weight=span_weight)

    def predict(model: RecordedModel, pool: ExpandedPool, history: Any) -> np.ndarray:
        assert np.all(pool.matrix[:, -1] == group_code(model.groups))
        return recorded.predict(model, pool, history)

    monkeypatch.setattr(train, 'fit_specialists', fit_specialists)
    monkeypatch.setattr(train, 'sequence_targets', lambda video: video.identity)
    monkeypatch.setattr(train, 'augment_pool', augment)
    monkeypatch.setattr(train, 'fit_layer', fit)
    monkeypatch.setattr(train, 'predict_layer', predict)
    pools, groups = training_pools(), tuple('ABCD')
    trainer = train.LayerTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0, roles=('serve',))
    models = trainer.fit(groups, 2)
    expected = {frozenset(subset) for size in (2, 3, 4) for subset in combinations(groups, size)}
    assert len(specialist_fits) == len(set(specialist_fits)) == 11
    assert set(specialist_fits) == expected
    assert len(recorded.models) == 6
    train.score_layers(models, make_pool('validation', 'V'))
    resumed = train.LayerTrainer(pools, TreeFitConfig(), tmp_path, span_weight=2.0, roles=('serve',))
    assert resumed.fit(groups, 2) == models
    assert len(specialist_fits) == 11


@pytest.mark.parametrize('first_depth', [1, 3])
@pytest.mark.parametrize('span_weight', [None, 371.86228729719033])
def test_cli_passes_only_training_pools_to_fitter_and_all_validation_pools_to_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_depth: int, span_weight: float | None,
) -> None:
    recorded = RecordingFits(monkeypatch)
    training = training_pools(videos_per_group=8)
    validation = {f'validation-{index}': make_pool(f'validation-{index}', 'ABCD'[index % 4]) for index in range(8)}
    pools = {**training, **validation}
    pool_dir, prepared_dir, output = tmp_path / 'pools', tmp_path / 'prepared', tmp_path / 'output'
    videos = [SimpleNamespace(identity=identity, split='train' if identity in training else 'validation')
              for identity in pools]
    original_read, original_load = train.read_json, train.joblib.load
    evaluations = []

    def read(path: Path) -> Any:
        if path == pool_dir / 'pools.json.gz':
            return {'prepared': str(prepared_dir)}
        if path == prepared_dir / 'preparation.json.gz':
            return {'inputs': {'paths': {'shots_master': str(tmp_path / 'shots.csv')},
                               'groups': {identity: pool.original.group for identity, pool in pools.items()}}}
        if path.parent == pool_dir / 'videos':
            identity = path.name.removesuffix('.json.gz')
            return {'split': 'train' if identity in training else 'validation'}
        return original_read(path)

    def load(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.parent == pool_dir / 'videos':
            return pools[path.stem]
        return original_load(path, *args, **kwargs)

    def evaluate(models: tuple[Any, ...], selected: dict[str, ExpandedPool], *_args: Any) -> dict[str, Any]:
        assert set(selected) == set(validation)
        assert not set(selected) & set(training)
        assert all(pool is validation[identity] for identity, pool in selected.items())
        evaluations.append(len(models))
        assert len(train.score_layers(models, selected['validation-0'])) == len(models)
        result = {'totals': {'fixture': len(models)}}
        _args[-1].mkdir(parents=True, exist_ok=True)
        train.write_json(_args[-1] / 'validation.json.gz', result)
        return result

    monkeypatch.setattr(train, 'read_json', read)
    monkeypatch.setattr(train.joblib, 'load', load)
    monkeypatch.setattr(train, 'load_original_videos', lambda _path: videos)
    monkeypatch.setattr(train, 'source_record', lambda: {'revision': 'fixture'})
    monkeypatch.setattr(train, 'evaluate', evaluate)
    command = [
        'train', '--pools', str(pool_dir), '--output', str(output),
        '--backend', 'hist_gradient_boosting', '--device', 'cpu', '--threads', '1',
    ]
    if span_weight is not None:
        command.extend(('--span-weight', str(span_weight)))
    for depth in dict.fromkeys((first_depth, 3)):
        monkeypatch.setattr(train.sys, 'argv', [*command, '--layers', str(depth)])
        train.main()
    expected_weight = 1.0 if span_weight is None else span_weight
    assert train.read_json(output / 'settings.json.gz')['span_weight'] == expected_weight
    assert train.read_json(output / 'fits/ABCD/layer-1.json.gz')['span_weight'] == expected_weight
    assert evaluations == [1, 2, 3]
    assert len(recorded.models) == 17
    monkeypatch.setattr(train.sys, 'argv', [*command, '--layers', '3', '--leaves', '31'])
    with pytest.raises(ValueError, match='Training configuration changed'):
        train.main()


@pytest.mark.parametrize('span_weight', ['0', '-1', 'nan', 'inf'])
def test_cli_rejects_invalid_span_weight(monkeypatch: pytest.MonkeyPatch, span_weight: str) -> None:
    monkeypatch.setattr(train.sys, 'argv', [
        'train', '--pools', 'unused', '--output', 'unused', '--span-weight', span_weight,
    ])
    with pytest.raises(SystemExit) as error:
        train.main()
    assert error.value.code == 2
