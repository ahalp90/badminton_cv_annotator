"""Layer context, per-rally weighting and jointly finished selections."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import ServeEditInputs
from annotator.sequence.refine import OptionPool
from annotator.training.sequences import SequenceTrainingVideo, TreeFitConfig
from experiments.annotator.tree_simplification import layers
from experiments.annotator.tree_simplification.options import ExpandedPool


def example_pool() -> ExpandedPool:
    first = ContactEvent(20, 0.8, Half.TOP)
    last = ContactEvent(40, 0.7, Half.TOP)
    added = ContactEvent(30, 0.6, Half.BOT)
    neighbour = ContactEvent(70, 0.9, Half.BOT)
    outside = ContactEvent(100, 0.6, Half.TOP)
    initial = (ContactSequence(0, 15, 45, (first, last)), ContactSequence(1, 65, 75, (neighbour,)))
    sequences = (
        initial[0], replace(initial[0], events=(first, added, last)),
        replace(initial[0], events=(last,)), initial[1],
    )
    legacy = OptionPool((), {}, ServeEditInputs({}, np.empty((0, 0))), np.empty((0, 0)), np.empty((0, 0)), {})
    original = SequenceTrainingVideo(
        'fixture', 'group', 30.0, initial, (first, last, neighbour, outside), legacy, (), 110,
    )
    return ExpandedPool(
        sequences, np.asarray([[0, 1], [1, 1], [0, 0], [2, 1]], dtype=np.float32),
        np.asarray([0, 1, -1, 0], dtype=np.int8), {0: np.asarray([0, 1, 2]), 1: np.asarray([3])}, original,
    )


def test_each_prior_layer_has_its_own_choice_context_and_stable_ties() -> None:
    pool = example_pool()
    first = np.asarray([0.5, 0.8, 0.8, 0.6])
    second = np.asarray([0.9, 0.2, 0.4, 0.1])
    assert layers.choose_rows(pool, first) == {0: 1, 1: 3}
    assert layers.choose_rows(pool, second) == {0: 0, 1: 3}
    assert layers.layer_features(pool) is pool.matrix
    actual = layers.layer_features(pool, (first, second))
    np.testing.assert_array_equal(actual[:, :2], pool.matrix)
    np.testing.assert_allclose(actual[:, 2:10], [
        [0.5, 0.8, 0, -0.3, 1, 0, 0, 1],
        [0.8, 0.8, 0, 0, 0, 1, 0, 0],
        [0.8, 0.8, 0, 0, 0.5, 0, 0, 2],
        [0.6, 0.6, 0, 0, 0, 1, 0, 0],
    ], atol=1e-7)
    np.testing.assert_allclose(actual[:, 10:], [
        [0.9, 0.9, 0.5, 0, 0, 1, 0, 0],
        [0.2, 0.9, 0.5, -0.7, 1, 0, 1, 0],
        [0.4, 0.9, 0.5, -0.5, 0.5, 0, 0, 1],
        [0.1, 0.1, 0, 0, 0, 1, 0, 0],
    ], atol=1e-7)
    assert actual.shape == (4, 2 + 2 * len(layers.CONTEXT_FEATURE_NAMES))


def test_fit_filters_unknown_targets_and_weights_spans_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = example_pool()
    unjudged = replace(pool, targets=np.full(4, -1, dtype=np.int8))
    sparse = replace(pool, targets=np.asarray([1, -1, -1, -1], dtype=np.int8))
    np.testing.assert_array_equal(layers.judged_weights(pool), [0.5, 0.5, 0, 1])
    np.testing.assert_array_equal(layers.judged_weights(unjudged), np.zeros(4))
    prior = np.asarray([0.5, 0.8, 0.7, 0.6])
    captured = {}

    def fit(matrix: np.ndarray, labels: np.ndarray, **kwargs: Any) -> object:
        captured.update(matrix=matrix, labels=labels, **kwargs)
        return captured

    monkeypatch.setattr(layers, 'fit_binary_tree', fit)
    model = layers.fit_layer([pool, sparse, unjudged], [(prior,), (prior,), (prior,)], TreeFitConfig(max_iter=3))
    assert model is captured
    np.testing.assert_array_equal(captured['labels'], [0, 1, 0, 1])
    fitted_weights = captured['sample_weight']
    np.testing.assert_allclose(fitted_weights, [2 / 3, 2 / 3, 4 / 3, 4 / 3])
    assert fitted_weights.mean() == pytest.approx(1.0)
    assert fitted_weights[:2].sum() == pytest.approx(fitted_weights[2])
    assert fitted_weights[2] == pytest.approx(fitted_weights[3])
    np.testing.assert_array_equal(captured['matrix'], layers.layer_features(pool, (prior,))[[0, 1, 3, 0]])
    assert captured['balanced'] is False
    assert captured['max_iter'] == 3


def test_fitted_layer_predicts_all_options_and_can_change_earlier_choices() -> None:
    pool = example_pool()
    prior = np.asarray([0.9, 0.1, 0.8, 0.6])
    with threadpool_limits(limits=1):
        model = layers.fit_layer([pool], [(prior,)], TreeFitConfig(max_iter=15, min_samples_leaf=1))
        scores = layers.predict_layer(model, pool, (prior,))
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert layers.choose_rows(pool, prior)[0] == 0
    assert layers.choose_rows(pool, scores)[0] == 1


def test_joint_finishing_expands_bounds_alternates_sides_and_keeps_external_events() -> None:
    pool = example_pool()
    sequences, events = layers.finish_selected(pool, np.asarray([0.2, 0.9, 0.1, 0.5]))
    assert [(sequence.start_frame, sequence.end_frame) for sequence in sequences] == [(10, 51), (60, 81)]
    assert tuple(event.frame for event in sequences[0].events) == (20, 30, 40)
    assert tuple(event.side for event in sequences[0].events) == (Half.TOP, Half.BOT, Half.TOP)
    assert tuple(event.frame for event in events) == (20, 30, 40, 70, 100)
    deleted, deletion_events = layers.finish_selected(pool, np.asarray([0.2, 0.1, 0.9, 0.5]))
    assert tuple(event.frame for event in deleted[0].events) == (40,)
    assert tuple(event.frame for event in deletion_events) == (40, 70, 100)
    earlier = ContactEvent(52, 0.8, Half.TOP)
    revised = replace(pool.sequences[3], start_frame=52, events=(earlier, *pool.sequences[3].events))
    interacting = replace(pool, sequences=(*pool.sequences[:3], revised))
    joint, _events = layers.finish_selected(interacting, np.asarray([0.2, 0.9, 0.1, 0.5]))
    assert joint[0].end_frame == joint[1].start_frame == 48


@pytest.mark.parametrize('scores', [np.ones(3), np.asarray([0.2, np.nan, 0.1, 0.5])])
def test_misaligned_or_nonfinite_scores_fail_loudly(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match='finite and align'):
        layers.choose_rows(example_pool(), scores)
