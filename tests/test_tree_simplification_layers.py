"""Layer context, per-rally weighting and jointly finished selections."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from annotator.contacts.score_candidates import positive_probabilities
from annotator.rallies import refine_rallies as rally_model
from annotator.rallies.build_contact_proposals import RallyContext
from annotator.training import rally as rally_training
from annotator.training.sequences import SequenceTrainingVideo
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from experiments.annotator.tree_simplification import margins
from experiments.annotator.tree_simplification.options import LabelledOptions


def example_pool() -> LabelledOptions:
    first = ContactEvent(20, 0.8, CourtHalf.TOP)
    last = ContactEvent(40, 0.7, CourtHalf.TOP)
    added = ContactEvent(30, 0.6, CourtHalf.BOT)
    neighbour = ContactEvent(70, 0.9, CourtHalf.BOT)
    outside = ContactEvent(100, 0.6, CourtHalf.TOP)
    initial = (ContactSequence(0, 15, 45, (first, last)), ContactSequence(1, 65, 75, (neighbour,)))
    sequences = (
        initial[0], replace(initial[0], events=(first, added, last)),
        replace(initial[0], events=(last,)), initial[1],
    )
    context = RallyContext(
        30.0, initial, (first, last, neighbour, outside), {}, {}, frame_count=110,
    )
    original = SequenceTrainingVideo('fixture', 'group', context, ())
    return LabelledOptions(
        sequences, np.asarray([[0, 1], [1, 1], [0, 0], [2, 1]], dtype=np.float32),
        np.asarray([0, 1, -1, 0], dtype=np.int8), {0: np.asarray([0, 1, 2]), 1: np.asarray([3])}, original,
    )


def test_final_margin_uses_initial_keep_without_changing_scores() -> None:
    pool = example_pool()
    scores = np.asarray([0.50, 0.54, 0.2, 0.1])
    assert margins.gated_rows(pool, scores, 0.0) == rally_model.choose_rows(pool.runtime, scores)
    assert margins.gated_rows(pool, scores, 0.05) == {0: 0, 1: 3}
    np.testing.assert_array_equal(scores, [0.50, 0.54, 0.2, 0.1])
    scores[1] = 0.56
    assert margins.gated_rows(pool, scores, 0.05) == {0: 1, 1: 3}
    scores[1] = scores[0]
    assert margins.gated_rows(pool, scores, 0.0) == {0: 0, 1: 3}


def test_margin_replay_rejects_an_incomplete_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr('sys.argv', ['margins', '--run', str(tmp_path), '--output', str(tmp_path / 'replay')])
    monkeypatch.setattr(margins, 'read_json', lambda path: {'complete': False} if path.name == 'result.json.gz' else {})
    with pytest.raises(SystemExit) as error:
        margins.main()
    assert error.value.code == 2
    assert '--run must name a completed configuration' in capsys.readouterr().err
    assert not (tmp_path / 'replay').exists()


def test_selector_keeps_stable_ties() -> None:
    pool = example_pool()
    scores = np.asarray([0.5, 0.8, 0.8, 0.6])
    assert rally_model.choose_rows(pool.runtime, scores) == {0: 1, 1: 3}


def test_fit_filters_unknown_targets_and_weights_spans_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = example_pool()
    unjudged = replace(pool, targets=np.full(4, -1, dtype=np.int8))
    sparse = replace(pool, targets=np.asarray([1, -1, -1, -1], dtype=np.int8))
    np.testing.assert_array_equal(rally_training.judged_weights(pool.runtime, pool.targets), [0.5, 0.5, 0, 1])
    np.testing.assert_array_equal(rally_training.judged_weights(unjudged.runtime, unjudged.targets), np.zeros(4))
    captured = {}

    def fit(matrix: np.ndarray, labels: np.ndarray, config: TreeFitConfig, **kwargs: Any) -> object:
        captured.update(matrix=matrix, labels=labels, config=config, **kwargs)
        return captured

    monkeypatch.setattr(rally_training, 'fit_binary_tree', fit)
    model = rally_training.fit_selector(
        [pool.runtime, sparse.runtime, unjudged.runtime], [pool.targets, sparse.targets, unjudged.targets],
        TreeFitConfig(max_iter=3), span_weight=1.5,
    )
    assert model is captured
    np.testing.assert_array_equal(captured['labels'], [0, 1, 0, 1])
    fitted_weights = captured['sample_weight']
    np.testing.assert_allclose(fitted_weights, [0.75, 0.75, 1.5, 1.5])
    assert fitted_weights[:2].sum() == pytest.approx(1.5)
    assert fitted_weights[:2].sum() == pytest.approx(fitted_weights[2])
    assert fitted_weights[2] == pytest.approx(fitted_weights[3])
    np.testing.assert_array_equal(captured['matrix'], pool.matrix[[0, 1, 3, 0]])
    assert captured['balanced'] is False
    assert captured['config'].max_iter == 3


def test_fixed_span_weight_is_identical_across_pool_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    original = example_pool()
    all_options = replace(
        original,
        sequences=(*original.sequences, *original.sequences[:2], *original.sequences[:2]),
        matrix=np.vstack((original.matrix, original.matrix[:2], original.matrix[:2])),
        targets=np.concatenate((original.targets, original.targets[:2], original.targets[:2])),
        span_rows={0: np.asarray([0, 1, 2, 4, 5, 6, 7]), 1: np.asarray([3])},
    )
    span_weight = 2.5

    def fit(_matrix: np.ndarray, _labels: np.ndarray, _config: TreeFitConfig, **kwargs: Any) -> np.ndarray:
        return kwargs['sample_weight']

    monkeypatch.setattr(rally_training, 'fit_binary_tree', fit)
    for pool in (original, all_options):
        judged = pool.targets >= 0
        passed = rally_training.fit_selector([pool.runtime], [pool.targets], TreeFitConfig(), span_weight=span_weight)
        full_weights = np.zeros(len(pool.targets))
        full_weights[judged] = passed
        assert len(passed) == int(judged.sum())
        for rows in pool.span_rows.values():
            assert full_weights[rows].sum() == pytest.approx(span_weight)


@pytest.mark.parametrize('span_weight', [0.0, -1.0, np.nan, np.inf])
def test_invalid_fixed_span_weight_fails_loudly(span_weight: float) -> None:
    with pytest.raises(ValueError, match='finite and positive'):
        rally_training.fit_selector([example_pool().runtime], [example_pool().targets], TreeFitConfig(), span_weight=span_weight)


def test_fitted_layer_predicts_all_options_and_can_change_earlier_choices() -> None:
    pool = example_pool()
    prior = np.asarray([0.9, 0.1, 0.8, 0.6])
    with threadpool_limits(limits=1):
        model = rally_training.fit_selector([pool.runtime], [pool.targets], TreeFitConfig(max_iter=15, min_child_weight=0), span_weight=1.5)
        scores = positive_probabilities(model, pool.matrix)
    assert scores.shape == (4,)
    assert np.isfinite(scores).all()
    assert rally_model.choose_rows(pool.runtime, prior)[0] == 0
    assert rally_model.choose_rows(pool.runtime, scores)[0] == 1


def test_joint_finishing_expands_bounds_alternates_sides_and_keeps_external_events() -> None:
    pool = example_pool()
    runtime = pool.runtime
    rallies, events = rally_model.finish_rallies(runtime, rally_model.choose_rows(runtime, np.asarray([0.2, 0.9, 0.1, 0.5])))
    sequences = tuple(rally.sequence for rally in rallies)
    assert [(sequence.start_frame, sequence.end_frame) for sequence in sequences] == [(10, 51), (60, 81)]
    assert tuple(event.frame for event in sequences[0].events) == (20, 30, 40)
    assert tuple(event.side for event in sequences[0].events) == (CourtHalf.TOP, CourtHalf.BOT, CourtHalf.TOP)
    assert tuple(event.frame for event in events) == (20, 30, 40, 70, 100)
    rallies, deletion_events = rally_model.finish_rallies(runtime, rally_model.choose_rows(runtime, np.asarray([0.2, 0.1, 0.9, 0.5])))
    deleted = tuple(rally.sequence for rally in rallies)
    assert tuple(event.frame for event in deleted[0].events) == (40,)
    assert tuple(event.frame for event in deletion_events) == (40, 70, 100)
    earlier = ContactEvent(52, 0.8, CourtHalf.TOP)
    revised = replace(pool.sequences[3], start_frame=52, events=(earlier, *pool.sequences[3].events))
    interacting = replace(pool, sequences=(*pool.sequences[:3], revised))
    runtime = interacting.runtime
    rallies, _events = rally_model.finish_rallies(runtime, rally_model.choose_rows(runtime, np.asarray([0.2, 0.9, 0.1, 0.5])))
    joint = tuple(rally.sequence for rally in rallies)
    assert joint[0].end_frame == joint[1].start_frame == 48


@pytest.mark.parametrize('scores', [np.ones(3), np.asarray([0.2, np.nan, 0.1, 0.5])])
def test_misaligned_or_nonfinite_scores_fail_loudly(scores: np.ndarray) -> None:
    with pytest.raises(ValueError, match='finite and align'):
        rally_model.choose_rows(example_pool().runtime, scores)
