"""Eight-model-stack specialist fitting and score alignment for labelled rally-option pools."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import ServeCandidate
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import (
    EditKind,
    SequenceEdit,
    SequenceOption,
    option_pool,
    sequence_edits,
)
from annotator.sequence.features import (
    insertion_features,
    positive_probabilities,
    sequence_features,
    serve_edit_inputs,
    serve_edit_key,
)
from annotator.sequence.refine import OptionPool
from annotator.training.sequences import (
    LabelledRally,
    SequenceFitConfig,
    SequenceTrainingVideo,
    TreeFitConfig,
    sequence_targets,
)
from annotator.training.trees import TreeBackend
from experiments.annotator.tree_simplification import specialists
from experiments.annotator.tree_simplification.options import (
    LabelledOptions,
    sequence_key,
)


def training_video() -> SequenceTrainingVideo:
    first = ContactSequence(0, 80, 200, (ContactEvent(130, 0.9, Half.BOT), ContactEvent(160, 0.9, Half.TOP)))
    second = ContactSequence(1, 280, 450, (ContactEvent(300, 0.9, Half.TOP), ContactEvent(360, 0.9, Half.TOP)))
    initial = (first, second)
    events = tuple(event for sequence in initial for event in sequence.events)
    candidates = {
        0: tuple(ServeCandidate(0, ContactEvent(frame, 0.8, Half.TOP), 130, (0.0,) * 9)
                 for frame in (75, 100, 115)),
        1: (ServeCandidate(1, ContactEvent(270, 0.8, Half.TOP), 300, (0.0,) * 9),),
    }
    later = {
        0: (ContactEvent(185, 0.6, Half.BOT),),
        1: (ContactEvent(330, 0.7, Half.BOT), ContactEvent(390, 0.6, Half.BOT)),
    }
    edits = {
        0: sequence_edits(first, candidates[0], events, -1),
        1: sequence_edits(second, candidates[1], events, 200),
    }
    options = option_pool(edits, later, 30.0)
    frames = {event.frame for event in events}
    frames.update(candidate.event.frame for rows in candidates.values() for candidate in rows)
    frames.update(event.frame for rows in later.values() for event in rows)
    physical = {frame: np.full(85, frame / 1000) for frame in frames}
    serve = serve_edit_inputs(candidates, physical)
    static = sequence_features(
        options, {sequence.span_id: sequence for sequence in initial}, serve,
        {event.frame: event for event in events}, physical, 30.0,
    )
    insertion = insertion_features(options, physical, 30.0)
    pool = OptionPool(options, later, serve, static, insertion, physical)
    rallies = (
        LabelledRally('first', (100, 130, 160), (Half.TOP, Half.BOT, Half.TOP)),
        LabelledRally('second', (300, 330, 360), (Half.TOP, Half.BOT, Half.TOP)),
    )
    return SequenceTrainingVideo('video', 'east', 30.0, initial, events, pool, rallies)


def test_fit_specialists_uses_exact_role_targets_and_only_requested_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = training_video()
    targets = sequence_targets(video)
    added = video.pool.has_added_contact
    assert np.any((targets.insertion == 1) & (targets.sequence == 0) & added)

    calls: list[tuple[np.ndarray, np.ndarray, TreeFitConfig]] = []

    def capture_fit(matrix: np.ndarray, labels: np.ndarray, config: TreeFitConfig) -> Any:
        calls.append((matrix.copy(), labels.copy(), config))
        return object()

    monkeypatch.setattr(specialists, 'fit_tree', capture_fit)
    config = SequenceFitConfig(
        serve=TreeFitConfig(max_iter=4),
        sequence=TreeFitConfig(max_iter=7),
    )
    models = specialists.fit_specialists((video,), config, ('insertion',))

    assert models.serve is None and models.insertion is not None
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], video.pool.insertion_features[added])
    np.testing.assert_array_equal(calls[0][1], targets.insertion[added])
    assert calls[0][2] == config.sequence

    calls.clear()
    models = specialists.fit_specialists((video,), config, ('serve',))
    assert models.serve is not None and models.insertion is None
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], video.pool.serve_inputs.matrix)
    np.testing.assert_array_equal(calls[0][1], targets.serve)
    assert calls[0][2] == config.serve


def test_fit_specialists_reuses_supplied_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    video = training_video()
    targets = sequence_targets(video)
    calls = []

    def capture_fit(matrix: np.ndarray, labels: np.ndarray, config: TreeFitConfig) -> Any:
        calls.append(labels.copy())
        return object()

    monkeypatch.setattr(specialists, 'fit_tree', capture_fit)

    def forbidden_target_read(_video: Any) -> Any:
        raise AssertionError('supplied targets should be reused')

    monkeypatch.setattr(specialists, 'sequence_targets', forbidden_target_read)
    models = specialists.fit_specialists(
        (video,), SequenceFitConfig(), ('insertion',), targets={video.identity: targets},
    )

    assert models.serve is None and models.insertion is not None
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], targets.insertion[video.pool.has_added_contact])


class ColumnProbability:
    def __init__(self, column: int) -> None:
        self.classes_ = np.asarray([0, 1])
        self.column = column

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = matrix[:, self.column]
        return np.column_stack((1.0 - positive, positive))


def test_augment_pool_max_merges_duplicate_routes_and_keeps_unmatched_nan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = training_video()
    original_pool = video.pool
    route_row = 3  # add serve candidate 75 followed by the useful later contact
    route = original_pool.options[route_row]
    candidate_key = next(
        key for key in original_pool.serve_inputs.rows_by_key
        if key == (route.span_id, 100, EditKind.ADD)
    )
    duplicate = SequenceOption(
        SequenceEdit(EditKind.ADD, candidate_key[1], None, route.edit.sequence),
        route.added_contact,
        route.sequence,
    )
    duplicate_row = len(original_pool.options)

    serve_matrix = original_pool.serve_inputs.matrix.copy()
    for key, probability in (
        (serve_edit_key(route), 0.2),
        (candidate_key, 0.8),
    ):
        assert key is not None
        serve_matrix[original_pool.serve_inputs.rows_by_key[key], 0] = probability
    insertion_matrix = np.vstack((original_pool.insertion_features, original_pool.insertion_features[route_row]))
    insertion_matrix[route_row, 1] = 0.2
    insertion_matrix[duplicate_row, 1] = 0.8
    modified_pool = replace(
        original_pool,
        options=(*original_pool.options, duplicate),
        serve_inputs=replace(original_pool.serve_inputs, matrix=serve_matrix),
        insertion_features=insertion_matrix,
    )
    original = replace(video, pool=modified_pool)
    unmatched = ContactSequence(
        route.span_id,
        route.sequence.start_frame,
        route.sequence.end_frame,
        (*route.sequence.events, ContactEvent(190, 0.4, None)),
    )
    options = LabelledOptions(
        sequences=(route.sequence, unmatched),
        matrix=np.zeros((2, 3), dtype=np.float64),
        targets=np.asarray([0, 1]),
        span_rows={route.span_id: np.asarray([0, 1])},
        original=original,
    )

    def forbidden_target_read(_video: Any) -> Any:
        raise AssertionError('augmentation must not read training targets')

    monkeypatch.setattr(specialists, 'sequence_targets', forbidden_target_read)
    augmented = specialists.augment_pool(
        options,
        specialists.SpecialistModels(ColumnProbability(0), ColumnProbability(1)),
    )

    assert augmented.matrix.shape == (2, 7)
    np.testing.assert_allclose(augmented.matrix[0, -4:], (0.8, 0.8, 0.8, 0.8))
    assert np.isnan(augmented.matrix[1, -4])
    assert np.isnan(augmented.matrix[1, -3])
    np.testing.assert_allclose(augmented.matrix[1, -2:], (0.8, 0.8))
    assert sequence_key(augmented.sequences[0]) == sequence_key(route.sequence)


def test_missing_specialist_roles_append_nan_columns() -> None:
    video = training_video()
    sequence = video.initial[0]
    pool = LabelledOptions(
        sequences=(sequence,),
        matrix=np.zeros((1, 2), dtype=np.float64),
        targets=np.asarray([0]),
        span_rows={sequence.span_id: np.asarray([0])},
        original=video,
    )

    augmented = specialists.augment_pool(pool, specialists.SpecialistModels())

    assert augmented.matrix.shape == (1, 6)
    assert np.isnan(augmented.matrix[0, -4:]).all()


def test_tiny_histogram_specialists_fit_and_score() -> None:
    video = training_video()
    config = TreeFitConfig(
        max_iter=3,
        max_leaf_nodes=3,
        min_samples_leaf=1,
        backend=TreeBackend.HIST_GRADIENT_BOOSTING,
    )
    models = specialists.fit_specialists(
        (video,), SequenceFitConfig(serve=config, sequence=config), ('serve', 'insertion'),
    )

    assert models.serve is not None and models.insertion is not None
    serve_predictions = positive_probabilities(models.serve, video.pool.serve_inputs.matrix)
    insertion_predictions = positive_probabilities(
        models.insertion,
        video.pool.insertion_features[video.pool.has_added_contact],
    )
    assert serve_predictions.shape == (len(video.pool.serve_inputs.matrix),)
    assert insertion_predictions.shape == (int(video.pool.has_added_contact.sum()),)
    assert np.isfinite(serve_predictions).all() and np.isfinite(insertion_predictions).all()


def test_unknown_or_repeated_roles_fail_early() -> None:
    config = SequenceFitConfig()
    with pytest.raises(ValueError, match='unknown specialist roles'):
        specialists.fit_specialists((), config, ('sequence',))
    with pytest.raises(ValueError, match='unique'):
        specialists.fit_specialists((), config, ('serve', 'serve'))
