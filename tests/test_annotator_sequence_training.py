"""Selected target rules and nested group exclusions for sequence model fitting."""

import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import ServeCandidate
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import option_pool, sequence_edits
from annotator.sequence.features import (
    insertion_features,
    sequence_features,
    serve_edit_inputs,
)
from annotator.sequence.refine import OptionPool
from annotator.training.sequences import (
    LabelledRally,
    SequenceFitConfig,
    SequenceTrainingVideo,
    TreeFitConfig,
    fit_sequence_models,
    match_contacts,
    sequence_targets,
)


def training_video(
    identity: str = 'video', group: str = 'east', marker: int = 1, fps: float = 30.0,
) -> SequenceTrainingVideo:
    first = ContactSequence(0, 80, 200, (ContactEvent(130, .9, Half.BOT), ContactEvent(160, .9, Half.TOP)))
    second = ContactSequence(1, 280, 450, (ContactEvent(300, .9, Half.TOP), ContactEvent(360, .9, Half.TOP)))
    initial = (first, second)
    events = tuple(event for sequence in initial for event in sequence.events)
    candidates = {
        0: tuple(ServeCandidate(0, ContactEvent(frame, .8, Half.TOP), 130, (0.0,) * 9) for frame in (75, 100, 115)),
        1: (ServeCandidate(1, ContactEvent(270, .8, Half.TOP), 300, (0.0,) * 9),),
    }
    later = {
        0: (ContactEvent(185, .6, Half.BOT),),
        1: (ContactEvent(330, .7, Half.BOT), ContactEvent(390, .6, Half.BOT)),
    }
    edits = {
        0: sequence_edits(first, candidates[0], events, -1),
        1: sequence_edits(second, candidates[1], events, 200),
    }
    options = option_pool(edits, later, fps)
    frames = {event.frame for event in events}
    frames.update(candidate.event.frame for rows in candidates.values() for candidate in rows)
    frames.update(event.frame for rows in later.values() for event in rows)
    physical = {frame: np.full(85, frame / 1000) for frame in frames}
    serve = serve_edit_inputs(candidates, physical)
    static = sequence_features(options, {sequence.span_id: sequence for sequence in initial}, serve,
                               {event.frame: event for event in events}, physical, fps)
    insertion = insertion_features(options, physical, fps)
    # One explicit group marker lets the fit/predict boundary reveal data leakage.
    serve.matrix[:, 0] = marker
    static[:, 0] = marker
    insertion[:, 0] = marker
    pool = OptionPool(options, later, serve, static, insertion, physical)
    rallies = (
        LabelledRally('first', (100, 130, 160), (Half.TOP, Half.BOT, Half.TOP)),
        LabelledRally('second', (300, 330, 360), (Half.TOP, Half.BOT, Half.TOP)),
    )
    return SequenceTrainingVideo(identity, group, fps, initial, events, pool, rallies)


REFERENCE_DIR = Path(__file__).parent / 'fixtures' / 'annotator_reference'


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
@pytest.mark.parametrize('missing_side', [False, True])
def test_targets_match_reference(fps: float, missing_side: bool) -> None:
    video = training_video(fps=fps)
    if missing_side:
        rally = replace(video.rallies[0], sides=(None, Half.BOT, Half.TOP))
        video = replace(video, rallies=(rally, video.rallies[1]))
    key = f'fps_{int(fps)}_' + ('missing' if missing_side else 'known')
    actual = sequence_targets(video)
    with np.load(REFERENCE_DIR / 'sequence_targets.npz') as reference:
        np.testing.assert_array_equal(actual.serve, reference[key + '_serve'])
        np.testing.assert_array_equal(actual.sequence, reference[key + '_sequence'])
        np.testing.assert_array_equal(actual.insertion, reference[key + '_insertion'])


def test_ambiguous_rally_and_empty_labels_are_excluded() -> None:
    video = training_video()
    empty = sequence_targets(replace(video, rallies=()))
    assert np.all(empty.serve == -1) and np.all(empty.sequence == -1) and np.all(empty.insertion == -1)
    # The same labelled rally touching two baseline sections makes both ambiguous.
    rally = LabelledRally('crosses', (130, 300), (Half.TOP, Half.BOT))
    ambiguous = sequence_targets(replace(video, rallies=(rally,)))
    assert np.all(ambiguous.sequence == -1)
    assert np.all(ambiguous.serve == -1)
    assert np.all(ambiguous.insertion == -1)


def test_matching_preserves_tie_and_duplicate_rules() -> None:
    cases = json.loads(gzip.decompress((REFERENCE_DIR / 'contact_matching.json.gz').read_bytes()))
    for case in cases:
        expected = [tuple(match) for match in case['matches']]
        assert match_contacts(case['labels'], case['predictions'], case['tolerance']) == expected


def test_grouped_smoke_fit_excludes_scored_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    original_fit = HistGradientBoostingClassifier.fit
    original_predict = HistGradientBoostingClassifier.predict_proba
    fits = []
    predictions = []
    fitted_groups = {}

    def observed_fit(
        model: HistGradientBoostingClassifier, matrix: np.ndarray, labels: np.ndarray, **kwargs: Any,
    ) -> HistGradientBoostingClassifier:
        groups = frozenset(matrix[:, 0].astype(int))
        result = original_fit(model, matrix, labels, **kwargs)
        fitted_groups[model] = groups
        fits.append((matrix.shape[1], groups, set(labels)))
        return result

    def checked_predict(model: HistGradientBoostingClassifier, matrix: np.ndarray) -> np.ndarray:
        row_groups = frozenset(matrix[:, 0].astype(int))
        assert row_groups.isdisjoint(fitted_groups[model])
        predictions.append((row_groups, fitted_groups[model]))
        return original_predict(model, matrix)

    monkeypatch.setattr(HistGradientBoostingClassifier, 'fit', observed_fit)
    monkeypatch.setattr(HistGradientBoostingClassifier, 'predict_proba', checked_predict)
    videos = [
        training_video(f'video{index}', group, index)
        for index, group in enumerate(('east', 'west', 'north'), start=1)
    ]
    tiny = TreeFitConfig(max_iter=3, max_leaf_nodes=3, min_samples_leaf=1)
    config = SequenceFitConfig(serve=replace(tiny, random_seed=20260824), sequence=tiny)
    with threadpool_limits(limits=1):
        result = fit_sequence_models(videos, config)
    assert result.models.whole_sequence.n_features_in_ == 304
    assert result.models.later_contact.n_features_in_ == 397
    assert result.models.scored_insertion.n_features_in_ == 398
    assert result.models.insertion.n_features_in_ == 93
    assert len(result.cross_fitted) == 3
    for video in videos:
        output = result.cross_fitted[video.identity]
        assert output.pool is video.pool
        insertion_model = result.cross_fitted_insertion_models[video.identity]
        assert video.pool.sequence_features[0, 0] not in fitted_groups[insertion_model]
        assert np.isfinite(output.scored_insertion.scores).all()
        assert output.sequences
    assert any(width == 10 and groups == frozenset({1}) for width, groups, _labels in fits)
    assert any(width == 398 and groups == frozenset({1, 2, 3}) for width, groups, _labels in fits)
    assert any(row_groups == frozenset({1}) and trained == frozenset({2}) for row_groups, trained in predictions)
    assert all(labels == {0, 1} for _width, _groups, labels in fits)


def test_selected_tree_defaults_and_input_failures() -> None:
    config = SequenceFitConfig()
    assert (config.serve.max_iter, config.serve.max_leaf_nodes, config.serve.random_seed) == (100, 7, 20260824)
    assert config.sequence.max_iter == 200
    assert config.sequence.max_leaf_nodes == 15
    assert config.sequence.random_seed == 20260905
    with pytest.raises(ValueError, match='three'):
        fit_sequence_models([training_video()])
    with pytest.raises(ValueError, match='distinct'):
        fit_sequence_models([training_video(group=group) for group in ('east', 'west', 'north')])
    with pytest.raises(ValueError, match='aligned'):
        LabelledRally('bad', (20, 10), (Half.TOP,))
