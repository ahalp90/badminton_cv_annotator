"""Compare review-ranking inputs with the original numerical builders."""

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence import OptionPool, RefinedSequences, StageChoices
from annotator.sequence.confidence import (
    BASE_FEATURE_NAMES,
    GAP_EVIDENCE_NAMES,
    build_confidence_features,
    discarded_contact_features,
    gap_contact_features,
    score_rally_confidence,
)
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import EditKind, SequenceEdit, SequenceOption
from annotator.sequence.features import PHYSICAL_FEATURE_NAMES, ServeEditInputs


@pytest.fixture
def reference() -> Iterator[Mapping[str, np.ndarray]]:
    path = Path(__file__).parent / "fixtures" / "annotator_reference" / "confidence.npz"
    with np.load(path, allow_pickle=False) as arrays:
        yield arrays


class TimingModel:
    classes_ = np.asarray([1, 0])

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        self.inputs.append(matrix.copy())
        scores = matrix[:, 1]
        return np.column_stack((scores, 1 - scores))


def example() -> tuple[SequenceOption, tuple[SequenceOption, ...], tuple[ContactEvent, ...], dict[int, np.ndarray]]:
    events = (ContactEvent(20, .8, Half.TOP), ContactEvent(80, .7, Half.TOP))
    sequence = ContactSequence(0, 0, 100, events)
    keep = SequenceOption(SequenceEdit(EditKind.KEEP, None, None, sequence), None, sequence)
    added = ContactEvent(50, .6, Half.BOT)
    inserted = SequenceOption(keep.edit, added, replace(sequence, events=(events[0], added, events[1])))
    serve_sequence = replace(sequence, events=(ContactEvent(10, .5, None), *events))
    serve = SequenceOption(SequenceEdit(EditKind.ADD, 10, None, serve_sequence), None, serve_sequence)
    candidates = (added, ContactEvent(65, .4, None), ContactEvent(80, .7, Half.TOP), ContactEvent(110, .2, None))
    physical = {event.frame: np.arange(len(PHYSICAL_FEATURE_NAMES), dtype=float) for event in candidates}
    physical[65][0] = np.nan
    return keep, (keep, inserted, serve), candidates, physical


@pytest.mark.parametrize("fps", [25., 30., 60.])
def test_base_features_match_original(fps: float, reference: Mapping[str, np.ndarray]) -> None:
    selected, options, candidates, physical = example()
    actual = discarded_contact_features(selected, options, [.8, .6, .5], candidates, physical, fps)
    assert tuple(reference["base_names"]) == BASE_FEATURE_NAMES
    np.testing.assert_allclose(actual, reference[f"base_{int(fps)}"][0], equal_nan=True)


@pytest.mark.parametrize("fps", [25., 30., 60.])
def test_gap_features_and_insertion_context_match_original(fps: float, reference: Mapping[str, np.ndarray]) -> None:
    selected, _, candidates, physical = example()
    model = TimingModel()
    actual = gap_contact_features(selected, candidates, physical, fps, model)
    assert tuple(reference["gap_evidence_names"]) == GAP_EVIDENCE_NAMES
    np.testing.assert_allclose(actual, reference[f"gap_{int(fps)}"][0], equal_nan=True)
    np.testing.assert_allclose(model.inputs[0], reference[f"insertion_{int(fps)}"], equal_nan=True)


def refined_example() -> RefinedSequences:
    selected, options, candidates, physical = example()
    # Finished sides alternate, although the two original guesses agreed.
    final_events = (selected.sequence.events[0], replace(selected.sequence.events[1], side=Half.BOT))
    final = replace(selected.sequence, start_frame=5, end_frame=95, events=final_events)
    pool = OptionPool(options, {0: candidates}, ServeEditInputs({}, np.empty((0, 0))),
                      np.empty((3, 0)), np.empty((3, 0)), physical)
    choices = StageChoices(np.asarray([.8, .6, .5]), {0: 0})
    return RefinedSequences((final,), final_events, pool, np.empty(3), choices, choices, choices)


@pytest.mark.parametrize("fps", [25., 30., 60.])
def test_finished_boundaries_keep_raw_side_votes(fps: float, reference: Mapping[str, np.ndarray]) -> None:
    refined = refined_example()
    model = TimingModel()
    features = build_confidence_features(refined, model, fps)
    np.testing.assert_allclose(features.base, reference[f"guarded_base_{int(fps)}"], equal_nan=True)
    np.testing.assert_allclose(features.gap, reference[f"guarded_gap_{int(fps)}"], equal_nan=True)
    np.testing.assert_allclose(model.inputs[0], reference[f"guarded_insertion_{int(fps)}"], equal_nan=True)
    assert features.span_ids == (0,)
    assert features.base[0, BASE_FEATURE_NAMES.index("adjacent_same_raw_side")] == 1.
    assert features.base[0, BASE_FEATURE_NAMES.index("output__section_duration_seconds")] == 90 / fps
    assert features.base[0, BASE_FEATURE_NAMES.index("output__start_to_first_seconds")] == 15 / fps
    assert features.base[0, BASE_FEATURE_NAMES.index("output__last_to_end_seconds")] == 15 / fps
    assert refined.sequences[0].events[1].side == Half.BOT


def test_current_fitted_models_score_without_editing_annotations() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    refined = refined_example()
    inputs = build_confidence_features(refined, TimingModel(), 30.)
    labels = np.tile([0, 1], 12)
    rng = np.random.default_rng(1)
    gap = HistGradientBoostingClassifier(max_iter=2, min_samples_leaf=2).fit(
        rng.normal(size=(24, inputs.gap.shape[1])), labels,
    )
    result = score_rally_confidence(refined, gap, TimingModel(), 30.)
    assert result.scores.shape == (1,)
    assert 0 <= result.scores[0] <= 1
    assert refined.sequences[0].events[1].side == Half.BOT


def test_empty_spans_and_missing_candidates(reference: Mapping[str, np.ndarray]) -> None:
    sequence = ContactSequence(0, 0, 100, ())
    option = SequenceOption(SequenceEdit(EditKind.KEEP, None, None, sequence), None, sequence)
    model = TimingModel()
    values = gap_contact_features(option, (), {}, 30., model)
    np.testing.assert_allclose(values, reference["empty_gap"][0], equal_nan=True)
    assert np.isnan(values[:8]).all()
    assert values[8:13] == (0., 0., 1., 1., 0.)
    assert not model.inputs
