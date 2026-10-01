"""The full refinement chain runs its stages in order from in-memory inputs.

Stub models score options from named feature columns, so each test states
which edit every stage should prefer and why.
"""

from collections.abc import Callable

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence import (
    CandidateScores,
    ContactEvent,
    SequenceModels,
    refine_contact_sequences,
)
from annotator.sequence.features import (
    INSERTION_FEATURE_NAMES,
    PHYSICAL_FEATURE_NAMES,
    SEQUENCE_FEATURE_NAMES,
    SERVE_EDIT_FEATURE_NAMES,
    SERVE_SCORE_FEATURE_NAMES,
)

TOP = Half.TOP
BOT = Half.BOT
DELETE = SEQUENCE_FEATURE_NAMES.index("action__delete")
ADD = SEQUENCE_FEATURE_NAMES.index("action__add")
SERVE_ALREADY_KEPT = SEQUENCE_FEATURE_NAMES.index("existing_start__candidate_already_kept")
DELETED_SCORE = SEQUENCE_FEATURE_NAMES.index("deleted_contact_score")
STATIC_WIDTH = len(SEQUENCE_FEATURE_NAMES) + len(SERVE_SCORE_FEATURE_NAMES)
HAS_ADDED = STATIC_WIDTH + INSERTION_FEATURE_NAMES.index("has_later_insertion")
ADDED_SCORE = STATIC_WIDTH + INSERTION_FEATURE_NAMES.index("later_score")
SERVE_PHYSICAL_WIDTH = len(SERVE_EDIT_FEATURE_NAMES) + 2 * len(PHYSICAL_FEATURE_NAMES)

# (frame, probability, kept, raw side); every row sits in search interval 0.
CANDIDATES = (
    (10, .5, False, TOP),
    (25, .6, False, None),
    (40, .9, True, TOP),
    (70, .6, False, BOT),
    (100, .8, True, BOT),
    (180, .85, True, TOP),  # kept between rallies; serve candidate for the third
    (205, .4, False, BOT),
    (220, .9, True, BOT),
    (250, .5, False, TOP),
    (280, .7, True, TOP),
    (310, .45, False, None),
    (390, .75, True, TOP),  # kept after the last rally
)
SPANS = ((30, 150), (160, 170), (200, 330))


class ColumnModel:
    """Binary classifier stub that scores rows with a supplied function."""

    classes_ = np.array([0, 1])

    def __init__(self, width: int, score: Callable[[np.ndarray], np.ndarray]):
        self.n_features_in_ = width
        self.score = score

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        assert matrix.shape[1] == self.n_features_in_
        positive = self.score(matrix)
        return np.column_stack((1.0 - positive, positive))


def _constant(width: int, value: float) -> ColumnModel:
    return ColumnModel(width, lambda matrix: np.full(len(matrix), value))


def _whole_score(matrix: np.ndarray) -> np.ndarray:
    """Prefer deletions, and adding a serve the detector already kept even more."""
    adds_kept_serve = matrix[:, ADD] * np.nan_to_num(matrix[:, SERVE_ALREADY_KEPT])
    return .5 + .1 * matrix[:, DELETE] + .2 * adds_kept_serve


def _later_score(matrix: np.ndarray) -> np.ndarray:
    """Add a later contact: clearly when it is strong, only slightly when it is weak."""
    has_added = matrix[:, HAS_ADDED] == 1.0
    strong_added = has_added & (matrix[:, ADDED_SCORE] >= .55)
    added_bonus = np.where(strong_added, .1, np.where(has_added, .04, 0.0))
    return _whole_score(matrix) + added_bonus


def _scored_insertion_score(matrix: np.ndarray) -> np.ndarray:
    """Slightly favour deleting a weaker contact.

    Weak added contacts also gain from the separate added-contact score.
    """
    has_added = matrix[:, HAS_ADDED] == 1.0
    weak_added = has_added & (matrix[:, ADDED_SCORE] < .55)
    deletes_weaker = matrix[:, DELETED_SCORE] < .85
    added_contact_score = np.nan_to_num(matrix[:, -1])
    return _later_score(matrix) + .04 * deletes_weaker + .1 * added_contact_score * weak_added


def _models() -> SequenceModels:
    insertion_width = len(INSERTION_FEATURE_NAMES)
    return SequenceModels(
        whole_sequence=ColumnModel(STATIC_WIDTH, _whole_score),
        later_contact=ColumnModel(STATIC_WIDTH + insertion_width, _later_score),
        scored_insertion=ColumnModel(STATIC_WIDTH + insertion_width + 1, _scored_insertion_score),
        insertion=_constant(insertion_width, .9),
        serve_summary=_constant(len(SERVE_EDIT_FEATURE_NAMES), .5),
        serve_physical=_constant(SERVE_PHYSICAL_WIDTH, .5),
    )


def _contact_features(frames: list[int]) -> np.ndarray:
    dtype = [("frame", "i4"), *((name, "f8") for name in PHYSICAL_FEATURE_NAMES)]
    features = np.zeros(len(frames), dtype=dtype)
    features["frame"] = frames
    features[PHYSICAL_FEATURE_NAMES[0]] = np.nan
    return features


def _refine(candidates, spans, side_for_frame, events=None):
    frames = [row[0] for row in candidates]
    scores = CandidateScores(
        np.asarray(frames, dtype=np.int32),
        np.zeros(len(candidates), dtype=np.int32),
        np.asarray([row[1] for row in candidates], dtype=np.float64),
        np.asarray([row[2] for row in candidates], dtype=bool),
    )
    if events is None:
        events = [ContactEvent(frame, probability, side) for frame, probability, kept, side in candidates if kept]
    return refine_contact_sequences(
        spans, events, scores, _contact_features(frames), [(0, 400)],
        fps=30.0, side_for_frame=side_for_frame, models=_models(),
    )


def _chosen(refined, stage) -> dict[int, tuple[str, int | None, list[int]]]:
    summary = {}
    for span_id, row in stage.chosen.items():
        option = refined.pool.options[row]
        added = None if option.added_contact is None else option.added_contact.frame
        summary[span_id] = (str(option.edit.kind), added, [event.frame for event in option.sequence.events])
    return summary


def test_each_stage_starts_from_the_previous_stage_choices():
    raw_sides = {row[0]: row[3] for row in CANDIDATES}
    asked = []

    def side_for_frame(frame: int) -> Half | None:
        asked.append(frame)
        return raw_sides[frame]

    refined = _refine(CANDIDATES, SPANS, side_for_frame)
    assert asked == [10, 25, 70, 205, 250, 310]
    assert _chosen(refined, refined.whole_sequence) == {
        0: ("delete", None, [100]), 1: ("keep", None, []), 2: ("add", None, [180, 220, 280]),
    }
    # The weak later contact in rally 2 gains 0.04, short of the 0.05 margin.
    assert _chosen(refined, refined.later_contact) == {
        0: ("delete", 70, [70, 100]), 1: ("keep", None, []), 2: ("add", None, [180, 220, 280]),
    }
    # Rally 0 keeps its later-contact choice: deleting contact 100 instead gains
    # only 0.04 over it, although it clears the margin over the first choice.
    assert _chosen(refined, refined.scored_insertion) == {
        0: ("delete", 70, [70, 100]), 1: ("keep", None, []), 2: ("add", 250, [180, 220, 250, 280]),
    }
    has_added = refined.pool.has_added_contact
    assert np.array_equal(np.isnan(refined.whole_sequence.scores), has_added)
    assert np.array_equal(np.isnan(refined.insertion_scores), ~has_added)


def test_finished_stream_keeps_out_of_span_contacts_and_tied_raw_sides():
    refined = _refine(CANDIDATES, SPANS, {row[0]: row[3] for row in CANDIDATES}.__getitem__)
    bounds = [(sequence.start_frame, sequence.end_frame) for sequence in refined.sequences]
    assert bounds == [(30, 150), (160, 170), (170, 330)]
    sides = [[(event.frame, event.side) for event in sequence.events] for sequence in refined.sequences]
    assert sides == [
        [(70, BOT), (100, BOT)],  # tied phases keep the raw guesses
        [],
        [(180, TOP), (220, BOT), (250, TOP), (280, BOT)],
    ]
    assert [(event.frame, event.side) for event in refined.events] == [
        (70, BOT), (100, BOT), (180, TOP), (220, BOT), (250, TOP), (280, BOT), (390, TOP),
    ]


def _no_side_lookup(frame: int) -> Half | None:
    raise AssertionError(f"side requested for frame {frame}")


def test_video_without_contacts_keeps_every_rally():
    candidates = [(15, .2, False, None), (240, .3, False, None)]
    refined = _refine(candidates, [(0, 100), (200, 300)], _no_side_lookup)
    assert [(sequence.start_frame, sequence.end_frame, sequence.events) for sequence in refined.sequences] == [
        (0, 100, ()), (200, 300, ()),
    ]
    assert refined.events == ()
    assert [str(option.edit.kind) for option in refined.pool.options] == ["keep", "keep"]


def test_video_without_rallies_keeps_its_contacts():
    refined = _refine([(15, .9, True, TOP), (240, .8, True, None)], [], _no_side_lookup)
    assert refined.sequences == ()
    assert [(event.frame, event.side) for event in refined.events] == [(15, TOP), (240, None)]


def test_initial_events_must_be_the_kept_candidates():
    candidates = [(15, .9, True, TOP), (240, .8, False, None)]
    with pytest.raises(ValueError, match="exactly the kept candidates"):
        _refine(candidates, [], _no_side_lookup, events=[ContactEvent(240, .8, None)])




@pytest.mark.parametrize("earlier", [[], [(10, .5, False, TOP)]])
def test_single_contact_rally_runs_with_fewer_serve_candidates(earlier):
    candidates = [*earlier, (40, .9, True, TOP)]
    refined = _refine(candidates, [(30, 100)], {row[0]: row[3] for row in candidates}.__getitem__)
    assert len(refined.sequences) == 1
    assert np.isfinite(refined.scored_insertion.scores).all()
    assert all(event.frame in {row[0] for row in candidates} for event in refined.events)
