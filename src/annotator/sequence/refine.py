"""Refine one video's rally contact sequences through the fitted chooser stages.

The stages run in a fixed order, and each later chooser starts from the
previous stage's choices:

1. the whole-sequence chooser picks a keep, serve-repair or deletion edit;
2. the later-contact chooser may add one missed contact, but only by a clear margin;
3. the scored-insertion chooser repeats that choice with a separate score for
   the added contact itself;
4. rally bounds widen around the chosen contacts without changing membership;
5. strikers alternate across each finished sequence.

All stages share one option pool and one set of feature columns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from annotator.outcomes.point_winner import Half
from annotator.sequence.boundaries import extend_boundaries
from annotator.sequence.candidates import (
    serve_candidates,
    shortlist_later_frames,
    shortlist_serve_frames,
)
from annotator.sequence.choices import (
    MIN_EDIT_ADVANTAGE,
    choose_edits,
    choose_with_margin,
)
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    initial_sequences,
)
from annotator.sequence.edits import (
    SequenceOption,
    apply_choices,
    option_pool,
    sequence_edits,
)
from annotator.sequence.features import (
    SERVE_EDIT_FEATURE_NAMES,
    ServeEditInputs,
    ServeEditKey,
    insertion_features,
    physical_rows,
    positive_probabilities,
    sequence_features,
    serve_edit_inputs,
    serve_score_features,
)
from annotator.sequence.sides import alternate_sides

MINIMUM_EDIT_SCORE = 0.0


@dataclass(frozen=True)
class SequenceModels:
    """Three sequence choosers sharing the same serve and insertion models."""

    whole_sequence: Any
    later_contact: Any
    scored_insertion: Any
    insertion: Any
    serve_summary: Any
    serve_physical: Any


@dataclass(frozen=True)
class StageChoices:
    """One chooser stage's scores over the shared option pool and its picks."""

    scores: np.ndarray  # (option,); NaN for options this stage does not score
    chosen: dict[int, int]  # pool row by span ID


@dataclass(frozen=True)
class OptionPool:
    """Every sequence option for one video and the model inputs describing them.

    Every chooser stage scores rows of these matrices, so later training can
    reuse them directly.
    """

    options: tuple[SequenceOption, ...]
    later_candidates: dict[int, tuple[ContactEvent, ...]]
    serve_inputs: ServeEditInputs
    sequence_features: np.ndarray  # (option, SEQUENCE_FEATURE_NAMES)
    insertion_features: np.ndarray  # (option, INSERTION_FEATURE_NAMES)
    physical: dict[int, np.ndarray]  # physical feature row by frame, for every frame read

    @property
    def has_added_contact(self) -> np.ndarray:
        return np.asarray([option.added_contact is not None for option in self.options], dtype=bool)


@dataclass(frozen=True)
class RefinedSequences:
    """Final sequences plus the option pool and scores behind them.

    ``sequences`` and ``events`` are the final output. The pool and stage
    choices let confidence ranking reuse each stage's work.
    """

    sequences: tuple[ContactSequence, ...]
    events: tuple[ContactEvent, ...]  # full stream, including contacts outside every sequence
    pool: OptionPool
    insertion_scores: np.ndarray  # (option,); NaN for options without an added contact
    whole_sequence: StageChoices
    later_contact: StageChoices
    scored_insertion: StageChoices


def check_events_match_kept_candidates(events: Sequence[ContactEvent], scores: CandidateScores) -> None:
    if np.any(np.diff(scores.frames) <= 0):
        raise ValueError("candidate scores must be in strictly increasing frame order")
    kept_frames = scores.frames[scores.kept].tolist()
    kept_probabilities = scores.probabilities[scores.kept].tolist()
    event_frames = [event.frame for event in events]
    event_probabilities = [event.probability for event in events]
    if event_frames != kept_frames or event_probabilities != kept_probabilities:
        raise ValueError("initial events must be exactly the kept candidates")


def score_serve_edits(models: SequenceModels, inputs: ServeEditInputs) -> dict[ServeEditKey, tuple[float, float]]:
    summary = positive_probabilities(models.serve_summary, inputs.matrix[:, :len(SERVE_EDIT_FEATURE_NAMES)])
    physical = positive_probabilities(models.serve_physical, inputs.matrix)
    scores = {}
    for key, row in inputs.rows_by_key.items():
        scores[key] = (float(summary[row]), float(physical[row]))
    return scores


def build_option_pool(
    initial: Sequence[ContactSequence],
    events: Sequence[ContactEvent],
    candidate_scores: CandidateScores,
    contact_features: np.ndarray,
    search_intervals: Sequence[tuple[int, int]],
    *,
    fps: float,
    side_for_frame: Callable[[int], Half | None],
) -> OptionPool:
    """Shortlist candidates, build every sequence option and describe it once.

    :param initial: Initial sequences in chronological order.
    :param events: Detector-selected contacts; they must be the kept candidates.
    :param candidate_scores: Every scored contact candidate, in frame order.
    :param contact_features: Structured contact feature rows, one per searched frame.
    :param search_intervals: Half-open intervals that the candidate interval IDs index.
    :param fps: Source frame rate.
    :param side_for_frame: Raw court-half guess, asked once per shortlisted
        frame that is not already a contact.
    :return: The shared option pool and its model inputs.
    """
    events_by_frame = {event.frame: event for event in events}
    serve_rows = shortlist_serve_frames(initial, candidate_scores, search_intervals, fps)
    later_frames = {sequence.span_id: shortlist_later_frames(sequence, candidate_scores, fps) for sequence in initial}
    serve_frames = {int(candidate_scores.frames[row]) for rows in serve_rows.values() for row in rows}
    later_frame_set = {frame for frames in later_frames.values() for frame in frames}
    sides = {}
    for frame in sorted((serve_frames | later_frame_set) - set(events_by_frame)):
        side = side_for_frame(frame)
        sides[frame] = None if side is None else Half(side)
    probability_by_frame = dict(zip(candidate_scores.frames.tolist(), candidate_scores.probabilities.tolist()))
    later_candidates = {}
    for span_id, frames in later_frames.items():
        contacts = [ContactEvent(frame, probability_by_frame[frame], sides[frame]) for frame in frames]
        later_candidates[span_id] = tuple(contacts)
    serve_by_span = serve_candidates(initial, serve_rows, candidate_scores, events_by_frame, sides, fps)

    physical = physical_rows(contact_features, set(events_by_frame) | serve_frames | later_frame_set)
    edits_by_span = {}
    previous_end = -1
    for sequence in initial:
        serve_options = serve_by_span.get(sequence.span_id, ())
        edits_by_span[sequence.span_id] = sequence_edits(sequence, serve_options, events, previous_end)
        previous_end = sequence.end_frame
    options = option_pool(edits_by_span, later_candidates, fps)
    serve_inputs = serve_edit_inputs(serve_by_span, physical)
    initial_by_span = {sequence.span_id: sequence for sequence in initial}
    return OptionPool(
        options=options,
        later_candidates=later_candidates,
        serve_inputs=serve_inputs,
        sequence_features=sequence_features(options, initial_by_span, serve_inputs, events_by_frame, physical, fps),
        insertion_features=insertion_features(options, physical, fps),
        physical=physical,
    )


def choose_sequences(
    pool: OptionPool, models: SequenceModels,
) -> tuple[StageChoices, StageChoices, StageChoices, np.ndarray]:
    """Run the three chooser stages in order, each starting from the previous choices.

    :return: Whole-sequence, later-contact and scored-insertion choices, then
        the added-contact score of each option (NaN without an added contact).
    """
    options = pool.options
    has_added_contact = pool.has_added_contact
    whole_rows = np.flatnonzero(~has_added_contact)
    serve_features = serve_score_features(pool.options, score_serve_edits(models, pool.serve_inputs))
    whole_matrix = np.column_stack((pool.sequence_features, serve_features))
    whole_scores = np.full(len(options), np.nan)
    whole_scores[whole_rows] = positive_probabilities(models.whole_sequence, whole_matrix[whole_rows])
    whole = StageChoices(whole_scores, choose_edits(options, whole_scores, whole_rows, MINIMUM_EDIT_SCORE))

    later_matrix = np.column_stack((whole_matrix, pool.insertion_features))
    later_scores = positive_probabilities(models.later_contact, later_matrix)
    later = StageChoices(later_scores, choose_with_margin(options, later_scores, whole.chosen, MIN_EDIT_ADVANTAGE))

    insertion_scores = np.full(len(options), np.nan)
    added_rows = pool.insertion_features[has_added_contact]
    insertion_scores[has_added_contact] = positive_probabilities(models.insertion, added_rows)
    scored_matrix = np.column_stack((later_matrix, insertion_scores))
    scored_scores = positive_probabilities(models.scored_insertion, scored_matrix)
    scored = StageChoices(scored_scores, choose_with_margin(options, scored_scores, later.chosen, MIN_EDIT_ADVANTAGE))
    return whole, later, scored, insertion_scores


def refine_contact_sequences(
    initial_spans: Sequence[tuple[int, int]],
    initial_events: Sequence[ContactEvent],
    candidate_scores: CandidateScores,
    contact_features: np.ndarray,
    search_intervals: Sequence[tuple[int, int]],
    *,
    fps: float,
    side_for_frame: Callable[[int], Half | None],
    models: SequenceModels,
    frame_count: int | None = None,
) -> RefinedSequences:
    """Run the three choosers, boundary widening and side alternation for one video.

    :param initial_spans: Heuristic half-open rally spans, chronological and disjoint.
    :param initial_events: Detector-selected contacts in frame order, including
        any outside a span, with raw side guesses.
    :param candidate_scores: Every scored contact candidate, in frame order.
    :param contact_features: Structured contact feature rows, one per searched frame.
    :param search_intervals: Half-open search intervals that ``candidate_scores``
        interval IDs index.
    :param fps: Source frame rate.
    :param side_for_frame: Raw court-half guess for a candidate frame; asked once
        per shortlisted frame that is not already a contact.
    :param models: Fitted chooser models.
    :return: Final sequences and events, plus the shared option pool and stage choices.
    """
    events = tuple(initial_events)
    check_events_match_kept_candidates(events, candidate_scores)
    initial = initial_sequences(initial_spans, events)
    pool = build_option_pool(
        initial, events, candidate_scores, contact_features, search_intervals, fps=fps, side_for_frame=side_for_frame,
    )
    return finish_sequences(initial, events, pool, models, fps=fps, frame_count=frame_count)


def finish_sequences(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent], pool: OptionPool,
    models: SequenceModels, *, fps: float, frame_count: int | None = None,
) -> RefinedSequences:
    """Apply shared choices, boundary rules and sides in both training and inference."""
    whole, later, scored, insertion_scores = choose_sequences(pool, models)
    chosen_sequences = {span_id: pool.options[row].sequence for span_id, row in scored.chosen.items()}
    chosen, final_events = apply_choices(initial, events, chosen_sequences)
    extended = extend_boundaries(chosen, final_events, fps, frame_count=frame_count)
    sequences, final_events = alternate_sides(extended, final_events)
    return RefinedSequences(sequences, final_events, pool, insertion_scores, whole, later, scored)
