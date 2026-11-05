"""Select rally contacts, extend their bounds and fit striker order.

The final sequences also supply features for review ranking.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.config import frames_at_fps
from annotator.outcomes.point_winner import (
    alternating_player_sides,
    fit_alternation,
)
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    FinishedRally,
    events_between,
)
from annotator.sequence.features import (
    SIDE_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    positive_probabilities,
    sequence_summary,
)
from annotator.sequence.rally_options import (
    FEATURE_NAMES,
    RallyOptions,
)

BOUNDARY_PADDING_AT_30_FPS = 10

REVIEW_OPTION_NAMES = (
    'added_count', 'deleted_count', *(f'after__{name}' for name in SIDE_FEATURE_NAMES),
    'unused_candidate_count', 'usable_candidate_count', 'positive_gap_count',
    'gaps_without_candidates', 'longest_gap_seconds', 'last_gap_has_candidate',
    'gap_probability_max', 'strongest_unused_left_gap_seconds', 'strongest_unused_right_gap_seconds',
)
REVIEW_OPTION_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in REVIEW_OPTION_NAMES)
REVIEW_FEATURE_NAMES = (
    *REVIEW_OPTION_NAMES,
    'option_score', 'top_margin',
    *(f'final__{name}' for name in SUMMARY_FEATURE_NAMES),
)


@dataclass(frozen=True)
class RallySelector:
    """Fitted selector and the proposal budgets used during training."""

    selector: Any
    max_edits: int = 3
    deletion_candidates: int = 6

    def __post_init__(self) -> None:
        if self.max_edits < 0 or self.deletion_candidates < 0:
            raise ValueError('edit and deletion budgets must be non-negative')


@dataclass(frozen=True)
class RallyPrediction:
    """Final annotations and the option scores that produced them."""

    rallies: tuple[FinishedRally, ...]
    events: tuple[ContactEvent, ...]
    pool: RallyOptions
    scores: np.ndarray
    chosen: dict[int, int]

    @property
    def sequences(self) -> tuple[ContactSequence, ...]:
        """Selected sequences for scoring and stream export."""
        return tuple(rally.sequence for rally in self.rallies)


@dataclass(frozen=True)
class RallyReviewFeatures:
    """Selected-option evidence with final boundaries, one row per output span."""

    span_ids: tuple[int, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class RallyReview:
    """Review priority scores; they do not change annotations or approve rallies."""

    features: RallyReviewFeatures
    scores: np.ndarray


def predict_rally_options(models: RallySelector, pool: RallyOptions) -> RallyPrediction:
    """Choose one option per rally, then finish its bounds and player sides."""
    scores = positive_probabilities(models.selector, pool.matrix)
    chosen = choose_rows(pool, scores)
    rallies, events = finish_rallies(pool, chosen)
    return RallyPrediction(rallies, events, pool, scores, chosen)


def choose_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, int]:
    """Choose each rally's highest-scoring option, retaining pool order for ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    chosen = {}
    for span_id, rows in pool.span_rows.items():
        chosen[span_id] = int(rows[np.argmax(scores[rows])])
    return chosen


def finish_rallies(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...]]:
    """Finish selected rallies and rebuild their full contact stream.

    Each chosen rally owns the contacts inside its bounds. Detector contacts
    outside every chosen rally survive in the full stream.
    """
    context = pool.context
    sequences = tuple(pool.sequences[rows[initial.span_id]] for initial in context.initial)
    events = [
        event for event in context.events
        if not any(sequence.start_frame <= event.frame < sequence.end_frame for sequence in sequences)
    ]
    events.extend(event for sequence in sequences for event in sequence.events)
    events.sort(key=lambda event: event.frame)
    if not context.bounds_supplied:
        sequences = extend_boundaries(sequences, events, context.fps, frame_count=context.frame_count)
    rallies = fit_rally_sides(sequences)
    fitted = {event.frame: event for rally in rallies for event in rally.sequence.events}
    return rallies, tuple(fitted.get(event.frame, event) for event in events)


def proposed_bounds(sequence: ContactSequence, padding: int) -> list[int]:
    if not sequence.events:
        return [sequence.start_frame, sequence.end_frame]
    first_frame = min(event.frame for event in sequence.events)
    last_frame = max(event.frame for event in sequence.events)
    start = max(0, min(sequence.start_frame, first_frame - padding))
    end = max(sequence.end_frame, last_frame + padding + 1)
    return [start, end]


def extend_boundaries(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent], fps: float,
    *, frame_count: int | None = None,
) -> tuple[ContactSequence, ...]:
    """Pad each sequence around its first and last contact, keeping its contacts.

    Padding never moves a bound inward and stops at the neighbouring
    sequences' current bounds. When both neighbours reach into one gap, the
    gap's midpoint divides it. A sequence whose padded bounds would take in or
    lose any full-stream contact keeps its current bounds instead.

    :param sequences: Chosen sequences in chronological order, disjoint.
    :param events: Full-stream contact events in frame order.
    :param fps: Source frame rate.
    :param frame_count: Video length; bound extensions stop at this exclusive end.
    :return: Sequences in the same order, with the same contacts.
    """
    padding = frames_at_fps(BOUNDARY_PADDING_AT_30_FPS, fps)
    bounds = [proposed_bounds(sequence, padding) for sequence in sequences]
    for position in range(len(sequences)):
        if frame_count is not None:
            bounds[position][1] = min(bounds[position][1], frame_count)
        if position:
            bounds[position][0] = max(bounds[position][0], sequences[position - 1].end_frame)
        if position + 1 < len(sequences):
            bounds[position][1] = min(bounds[position][1], sequences[position + 1].start_frame)
    for left_position in range(len(sequences) - 1):
        right_position = left_position + 1
        if bounds[left_position][1] <= bounds[right_position][0]:
            continue
        midpoint = (sequences[left_position].end_frame + sequences[right_position].start_frame) // 2
        bounds[left_position][1] = min(bounds[left_position][1], midpoint)
        bounds[right_position][0] = max(bounds[right_position][0], midpoint)

    extended = []
    for sequence, (start_frame, end_frame) in zip(sequences, bounds, strict=True):
        if events_between(events, start_frame, end_frame) == sequence.events:
            extended.append(replace(sequence, start_frame=start_frame, end_frame=end_frame))
        else:
            extended.append(sequence)
    return tuple(extended)


def fit_rally_sides(sequences: Sequence[ContactSequence]) -> tuple[FinishedRally, ...]:
    """Fit each rally once and keep its resolved final side with its sequence."""
    rallies = []
    for sequence in sequences:
        final_half = fit_alternation([event.side for event in sequence.events])
        if final_half is not None:
            fitted = alternating_player_sides(final_half, len(sequence.events))
            events = tuple(replace(event, side=side) for event, side in zip(sequence.events, fitted, strict=True))
            sequence = replace(sequence, events=events)
        rallies.append(FinishedRally(sequence, final_half))
    return tuple(rallies)


def build_review_features(prediction: RallyPrediction) -> RallyReviewFeatures:
    """Measure the selected proposal and its final rally bounds for review ranking.

    Keep the proposed contact sides so the ranker sees uncertainty that final
    player alternation may resolve. Use final bounds for the timing summaries.
    """
    pool = prediction.pool
    span_ids = tuple(prediction.chosen)
    selected = np.asarray(list(prediction.chosen.values()), dtype=np.int64)
    option_features = pool.matrix[np.ix_(selected, REVIEW_OPTION_COLUMNS)]
    final_by_span = {sequence.span_id: sequence for sequence in prediction.sequences}
    summaries, scores = [], []
    for span_id, row in prediction.chosen.items():
        proposed_sequence = pool.sequences[row]
        final_sequence = final_by_span[span_id]
        proposal_with_final_bounds = replace(
            proposed_sequence, start_frame=final_sequence.start_frame, end_frame=final_sequence.end_frame,
        )
        summaries.append(sequence_summary(proposal_with_final_bounds, pool.context.fps))
        values = prediction.scores[pool.span_rows[span_id]]
        best = np.partition(values, -2)[-2:] if len(values) > 1 else values
        margin = float(best.max() - best.min())
        scores.append((float(prediction.scores[row]), margin))
    sequence_summaries = np.asarray(summaries, dtype=np.float64).reshape(len(span_ids), len(SUMMARY_FEATURE_NAMES))
    scores = np.asarray(scores, dtype=np.float64).reshape(len(span_ids), 2)
    return RallyReviewFeatures(span_ids, np.column_stack((option_features, scores, sequence_summaries)))


def score_rally_review(prediction: RallyPrediction, model: Any) -> RallyReview:
    features = build_review_features(prediction)
    scores = positive_probabilities(model, features.matrix)
    return RallyReview(features, scores)


def score_selector_review(prediction: RallyPrediction) -> RallyReview:
    """Expose selector scores for experimental ranking comparisons."""
    features = build_review_features(prediction)
    scores = prediction.scores[list(prediction.chosen.values())]
    return RallyReview(features, scores)
