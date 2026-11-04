"""Select rally contacts, extend their bounds and fit striker order.

The final sequences also supply features for review ranking.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.outcomes.point_winner import (
    Half,
    alternating_player_sides,
    fit_alternation,
)
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    FinishedRally,
    apply_choices,
    events_between,
    initial_sequences,
    scale_frames,
)
from annotator.sequence.features import (
    SIDE_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    positive_probabilities,
    sequence_summary,
)
from annotator.sequence.rally_options import (
    FEATURE_NAMES,
    RallyContext,
    RallyOptions,
    build_rally_options,
)
from annotator.sequence.repairs import (
    build_rally_repairs,
    check_events_match_kept_candidates,
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


def select_rally_sequences(
    spans: Sequence[tuple[int, int]], events: tuple[ContactEvent, ...], scores: CandidateScores,
    features: np.ndarray, search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None], models: RallySelector, frame_count: int | None = None,
    bounds_supplied: bool = False,
) -> RallyPrediction:
    """Build rally alternatives from detector evidence and apply the fitted selectors."""
    check_events_match_kept_candidates(events, scores)
    initial = initial_sequences(spans, events)
    repairs = build_rally_repairs(
        initial, events, scores, features, search_intervals, fps=fps, side_for_frame=side_for_frame,
    )
    context = RallyContext(fps, initial, events, repairs, frame_count, bounds_supplied)
    pool = build_rally_options(context, models.max_edits, models.deletion_candidates)
    return predict_rally_options(models, pool)


def predict_rally_options(models: RallySelector, pool: RallyOptions) -> RallyPrediction:
    """Choose one option per rally, then finish its bounds and player sides."""
    scores = positive_probabilities(models.selector, pool.matrix)
    chosen = choose_rows(pool, scores)
    rallies, events = finish_rallies(pool, chosen)
    return RallyPrediction(rallies, events, pool, scores, chosen)


def choose_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, int]:
    """Choose each rally's highest-scoring option, retaining pool order for ties."""
    return {span_id: int(rows[0]) for span_id, rows in ordered_span_rows(pool, scores).items()}


def finish_rows(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Finish the chosen rallies through the same path used for training labels."""
    rallies, events = finish_rallies(pool, rows)
    return tuple(rally.sequence for rally in rallies), events


def finish_rallies(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...]]:
    """Finish selected rallies and retain the result of fitting their sides."""
    choices = {span_id: pool.sequences[row] for span_id, row in rows.items()}
    context = pool.context
    sequences, events = finish_choices(
        context.initial, context.events, choices, context.fps,
        frame_count=context.frame_count, bounds_supplied=context.bounds_supplied,
    )
    return fit_rally_sides(sequences, events)


def finish_choices(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent], choices: dict[int, ContactSequence], fps: float,
    *, frame_count: int | None = None, bounds_supplied: bool = False,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Rebuild owned contacts, then extend discovery bounds around chosen contacts.

    Chosen spans retain their context. Dataset clip padding belongs to export;
    discovery only adds the small contact margin checked against neighbours.
    """
    sequences, stream = apply_choices(initial, events, choices)
    if not bounds_supplied:
        sequences = extend_boundaries(sequences, stream, fps, frame_count=frame_count)
    return sequences, stream


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
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, fps)
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


def alternate_sides(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Fit alternating player sides to each sequence's per-contact guesses.

    Choose the assignment that agrees with more guesses. A tie keeps the original
    per-contact guesses.
    Contacts outside every sequence keep their raw guesses.

    :param sequences: Final sequences; each contact belongs to at most one.
    :param events: Full-stream contact events in frame order.
    :return: Sequences and full-stream events with the fitted sides.
    """
    rallies, revised = fit_rally_sides(sequences, events)
    return tuple(rally.sequence for rally in rallies), revised


def fit_rally_sides(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent],
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...]]:
    """Fit each rally once and keep its resolved final side with its sequence."""
    side_by_frame: dict[int, Half | None] = {}
    final_halves = []
    for sequence in sequences:
        final_half = fit_alternation([event.side for event in sequence.events])
        final_halves.append(final_half)
        if final_half is None:
            continue
        fitted = alternating_player_sides(final_half, len(sequence.events))
        for event, side in zip(sequence.events, fitted, strict=True):
            side_by_frame[event.frame] = side
    revised = tuple(replace(event, side=side_by_frame.get(event.frame, event.side)) for event in events)
    by_frame = {event.frame: event for event in revised}
    revised_sequences = tuple(
        replace(sequence, events=tuple(by_frame[event.frame] for event in sequence.events)) for sequence in sequences
    )
    rallies = tuple(
        FinishedRally(sequence, final_half)
        for sequence, final_half in zip(revised_sequences, final_halves, strict=True)
    )
    return rallies, revised


def ordered_span_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, np.ndarray]:
    """Rank every span's rows by descending score, retaining pool order for ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    return {span_id: rows[np.argsort(-scores[rows], kind='stable')] for span_id, rows in pool.span_rows.items()}


def build_review_features(prediction: RallyPrediction) -> RallyReviewFeatures:
    """Reuse raw option evidence; only the extra boundary summary uses final bounds."""
    pool = prediction.pool
    span_ids = tuple(prediction.chosen)
    selected = np.asarray(list(prediction.chosen.values()), dtype=np.int64)
    option_features = pool.matrix[np.ix_(selected, REVIEW_OPTION_COLUMNS)]
    final_by_span = {sequence.span_id: sequence for sequence in prediction.sequences}
    summaries, scores = [], []
    for span_id, row in prediction.chosen.items():
        raw = pool.sequences[row]
        final = final_by_span[span_id]
        bounded = replace(raw, start_frame=final.start_frame, end_frame=final.end_frame)
        summaries.append(sequence_summary(bounded, pool.context.fps))
        values = prediction.scores[pool.span_rows[span_id]]
        best = np.partition(values, -2)[-2:] if len(values) > 1 else values
        margin = float(best.max() - best.min())
        scores.append((float(prediction.scores[row]), margin))
    boundaries = np.asarray(summaries, dtype=np.float64).reshape(len(span_ids), len(SUMMARY_FEATURE_NAMES))
    scores = np.asarray(scores, dtype=np.float64).reshape(len(span_ids), 2)
    return RallyReviewFeatures(span_ids, np.column_stack((option_features, scores, boundaries)))


def score_rally_review(prediction: RallyPrediction, model: Any) -> RallyReview:
    features = build_review_features(prediction)
    scores = positive_probabilities(model, features.matrix)
    return RallyReview(features, scores)


def score_selector_review(prediction: RallyPrediction) -> RallyReview:
    """Expose selector scores for experimental ranking comparisons."""
    features = build_review_features(prediction)
    scores = prediction.scores[list(prediction.chosen.values())]
    return RallyReview(features, scores)
