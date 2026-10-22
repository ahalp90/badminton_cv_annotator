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

CONTEXT_FEATURE_NAMES = (
    'option_score', 'chosen_score', 'top_margin', 'option_minus_chosen',
    'normalised_rank', 'is_chosen', 'added_vs_chosen', 'deleted_vs_chosen',
)

REVIEW_OPTION_NAMES = (
    'added_count', 'deleted_count', *(f'after__{name}' for name in SIDE_FEATURE_NAMES),
    'unused_candidate_count', 'usable_candidate_count', 'positive_gap_count',
    'gaps_without_candidates', 'longest_gap_seconds', 'last_gap_has_candidate',
    'gap_probability_max', 'strongest_unused_left_gap_seconds', 'strongest_unused_right_gap_seconds',
)
REVIEW_OPTION_COLUMNS = tuple(FEATURE_NAMES.index(name) for name in REVIEW_OPTION_NAMES)


@dataclass(frozen=True)
class RallyLayerModels:
    """Fitted selectors and the proposal budgets used during training."""

    layers: tuple[Any, ...]
    max_edits: int = 3
    deletion_candidates: int = 6

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError('a rally stack needs at least one fitted layer')
        if self.max_edits < 0 or self.deletion_candidates < 0:
            raise ValueError('edit and deletion budgets must be non-negative')


@dataclass(frozen=True)
class RallyLayerPrediction:
    """Final annotations and the option scores that produced them."""

    sequences: tuple[ContactSequence, ...]
    events: tuple[ContactEvent, ...]
    pool: RallyOptions
    scores: tuple[np.ndarray, ...]
    chosen: dict[int, int]


@dataclass(frozen=True)
class LayerReviewFeatures:
    """Selected-option evidence with final boundaries, one row per output span."""

    span_ids: tuple[int, ...]
    matrix: np.ndarray


@dataclass(frozen=True)
class RallyReview:
    """Review priority scores; they do not change annotations or approve rallies."""

    features: LayerReviewFeatures
    scores: np.ndarray


def select_rally_sequences(
    spans: Sequence[tuple[int, int]], events: tuple[ContactEvent, ...], scores: CandidateScores,
    features: np.ndarray, search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None], models: RallyLayerModels, frame_count: int | None = None,
) -> RallyLayerPrediction:
    """Build rally alternatives from detector evidence and apply the fitted selectors."""
    check_events_match_kept_candidates(events, scores)
    initial = initial_sequences(spans, events)
    repairs = build_rally_repairs(
        initial, events, scores, features, search_intervals, fps=fps, side_for_frame=side_for_frame,
    )
    context = RallyContext(fps, initial, events, repairs, frame_count)
    pool = build_rally_options(context, models.max_edits, models.deletion_candidates)
    return predict_rally_options(models, pool)


def predict_rally_options(models: RallyLayerModels, pool: RallyOptions) -> RallyLayerPrediction:
    """Choose one option per rally, then finish its bounds and player sides."""
    scores = score_layers(models.layers, pool)
    chosen = choose_rows(pool, scores[-1])
    sequences, events = finish_rows(pool, chosen)
    return RallyLayerPrediction(sequences, events, pool, scores, chosen)


def score_layers(models: tuple[Any, ...], pool: RallyOptions) -> tuple[np.ndarray, ...]:
    """Score every option, passing earlier selectors' scores to later selectors."""
    scores = []
    for model in models:
        scores.append(predict_layer(model, pool, scores))
    return tuple(scores)


def predict_layer(model: Any, pool: RallyOptions, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Estimate each option's probability using its evidence and earlier scores."""
    return positive_probabilities(model, layer_features(pool, prior_scores))


def choose_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, int]:
    """Choose each rally's highest-scoring option, retaining pool order for ties."""
    return {span_id: int(rows[0]) for span_id, rows in ordered_span_rows(pool, scores).items()}


def finish_rows(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Apply selected contacts, extend rally bounds and fit alternating sides."""
    choices = {span_id: pool.sequences[row] for span_id, row in rows.items()}
    context = pool.context
    sequences, events = apply_choices(context.initial, context.events, choices)
    extended = extend_boundaries(sequences, events, context.fps, frame_count=context.frame_count)
    return alternate_sides(extended, events)


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
    """Replace each sequence's raw side guesses with its better alternating phase.

    A tied phase vote leaves that sequence's raw per-contact guesses unchanged.
    Contacts outside every sequence keep their raw guesses.

    :param sequences: Final sequences; each contact belongs to at most one.
    :param events: Full-stream contact events in frame order.
    :return: Sequences and full-stream events with the fitted sides.
    """
    side_by_frame: dict[int, Half | None] = {}
    for sequence in sequences:
        final_half = fit_alternation([event.side for event in sequence.events])
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
    return revised_sequences, revised


def layer_features(pool: RallyOptions, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    """Append each earlier layer's scores and choice context to the fixed evidence.

    Rank runs from zero for the first choice to one for the last. Ties use pool
    order. A singleton span has rank zero and margin zero. Added/deleted counts
    compare this option's contact frames with that earlier layer's chosen frames.
    """
    if not prior_scores:
        return pool.matrix
    frames = tuple({event.frame for event in sequence.events} for sequence in pool.sequences)
    blocks = [pool.matrix]
    for scores in prior_scores:
        choice_features = np.empty((len(pool.sequences), len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)
        for ranked in ordered_span_rows(pool, scores).values():
            chosen = int(ranked[0])
            chosen_score = scores[chosen]
            margin = chosen_score - scores[ranked[1]] if len(ranked) > 1 else 0.0
            for rank, row in enumerate(ranked):
                choice_features[row] = (
                    scores[row], chosen_score, margin, scores[row] - chosen_score,
                    rank / max(1, len(ranked) - 1), float(row == chosen),
                    len(frames[row] - frames[chosen]), len(frames[chosen] - frames[row]),
                )
        blocks.append(choice_features)
    return np.column_stack(blocks)


def ordered_span_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, np.ndarray]:
    """Rank every span's rows by descending score, retaining pool order for ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    return {span_id: rows[np.argsort(-scores[rows], kind='stable')] for span_id, rows in pool.span_rows.items()}


def review_feature_names(depth: int) -> tuple[str, ...]:
    return (
        *REVIEW_OPTION_NAMES,
        *(f'layer_{layer}__{name}' for layer in range(1, depth + 1) for name in ('option_score', 'top_margin')),
        *(f'final__{name}' for name in SUMMARY_FEATURE_NAMES),
    )


def build_layer_review_features(prediction: RallyLayerPrediction) -> LayerReviewFeatures:
    """Reuse raw option evidence; only the extra boundary summary uses final bounds."""
    pool = prediction.pool
    span_ids = tuple(prediction.chosen)
    selected = np.asarray(list(prediction.chosen.values()), dtype=np.int64)
    option_features = pool.matrix[np.ix_(selected, REVIEW_OPTION_COLUMNS)]
    final_by_span = {sequence.span_id: sequence for sequence in prediction.sequences}
    summaries, histories = [], []
    for span_id, row in prediction.chosen.items():
        raw = pool.sequences[row]
        final = final_by_span[span_id]
        bounded = replace(raw, start_frame=final.start_frame, end_frame=final.end_frame)
        summaries.append(sequence_summary(bounded, pool.context.fps))
        history = []
        for scores in prediction.scores:
            values = scores[pool.span_rows[span_id]]
            best = np.partition(values, -2)[-2:] if len(values) > 1 else values
            margin = float(best.max() - best.min())
            history.extend((float(scores[row]), margin))
        histories.append(history)
    boundaries = np.asarray(summaries, dtype=np.float64).reshape(len(span_ids), len(SUMMARY_FEATURE_NAMES))
    scores = np.asarray(histories, dtype=np.float64).reshape(len(span_ids), 2 * len(prediction.scores))
    return LayerReviewFeatures(span_ids, np.column_stack((option_features, scores, boundaries)))


def score_rally_review(prediction: RallyLayerPrediction, model: Any | None) -> RallyReview:
    features = build_layer_review_features(prediction)
    if model is None:
        scores = prediction.scores[-1][list(prediction.chosen.values())]
    else:
        scores = positive_probabilities(model, features.matrix)
    return RallyReview(features, scores)
