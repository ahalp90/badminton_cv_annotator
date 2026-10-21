"""Run a fitted rally-layer stack and expose evidence for review ranking."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    initial_sequences,
)
from annotator.sequence.features import (
    SIDE_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    positive_probabilities,
    sequence_summary,
)
from annotator.sequence.rally_layers import (
    choose_rows,
    finish_rows,
    predict_layer,
)
from annotator.sequence.rally_options import (
    FEATURE_NAMES,
    RallyContext,
    RallyOptions,
    build_rally_options,
)
from annotator.sequence.refine import (
    build_option_pool,
    check_events_match_kept_candidates,
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
class LayerConfidence:
    """Review priority scores; they do not change annotations or approve rallies."""

    features: LayerReviewFeatures
    scores: np.ndarray


def review_feature_names(depth: int) -> tuple[str, ...]:
    return (
        *REVIEW_OPTION_NAMES,
        *(f'layer_{layer}__{name}' for layer in range(1, depth + 1) for name in ('option_score', 'top_margin')),
        *(f'final__{name}' for name in SUMMARY_FEATURE_NAMES),
    )


def score_layers(models: tuple[Any, ...], pool: RallyOptions) -> tuple[np.ndarray, ...]:
    scores = []
    for model in models:
        scores.append(predict_layer(model, pool, scores))
    return tuple(scores)


def predict_rally_options(models: RallyLayerModels, pool: RallyOptions) -> RallyLayerPrediction:
    scores = score_layers(models.layers, pool)
    chosen = choose_rows(pool, scores[-1])
    sequences, events = finish_rows(pool, chosen)
    return RallyLayerPrediction(sequences, events, pool, scores, chosen)


def refine_rally_layers(
    spans: Sequence[tuple[int, int]], events: tuple[ContactEvent, ...], scores: CandidateScores,
    features: np.ndarray, search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None], models: RallyLayerModels, frame_count: int | None = None,
) -> RallyLayerPrediction:
    check_events_match_kept_candidates(events, scores)
    initial = initial_sequences(spans, events)
    original = build_option_pool(
        initial, events, scores, features, search_intervals, fps=fps, side_for_frame=side_for_frame,
    )
    context = RallyContext(fps, initial, events, original, frame_count)
    pool = build_rally_options(context, models.max_edits, models.deletion_candidates)
    return predict_rally_options(models, pool)


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
        summaries.append(sequence_summary(bounded, pool.original.fps))
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


def score_layer_confidence(prediction: RallyLayerPrediction, model: Any | None) -> LayerConfidence:
    features = build_layer_review_features(prediction)
    if model is None:
        scores = prediction.scores[-1][list(prediction.chosen.values())]
    else:
        scores = positive_probabilities(model, features.matrix)
    return LayerConfidence(features, scores)
