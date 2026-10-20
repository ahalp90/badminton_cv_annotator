"""Score rally options with fitted correction layers and finish their choices."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from annotator.sequence.boundaries import extend_boundaries
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import apply_choices
from annotator.sequence.features import positive_probabilities
from annotator.sequence.rally_options import RallyOptions
from annotator.sequence.sides import alternate_sides

CONTEXT_FEATURE_NAMES = (
    'option_score', 'chosen_score', 'top_margin', 'option_minus_chosen',
    'normalised_rank', 'is_chosen', 'added_vs_chosen', 'deleted_vs_chosen',
)


def ordered_span_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, np.ndarray]:
    """Rank every span's rows by descending score, retaining pool order for ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    return {span_id: rows[np.argsort(-scores[rows], kind='stable')] for span_id, rows in pool.span_rows.items()}


def choose_rows(pool: RallyOptions, scores: np.ndarray) -> dict[int, int]:
    return {span_id: int(rows[0]) for span_id, rows in ordered_span_rows(pool, scores).items()}


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
        context = np.empty((len(pool.sequences), len(CONTEXT_FEATURE_NAMES)), dtype=np.float32)
        for ranked in ordered_span_rows(pool, scores).values():
            chosen = int(ranked[0])
            chosen_score = scores[chosen]
            margin = chosen_score - scores[ranked[1]] if len(ranked) > 1 else 0.0
            for rank, row in enumerate(ranked):
                context[row] = (
                    scores[row], chosen_score, margin, scores[row] - chosen_score,
                    rank / max(1, len(ranked) - 1), float(row == chosen),
                    len(frames[row] - frames[chosen]), len(frames[chosen] - frames[row]),
                )
        blocks.append(context)
    return np.column_stack(blocks)


def predict_layer(model: Any, pool: RallyOptions, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    return positive_probabilities(model, layer_features(pool, prior_scores))


def finish_selected(
    pool: RallyOptions, scores: np.ndarray,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Finish all selected spans together using the production boundary/side rules."""
    return finish_rows(pool, choose_rows(pool, scores))


def finish_rows(
    pool: RallyOptions, rows: dict[int, int],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Apply the chosen rows together before canonical boundary and side finishing."""
    choices = {span_id: pool.sequences[row] for span_id, row in rows.items()}
    video = pool.original
    sequences, events = apply_choices(video.initial, video.events, choices)
    extended = extend_boundaries(sequences, events, video.fps, frame_count=video.frame_count)
    return alternate_sides(extended, events)
