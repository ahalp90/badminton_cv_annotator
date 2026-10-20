"""Rally-layer fitting and selection, independent of cross-validation scheduling."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from annotator.sequence.boundaries import extend_boundaries
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import apply_choices
from annotator.sequence.features import positive_probabilities
from annotator.sequence.sides import alternate_sides
from annotator.training.sequences import TreeFitConfig
from annotator.training.trees import fit_binary_tree
from experiments.annotator.tree_simplification.options import ExpandedPool

DEFAULT_LAYER_CONFIG = TreeFitConfig()

CONTEXT_FEATURE_NAMES = (
    'option_score', 'chosen_score', 'top_margin', 'option_minus_chosen',
    'normalised_rank', 'is_chosen', 'added_vs_chosen', 'deleted_vs_chosen',
)


def ordered_span_rows(pool: ExpandedPool, scores: np.ndarray) -> dict[int, np.ndarray]:
    """Rank every span's rows by descending score, retaining pool order for ties."""
    if scores.shape != (len(pool.sequences),) or not np.isfinite(scores).all():
        raise ValueError('scores must be finite and align with every pool option')
    return {span_id: rows[np.argsort(-scores[rows], kind='stable')] for span_id, rows in pool.span_rows.items()}


def choose_rows(pool: ExpandedPool, scores: np.ndarray) -> dict[int, int]:
    return {span_id: int(rows[0]) for span_id, rows in ordered_span_rows(pool, scores).items()}


def layer_features(pool: ExpandedPool, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
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


def judged_weights(pool: ExpandedPool) -> np.ndarray:
    """Give each span total weight one across its judged options."""
    weights = np.zeros(len(pool.targets), dtype=np.float64)
    for rows in pool.span_rows.values():
        judged = rows[pool.targets[rows] >= 0]
        if len(judged):
            weights[judged] = 1.0 / len(judged)
    return weights


def fit_layer(
    pools: Sequence[ExpandedPool], histories: Sequence[Sequence[np.ndarray]],
    config: TreeFitConfig = DEFAULT_LAYER_CONFIG,
) -> Any:
    """Fit one layer from caller-supplied pools and aligned earlier-layer scores.

    The caller owns data splits and must supply held-out predictions for earlier
    layers. Every span contributes equal total weight regardless of pool size.
    Unknown targets are excluded; class balancing would change the span totals.
    """
    matrices, labels, weights = [], [], []
    for pool, history in zip(pools, histories, strict=True):
        judged = pool.targets >= 0
        matrices.append(layer_features(pool, history)[judged])
        labels.append(pool.targets[judged])
        weights.append(judged_weights(pool)[judged])
    if not labels:
        raise ValueError('supply at least one training pool')
    targets = np.concatenate(labels)
    if set(targets.tolist()) != {0, 1}:
        raise ValueError('layer training needs both correct and incorrect judged options')
    return fit_binary_tree(
        np.concatenate(matrices), targets, sample_weight=np.concatenate(weights), balanced=False,
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes, learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf, l2_regularization=config.l2_regularization,
        random_seed=config.random_seed, backend=config.backend, device=config.device,
        min_child_weight=config.min_child_weight,
    )


def predict_layer(model: Any, pool: ExpandedPool, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    return positive_probabilities(model, layer_features(pool, prior_scores))


def finish_selected(
    pool: ExpandedPool, scores: np.ndarray,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Finish all selected spans together using the production boundary/side rules."""
    choices = {span_id: pool.sequences[row] for span_id, row in choose_rows(pool, scores).items()}
    video = pool.original
    sequences, events = apply_choices(video.initial, video.events, choices)
    extended = extend_boundaries(sequences, events, video.fps, frame_count=video.frame_count)
    return alternate_sides(extended, events)
