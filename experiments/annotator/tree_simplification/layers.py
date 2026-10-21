"""Fit experimental rally layers using the production feature and inference path."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from annotator.sequence import rally_layers
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.rally_layers import CONTEXT_FEATURE_NAMES
from annotator.training import rally_layers as rally_training
from annotator.training.trees import TreeFitConfig
from experiments.annotator.tree_simplification.options import LabelledOptions

__all__ = [
    'CONTEXT_FEATURE_NAMES', 'choose_rows', 'finish_rows', 'finish_selected', 'fit_layer',
    'judged_weights', 'layer_features', 'ordered_span_rows', 'predict_layer',
]

DEFAULT_LAYER_CONFIG = TreeFitConfig()


def ordered_span_rows(pool: LabelledOptions, scores: np.ndarray) -> dict[int, np.ndarray]:
    return rally_layers.ordered_span_rows(pool.runtime, scores)


def choose_rows(pool: LabelledOptions, scores: np.ndarray) -> dict[int, int]:
    return rally_layers.choose_rows(pool.runtime, scores)


def layer_features(pool: LabelledOptions, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    return rally_layers.layer_features(pool.runtime, prior_scores)


def judged_weights(pool: LabelledOptions) -> np.ndarray:
    return rally_training.judged_weights(pool.runtime, pool.targets)


def fit_layer(
    pools: Sequence[LabelledOptions], histories: Sequence[Sequence[np.ndarray]],
    config: TreeFitConfig = DEFAULT_LAYER_CONFIG, *, span_weight: float,
) -> Any:
    return rally_training.fit_layer(
        [pool.runtime for pool in pools], [pool.targets for pool in pools], histories,
        config, span_weight=span_weight,
    )


def predict_layer(model: Any, pool: LabelledOptions, prior_scores: Sequence[np.ndarray] = ()) -> np.ndarray:
    return rally_layers.predict_layer(model, pool.runtime, prior_scores)


def finish_selected(
    pool: LabelledOptions, scores: np.ndarray,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    return rally_layers.finish_selected(pool.runtime, scores)


def finish_rows(
    pool: LabelledOptions, rows: dict[int, int],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    return rally_layers.finish_rows(pool.runtime, rows)
