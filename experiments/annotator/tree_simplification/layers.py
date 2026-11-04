"""Fit and apply the production rally selector to prepared option pools."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from annotator.sequence import rally_model
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import positive_probabilities
from annotator.training import rally_layers as rally_training
from annotator.training.trees import TreeFitConfig
from experiments.annotator.tree_simplification.options import LabelledOptions

DEFAULT_LAYER_CONFIG = TreeFitConfig()


def ordered_span_rows(pool: LabelledOptions, scores: np.ndarray) -> dict[int, np.ndarray]:
    return rally_model.ordered_span_rows(pool.runtime, scores)


def choose_rows(pool: LabelledOptions, scores: np.ndarray) -> dict[int, int]:
    return rally_model.choose_rows(pool.runtime, scores)


def judged_weights(pool: LabelledOptions) -> np.ndarray:
    return rally_training.judged_weights(pool.runtime, pool.targets)


def fit_selector(
    pools: Sequence[LabelledOptions], config: TreeFitConfig = DEFAULT_LAYER_CONFIG, *, span_weight: float,
) -> Any:
    return rally_training.fit_selector(
        [pool.runtime for pool in pools], [pool.targets for pool in pools], config, span_weight=span_weight,
    )


def predict_selector(model: Any, pool: LabelledOptions) -> np.ndarray:
    return positive_probabilities(model, pool.matrix)


def finish_selected(
    pool: LabelledOptions, scores: np.ndarray,
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    runtime = pool.runtime
    return rally_model.finish_rows(runtime, rally_model.choose_rows(runtime, scores))


def finish_rows(
    pool: LabelledOptions, rows: dict[int, int],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    return rally_model.finish_rows(pool.runtime, rows)
