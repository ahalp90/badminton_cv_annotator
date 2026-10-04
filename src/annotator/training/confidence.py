"""Fit the review-ranking tree from held-group annotation predictions."""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from annotator.sequence.confidence import (
    GAP_FEATURE_NAMES,
    ConfidenceFeatures,
)


@dataclass(frozen=True)
class ConfidenceFitConfig:
    """Tree settings for the gap-evidence review ranking."""

    max_iter: int = 100
    max_leaf_nodes: int = 7
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_seed: int = 20260905


DEFAULT_CONFIDENCE_FIT_CONFIG = ConfidenceFitConfig()


def fit_confidence_model(
    features: ConfidenceFeatures,
    correctness: np.ndarray,
    config: ConfidenceFitConfig = DEFAULT_CONFIDENCE_FIT_CONFIG,
) -> HistGradientBoostingClassifier:
    """Fit the ranking, excluding sections that cannot be judged.

    Training rows must come from held-group sequence predictions: the sequence
    models producing each row must not have trained on that video's group.
    The caller creates those predictions and evaluates held-out videos outside
    this fit. Labels describe whole-section correctness at the chosen tolerance.

    :param features: Gap rows from ``build_confidence_features``, including base evidence.
    :param correctness: One label per row: 1 correct, 0 wrong, -1 unjudgeable.
    :param config: Tree settings for a dataset retune.
    :return: Fitted model ready for ``score_rally_confidence``.
    """
    row_count = len(features.span_ids)
    labels = np.asarray(correctness)
    if labels.shape != (row_count,):
        raise ValueError("correctness labels must align with confidence feature rows")
    if not np.isin(labels, (-1, 0, 1)).all():
        raise ValueError("correctness labels must be -1, 0 or 1")
    if features.gap.shape != (row_count, len(GAP_FEATURE_NAMES)):
        raise ValueError("gap confidence features do not match the supported columns")
    judged = labels != -1
    judged_labels = labels[judged].astype(np.int64)
    if set(judged_labels.tolist()) != {0, 1}:
        raise ValueError("confidence training needs both correct and wrong judged sections")
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        random_state=config.random_seed,
        class_weight=None,
        early_stopping=False,
    )
    model.fit(features.gap[judged], judged_labels)
    return model
