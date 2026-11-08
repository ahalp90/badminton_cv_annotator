"""Fit the annotator's XGBoost trees with fixed weighting and stopping rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

if TYPE_CHECKING:
    from annotator.training.contact import ContactFitConfig


@dataclass(frozen=True)
class TreeFitConfig:
    """XGBoost settings shared by rally selection and review ranking."""

    max_iter: int = 200
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    random_seed: int = 20260905
    device: str = 'cuda'
    min_child_weight: float = 5.0


AUTO_STOPPING_ROWS = 10_000
VALIDATION_FRACTION = 0.1
STOPPING_ROUNDS = 10


def fit_binary_tree(
    matrix: np.ndarray,
    labels: np.ndarray,
    config: TreeFitConfig | ContactFitConfig,
    *,
    balanced: bool,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool | Literal['auto'] = False,
) -> Any:
    """Fit already selected rows containing both binary target classes.

    Stopping uses a stratified holdout from these training rows. Class weights
    are computed before that split, then multiplied by any supplied row weights.

    :param matrix: One feature row per selected training example; NaNs are allowed.
    :param labels: Aligned labels, with both 0 and 1 present.
    :param config: Flat tree settings shared by the contact, selector and review fits.
    :param sample_weight: Optional per-row weights, multiplied by class weights when balanced.
    :return: The fitted XGBoost estimator, exposing ``classes_`` and ``predict_proba``.
    """
    from xgboost import XGBClassifier

    stopping = len(labels) > AUTO_STOPPING_ROWS if early_stopping == 'auto' else early_stopping
    weights = sample_weight
    if balanced:
        class_weights = compute_sample_weight('balanced', labels)
        weights = class_weights if weights is None else weights * class_weights
    training_matrix, training_labels, training_weights = matrix, labels, weights
    validation = None
    validation_weights = None
    if stopping:
        training_indices, validation_indices = train_test_split(
            np.arange(len(labels)), test_size=VALIDATION_FRACTION,
            stratify=labels, random_state=config.random_seed,
        )
        training_matrix, training_labels = matrix[training_indices], labels[training_indices]
        validation = [(matrix[validation_indices], labels[validation_indices])]
        if weights is not None:
            training_weights = weights[training_indices]
            validation_weights = [weights[validation_indices]]
    xgb_model = XGBClassifier(
        objective='binary:logistic', tree_method='hist', grow_policy='lossguide',
        max_depth=0, max_leaves=config.max_leaf_nodes, max_bin=255,
        learning_rate=config.learning_rate, n_estimators=config.max_iter, reg_lambda=config.l2_regularization,
        min_child_weight=config.min_child_weight, random_state=config.random_seed, device=config.device,
        n_jobs=2, eval_metric='logloss',
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
    )
    xgb_model.fit(
        training_matrix, training_labels, sample_weight=training_weights,
        eval_set=validation, sample_weight_eval_set=validation_weights, verbose=False,
    )
    return xgb_model
