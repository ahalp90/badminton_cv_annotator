"""Fit the annotator's XGBoost trees with fixed weighting and stopping rules."""

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


@dataclass(frozen=True)
class TreeFitConfig:
    """XGBoost settings shared by rally selection and review ranking."""

    max_iter: int = 200
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    l2_regularization: float = 1.0
    random_seed: int = 20260905
    device: str = 'cpu'
    min_child_weight: float = 5.0


AUTO_STOPPING_ROWS = 10_000
VALIDATION_FRACTION = 0.1
STOPPING_ROUNDS = 10


def fit_binary_tree(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int,
    max_leaf_nodes: int,
    learning_rate: float,
    l2_regularization: float,
    random_seed: int,
    balanced: bool,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool | Literal['auto'] = False,
    device: str = 'cpu',
    min_child_weight: float = 5.0,
) -> Any:
    """Fit already selected rows containing both binary target classes.

    Stopping uses a stratified holdout from these training rows. Class weights
    are computed before that split, then multiplied by any supplied row weights.

    :param matrix: One feature row per selected training example; NaNs are allowed.
    :param labels: Aligned labels, with both 0 and 1 present.
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
            stratify=labels, random_state=random_seed,
        )
        training_matrix, training_labels = matrix[training_indices], labels[training_indices]
        validation = [(matrix[validation_indices], labels[validation_indices])]
        if weights is not None:
            training_weights = weights[training_indices]
            validation_weights = [weights[validation_indices]]
    xgb_model = XGBClassifier(
        objective='binary:logistic', tree_method='hist', grow_policy='lossguide',
        max_depth=0, max_leaves=max_leaf_nodes, max_bin=255,
        learning_rate=learning_rate, n_estimators=max_iter, reg_lambda=l2_regularization,
        min_child_weight=min_child_weight, random_state=random_seed, device=device,
        n_jobs=2, eval_metric='logloss',
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
    )
    xgb_model.fit(
        training_matrix, training_labels, sample_weight=training_weights,
        eval_set=validation, sample_weight_eval_set=validation_weights, verbose=False,
    )
    # Annotation passes NumPy arrays and also runs on machines without CUDA.
    xgb_model.set_params(device='cpu')
    return xgb_model
