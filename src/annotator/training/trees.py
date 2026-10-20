"""Fit the annotator's binary trees with an explicitly selected backend."""

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight


class TreeBackend(StrEnum):
    HIST_GRADIENT_BOOSTING = 'hist_gradient_boosting'
    XGBOOST = 'xgboost'


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
    min_samples_leaf: int,
    l2_regularization: float,
    random_seed: int,
    balanced: bool,
    sample_weight: np.ndarray | None = None,
    early_stopping: bool | Literal['auto'] = False,
    backend: TreeBackend = TreeBackend.HIST_GRADIENT_BOOSTING,
    device: str = 'cpu',
    min_child_weight: float = 5.0,
) -> Any:
    """Fit already selected rows containing both binary target classes.

    XGBoost uses a Hessian-weight minimum, ``min_child_weight``, instead of
    HGB's row-count minimum, ``min_samples_leaf``. These are separate controls.
    Its stopping split comes solely from these training rows. The automatic
    threshold, holdout fraction and patience follow HGB defaults, but the
    backend's optimisation and stopping rules are not numerically equivalent.

    :param matrix: One feature row per selected training example; NaNs are allowed.
    :param labels: Aligned labels, with both 0 and 1 present.
    :param sample_weight: Optional per-row weights, multiplied by class weights when balanced.
    :return: The fitted backend estimator, exposing ``classes_`` and ``predict_proba``.
    """
    if backend == TreeBackend.HIST_GRADIENT_BOOSTING:
        model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_seed,
            class_weight='balanced' if balanced else None,
            early_stopping=early_stopping,
        )
        if sample_weight is None:
            model.fit(matrix, labels)
        else:
            model.fit(matrix, labels, sample_weight=sample_weight)
        return model
    if backend != TreeBackend.XGBOOST:
        raise ValueError(f'unknown tree backend: {backend}')

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
        n_jobs=1, eval_metric='logloss',
        early_stopping_rounds=STOPPING_ROUNDS if stopping else None,
    )
    xgb_model.fit(
        training_matrix, training_labels, sample_weight=training_weights,
        eval_set=validation, sample_weight_eval_set=validation_weights, verbose=False,
    )
    return xgb_model
