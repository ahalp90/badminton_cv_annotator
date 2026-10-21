"""XGBoost weighting and stopping use the same selected training rows."""

from typing import Any, Literal

import numpy as np
import pytest
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from annotator.training.trees import fit_binary_tree

FIT_SETTINGS = {
    'max_iter': 8, 'max_leaf_nodes': 7, 'learning_rate': 0.15,
    'l2_regularization': 1.0, 'random_seed': 7,
}


def examples(row_count: int = 120) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(3)
    matrix = random.normal(size=(row_count, 4)).astype(np.float32)
    labels = (matrix[:, 1] > 0.7).astype(np.uint8)
    matrix[:, 0] = np.arange(row_count)
    matrix[::7, 2] = np.nan
    return matrix, labels


@pytest.mark.parametrize(('balanced', 'weighted'), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize(('row_count', 'early_stopping', 'stopping'), [
    (10_000, 'auto', False), (10_001, 'auto', True),
    (120, True, True), (10_001, False, False),
])
def test_xgb_weights_and_internal_stopping_split(
    monkeypatch: pytest.MonkeyPatch, row_count: int,
    early_stopping: bool | Literal['auto'], stopping: bool, balanced: bool, weighted: bool,
) -> None:
    xgb = pytest.importorskip('xgboost')
    matrix, labels = examples(row_count)
    weights = 1.0 / (1 + np.arange(row_count) % 4) if weighted else None
    original_fit = xgb.XGBClassifier.fit
    seen = []

    def observe_fit(model: Any, training: np.ndarray, targets: np.ndarray, **kwargs: Any) -> Any:
        training_indices = training[:, 0].astype(int)
        np.testing.assert_array_equal(targets, labels[training_indices])
        expected_weights = weights
        if balanced:
            class_weights = compute_sample_weight('balanced', labels)
            expected_weights = class_weights if weights is None else weights * class_weights
        if expected_weights is None:
            assert kwargs['sample_weight'] is None
        else:
            np.testing.assert_array_equal(kwargs['sample_weight'], expected_weights[training_indices])
        if stopping:
            validation, validation_labels = kwargs['eval_set'][0]
            validation_indices = validation[:, 0].astype(int)
            assert not set(training_indices) & set(validation_indices)
            assert sorted([*training_indices, *validation_indices]) == list(range(row_count))
            assert len(validation) == int(np.ceil(row_count * 0.1))
            assert set(targets) == set(validation_labels) == {0, 1}
            np.testing.assert_array_equal(validation_labels, labels[validation_indices])
            if expected_weights is None:
                assert kwargs['sample_weight_eval_set'] is None
            else:
                np.testing.assert_array_equal(kwargs['sample_weight_eval_set'][0], expected_weights[validation_indices])
        else:
            np.testing.assert_array_equal(training, matrix)
            assert kwargs['eval_set'] is None
            assert kwargs['sample_weight_eval_set'] is None
        seen.append(training_indices)
        return original_fit(model, training, targets, **kwargs)

    monkeypatch.setattr(xgb.XGBClassifier, 'fit', observe_fit)
    with threadpool_limits(limits=1):
        model = fit_binary_tree(
            matrix, labels, balanced=balanced, sample_weight=weights, early_stopping=early_stopping,
            **FIT_SETTINGS,
        )
        repeated = fit_binary_tree(
            matrix, labels, balanced=balanced, sample_weight=weights, early_stopping=early_stopping,
            **FIT_SETTINGS,
        )
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(model.predict_proba(matrix), repeated.predict_proba(matrix))
    assert model.get_params()['early_stopping_rounds'] == (10 if stopping else None)
    assert np.mean(model.predict_proba(matrix)[labels == 1, 1]) > np.mean(model.predict_proba(matrix)[labels == 0, 1])


def test_xgb_stops_when_internal_holdout_loss_cannot_improve() -> None:
    pytest.importorskip('xgboost')
    matrix = np.zeros((200, 3), dtype=np.float32)
    labels = np.tile([0, 1], 100)
    model = fit_binary_tree(
        matrix, labels, balanced=True, early_stopping=True,
        **{**FIT_SETTINGS, 'max_iter': 40},
    )
    assert model.best_iteration == 0
    assert model.get_booster().num_boosted_rounds() == 11
    np.testing.assert_allclose(model.predict_proba(matrix), 0.5)
