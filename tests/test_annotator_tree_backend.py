"""Default HGB parity and opt-in XGBoost weighting and stopping boundaries."""

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from annotator.contacts.features import REGION_FIELDS
from annotator.contacts.model import CONTACT_FEATURE_NAMES, contact_feature_matrix
from annotator.sequence.confidence import (
    BASE_FEATURE_NAMES,
    GAP_FEATURE_NAMES,
    ConfidenceFeatures,
)
from annotator.training.confidence import ConfidenceFitConfig, fit_confidence_model
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from annotator.training.sequences import TreeFitConfig, fit_tree
from annotator.training.trees import TreeBackend, fit_binary_tree

FIT_SETTINGS = {
    'max_iter': 8, 'max_leaf_nodes': 7, 'learning_rate': 0.15,
    'min_samples_leaf': 2, 'l2_regularization': 1.0, 'random_seed': 7,
}


def examples(row_count: int = 120) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(3)
    matrix = random.normal(size=(row_count, 4)).astype(np.float32)
    labels = (matrix[:, 1] > 0.7).astype(np.uint8)
    matrix[:, 0] = np.arange(row_count)
    matrix[::7, 2] = np.nan
    return matrix, labels


@pytest.mark.parametrize(('balanced', 'weighted'), [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('early_stopping', [False, True, 'auto'])
def test_default_backend_matches_previous_hgb_constructor(
    balanced: bool, weighted: bool, early_stopping: bool | Literal['auto'],
) -> None:
    matrix, labels = examples()
    weights = 1.0 / (1 + np.arange(len(labels)) % 4) if weighted else None
    with threadpool_limits(limits=1):
        actual = fit_binary_tree(
            matrix, labels, balanced=balanced, sample_weight=weights, early_stopping=early_stopping, **FIT_SETTINGS,
        )
        expected = HistGradientBoostingClassifier(
            max_iter=8, max_leaf_nodes=7, learning_rate=0.15, min_samples_leaf=2,
            l2_regularization=1.0, random_state=7,
            class_weight='balanced' if balanced else None, early_stopping=early_stopping,
        ).fit(matrix, labels, sample_weight=weights)
        assert type(actual) is HistGradientBoostingClassifier
        assert actual.get_params() == expected.get_params()
        np.testing.assert_array_equal(actual.predict_proba(matrix), expected.predict_proba(matrix))


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
            backend=TreeBackend.XGBOOST, **FIT_SETTINGS,
        )
        repeated = fit_binary_tree(
            matrix, labels, balanced=balanced, sample_weight=weights, early_stopping=early_stopping,
            backend=TreeBackend.XGBOOST, **FIT_SETTINGS,
        )
    np.testing.assert_array_equal(seen[0], seen[1])
    np.testing.assert_array_equal(model.predict_proba(matrix), repeated.predict_proba(matrix))
    assert model.get_params()['early_stopping_rounds'] == (10 if stopping else None)
    assert np.mean(model.predict_proba(matrix)[labels == 1, 1]) > np.mean(model.predict_proba(matrix)[labels == 0, 1])


def test_xgb_training_entry_points_and_models_round_trip(tmp_path: Path) -> None:
    xgb = pytest.importorskip('xgboost')
    matrix, labels = examples()
    targets = labels.astype(int)
    targets[::9] = -1
    sequence = fit_tree(matrix, targets, TreeFitConfig(max_iter=8, backend=TreeBackend.XGBOOST))
    gap = np.tile(matrix[:, 1:2], (1, len(GAP_FEATURE_NAMES)))
    features = ConfidenceFeatures(tuple(range(len(matrix))), gap[:, :len(BASE_FEATURE_NAMES)], gap)
    confidence = fit_confidence_model(features, targets, ConfidenceFitConfig(max_iter=8, backend=TreeBackend.XGBOOST))
    fields = tuple(dict.fromkeys(('frame', *CONTACT_FEATURE_NAMES, *REGION_FIELDS)))
    rows = np.zeros(len(matrix), dtype=[(name, np.float32) for name in fields])
    rows['frame'] = np.arange(len(rows))
    for name in REGION_FIELDS:
        rows[name] = 1
    video = ContactTrainingVideo('training', rows, np.arange(10, len(rows), 20), 30.0)
    contact = fit_contact_model([video], ContactFitConfig(max_iter=8, backend=TreeBackend.XGBOOST))
    contact_matrix = contact_feature_matrix(contact.selection.rows)
    for name, model, inputs in (
        ('sequence', sequence, matrix), ('confidence', confidence, gap), ('contact', contact.model, contact_matrix),
    ):
        assert isinstance(model, xgb.XGBClassifier)
        path = tmp_path / f'{name}.joblib'
        joblib.dump(model, path)
        loaded = joblib.load(path)
        np.testing.assert_array_equal(loaded.classes_, [0, 1])
        np.testing.assert_array_equal(loaded.predict_proba(inputs), model.predict_proba(inputs))
        params = model.get_params()
        assert params['objective'] == 'binary:logistic'
        assert params['tree_method'] == 'hist'
        assert params['grow_policy'] == 'lossguide'
        assert params['max_depth'] == 0
        assert params['max_bin'] == 255
        assert params['min_child_weight'] == 5.0
        assert params['device'] == 'cpu'


def test_xgb_stops_when_internal_holdout_loss_cannot_improve() -> None:
    pytest.importorskip('xgboost')
    matrix = np.zeros((200, 3), dtype=np.float32)
    labels = np.tile([0, 1], 100)
    model = fit_binary_tree(
        matrix, labels, balanced=True, early_stopping=True, backend=TreeBackend.XGBOOST,
        **{**FIT_SETTINGS, 'max_iter': 40},
    )
    assert model.best_iteration == 0
    assert model.get_booster().num_boosted_rounds() == 11
    np.testing.assert_allclose(model.predict_proba(matrix), 0.5)
