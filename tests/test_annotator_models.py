"""A fitted bundle carries its preprocessing rules into the current runtime."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from annotator.models import (
    DEFAULT_MODEL_DIRECTORY,
    AnnotatorModels,
    fitted_trees,
    load_models,
    save_models,
)
from annotator.rallies.refine_rallies import RallySelector


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
def test_default_bundle_loads_and_predicts_with_all_four_models(device):
    models = load_models(DEFAULT_MODEL_DIRECTORY, device=device)
    assert len(fitted_trees(models)) == 4
    for estimator in fitted_trees(models):
        assert estimator.get_params()['device'] == device
        probabilities = estimator.predict_proba(np.zeros((1, estimator.n_features_in_)))
        assert probabilities.shape == (1, 2)
        assert np.isfinite(probabilities).all()


def test_loading_uses_runtime_device_instead_of_saved_device(tmp_path):
    models = load_models(DEFAULT_MODEL_DIRECTORY, device='cpu')
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    for original, loaded in zip(fitted_trees(models), fitted_trees(restored), strict=True):
        assert loaded.get_params()['device'] == 'cuda'
        matrix = np.zeros((3, loaded.n_features_in_), dtype=np.float32)
        np.testing.assert_allclose(loaded.predict_proba(matrix), original.predict_proba(matrix), rtol=1e-6)


def test_previous_schema_requires_matching_historical_checkout():
    historical = (
        Path(__file__).resolve().parents[1]
        / 'experiments/annotator/tree_simplification/results/discovery_context/no_max/models/raw_baseline-seed-0'
    )
    with pytest.raises(ValueError, match='Unsupported annotator model bundle'):
        load_models(historical)


def test_bundle_roundtrip_keeps_predictions_and_preprocessing(tmp_path):
    matrix = np.zeros((3, len(CONTACT_FEATURE_NAMES)))
    tree = DummyClassifier(strategy='prior').fit(matrix, [0, 1, 1])
    followup = DummyClassifier(strategy='prior').fit(matrix, [0, 0, 1])
    models = AnnotatorModels(
        tree, RallySelector(tree), RallySelector(followup, max_edits=2), tree,
    )
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert restored.preprocessing == models.preprocessing
    np.testing.assert_array_equal(restored.contact_model.predict_proba(matrix), tree.predict_proba(matrix))
    np.testing.assert_array_equal(restored.followup_selector.selector.predict_proba(matrix), followup.predict_proba(matrix))
    assert restored.followup_selector.max_edits == 2
    assert len(fitted_trees(restored)) == 4


def test_missing_bundle_explains_retune_requirement(tmp_path):
    with pytest.raises(FileNotFoundError, match='model directory produced by the annotator retune'):
        load_models(tmp_path)


def test_library_version_metadata_does_not_block_loading(tmp_path, annotator_models):
    save_models(annotator_models, tmp_path)
    metadata_path = tmp_path / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata.update(scikit_learn='0.0', xgboost='0.0')
    metadata_path.write_text(json.dumps(metadata))

    assert load_models(tmp_path).preprocessing == annotator_models.preprocessing


@pytest.mark.parametrize('schema', ['annotator-models/1', 'annotator-rally-layers/1', 'annotator-three-models/1', 'annotator-three-models/2', 'annotator-three-models/3', 'annotator-three-models/6'])
def test_retired_schema_is_rejected_before_unpickling(tmp_path, schema):
    (tmp_path / 'models.joblib').write_bytes(b'not loaded')
    (tmp_path / 'metadata.json').write_text(json.dumps({'schema': schema}))
    with pytest.raises(ValueError, match='Unsupported annotator model bundle'):
        load_models(tmp_path)
