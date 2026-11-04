"""A fitted bundle carries its preprocessing rules into the current runtime."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from annotator.models import (
    DEFAULT_MODEL_DIRECTORY,
    AnnotatorModels,
    SideGeometry,
    fitted_trees,
    load_models,
    save_models,
)
from annotator.sequence.rally_model import RallySelector


def test_default_bundle_loads_and_predicts_with_all_three_models():
    models = load_models(DEFAULT_MODEL_DIRECTORY)
    for estimator in fitted_trees(models):
        probabilities = estimator.predict_proba(np.zeros((1, estimator.n_features_in_)))
        assert probabilities.shape == (1, 2)
        assert np.isfinite(probabilities).all()


def test_previous_schema_requires_matching_historical_checkout():
    historical = (
        Path(__file__).resolve().parents[1]
        / 'experiments/annotator/tree_simplification/results/discovery_context/no_max/models/raw_baseline-seed-0'
    )
    with pytest.raises(ValueError, match='Unsupported annotator model bundle'):
        load_models(historical)


def test_bundle_roundtrip_keeps_predictions_and_geometry(tmp_path):
    tree = DummyClassifier(strategy='prior').fit(np.zeros((3, 85)), [0, 1, 1])
    models = AnnotatorModels(
        tree, RallySelector(tree), tree, side_geometry=SideGeometry.SCENE,
    )
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert restored.side_geometry is SideGeometry.SCENE
    assert restored.preprocessing == models.preprocessing
    np.testing.assert_array_equal(restored.contact.predict_proba([[0] * 85]), tree.predict_proba([[0] * 85]))


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


@pytest.mark.parametrize('schema', ['annotator-models/1', 'annotator-rally-layers/1'])
def test_retired_schema_is_rejected_before_unpickling(tmp_path, schema):
    (tmp_path / 'models.joblib').write_bytes(b'not loaded')
    (tmp_path / 'metadata.json').write_text(json.dumps({'schema': schema}))
    with pytest.raises(ValueError, match='Unsupported annotator model bundle'):
        load_models(tmp_path)
