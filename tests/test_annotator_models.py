"""A fitted bundle carries its preprocessing rules into the current runtime."""

import json

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.sequence import SequenceModels


def test_bundle_roundtrip_keeps_predictions_and_geometry(tmp_path):
    tree = DummyClassifier(strategy='prior').fit(np.zeros((3, 85)), [0, 1, 1])
    models = AnnotatorModels(
        tree, SequenceModels(tree, tree, tree, tree, tree, tree), tree, side_geometry=SideGeometry.SCENE,
    )
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert restored.side_geometry is SideGeometry.SCENE
    assert restored.preprocessing == models.preprocessing
    np.testing.assert_array_equal(restored.contact.predict_proba([[0] * 85]), tree.predict_proba([[0] * 85]))


def test_missing_bundle_explains_retune_requirement(tmp_path):
    with pytest.raises(FileNotFoundError, match='model directory produced by the annotator retune'):
        load_models(tmp_path)


def test_incompatible_runtime_is_rejected_before_loading_models(tmp_path):
    (tmp_path / 'models.joblib').write_bytes(b'not loaded')
    (tmp_path / 'metadata.json').write_text(json.dumps({
        'schema': 'annotator-models/1', 'scikit_learn': '0.0',
    }))
    with pytest.raises(ValueError, match='Refit the bundle'):
        load_models(tmp_path)
