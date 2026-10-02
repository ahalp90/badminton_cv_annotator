"""A fitted bundle carries its preprocessing rules into the current runtime."""

import json
from dataclasses import replace

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

from annotator.contacts.model import ContactModelConfig
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


@pytest.mark.parametrize('enabled', [False, True])
def test_bundle_keeps_masked_candidate_policy(tmp_path, annotator_models, enabled):
    models = replace(annotator_models, contact_settings=ContactModelConfig(reject_masked_without_player=enabled))
    save_models(models, tmp_path)
    restored = load_models(tmp_path)
    assert restored.contact_settings.reject_masked_without_player is enabled


def test_bundle_without_new_policy_field_defaults_to_baseline(tmp_path, annotator_models):
    save_models(annotator_models, tmp_path)
    old_models = joblib.load(tmp_path / 'models.joblib')
    object.__delattr__(old_models.contact_settings, 'reject_masked_without_player')
    joblib.dump(old_models, tmp_path / 'models.joblib')
    assert not load_models(tmp_path).contact_settings.reject_masked_without_player
