"""Configured tree settings and proposal limits match five accepted XGBoost fits."""

import json
from dataclasses import asdict, replace

import pytest

from annotator.models import DEFAULT_MODEL_DIRECTORY, load_models
from experiments.annotator.development import (
    contact_stage_settings,
    settings_for,
    xgboost_settings,
)
from experiments.annotator.heuristic_only_tune.run import read_json


@pytest.mark.parametrize('seed', [0, 100, 200, 300, 400])
def test_tree_settings_and_proposal_limits_match_saved_fits(seed: int) -> None:
    results = DEFAULT_MODEL_DIRECTORY.parents[1]
    models = load_models(DEFAULT_MODEL_DIRECTORY.parent / f'raw_baseline-seed-{seed}')
    stage = read_json(results / 'fit-settings.json.gz')['bundles'][f'raw_baseline-seed-{seed}']
    settings = replace(settings_for('cuda', 5.0, seed_offset=seed), preprocessing=models.preprocessing)
    record = json.loads(json.dumps(asdict(settings), default=sorted))
    assert contact_stage_settings(record) == stage['training_settings']
    selector = stage['selector']
    assert asdict(settings.rally_layers.tree) == selector['tree']
    assert settings.rally_layers.depth == selector['layers']
    assert settings.rally_layers.max_edits == stage['max_edits']
    assert settings.rally_layers.deletion_candidates == stage['deletion_candidates']


def test_saved_tree_settings_drop_only_retired_fields() -> None:
    record = {'backend': 'xgboost', 'min_samples_leaf': 40, 'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert xgboost_settings(record) == {'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert record['min_samples_leaf'] == 40
    with pytest.raises(ValueError, match='HGB fitting is retired'):
        xgboost_settings({'backend': 'hist_gradient_boosting'})
