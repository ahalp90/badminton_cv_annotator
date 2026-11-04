"""Configured tree settings and proposal limits match five accepted XGBoost fits."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from experiments.annotator.development import (
    contact_stage_settings,
    settings_for,
    xgboost_settings,
)
from experiments.annotator.heuristic_only_tune.run import read_json


@pytest.mark.parametrize('seed', [0, 100, 200, 300, 400])
def test_tree_settings_and_proposal_limits_match_saved_fits(seed: int) -> None:
    results = (
        Path(__file__).resolve().parents[1]
        / 'experiments/annotator/tree_simplification/results/discovery_context/no_max'
    )
    stage = read_json(results / 'fit-settings.json.gz')['bundles'][f'raw_baseline-seed-{seed}']
    settings = settings_for('cuda', 5.0, seed_offset=seed)
    record = json.loads(json.dumps(asdict(settings), default=sorted))
    record['contact'].pop('positive_radius_at_30_fps')
    record['contact'].pop('ignored_radius_at_30_fps')
    assert contact_stage_settings(record) == stage['training_settings']
    selector = stage['selector']
    assert asdict(settings.rally_layers.tree) == selector['tree']
    assert selector['layers'] == 1
    assert settings.rally_layers.max_edits == stage['max_edits']
    assert settings.rally_layers.deletion_candidates == stage['deletion_candidates']


def test_saved_tree_settings_drop_only_retired_fields() -> None:
    record = {'backend': 'xgboost', 'min_samples_leaf': 40, 'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert xgboost_settings(record) == {'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert record['min_samples_leaf'] == 40
    with pytest.raises(ValueError, match='HGB fitting is retired'):
        xgboost_settings({'backend': 'hist_gradient_boosting'})
