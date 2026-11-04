"""Experiment readers preserve the supported tree controls."""

import pytest

from experiments.annotator.development import xgboost_settings


def test_saved_tree_settings_drop_only_retired_fields() -> None:
    record = {'backend': 'xgboost', 'min_samples_leaf': 40, 'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert xgboost_settings(record) == {'random_seed': 7, 'device': 'cuda', 'extra_setting': 3}
    assert record['min_samples_leaf'] == 40
    with pytest.raises(ValueError, match='HGB fitting is retired'):
        xgboost_settings({'backend': 'hist_gradient_boosting'})
