from __future__ import annotations

import json
from pathlib import Path

import pytest

from scratch.contact_det_full_ds_fit.scripts import rally_start_model_config as config

CONFIG_PATH = Path(__file__).parents[1] / "records/rally_start_model_runs.json"


def test_fixed_model_comparison_loads() -> None:
    loaded = config.load_rally_start_model_config(CONFIG_PATH)

    assert loaded.feature_names == config.FEATURE_NAMES
    assert tuple(model.model_id for model in loaded.models) == config.MODEL_IDS
    assert loaded.selection_cutoffs == (0.5, 0.7, 0.9)
    assert loaded.training_groups == ("A", "B", "C", "D")
    assert loaded.training_gate.minimum_new_fully_correct_sections == 10
    assert loaded.validation_gate.minimum_new_fully_correct_sections == 5


def test_changed_model_comparison_is_rejected(tmp_path) -> None:
    changed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed["selection_cutoffs"].append(0.95)
    path = tmp_path / CONFIG_PATH.name
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        config.load_rally_start_model_config(path)
