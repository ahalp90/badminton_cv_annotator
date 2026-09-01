from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from scratch.contact_det_full_ds_fit.scripts.baseline_config import (
    BASELINE_RUNS_SCHEMA,
    ModelKind,
    MotionMode,
    load_baseline_config,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_MANIFEST = EXPERIMENT_ROOT / "records/baseline_runs.json"


def _canonical_payload() -> dict[str, object]:
    return json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "baseline_runs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_canonical_menu_loads_with_resolved_run_settings() -> None:
    config = load_baseline_config(CANONICAL_MANIFEST)

    assert config.schema == BASELINE_RUNS_SCHEMA
    assert config.planned_run_count == len(config.runs) == 9
    assert config.models["hgb_reference"].kind is ModelKind.HISTOGRAM_GRADIENT_BOOSTING
    assert config.models["rf_reference"].kind is ModelKind.RANDOM_FOREST
    assert config.runs[0].motion_mode is MotionMode.RAW_PER_FRAME
    assert config.runs[0].model_settings["max_leaf_nodes"] == 31
    assert config.runs[0].class_weight_value == "balanced"
    assert config.runs[-1].negative_limit == 24


def test_loaded_records_are_immutable() -> None:
    config = load_baseline_config(CANONICAL_MANIFEST)

    with pytest.raises(TypeError):
        config.models["hgb_reference"].settings["max_iter"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        config.runs += (config.runs[0],)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "other/1", "version differs"),
        ("planned_run_count", 8, "match runs"),
        ("run_limit", 8, "too large"),
        ("score_cutoffs", [0.1, 0.1], "strictly increasing"),
        ("duplicate_distances_at_30_fps", [6, 4], "sorted and unique"),
    ],
)
def test_menu_rejects_bad_shared_settings(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _canonical_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_more_than_twelve_runs(tmp_path: Path) -> None:
    payload = _canonical_payload()
    runs = payload["runs"]
    assert isinstance(runs, list)
    payload["planned_run_count"] = 13
    payload["run_limit"] = 13
    runs.extend(copy.deepcopy(runs[:4]))

    with pytest.raises(ValueError, match="too large"):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    payload = _canonical_payload()
    runs = payload["runs"]
    assert isinstance(runs, list)
    runs[1]["run_id"] = runs[0]["run_id"]

    with pytest.raises(ValueError, match="run IDs must be unique"):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_bad_radius_order(tmp_path: Path) -> None:
    payload = _canonical_payload()
    payload["ignored_radius_at_30_fps"] = payload["positive_radius_at_30_fps"]

    with pytest.raises(ValueError, match="radii must be in increasing order"):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_unknown_model_kind_and_wrong_settings(tmp_path: Path) -> None:
    payload = _canonical_payload()
    models = payload["models"]
    assert isinstance(models, dict)
    models["rf_reference"]["kind"] = "xgboost"

    with pytest.raises(ValueError, match="supported model"):
        load_baseline_config(_write_config(tmp_path, payload))

    payload = _canonical_payload()
    models = payload["models"]
    assert isinstance(models, dict)
    models["hgb_reference"]["max_iter"] = 0

    with pytest.raises(ValueError, match="must be positive"):
        load_baseline_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "missing_model", "model is unknown"),
        ("class_weight", "missing_weight", "class_weight is unknown"),
        ("negative_rule", "missing_rule", "negative_rule is unknown"),
        ("motion_mode", "missing_motion", "motion_mode is not supported"),
    ],
)
def test_menu_rejects_unknown_run_references(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = _canonical_payload()
    runs = payload["runs"]
    assert isinstance(runs, list)
    runs[0][field] = value

    with pytest.raises(ValueError, match=message):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_extra_run_fields(tmp_path: Path) -> None:
    payload = _canonical_payload()
    runs = payload["runs"]
    assert isinstance(runs, list)
    runs[0]["comment"] = "extra"

    with pytest.raises(ValueError, match=r"runs\[0\] fields differ"):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_a_valid_but_unplanned_setting(tmp_path: Path) -> None:
    payload = _canonical_payload()
    models = payload["models"]
    assert isinstance(models, dict)
    models["hgb_reference"]["max_iter"] = 181

    with pytest.raises(ValueError, match="run menu differs from the fixed comparison"):
        load_baseline_config(_write_config(tmp_path, payload))


def test_menu_rejects_a_different_tie_order(tmp_path: Path) -> None:
    payload = _canonical_payload()
    selection_order = payload["selection_order"]
    assert isinstance(selection_order, list)
    selection_order[-2:] = reversed(selection_order[-2:])

    with pytest.raises(ValueError, match="selection order differs"):
        load_baseline_config(_write_config(tmp_path, payload))
