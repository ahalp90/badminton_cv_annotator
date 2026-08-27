"""Load and check the fixed full-dataset contact-model runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

BASELINE_RUNS_SCHEMA = "full-dataset-contact-model-runs/1"
MAX_PLANNED_RUNS = 12
FIXED_MENU_SHA256 = "635b0e50fceff25cae5186d0b66e48cb7bda6d81e996b8719332b4b27a3e739e"
MOTION_MODE_NAMES = frozenset({"raw_per_frame", "base30_per_frame"})
EXPECTED_SELECTION_ORDER = (
    "higher timing F1",
    "higher timing recall",
    "higher timing precision",
    "larger duplicate distance",
    "higher score cutoff",
)
FIXED_RUN_IDS = (
    "hgb_reference_raw_balanced",
    "hgb_reference_common30_balanced",
    "rf_reference_raw_balanced",
    "rf_reference_common30_balanced",
    "hgb_reference_raw_no_weight",
    "rf_reference_raw_no_weight",
    "hgb_15_leaves_raw_balanced",
    "hgb_learning_rate_004_raw_balanced",
    "hgb_reference_raw_more_negatives",
)
MODEL_SETTINGS_FIELDS = {
    "histogram_gradient_boosting": frozenset(
        {"kind", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    ),
    "random_forest": frozenset({"kind", "n_estimators", "max_depth", "min_samples_leaf", "max_features"}),
}


class ModelKind(StrEnum):
    """The two model kinds allowed in the first comparison."""

    HISTOGRAM_GRADIENT_BOOSTING = "histogram_gradient_boosting"
    RANDOM_FOREST = "random_forest"


class MotionMode(StrEnum):
    """How motion values are represented in a run."""

    RAW_PER_FRAME = "raw_per_frame"
    BASE30_PER_FRAME = "base30_per_frame"


@dataclass(frozen=True)
class ModelSpec:
    """One named model and its settings."""

    name: str
    kind: ModelKind
    settings: Mapping[str, float | int | str]

    @property
    def parameters(self) -> Mapping[str, float | int | str]:
        """Return settings ready to pass to a model constructor."""
        return self.settings


@dataclass(frozen=True)
class BaselineRun:
    """One run with all named settings resolved for the scorer."""

    run_id: str
    motion_mode: MotionMode
    model: ModelSpec
    class_weight: str
    class_weight_value: str | None
    negative_rule: str
    negative_limit: int

    @property
    def model_name(self) -> str:
        """Return the model name from the run menu."""
        return self.model.name

    @property
    def model_kind(self) -> ModelKind:
        """Return the resolved model kind."""
        return self.model.kind

    @property
    def model_settings(self) -> Mapping[str, float | int | str]:
        """Return the resolved model settings."""
        return self.model.settings


@dataclass(frozen=True)
class BaselineConfig:
    """The checked run menu and shared settings for the first comparison."""

    schema: str
    run_limit: int
    planned_run_count: int
    random_seed: int
    feature_inputs: tuple[str, ...]
    motion_modes: Mapping[str, str]
    positive_radius_at_30_fps: int
    ignored_radius_at_30_fps: int
    hard_negative_radius_at_30_fps: int
    negative_rules: Mapping[str, int]
    score_cutoffs: tuple[float, ...]
    duplicate_distances_at_30_fps: tuple[int, ...]
    timing_tolerance_at_30_fps: int
    selection_order: tuple[str, ...]
    models: Mapping[str, ModelSpec]
    class_weight_values: Mapping[str, Mapping[str, str | None]]
    runs: tuple[BaselineRun, ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def _positive_integer(value: object, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = tuple(_non_empty_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    if not result:
        raise ValueError(f"{label} must not be empty")
    return result


def _strictly_increasing_numbers(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if not result or any(cutoff <= 0 or cutoff >= 1 for cutoff in result):
        raise ValueError(f"{label} must be between 0 and 1")
    if result != tuple(sorted(result)) or any(left >= right for left, right in pairwise(result)):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def _positive_sorted_integers(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = tuple(_positive_integer(item, f"{label}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{label} must not be empty")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _model_kind(value: object, label: str) -> ModelKind:
    try:
        return ModelKind(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a supported model") from error


def _model_spec(raw: object, name: str) -> ModelSpec:
    row = _mapping(raw, f"models.{name}")
    model_kind = _model_kind(row.get("kind"), f"models.{name}.kind")
    expected_fields = MODEL_SETTINGS_FIELDS[model_kind.value]
    if set(row) != expected_fields:
        raise ValueError(f"models.{name} fields differ")

    if model_kind is ModelKind.HISTOGRAM_GRADIENT_BOOSTING:
        settings: dict[str, float | int | str] = {
            "learning_rate": _positive_number(row["learning_rate"], f"models.{name}.learning_rate"),
            "max_iter": _positive_integer(row["max_iter"], f"models.{name}.max_iter"),
            "max_leaf_nodes": _positive_integer(row["max_leaf_nodes"], f"models.{name}.max_leaf_nodes"),
            "min_samples_leaf": _positive_integer(row["min_samples_leaf"], f"models.{name}.min_samples_leaf"),
            "l2_regularization": _positive_number(
                row["l2_regularization"], f"models.{name}.l2_regularization"
            ),
        }
    else:
        max_features = row["max_features"]
        if max_features != "sqrt":
            raise ValueError(f"models.{name}.max_features must be sqrt")
        settings = {
            "n_estimators": _positive_integer(row["n_estimators"], f"models.{name}.n_estimators"),
            "max_depth": _positive_integer(row["max_depth"], f"models.{name}.max_depth"),
            "min_samples_leaf": _positive_integer(row["min_samples_leaf"], f"models.{name}.min_samples_leaf"),
            "max_features": max_features,
        }
    return ModelSpec(name=name, kind=model_kind, settings=MappingProxyType(settings))


def _model_specs(value: object) -> Mapping[str, ModelSpec]:
    raw_models = _mapping(value, "models")
    if not raw_models:
        raise ValueError("models must not be empty")
    models = {
        _non_empty_string(name, "model name"): _model_spec(raw_model, name)
        for name, raw_model in raw_models.items()
    }
    kinds = {model.kind for model in models.values()}
    expected_kinds = set(ModelKind)
    if kinds != expected_kinds:
        raise ValueError("models must contain HGB and RF")
    return MappingProxyType(models)


def _class_weight_values(value: object) -> Mapping[str, Mapping[str, str | None]]:
    raw_values = _mapping(value, "class_weight_values")
    if not raw_values:
        raise ValueError("class_weight_values must not be empty")
    values: dict[str, Mapping[str, str | None]] = {}
    expected_kinds = {kind.value for kind in ModelKind}
    for name, raw_by_kind in raw_values.items():
        weight_name = _non_empty_string(name, "class weight name")
        by_kind = _mapping(raw_by_kind, f"class_weight_values.{weight_name}")
        if set(by_kind) != expected_kinds:
            raise ValueError(f"class_weight_values.{weight_name} model kinds differ")
        resolved: dict[str, str | None] = {}
        for kind_name, raw_weight in by_kind.items():
            if raw_weight is not None:
                resolved[kind_name] = _non_empty_string(
                    raw_weight, f"class_weight_values.{weight_name}.{kind_name}"
                )
            else:
                resolved[kind_name] = None
        values[weight_name] = MappingProxyType(resolved)
    return MappingProxyType(values)


def _negative_rules(value: object) -> Mapping[str, int]:
    raw_rules = _mapping(value, "negative_rules")
    if not raw_rules:
        raise ValueError("negative_rules must not be empty")
    rules = {
        _non_empty_string(name, "negative rule name"): _positive_integer(limit, f"negative_rules.{name}")
        for name, limit in raw_rules.items()
    }
    return MappingProxyType(rules)


def _run(
    raw: object,
    index: int,
    models: Mapping[str, ModelSpec],
    weights: Mapping[str, Mapping[str, str | None]],
    rules: Mapping[str, int],
    descriptions: Mapping[str, str],
) -> BaselineRun:
    row = _mapping(raw, f"runs[{index}]")
    expected_fields = {"run_id", "motion_mode", "model", "class_weight", "negative_rule"}
    if set(row) != expected_fields:
        raise ValueError(f"runs[{index}] fields differ")
    run_id = _non_empty_string(row["run_id"], f"runs[{index}].run_id")
    try:
        motion_mode = MotionMode(row["motion_mode"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"runs[{index}].motion_mode is not supported") from error
    if motion_mode.value not in descriptions:
        raise ValueError(f"runs[{index}].motion_mode is not described")
    model_name = _non_empty_string(row["model"], f"runs[{index}].model")
    if model_name not in models:
        raise ValueError(f"runs[{index}].model is unknown")
    class_weight = _non_empty_string(row["class_weight"], f"runs[{index}].class_weight")
    if class_weight not in weights:
        raise ValueError(f"runs[{index}].class_weight is unknown")
    negative_rule = _non_empty_string(row["negative_rule"], f"runs[{index}].negative_rule")
    if negative_rule not in rules:
        raise ValueError(f"runs[{index}].negative_rule is unknown")
    model = models[model_name]
    class_weight_value = weights[class_weight][model.kind.value]
    return BaselineRun(
        run_id=run_id,
        motion_mode=motion_mode,
        model=model,
        class_weight=class_weight,
        class_weight_value=class_weight_value,
        negative_rule=negative_rule,
        negative_limit=rules[negative_rule],
    )


def load_baseline_config(path: Path) -> BaselineConfig:
    """Load the run menu and check every setting and reference."""
    config_path = Path(path)
    payload = _mapping(json.loads(config_path.read_text(encoding="utf-8")), "config file")
    expected_fields = {
        "schema",
        "run_limit",
        "planned_run_count",
        "random_seed",
        "feature_inputs",
        "motion_modes",
        "positive_radius_at_30_fps",
        "ignored_radius_at_30_fps",
        "hard_negative_radius_at_30_fps",
        "negative_rules",
        "score_cutoffs",
        "duplicate_distances_at_30_fps",
        "timing_tolerance_at_30_fps",
        "selection_order",
        "models",
        "class_weight_values",
        "runs",
    }
    if set(payload) != expected_fields:
        raise ValueError("config file fields differ")
    if payload["schema"] != BASELINE_RUNS_SCHEMA:
        raise ValueError("config file version differs")

    run_limit = _positive_integer(payload["run_limit"], "run_limit")
    planned_run_count = _positive_integer(payload["planned_run_count"], "planned_run_count")
    raw_runs = payload["runs"]
    if not isinstance(raw_runs, list):
        raise TypeError("runs must be a list")
    if planned_run_count != len(raw_runs):
        raise ValueError("planned run count must match runs")
    if planned_run_count > run_limit or planned_run_count > MAX_PLANNED_RUNS:
        raise ValueError("planned run count is too large")

    raw_motion_modes = _mapping(payload["motion_modes"], "motion_modes")
    if set(raw_motion_modes) != MOTION_MODE_NAMES:
        raise ValueError("motion modes differ")
    motion_modes = MappingProxyType(
        {
            _non_empty_string(name, "motion mode name"): _non_empty_string(description, f"motion_modes.{name}")
            for name, description in raw_motion_modes.items()
        }
    )
    feature_inputs = _string_list(payload["feature_inputs"], "feature_inputs")
    random_seed = _integer(payload["random_seed"], "random_seed")
    positive_radius = _positive_integer(payload["positive_radius_at_30_fps"], "positive radius")
    ignored_radius = _positive_integer(payload["ignored_radius_at_30_fps"], "ignored radius")
    hard_negative_radius = _positive_integer(payload["hard_negative_radius_at_30_fps"], "hard negative radius")
    if not positive_radius < ignored_radius < hard_negative_radius:
        raise ValueError("radii must be in increasing order")
    negative_rules = _negative_rules(payload["negative_rules"])
    score_cutoffs = _strictly_increasing_numbers(payload["score_cutoffs"], "score_cutoffs")
    duplicate_distances = _positive_sorted_integers(
        payload["duplicate_distances_at_30_fps"], "duplicate distances"
    )
    timing_tolerance = _positive_integer(payload["timing_tolerance_at_30_fps"], "timing tolerance")
    selection_order = _string_list(payload["selection_order"], "selection_order")
    if selection_order != EXPECTED_SELECTION_ORDER:
        raise ValueError("selection order differs from the fixed comparison")
    models = _model_specs(payload["models"])
    class_weight_values = _class_weight_values(payload["class_weight_values"])
    runs = tuple(
        _run(raw_run, index, models, class_weight_values, negative_rules, motion_modes)
        for index, raw_run in enumerate(raw_runs)
    )
    run_ids = tuple(run.run_id for run in runs)
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run IDs must be unique")
    if run_ids != FIXED_RUN_IDS:
        raise ValueError("run IDs differ from the fixed comparison")
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != FIXED_MENU_SHA256:
        raise ValueError("run menu differs from the fixed comparison")

    return BaselineConfig(
        schema=BASELINE_RUNS_SCHEMA,
        run_limit=run_limit,
        planned_run_count=planned_run_count,
        random_seed=random_seed,
        feature_inputs=feature_inputs,
        motion_modes=motion_modes,
        positive_radius_at_30_fps=positive_radius,
        ignored_radius_at_30_fps=ignored_radius,
        hard_negative_radius_at_30_fps=hard_negative_radius,
        negative_rules=negative_rules,
        score_cutoffs=score_cutoffs,
        duplicate_distances_at_30_fps=duplicate_distances,
        timing_tolerance_at_30_fps=timing_tolerance,
        selection_order=selection_order,
        models=models,
        class_weight_values=class_weight_values,
        runs=runs,
    )


load_baseline_runs = load_baseline_config
