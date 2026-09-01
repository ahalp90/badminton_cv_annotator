"""Load the fixed rally-start contact model comparison."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

CONFIG_SCHEMA = "rally-start-contact-model-runs/1"
CONFIG_SHA256 = "8394fb9ba2a41df8384f8ebe07ed906ab431413f6045c64826cbbeb39897761c"
FEATURE_NAMES = (
    "candidate_contact_score",
    "fixed_contact_score",
    "frames_before_fixed_at_30_fps",
    "candidate_from_section_start_at_30_fps",
    "section_length_at_30_fps",
    "candidate_already_kept",
    "candidate_side_known",
    "fixed_side_known",
    "candidate_and_fixed_side_match",
)
MODEL_IDS = ("logistic_regression", "shallow_hgb")
SELECTION_CUTOFFS = (0.5, 0.7, 0.9)
TRAINING_GROUPS = ("A", "B", "C", "D")
CHOICE_ORDER = (
    "more fully correct sections",
    "fewer added contacts",
    "higher correct addition rate",
    "logistic regression",
    "higher selection cut-off",
)
TOP_LEVEL_FIELDS = {
    "schema",
    "random_seed",
    "feature_names",
    "models",
    "selection_cutoffs",
    "training_groups",
    "training_gate",
    "validation_gate",
    "choice_order",
}


class ModelKind(StrEnum):
    """The two model types in the fixed comparison."""

    LOGISTIC_REGRESSION = "logistic_regression"
    HISTOGRAM_GRADIENT_BOOSTING = "histogram_gradient_boosting"


@dataclass(frozen=True)
class ModelSpec:
    """One fixed model and its constructor settings."""

    model_id: str
    kind: ModelKind
    settings: Mapping[str, object]


@dataclass(frozen=True)
class ResultGate:
    """Minimum result needed to keep a model."""

    minimum_new_fully_correct_sections: int
    maximum_lost_fully_correct_sections: int
    minimum_correct_addition_rate: float
    minimum_recovery_rate: float
    minimum_contact_f1_change_at_10_frames: float | None = None
    allow_per_video_fully_correct_loss: bool | None = None


@dataclass(frozen=True)
class RallyStartModelConfig:
    """The complete fixed model comparison."""

    random_seed: int
    feature_names: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    selection_cutoffs: tuple[float, ...]
    training_groups: tuple[str, ...]
    training_gate: ResultGate
    validation_gate: ResultGate
    choice_order: tuple[str, ...]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be an object with string keys")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"{label} must be a string list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not repeat values")
    return result


def _model_spec(value: object, expected_id: str) -> ModelSpec:
    row = _mapping(value, f"model {expected_id}")
    if set(row) != {"model_id", "kind", "settings"} or row["model_id"] != expected_id:
        raise ValueError(f"model {expected_id} fields differ")
    try:
        kind = ModelKind(row["kind"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"model {expected_id} kind differs") from error
    settings = _mapping(row["settings"], f"model {expected_id} settings")
    return ModelSpec(expected_id, kind, MappingProxyType(dict(settings)))


def _gate(value: object, label: str, *, validation: bool) -> ResultGate:
    row = _mapping(value, label)
    common_fields = {
        "minimum_new_fully_correct_sections",
        "maximum_lost_fully_correct_sections",
        "minimum_correct_addition_rate",
        "minimum_recovery_rate",
    }
    validation_fields = {
        "minimum_contact_f1_change_at_10_frames",
        "allow_per_video_fully_correct_loss",
    }
    expected_fields = common_fields | validation_fields if validation else common_fields
    if set(row) != expected_fields:
        raise ValueError(f"{label} fields differ")
    new_count = row["minimum_new_fully_correct_sections"]
    lost_count = row["maximum_lost_fully_correct_sections"]
    correct_rate = row["minimum_correct_addition_rate"]
    recovery_rate = row["minimum_recovery_rate"]
    if (
        type(new_count) is not int
        or type(lost_count) is not int
        or new_count < 0
        or lost_count < 0
        or isinstance(correct_rate, bool)
        or not isinstance(correct_rate, (int, float))
        or isinstance(recovery_rate, bool)
        or not isinstance(recovery_rate, (int, float))
        or not 0.0 <= float(correct_rate) <= 1.0
        or not 0.0 <= float(recovery_rate) <= 1.0
    ):
        raise ValueError(f"{label} values differ")
    contact_change = row.get("minimum_contact_f1_change_at_10_frames")
    per_video_loss = row.get("allow_per_video_fully_correct_loss")
    if validation and (
        isinstance(contact_change, bool)
        or not isinstance(contact_change, (int, float))
        or type(per_video_loss) is not bool
    ):
        raise ValueError(f"{label} validation values differ")
    return ResultGate(
        new_count,
        lost_count,
        float(correct_rate),
        float(recovery_rate),
        None if contact_change is None else float(contact_change),
        per_video_loss if isinstance(per_video_loss, bool) else None,
    )


def load_rally_start_model_config(path: Path) -> RallyStartModelConfig:
    """Load the exact reviewed comparison file."""
    source_path = Path(path)
    if _sha256(source_path) != CONFIG_SHA256:
        raise ValueError("rally-start model comparison file hash differs")
    payload = _mapping(
        json.loads(source_path.read_text(encoding="utf-8")),
        "rally-start model comparison",
    )
    if set(payload) != TOP_LEVEL_FIELDS or payload["schema"] != CONFIG_SCHEMA:
        raise ValueError("rally-start model comparison fields differ")
    if payload["random_seed"] != 20260824:
        raise ValueError("rally-start model random seed differs")
    feature_names = _strict_string_list(payload["feature_names"], "feature names")
    cutoffs = tuple(float(value) for value in payload["selection_cutoffs"])
    groups = _strict_string_list(payload["training_groups"], "training groups")
    choice_order = _strict_string_list(payload["choice_order"], "choice order")
    raw_models = payload["models"]
    if not isinstance(raw_models, list) or len(raw_models) != len(MODEL_IDS):
        raise ValueError("rally-start model list differs")
    models = tuple(
        _model_spec(value, expected_id)
        for value, expected_id in zip(raw_models, MODEL_IDS, strict=True)
    )
    if (
        feature_names != FEATURE_NAMES
        or cutoffs != SELECTION_CUTOFFS
        or groups != TRAINING_GROUPS
        or choice_order != CHOICE_ORDER
    ):
        raise ValueError("rally-start model comparison choices differ")
    return RallyStartModelConfig(
        20260824,
        feature_names,
        models,
        cutoffs,
        groups,
        _gate(payload["training_gate"], "training gate", validation=False),
        _gate(payload["validation_gate"], "validation gate", validation=True),
        choice_order,
    )
