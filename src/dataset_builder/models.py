"""Typed run-manifest contracts for the dataset builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import Any, TypeAlias

from frozendict import frozendict


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue: TypeAlias = JsonScalar | tuple["FrozenJsonValue", ...] | frozendict[str, "FrozenJsonValue"]
FrozenJsonObject: TypeAlias = frozendict[str, FrozenJsonValue]

RUN_MANIFEST_SCHEMA = "dataset-builder-run/0.1"
_MD5_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class StageOutcome(StrEnum):
    """Terminal outcome of one stage invocation."""

    PROCESSED = "processed"
    SKIPPED = "skipped"
    EXCLUDED = "excluded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ArtifactIntegrity:
    """Persisted identity and integrity for one regular file."""

    name: str
    path: str
    md5: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("artifact name must be non-empty")
        if not self.path:
            raise ValueError("artifact path must be non-empty")
        if not _MD5_PATTERN.fullmatch(self.md5):
            raise ValueError(f"artifact md5 must be 32 lowercase hexadecimal characters: {self.md5!r}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError(f"artifact size_bytes must be a non-negative integer: {self.size_bytes!r}")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "path": self.path,
            "md5": self.md5,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: object) -> ArtifactIntegrity:
        record = _object(payload, "artifact")
        _require_exact_keys(record, {"name", "path", "md5", "size_bytes"}, "artifact")
        return cls(
            name=_string(record["name"], "artifact.name"),
            path=_string(record["path"], "artifact.path"),
            md5=_string(record["md5"], "artifact.md5"),
            size_bytes=_integer(record["size_bytes"], "artifact.size_bytes"),
        )


@dataclass(frozen=True)
class InterpreterIdentity:
    """Resolved executable and its reported version."""

    path: str
    version: str

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("interpreter path must be non-empty")
        if not self.version:
            raise ValueError("interpreter version must be non-empty")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"path": self.path, "version": self.version}

    @classmethod
    def from_dict(cls, payload: object) -> InterpreterIdentity:
        record = _object(payload, "interpreter")
        _require_exact_keys(record, {"path", "version"}, "interpreter")
        return cls(
            path=_string(record["path"], "interpreter.path"),
            version=_string(record["version"], "interpreter.version"),
        )


@dataclass(frozen=True)
class StageFingerprint:
    """Complete identity of the code, configuration, runtime and inputs."""

    digest: str
    source_commit: str
    contract_version: str
    configuration_sha256: str
    interpreter: InterpreterIdentity
    model_weights: tuple[ArtifactIntegrity, ...]
    inputs: tuple[ArtifactIntegrity, ...]

    def __post_init__(self) -> None:
        if not _SHA256_PATTERN.fullmatch(self.digest):
            raise ValueError(f"fingerprint digest must be 64 lowercase hexadecimal characters: {self.digest!r}")
        if not self.source_commit:
            raise ValueError("fingerprint source_commit must be non-empty")
        if not self.contract_version:
            raise ValueError("fingerprint contract_version must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.configuration_sha256):
            raise ValueError("fingerprint configuration_sha256 must be 64 lowercase hexadecimal characters")
        _require_unique_artifact_names(self.model_weights, "model_weights")
        _require_unique_artifact_names(self.inputs, "inputs")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "digest": self.digest,
            "source_commit": self.source_commit,
            "contract_version": self.contract_version,
            "configuration_sha256": self.configuration_sha256,
            "interpreter": self.interpreter.to_dict(),
            "model_weights": [artifact.to_dict() for artifact in self.model_weights],
            "inputs": [artifact.to_dict() for artifact in self.inputs],
        }

    @classmethod
    def from_dict(cls, payload: object) -> StageFingerprint:
        record = _object(payload, "fingerprint")
        keys = {
            "digest",
            "source_commit",
            "contract_version",
            "configuration_sha256",
            "interpreter",
            "model_weights",
            "inputs",
        }
        _require_exact_keys(record, keys, "fingerprint")
        return cls(
            digest=_string(record["digest"], "fingerprint.digest"),
            source_commit=_string(record["source_commit"], "fingerprint.source_commit"),
            contract_version=_string(record["contract_version"], "fingerprint.contract_version"),
            configuration_sha256=_string(
                record["configuration_sha256"], "fingerprint.configuration_sha256"
            ),
            interpreter=InterpreterIdentity.from_dict(record["interpreter"]),
            model_weights=_artifacts(record["model_weights"], "fingerprint.model_weights"),
            inputs=_artifacts(record["inputs"], "fingerprint.inputs"),
        )


@dataclass(frozen=True)
class SemanticValidation:
    """Named semantic check captured after a stage writes its outputs."""

    name: str
    passed: bool
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("semantic validation name must be non-empty")
        if not isinstance(self.passed, bool):
            raise ValueError(f"semantic validation passed must be bool: {self.passed!r}")
        if self.detail is not None and not self.detail:
            raise ValueError("semantic validation detail must be non-empty when present")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}

    @classmethod
    def from_dict(cls, payload: object) -> SemanticValidation:
        record = _object(payload, "semantic validation")
        _require_exact_keys(record, {"name", "passed", "detail"}, "semantic validation")
        detail = record["detail"]
        if detail is not None:
            detail = _string(detail, "semantic_validation.detail")
        passed = record["passed"]
        if not isinstance(passed, bool):
            raise ValueError(f"semantic_validation.passed must be bool, got {passed!r}")
        return cls(
            name=_string(record["name"], "semantic_validation.name"),
            passed=passed,
            detail=detail,
        )


@dataclass(frozen=True)
class StageRecord:
    """Auditable terminal record for one named stage."""

    name: str
    outcome: StageOutcome
    fingerprint: StageFingerprint
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    configuration: FrozenJsonObject
    outputs: tuple[ArtifactIntegrity, ...]
    counts: tuple[tuple[str, int], ...]
    elapsed_seconds: float
    semantic_validation: tuple[SemanticValidation, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("stage name must be non-empty")
        if not isinstance(self.outcome, StageOutcome):
            raise ValueError(f"stage outcome must be StageOutcome, got {self.outcome!r}")
        object.__setattr__(self, "configuration", freeze_json_object(self.configuration))
        _require_unique_strings(self.dependencies, "stage dependencies")
        if any(not dependency for dependency in self.dependencies):
            raise ValueError("stage dependencies must be non-empty")
        if self.name in self.dependencies:
            raise ValueError(f"stage {self.name!r} cannot depend on itself")
        if not self.command or any(not part for part in self.command):
            raise ValueError("stage command arguments must be non-empty")
        _require_unique_artifact_names(self.outputs, "outputs")
        count_names = tuple(name for name, _ in self.counts)
        _require_unique_strings(count_names, "stage count names")
        for name, value in self.counts:
            if not name:
                raise ValueError("stage count names must be non-empty")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"stage count {name!r} must be a non-negative integer: {value!r}")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError(f"stage elapsed_seconds must be finite and non-negative: {self.elapsed_seconds!r}")
        _require_unique_strings(
            tuple(validation.name for validation in self.semantic_validation),
            "semantic validation names",
        )
        if self.outcome in {
            StageOutcome.SKIPPED,
            StageOutcome.EXCLUDED,
            StageOutcome.FAILED,
            StageOutcome.UNAVAILABLE,
        }:
            if not self.reason:
                raise ValueError(f"stage outcome {self.outcome.value!r} requires a reason")
        elif self.reason is not None and not self.reason:
            raise ValueError("stage reason must be non-empty when present")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "outcome": self.outcome.value,
            "fingerprint": self.fingerprint.to_dict(),
            "dependencies": list(self.dependencies),
            "command": list(self.command),
            "configuration": thaw_json_object(self.configuration),
            "outputs": [artifact.to_dict() for artifact in self.outputs],
            "counts": {name: value for name, value in self.counts},
            "elapsed_seconds": self.elapsed_seconds,
            "semantic_validation": [validation.to_dict() for validation in self.semantic_validation],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StageRecord:
        record = _object(payload, "stage")
        keys = {
            "name",
            "outcome",
            "fingerprint",
            "dependencies",
            "command",
            "configuration",
            "outputs",
            "counts",
            "elapsed_seconds",
            "semantic_validation",
            "reason",
        }
        _require_exact_keys(record, keys, "stage")
        outcome_text = _string(record["outcome"], "stage.outcome")
        try:
            outcome = StageOutcome(outcome_text)
        except ValueError as exc:
            raise ValueError(f"unknown stage outcome: {outcome_text!r}") from exc
        reason = record["reason"]
        if reason is not None:
            reason = _string(reason, "stage.reason")
        counts = _object(record["counts"], "stage.counts")
        configuration = freeze_json_object(_json_object(record["configuration"], "stage.configuration"))
        return cls(
            name=_string(record["name"], "stage.name"),
            outcome=outcome,
            fingerprint=StageFingerprint.from_dict(record["fingerprint"]),
            dependencies=_strings(record["dependencies"], "stage.dependencies"),
            command=_strings(record["command"], "stage.command"),
            configuration=configuration,
            outputs=_artifacts(record["outputs"], "stage.outputs"),
            counts=tuple(sorted(
                (_string(name, "stage.counts key"), _integer(value, f"stage.counts.{name}"))
                for name, value in counts.items()
            )),
            elapsed_seconds=_number(record["elapsed_seconds"], "stage.elapsed_seconds"),
            semantic_validation=_validations(record["semantic_validation"]),
            reason=reason,
        )


@dataclass(frozen=True)
class RunManifest:
    """Immutable snapshot of a dataset-builder run manifest."""

    run_id: str
    created_at_utc: str
    stages: tuple[StageRecord, ...] = ()
    schema: str = RUN_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.created_at_utc:
            raise ValueError("created_at_utc must be non-empty")
        if self.schema != RUN_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported run manifest schema: {self.schema!r}")
        _require_unique_strings(tuple(stage.name for stage in self.stages), "stage names")
        _validate_stage_graph(self.stages)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, payload: object) -> RunManifest:
        record = _object(payload, "run manifest")
        _require_exact_keys(record, {"schema", "run_id", "created_at_utc", "stages"}, "run manifest")
        stages = record["stages"]
        if not isinstance(stages, list):
            raise ValueError(f"run manifest stages must be a list, got {type(stages).__name__}")
        return cls(
            schema=_string(record["schema"], "run_manifest.schema"),
            run_id=_string(record["run_id"], "run_manifest.run_id"),
            created_at_utc=_string(record["created_at_utc"], "run_manifest.created_at_utc"),
            stages=tuple(StageRecord.from_dict(stage) for stage in stages),
        )


@dataclass(frozen=True)
class ReuseDecision:
    """Result of checking one recorded stage for safe reuse."""

    reusable: bool
    reason: str
    invalidated_stages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reuse decision reason must be non-empty")
        if self.reusable and self.invalidated_stages:
            raise ValueError("a reusable stage cannot invalidate stage records")


def freeze_json_object(payload: Mapping[str, object]) -> FrozenJsonObject:
    """Recursively copy a JSON object into immutable mappings and tuples."""
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("frozen JSON object keys must be strings")
    return frozendict({key: _freeze_json_value(value, f"configuration.{key}") for key, value in payload.items()})


def thaw_json_object(payload: Mapping[str, FrozenJsonValue]) -> dict[str, JsonValue]:
    """Materialise a fresh mutable JSON tree for serialization."""
    return {key: _thaw_json_value(value) for key, value in payload.items()}


def _object(payload: object, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be a JSON object")
    return payload


def _json_object(payload: object, name: str) -> dict[str, JsonValue]:
    record = _object(payload, name)
    return {key: _json_value(value, f"{name}.{key}") for key, value in record.items()}


def _json_value(payload: object, name: str) -> JsonValue:
    if payload is None or isinstance(payload, (bool, str)):
        return payload
    if isinstance(payload, int):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError(f"{name} must be finite")
        return payload
    if isinstance(payload, list):
        return [_json_value(value, f"{name}[]") for value in payload]
    if isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        return {key: _json_value(value, f"{name}.{key}") for key, value in payload.items()}
    raise ValueError(f"{name} is not a JSON value: {payload!r}")


def _freeze_json_value(payload: object, name: str) -> FrozenJsonValue:
    if payload is None or isinstance(payload, (bool, str)):
        return payload
    if isinstance(payload, int):
        return payload
    if isinstance(payload, float):
        if not math.isfinite(payload):
            raise ValueError(f"{name} must be finite")
        return payload
    if isinstance(payload, (list, tuple)):
        return tuple(_freeze_json_value(value, f"{name}[]") for value in payload)
    if isinstance(payload, Mapping):
        if any(not isinstance(key, str) for key in payload):
            raise ValueError(f"{name} keys must be strings")
        return frozendict({
            key: _freeze_json_value(value, f"{name}.{key}")
            for key, value in payload.items()
        })
    raise ValueError(f"{name} is not a JSON value: {payload!r}")


def _thaw_json_value(payload: FrozenJsonValue) -> JsonValue:
    if isinstance(payload, tuple):
        return [_thaw_json_value(value) for value in payload]
    if isinstance(payload, Mapping):
        return {key: _thaw_json_value(value) for key, value in payload.items()}
    return payload


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(f"{name} keys differ: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return float(value)


def _strings(payload: object, name: str) -> tuple[str, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return tuple(_string(value, f"{name}[]") for value in payload)


def _artifacts(payload: object, name: str) -> tuple[ArtifactIntegrity, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return tuple(ArtifactIntegrity.from_dict(value) for value in payload)


def _validations(payload: object) -> tuple[SemanticValidation, ...]:
    if not isinstance(payload, list):
        raise ValueError("stage.semantic_validation must be a list")
    return tuple(SemanticValidation.from_dict(value) for value in payload)


def _require_unique_strings(values: tuple[str, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique: {values!r}")


def _require_unique_artifact_names(artifacts: tuple[ArtifactIntegrity, ...], name: str) -> None:
    _require_unique_strings(tuple(artifact.name for artifact in artifacts), f"{name} artifact names")


def _validate_stage_graph(stages: tuple[StageRecord, ...]) -> None:
    known: set[str] = set()
    for stage in stages:
        missing = set(stage.dependencies) - known
        if missing:
            raise ValueError(
                f"stage {stage.name!r} has dependencies that are absent or ordered after it: {sorted(missing)}"
            )
        known.add(stage.name)
