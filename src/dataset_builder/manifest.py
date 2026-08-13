"""Atomic persistence, fingerprinting and resume checks for dataset-builder runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4

from dataset_builder.models import (
    ArtifactIntegrity,
    InterpreterIdentity,
    JsonValue,
    ReuseDecision,
    RunManifest,
    SemanticValidation,
    StageFingerprint,
    StageOutcome,
    StageRecord,
    freeze_json_object,
)


MANIFEST_FILENAME = "run_manifest.json.gz"
REDACTED = "<redacted>"
_SECRET_KEY_MARKERS = (
    "api_key",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_REUSABLE_OUTCOMES = frozenset({StageOutcome.PROCESSED, StageOutcome.SKIPPED, StageOutcome.EXCLUDED})

SemanticValidator = Callable[[Path], bool]


def normalise_configuration(configuration: Mapping[str, object]) -> dict[str, JsonValue]:
    """Convert an effective configuration into deterministic JSON values."""
    return {
        key: _normalise_json(value, f"configuration.{key}")
        for key, value in sorted(configuration.items())
    }


def run_manifest_sha256(manifest: RunManifest) -> str:
    """Return the canonical SHA-256 identity of a complete manifest snapshot."""
    if not isinstance(manifest, RunManifest):
        raise TypeError("manifest must be RunManifest")
    return _sha256_json(manifest.to_dict())


def redact_configuration(
    configuration: Mapping[str, object],
    *,
    secret_keys: Sequence[str] = (),
    secret_values: Sequence[str] = (),
) -> dict[str, JsonValue]:
    """Normalise a configuration and replace secret keys and values."""
    normalised = normalise_configuration(configuration)
    explicit_keys = {_normalise_key(key) for key in secret_keys}
    values = tuple(value for value in secret_values if value)
    return _redact_mapping(normalised, explicit_keys, values)


def redact_command(
    command: Sequence[str],
    *,
    secret_options: Sequence[str] = (),
    secret_values: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return command arguments with named option values and literal secrets removed."""
    explicit_options = {_normalise_option(option) for option in secret_options}
    explicit_short_options = tuple(
        option for option in secret_options if option.startswith("-") and not option.startswith("--")
    )
    values = tuple(value for value in secret_values if value)
    redacted: list[str] = []
    hide_next = False
    for raw_argument in command:
        argument = str(raw_argument)
        if hide_next:
            redacted.append(REDACTED)
            hide_next = False
            continue
        attached_short_option: str | None = None
        for short_option in explicit_short_options:
            if argument.startswith(short_option) and len(argument) > len(short_option):
                attached_short_option = short_option
                break
        if attached_short_option is not None and not argument[len(attached_short_option):].startswith("="):
            redacted.append(f"{attached_short_option}{REDACTED}")
            continue
        option, separator, _ = argument.partition("=")
        normalised_option = _normalise_option(option)
        explicitly_secret = option.startswith("-") and normalised_option in explicit_options
        automatically_secret = option.startswith("--") and _is_secret_key(normalised_option)
        if explicitly_secret or automatically_secret:
            if separator:
                redacted.append(f"{option}={REDACTED}")
            else:
                redacted.append(option)
                hide_next = True
            continue
        if separator and not option.startswith("--") and _is_secret_key(_normalise_key(option)):
            redacted.append(f"{option}={REDACTED}")
            continue
        redacted.append(_redact_string(argument, values))
    return tuple(redacted)


def resolve_interpreter(
    executable: str | Path,
    *,
    version_option: str = "--version",
) -> InterpreterIdentity:
    """Resolve an executable and capture its version string."""
    requested = os.fspath(executable)
    located = shutil.which(requested)
    if located is None:
        candidate = Path(requested)
        if not candidate.is_file():
            raise FileNotFoundError(f"interpreter is not an executable file: {requested}")
        located = os.fspath(candidate)
    path = Path(located).resolve(strict=True)
    if not os.access(path, os.X_OK):
        raise PermissionError(f"interpreter is not executable: {path}")
    try:
        completed = subprocess.run(
            [os.fspath(path), version_option],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError as exc:
        raise RuntimeError(f"could not run interpreter {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"interpreter {path} {version_option} failed with exit status "
            f"{completed.returncode}: {detail}"
        )
    version = (completed.stdout or completed.stderr).strip()
    if not version:
        raise RuntimeError(f"interpreter {path} {version_option} returned no version text")
    return InterpreterIdentity(path=os.fspath(path), version=version)


def artifact_integrity(name: str, path: Path, *, relative_to: Path | None = None) -> ArtifactIntegrity:
    """Hash one regular file after resolving its persisted manifest path."""
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {candidate}")
    if relative_to is not None and candidate.is_symlink():
        raise FileNotFoundError(f"output artifact must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    stored_path = os.fspath(resolved)
    if relative_to is not None:
        root = Path(relative_to).resolve(strict=True)
        try:
            stored_path = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"artifact {resolved} is outside run directory {root}") from exc
    return ArtifactIntegrity(
        name=name,
        path=stored_path,
        md5=_md5_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def build_stage_fingerprint(
    *,
    source_commit: str,
    contract_version: str,
    effective_configuration: Mapping[str, object],
    interpreter: InterpreterIdentity,
    model_weights: Mapping[str, Path] | None = None,
    inputs: Mapping[str, Path] | None = None,
) -> StageFingerprint:
    """Build the stable identity that gates reuse of one stage."""
    configuration = normalise_configuration(effective_configuration)
    configuration_sha256 = _sha256_json(configuration)
    weight_records = _artifact_records(model_weights or {})
    input_records = _artifact_records(inputs or {})
    components: dict[str, JsonValue] = {
        "source_commit": _nonempty(source_commit, "source_commit"),
        "contract_version": _nonempty(contract_version, "contract_version"),
        "configuration_sha256": configuration_sha256,
        "interpreter": interpreter.to_dict(),
        "model_weights": [record.to_dict() for record in weight_records],
        "inputs": [record.to_dict() for record in input_records],
    }
    return StageFingerprint(
        digest=_sha256_json(components),
        source_commit=source_commit,
        contract_version=contract_version,
        configuration_sha256=configuration_sha256,
        interpreter=interpreter,
        model_weights=weight_records,
        inputs=input_records,
    )


def make_stage_record(
    *,
    name: str,
    outcome: StageOutcome,
    fingerprint: StageFingerprint,
    run_dir: Path,
    command: Sequence[str],
    effective_configuration: Mapping[str, object],
    outputs: Mapping[str, Path] | None = None,
    dependencies: Sequence[str] = (),
    counts: Mapping[str, int] | None = None,
    elapsed_seconds: float,
    semantic_validation: Sequence[SemanticValidation] = (),
    reason: str | None = None,
    secret_keys: Sequence[str] = (),
    secret_options: Sequence[str] = (),
    secret_values: Sequence[str] = (),
) -> StageRecord:
    """Capture a redacted terminal stage record after output publication."""
    configuration_sha256 = _sha256_json(normalise_configuration(effective_configuration))
    if configuration_sha256 != fingerprint.configuration_sha256:
        raise ValueError("effective configuration does not match the stage fingerprint")
    output_records = _artifact_records(outputs or {}, relative_to=run_dir)
    values = tuple(value for value in secret_values if value)
    redacted_validations: list[SemanticValidation] = []
    for validation in semantic_validation:
        redacted_validations.append(replace(
            validation,
            detail=None if validation.detail is None else _redact_string(validation.detail, values),
        ))
    redacted_reason = None if reason is None else _redact_string(reason, values)
    return StageRecord(
        name=name,
        outcome=outcome,
        fingerprint=fingerprint,
        dependencies=tuple(dependencies),
        command=redact_command(command, secret_options=secret_options, secret_values=secret_values),
        configuration=freeze_json_object(redact_configuration(
            effective_configuration,
            secret_keys=secret_keys,
            secret_values=secret_values,
        )),
        outputs=output_records,
        counts=tuple(sorted((counts or {}).items())),
        elapsed_seconds=elapsed_seconds,
        semantic_validation=tuple(redacted_validations),
        reason=redacted_reason,
    )


def start_or_resume_run(
    run_dir: Path,
    *,
    run_id_factory: Callable[[], str] | None = None,
    now: datetime | None = None,
) -> RunManifest:
    """Create one run identity or load the identity already in the run directory."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        return load_run_manifest(run_dir)
    if run_dir.exists() and not run_dir.is_dir():
        raise NotADirectoryError(f"run directory path is not a directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    factory = run_id_factory or (lambda: uuid4().hex)
    run_id = factory()
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("run creation time must include a timezone")
    manifest = RunManifest(
        run_id=_nonempty(run_id, "run_id"),
        created_at_utc=current.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    write_run_manifest(run_dir, manifest)
    return manifest


def load_run_manifest(run_dir: Path) -> RunManifest:
    """Read and validate a compressed run manifest."""
    path = Path(run_dir) / MANIFEST_FILENAME
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"could not read run manifest {path}: {exc}") from exc
    return RunManifest.from_dict(payload)


def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    """Atomically persist a manifest without allowing its run identity to change."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / MANIFEST_FILENAME
    if path.exists():
        existing = load_run_manifest(run_dir)
        if existing.run_id != manifest.run_id:
            raise ValueError(
                f"run_id is immutable for {run_dir}: existing {existing.run_id!r}, new {manifest.run_id!r}"
            )
        if existing.created_at_utc != manifest.created_at_utc:
            raise ValueError("created_at_utc is immutable for an existing run directory")
    payload = json.dumps(
        manifest.to_dict(),
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ).encode("utf-8") + b"\n"
    _atomic_write(path, gzip.compress(payload, compresslevel=9, mtime=0))
    return path


def record_stage(run_dir: Path, record: StageRecord) -> RunManifest:
    """Replace one stage, discard its stale dependants and persist the result."""
    manifest = load_run_manifest(run_dir)
    invalidated = _invalidated_stage_names(manifest, record.name)
    retained = tuple(stage for stage in manifest.stages if stage.name not in invalidated)
    available = {stage.name for stage in retained}
    missing = set(record.dependencies) - available
    if missing:
        raise ValueError(f"stage {record.name!r} has unrecorded dependencies: {sorted(missing)}")
    updated = replace(manifest, stages=(*retained, record))
    write_run_manifest(run_dir, updated)
    return updated


def reuse_or_invalidate_stage(
    run_dir: Path,
    stage_name: str,
    expected_fingerprint: StageFingerprint,
    *,
    semantic_validators: Mapping[str, SemanticValidator] | None = None,
    reuse_unavailable: bool = False,
) -> tuple[RunManifest, ReuseDecision]:
    """Reuse a valid stage or atomically remove it and all recorded dependants."""
    manifest = load_run_manifest(run_dir)
    record = next((stage for stage in manifest.stages if stage.name == stage_name), None)
    reason = _non_reuse_reason(
        Path(run_dir),
        record,
        expected_fingerprint,
        semantic_validators or {},
        reuse_unavailable=reuse_unavailable,
    )
    if reason is None:
        return manifest, ReuseDecision(True, "fingerprint, output integrity and semantic validation match")
    invalidated = _invalidated_stage_names(manifest, stage_name)
    if invalidated:
        updated = replace(
            manifest,
            stages=tuple(stage for stage in manifest.stages if stage.name not in invalidated),
        )
        write_run_manifest(run_dir, updated)
    else:
        updated = manifest
    return updated, ReuseDecision(False, reason, invalidated)


def _normalise_json(value: object, name: str) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        return os.fspath(value.expanduser().resolve(strict=False))
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} keys must be strings")
        return {
            key: _normalise_json(item, f"{name}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item, f"{name}[]") for item in value]
    raise ValueError(f"{name} is not a supported configuration value: {value!r}")


def _redact_mapping(
    mapping: Mapping[str, JsonValue], explicit_keys: set[str], secret_values: tuple[str, ...]
) -> dict[str, JsonValue]:
    redacted: dict[str, JsonValue] = {}
    for key, value in mapping.items():
        normalised_key = _normalise_key(key)
        if normalised_key in explicit_keys or _is_secret_key(normalised_key):
            redacted[key] = REDACTED
        else:
            redacted[key] = _redact_value(value, explicit_keys, secret_values)
    return redacted


def _redact_value(value: JsonValue, explicit_keys: set[str], secret_values: tuple[str, ...]) -> JsonValue:
    if isinstance(value, str):
        return _redact_string(value, secret_values)
    if isinstance(value, list):
        return [_redact_value(item, explicit_keys, secret_values) for item in value]
    if isinstance(value, dict):
        return _redact_mapping(value, explicit_keys, secret_values)
    return value


def _redact_string(value: str, secret_values: tuple[str, ...]) -> str:
    for secret in secret_values:
        value = value.replace(secret, REDACTED)
    return value


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _normalise_option(value: str) -> str:
    return _normalise_key(value.removeprefix("--"))


def _is_secret_key(value: str) -> bool:
    padded = f"_{value}_"
    for marker in _SECRET_KEY_MARKERS:
        if f"_{marker}_" in padded:
            return True
    return False


def _artifact_records(
    files: Mapping[str, Path], *, relative_to: Path | None = None
) -> tuple[ArtifactIntegrity, ...]:
    records: list[ArtifactIntegrity] = []
    for name, path in sorted(files.items()):
        records.append(artifact_integrity(name, Path(path), relative_to=relative_to))
    return tuple(records)


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: JsonValue) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _non_reuse_reason(
    run_dir: Path,
    record: StageRecord | None,
    expected_fingerprint: StageFingerprint,
    validators: Mapping[str, SemanticValidator],
    *,
    reuse_unavailable: bool,
) -> str | None:
    if record is None:
        return "stage has no recorded result"
    reusable_outcome = (
        record.outcome in _REUSABLE_OUTCOMES
        or reuse_unavailable and record.outcome is StageOutcome.UNAVAILABLE
    )
    if not reusable_outcome:
        return f"recorded outcome {record.outcome.value!r} is not reusable"
    if record.fingerprint != expected_fingerprint:
        return "stage fingerprint changed"
    output_reason = _output_integrity_reason(run_dir, record.outputs)
    if output_reason is not None:
        return output_reason
    failed = [validation.name for validation in record.semantic_validation if not validation.passed]
    if failed:
        return f"stored semantic validation failed: {', '.join(failed)}"
    recorded_names = {validation.name for validation in record.semantic_validation}
    if set(validators) != recorded_names:
        missing = sorted(recorded_names - set(validators))
        extra = sorted(set(validators) - recorded_names)
        return f"semantic validators changed: missing={missing}, extra={extra}"
    for name, validator in validators.items():
        try:
            passed = validator(run_dir)
        except Exception as exc:
            return f"semantic validation {name!r} raised {type(exc).__name__}: {exc}"
        if not isinstance(passed, bool):
            return f"semantic validation {name!r} did not return bool"
        if not passed:
            return f"semantic validation {name!r} no longer passes"
    return None


def _output_integrity_reason(run_dir: Path, outputs: tuple[ArtifactIntegrity, ...]) -> str | None:
    root = run_dir.resolve(strict=True)
    for artifact in outputs:
        relative = Path(artifact.path)
        if relative.is_absolute():
            return f"output {artifact.name!r} has a non-relative manifest path"
        candidate = root / relative
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            return f"output {artifact.name!r} is missing"
        if candidate.is_symlink() or not resolved.is_file() or root not in resolved.parents:
            return f"output {artifact.name!r} is not a regular file inside the run directory"
        if resolved.stat().st_size != artifact.size_bytes:
            return f"output {artifact.name!r} size changed"
        if _md5_file(resolved) != artifact.md5:
            return f"output {artifact.name!r} content changed"
    return None


def _invalidated_stage_names(manifest: RunManifest, stage_name: str) -> tuple[str, ...]:
    invalidated = {stage_name}
    changed = True
    while changed:
        changed = False
        for stage in manifest.stages:
            if stage.name not in invalidated and invalidated.intersection(stage.dependencies):
                invalidated.add(stage.name)
                changed = True
    return tuple(stage.name for stage in manifest.stages if stage.name in invalidated)


def _nonempty(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value
