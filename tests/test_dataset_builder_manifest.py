"""Run identity, fingerprint and resume contracts for the dataset builder."""

from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path

import pytest
from frozendict import frozendict

from dataset_builder.manifest import (
    MANIFEST_FILENAME,
    build_stage_fingerprint,
    make_stage_record,
    record_stage,
    resolve_interpreter,
    reuse_or_invalidate_stage,
    start_or_resume_run,
    write_run_manifest,
)
from dataset_builder.models import (
    InterpreterIdentity,
    RunManifest,
    SemanticValidation,
    StageFingerprint,
    StageOutcome,
)


SOURCE_COMMIT = "a" * 40
INTERPRETER = InterpreterIdentity("/resolved/python", "Python 3.11.9")
CONFIGURATION = {"threshold": 3, "nested": {"enabled": True}}


def _write(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _fingerprint(
    input_path: Path,
    weight_path: Path,
    *,
    source_commit: str = SOURCE_COMMIT,
    configuration: dict[str, object] | None = None,
    interpreter: InterpreterIdentity = INTERPRETER,
) -> StageFingerprint:
    return build_stage_fingerprint(
        source_commit=source_commit,
        contract_version="extract/1",
        effective_configuration=CONFIGURATION if configuration is None else configuration,
        interpreter=interpreter,
        model_weights={"tracknet": weight_path},
        inputs={"video": input_path},
    )


def _record_two_stage_run(tmp_path: Path) -> tuple[Path, Path, Path, StageFingerprint, RunManifest]:
    run_dir = tmp_path / "run"
    start_or_resume_run(run_dir, run_id_factory=lambda: "run-15")
    input_path = _write(tmp_path / "input.mp4", b"video-input")
    weight_path = _write(tmp_path / "weights.pt", b"model-weights")
    first_output = _write(run_dir / "vision" / "track.npy.xz", b"array-one")
    fingerprint = _fingerprint(input_path, weight_path)
    first = make_stage_record(
        name="extract",
        outcome=StageOutcome.PROCESSED,
        fingerprint=fingerprint,
        run_dir=run_dir,
        command=["python", "extract.py", "--token", "secret-value"],
        effective_configuration=CONFIGURATION,
        outputs={"track": first_output},
        counts={"frames": 10},
        elapsed_seconds=1.25,
        semantic_validation=[SemanticValidation("shape", True, "(10, 3)")],
        secret_values=["secret-value"],
    )
    record_stage(run_dir, first)

    second_output = _write(run_dir / "records.json.gz", b"records")
    second_fingerprint = build_stage_fingerprint(
        source_commit=SOURCE_COMMIT,
        contract_version="assemble/1",
        effective_configuration={},
        interpreter=INTERPRETER,
        inputs={"track": first_output},
    )
    second = make_stage_record(
        name="assemble",
        outcome=StageOutcome.PROCESSED,
        fingerprint=second_fingerprint,
        run_dir=run_dir,
        command=["python", "assemble.py"],
        effective_configuration={},
        outputs={"records": second_output},
        dependencies=["extract"],
        counts={"rallies": 2},
        elapsed_seconds=0.5,
        semantic_validation=[SemanticValidation("row_count", True)],
    )
    manifest = record_stage(run_dir, second)
    return run_dir, input_path, weight_path, fingerprint, manifest


def test_manifest_round_trip_reuses_one_immutable_nonempty_run_id(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    created = start_or_resume_run(run_dir, run_id_factory=lambda: "run-15")
    def unexpected_run_id() -> str:
        raise AssertionError("must not create another run id")

    resumed = start_or_resume_run(run_dir, run_id_factory=unexpected_run_id)

    assert resumed == created
    assert resumed.run_id == "run-15"
    assert not list(run_dir.glob(f".{MANIFEST_FILENAME}.*.tmp"))
    with pytest.raises(ValueError, match="run_id is immutable"):
        write_run_manifest(run_dir, replace(created, run_id="different-run"))
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        start_or_resume_run(tmp_path / "empty", run_id_factory=lambda: "")


def test_stage_record_round_trips_redacted_audit_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest = start_or_resume_run(run_dir, run_id_factory=lambda: "run-15")
    source = _write(tmp_path / "input.mp4", b"source")
    output = _write(run_dir / "out.bin", b"output")
    fingerprint = build_stage_fingerprint(
        source_commit=SOURCE_COMMIT,
        contract_version="stage/1",
        effective_configuration={
            "client_secret_value": "secret-value",
            "limit": 5,
            "nested": {"service_token_value": "secret-value"},
        },
        interpreter=INTERPRETER,
        inputs={"source": source},
    )
    record = make_stage_record(
        name="stage",
        outcome=StageOutcome.FAILED,
        fingerprint=fingerprint,
        run_dir=run_dir,
        command=["python", "stage.py", "-psecret-value", "--api-key=secret-value"],
        effective_configuration={
            "client_secret_value": "secret-value",
            "limit": 5,
            "nested": {"service_token_value": "secret-value"},
        },
        outputs={"result": output},
        counts={"items": 5},
        elapsed_seconds=2.0,
        semantic_validation=[SemanticValidation("schema", False, "server echoed secret-value")],
        reason="request failed for secret-value",
        secret_options=["-p"],
        secret_values=["secret-value"],
    )
    updated = record_stage(run_dir, record)

    with gzip.open(run_dir / MANIFEST_FILENAME, "rt", encoding="utf-8") as handle:
        on_disk = json.load(handle)
    assert "secret-value" not in json.dumps(on_disk)
    assert on_disk["stages"][0]["command"][-2:] == ["-p<redacted>", "--api-key=<redacted>"]
    assert on_disk["stages"][0]["configuration"]["client_secret_value"] == "<redacted>"
    assert on_disk["stages"][0]["configuration"]["nested"]["service_token_value"] == "<redacted>"
    assert on_disk["stages"][0]["reason"] == "request failed for <redacted>"
    assert on_disk["stages"][0]["semantic_validation"][0]["detail"] == "server echoed <redacted>"
    assert updated.stages[0].outputs[0].path == "out.bin"
    assert start_or_resume_run(run_dir) == updated
    assert manifest.run_id == updated.run_id


def test_manifest_configuration_snapshot_and_serialized_tree_are_immutable(tmp_path: Path) -> None:
    _, _, _, _, manifest = _record_two_stage_run(tmp_path)
    record = manifest.stages[0]
    serialized = record.to_dict()
    configuration = serialized["configuration"]
    assert isinstance(configuration, dict)
    configuration["threshold"] = 99
    nested = configuration["nested"]
    assert isinstance(nested, dict)
    nested["enabled"] = False

    assert isinstance(record.configuration, frozendict)
    assert isinstance(record.configuration["nested"], frozendict)
    assert record.configuration["threshold"] == 3
    assert record.configuration["nested"]["enabled"] is True


def test_unchanged_stage_is_reusable_only_after_live_semantic_validation(tmp_path: Path) -> None:
    run_dir, _, _, fingerprint, _ = _record_two_stage_run(tmp_path)

    manifest, decision = reuse_or_invalidate_stage(
        run_dir,
        "extract",
        fingerprint,
        semantic_validators={"shape": lambda _: True},
    )

    assert decision.reusable
    assert decision.invalidated_stages == ()
    assert [stage.name for stage in manifest.stages] == ["extract", "assemble"]


@pytest.mark.parametrize("change", ["source", "configuration", "interpreter", "weight", "input"])
def test_fingerprint_changes_invalidate_the_stage_and_dependants(tmp_path: Path, change: str) -> None:
    run_dir, input_path, weight_path, fingerprint, _ = _record_two_stage_run(tmp_path)
    source_commit = SOURCE_COMMIT
    configuration = CONFIGURATION
    interpreter = INTERPRETER
    if change == "source":
        source_commit = "b" * 40
    elif change == "configuration":
        configuration = {"threshold": 4, "nested": {"enabled": True}}
    elif change == "interpreter":
        interpreter = InterpreterIdentity("/resolved/python", "Python 3.12.4")
    elif change == "weight":
        weight_path.write_bytes(b"changed-weights")
    elif change == "input":
        input_path.write_bytes(b"changed-video")
    else:  # pragma: no cover - parametrization is the exhaustive contract.
        raise AssertionError(change)
    changed = _fingerprint(
        input_path,
        weight_path,
        source_commit=source_commit,
        configuration=configuration,
        interpreter=interpreter,
    )
    assert changed != fingerprint

    manifest, decision = reuse_or_invalidate_stage(
        run_dir,
        "extract",
        changed,
        semantic_validators={"shape": lambda _: True},
    )

    assert not decision.reusable
    assert decision.reason == "stage fingerprint changed"
    assert decision.invalidated_stages == ("extract", "assemble")
    assert manifest.stages == ()


@pytest.mark.parametrize("damage", ["missing", "same-size-corruption"])
def test_output_integrity_invalidates_a_stage_even_when_its_declared_shape_is_unchanged(
    tmp_path: Path, damage: str
) -> None:
    run_dir, _, _, fingerprint, manifest = _record_two_stage_run(tmp_path)
    output = run_dir / manifest.stages[0].outputs[0].path
    if damage == "missing":
        output.unlink()
    else:
        output.write_bytes(b"array-two")

    updated, decision = reuse_or_invalidate_stage(
        run_dir,
        "extract",
        fingerprint,
        semantic_validators={"shape": lambda _: True},
    )

    assert not decision.reusable
    assert "output 'track'" in decision.reason
    assert decision.invalidated_stages == ("extract", "assemble")
    assert updated.stages == ()


def test_live_semantic_failure_invalidates_stage_and_dependants(tmp_path: Path) -> None:
    run_dir, _, _, fingerprint, _ = _record_two_stage_run(tmp_path)

    manifest, decision = reuse_or_invalidate_stage(
        run_dir,
        "extract",
        fingerprint,
        semantic_validators={"shape": lambda _: False},
    )

    assert not decision.reusable
    assert decision.reason == "semantic validation 'shape' no longer passes"
    assert decision.invalidated_stages == ("extract", "assemble")
    assert manifest.stages == ()


def test_resolved_interpreter_captures_absolute_path_and_version() -> None:
    identity = resolve_interpreter("python")

    assert Path(identity.path).is_absolute()
    assert identity.version.startswith("Python ")


def test_resolved_interpreter_supports_a_single_dash_version_option(tmp_path: Path) -> None:
    executable = tmp_path / "ffmpeg-fixture"
    executable.write_text(
        "#!/bin/sh\n"
        "test \"$1\" = -version || exit 2\n"
        "printf 'ffmpeg fixture 1.0\\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)

    identity = resolve_interpreter(executable, version_option="-version")

    assert identity.path == str(executable.resolve())
    assert identity.version == "ffmpeg fixture 1.0"


def test_stage_outcomes_have_the_approved_wire_values() -> None:
    assert {outcome.value for outcome in StageOutcome} == {
        "processed",
        "skipped",
        "excluded",
        "failed",
        "unavailable",
    }
