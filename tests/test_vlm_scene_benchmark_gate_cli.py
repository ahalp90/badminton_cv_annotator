"""Tests for truth-free VLM deployment gating."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from annotator.vlm_scene_benchmark.backends import backend_spec
from annotator.vlm_scene_benchmark.contracts import (
    BenchmarkRunRecord,
    BroadcastPhase,
    CameraView,
    Continuity,
    DataUse,
    ModelIdentity,
    Playback,
    PredictionSegment,
    RunOutcome,
    RuntimeTelemetry,
    SamplingObservation,
    SamplingRequest,
    SceneTruth,
    ShardSpec,
    read_run_record,
    write_run_record,
)
from annotator.vlm_scene_benchmark.gate_cli import RecordTarget, gate_failures, main
from annotator.vlm_scene_benchmark.runtime import sha256_bytes


SHA256 = "a" * 64


def _record(raw_digest: str) -> BenchmarkRunRecord:
    spec = backend_spec("internvideo3")
    package_versions = tuple(
        (
            package_name,
            spec.expected_backend_version
            if package_name == spec.backend_distribution
            else "test-version",
        )
        for package_name in spec.package_names
    )
    segment = PredictionSegment(
        start_frame=10,
        end_frame=60,
        scene_label=SceneTruth.LIVE,
        broadcast_phase=BroadcastPhase.LIVE_RALLY,
        view=CameraView.FULL_COURT,
        playback=Playback.REAL_TIME,
        continuity_from_previous=Continuity.SAME_RALLY,
        data_use=DataUse.USABLE_STANDARD,
        confidence=0.9,
        evidence_frames=(10,),
        reason="Visible standard court view.",
    )
    return BenchmarkRunRecord(
        run_id="internvideo3-smoke-test",
        outcome=RunOutcome.SUCCEEDED,
        model=ModelIdentity(spec.model_id, spec.model_revision, spec.backend_name, "4.57.3"),
        shard=ShardSpec("sset_15", "source.mp4", SHA256, "model.mp4", "b" * 64, 25.0, 100, 10, 60),
        requested_sampling=SamplingRequest(1.0, 512, 288),
        observed_sampling=SamplingObservation((10, 35, 59), 512, 288, 720, 1000, True, True),
        runtime=RuntimeTelemetry(
            "carmack", "NVIDIA L40", 22000.0, 10.0, False, "bfloat16", package_versions
        ),
        attempt_count=1,
        first_attempt_valid_json=True,
        first_attempt_valid_prediction=True,
        raw_response_sha256=raw_digest,
        failure_reason=None,
        segments=(segment,),
        attempt_response_sha256s=(raw_digest,),
    )


def _write_record(tmp_path: Path) -> Path:
    provisional = _record(SHA256)
    raw = (
        json.dumps(
            {"segments": [segment.to_json() for segment in provisional.segments]},
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    path = tmp_path / "internvideo3-smoke.json"
    write_run_record(path, _record(sha256_bytes(raw)))
    path.with_name("internvideo3-smoke.attempt-1.txt").write_bytes(raw)
    return path


def test_gate_accepts_exact_model_and_retained_response(tmp_path: Path) -> None:
    path = _write_record(tmp_path)

    assert gate_failures(RecordTarget("internvideo3", path)) == []
    assert main([f"internvideo3={path}"]) == 0


def test_gate_reports_identity_and_response_digest_failures(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    record = replace(_record("c" * 64), model=replace(_record("c" * 64).model, model_revision="wrong"))
    write_run_record(path, record)

    failures = gate_failures(RecordTarget("internvideo3", path))

    assert any("model revision" in failure for failure in failures)
    assert any("response digest differs" in failure for failure in failures)
    assert main([f"internvideo3={path}"]) == 4


def test_gate_reports_backend_and_runtime_version_failures(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    original = _record(SHA256)
    changed_versions = dict(original.runtime.package_versions)
    del changed_versions["accelerate"]
    changed_versions["av"] = "not-installed"
    changed_versions["transformers"] = "wrong"
    record = replace(
        original,
        model=replace(original.model, backend_version="wrong"),
        runtime=replace(original.runtime, package_versions=tuple(changed_versions.items())),
    )
    write_run_record(path, record)

    failures = gate_failures(RecordTarget("internvideo3", path))

    assert any("backend version" in failure for failure in failures)
    assert "runtime package is missing: accelerate" in failures
    assert "runtime package is not installed: av" in failures
    assert any("runtime transformers version" in failure for failure in failures)


def test_gate_reports_non_bf16_cache_dtype(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    original = read_run_record(path)
    write_run_record(
        path,
        replace(original, runtime=replace(original.runtime, cache_dtype="float16")),
    )

    failures = gate_failures(RecordTarget("internvideo3", path))

    assert "cache dtype 'float16' differs from 'bfloat16'" in failures


def test_gate_reports_raw_json_flag_and_prediction_mismatches(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    original = _record(sha256_bytes(path.with_name("internvideo3-smoke.attempt-1.txt").read_bytes()))
    changed_segment = replace(original.segments[0], scene_label=SceneTruth.CUTAWAY)
    write_run_record(
        path,
        replace(
            original,
            first_attempt_valid_json=False,
            segments=(changed_segment,),
        ),
    )

    failures = gate_failures(RecordTarget("internvideo3", path))

    assert "first_attempt_valid_json differs from the retained first response" in failures
    assert "retained model response differs from recorded predictions" in failures


def test_gate_authenticates_every_retry_response(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    original = read_run_record(path)
    final_path = path.with_name("internvideo3-smoke.attempt-2.txt")
    path.with_name("internvideo3-smoke.attempt-1.txt").write_text("not JSON", encoding="utf-8")
    final_path.write_text(
        json.dumps({"segments": [segment.to_json() for segment in original.segments]}),
        encoding="utf-8",
    )
    first_digest = sha256_bytes(b"not JSON")
    final_digest = sha256_bytes(final_path.read_bytes())
    write_run_record(
        path,
        replace(
            original,
            attempt_count=2,
            first_attempt_valid_json=False,
            first_attempt_valid_prediction=False,
            raw_response_sha256=final_digest,
            attempt_response_sha256s=(first_digest, final_digest),
        ),
    )
    assert gate_failures(RecordTarget("internvideo3", path)) == []

    path.with_name("internvideo3-smoke.attempt-1.txt").write_text("different invalid", encoding="utf-8")

    failures = gate_failures(RecordTarget("internvideo3", path))
    assert "retained model response digest differs: internvideo3-smoke.attempt-1.txt" in failures


def test_gate_verifies_first_prediction_validity_flag(tmp_path: Path) -> None:
    path = _write_record(tmp_path)
    original = read_run_record(path)
    first_path = path.with_name("internvideo3-smoke.attempt-1.txt")
    second_path = path.with_name("internvideo3-smoke.attempt-2.txt")
    second_path.write_bytes(first_path.read_bytes())
    digest = sha256_bytes(first_path.read_bytes())
    write_run_record(
        path,
        replace(
            original,
            attempt_count=2,
            first_attempt_valid_prediction=False,
            raw_response_sha256=digest,
            attempt_response_sha256s=(digest, digest),
        ),
    )

    failures = gate_failures(RecordTarget("internvideo3", path))

    assert (
        "first_attempt_valid_prediction differs from the retained first response"
        in failures
    )


def test_gate_reports_missing_record_without_traceback(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "missing.json"

    assert main([f"qwen3-vl={missing}"]) == 4
    assert "cannot reload record" in capsys.readouterr().out
