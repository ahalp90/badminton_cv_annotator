"""Contract tests for reloadable VLM scene benchmark records."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from annotator.broadcast_timeline_labels import SceneTruth
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
    ShardSpec,
    read_run_record,
    write_run_record,
)


SHA256 = "a" * 64


def _segment(start: int, end: int, label: SceneTruth = SceneTruth.LIVE) -> PredictionSegment:
    return PredictionSegment(
        start_frame=start,
        end_frame=end,
        scene_label=label,
        broadcast_phase=BroadcastPhase.LIVE_RALLY,
        view=CameraView.FULL_COURT,
        playback=Playback.REAL_TIME,
        continuity_from_previous=Continuity.SAME_RALLY,
        data_use=DataUse.USABLE_STANDARD,
        confidence=0.8,
        evidence_frames=(start,),
        reason="Visible badminton broadcast scene.",
    )


def valid_record() -> BenchmarkRunRecord:
    return BenchmarkRunRecord(
        run_id="internvideo3-smoke-001",
        outcome=RunOutcome.SUCCEEDED,
        model=ModelIdentity("model/id", "revision", "transformers", "4.57.3"),
        shard=ShardSpec(
            "sset_15", "sset_15.mp4", SHA256, "sset_15_1fps.mp4", "c" * 64, 25.0, 100, 10, 60
        ),
        requested_sampling=SamplingRequest(1.0, 512, 288),
        observed_sampling=SamplingObservation((10, 35, 59), 512, 288, 100, 120, True, True),
        runtime=RuntimeTelemetry(
            "carmack",
            "NVIDIA L40",
            21000.0,
            12.5,
            False,
            "bfloat16",
            (("torch", "2.8.0"),),
        ),
        attempt_count=1,
        first_attempt_valid_json=True,
        first_attempt_valid_prediction=True,
        raw_response_sha256="b" * 64,
        failure_reason=None,
        segments=(_segment(10, 30), _segment(30, 60, SceneTruth.CUTAWAY)),
        attempt_response_sha256s=("b" * 64,),
    )


def test_run_record_round_trip_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    record = valid_record()

    write_run_record(path, record)
    first = path.read_bytes()
    write_run_record(path, record)

    assert read_run_record(path) == record
    assert path.read_bytes() == first
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ((_segment(11, 60),), "starts at"),
        ((_segment(10, 30), _segment(31, 60)), "gap or overlap"),
        ((_segment(10, 30), _segment(29, 60)), "gap or overlap"),
        ((_segment(10, 59),), "ends at"),
    ],
)
def test_successful_record_rejects_incomplete_prediction_partition(
    segments: tuple[PredictionSegment, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(valid_record(), segments=segments)


def test_failed_record_keeps_failure_evidence_without_predictions() -> None:
    record = valid_record()

    failed = replace(
        record,
        outcome=RunOutcome.FAILED,
        observed_sampling=None,
        failure_reason="CUDA out of memory",
        segments=(),
    )

    assert failed.failure_reason == "CUDA out of memory"
    with pytest.raises(ValueError, match="failed run cannot contain"):
        replace(failed, segments=(_segment(10, 60),))


def test_successful_normalized_first_response_does_not_claim_valid_json() -> None:
    record = replace(valid_record(), first_attempt_valid_json=False, attempt_count=1)

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 1
    assert record.first_attempt_valid_json is False


def test_successful_invalid_first_prediction_requires_correction_retry() -> None:
    with pytest.raises(ValueError, match="requires the correction retry"):
        replace(valid_record(), first_attempt_valid_prediction=False)


def test_successful_valid_first_response_rejects_correction_retry() -> None:
    with pytest.raises(ValueError, match="requires exactly one attempt"):
        replace(
            valid_record(),
            attempt_count=2,
            attempt_response_sha256s=("b" * 64, "b" * 64),
        )


def test_reader_keeps_schema_one_evidence_compatible(tmp_path: Path) -> None:
    value = valid_record().to_json()
    value["schema_version"] = 1
    del value["first_attempt_valid_prediction"]
    del value["attempt_response_sha256s"]
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    record = read_run_record(path)

    assert record.schema_version == 1
    assert record.first_attempt_valid_prediction is True
    assert record.attempt_response_sha256s is None


def test_record_rejects_unknown_fields_and_enum_values(tmp_path: Path) -> None:
    value = valid_record().to_json()
    value["unexpected"] = True
    path = tmp_path / "extra.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*unexpected"):
        read_run_record(path)

    del value["unexpected"]
    value["segments"][0]["scene_label"] = "maybe-live"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown value"):
        read_run_record(path)


@pytest.mark.parametrize("field", ["run_id", "scene_label"])
def test_reader_rejects_duplicate_json_keys(tmp_path: Path, field: str) -> None:
    encoded = json.dumps(valid_record().to_json())
    original = '"run_id": "internvideo3-smoke-001"' if field == "run_id" else '"scene_label": "live"'
    duplicate = f'{original}, "{field}": "conflicting-value"'
    path = tmp_path / "duplicate.json"
    path.write_text(encoded.replace(original, duplicate, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"duplicate JSON key '{field}'"):
        read_run_record(path)


def test_segment_rejects_out_of_range_confidence_and_evidence() -> None:
    segment = _segment(10, 20)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(segment, confidence=1.01)
    with pytest.raises(ValueError, match="outside"):
        replace(segment, evidence_frames=(20,))


def test_observed_frames_must_be_ordered_and_inside_shard() -> None:
    record = valid_record()
    assert record.observed_sampling is not None
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(record.observed_sampling, sampled_source_frames=(10, 35, 35))
    with pytest.raises(ValueError, match="outside"):
        observed = replace(record.observed_sampling, sampled_source_frames=(9, 35, 59))
        replace(record, observed_sampling=observed)
