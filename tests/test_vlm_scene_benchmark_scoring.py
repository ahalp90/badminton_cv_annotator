"""Hand-calculated scoring tests for the VLM scene benchmark."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from annotator.broadcast_timeline_labels import SceneTruth, VideoMetadata, make_interval, write_label_csv
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
)
from annotator.vlm_scene_benchmark.score_cli import main as score_main
from annotator.vlm_scene_benchmark.scoring import boundary_metrics, score_run_record


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


def _valid_record() -> BenchmarkRunRecord:
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


def _truth():
    metadata = VideoMetadata("sset_15", 25.0, 100)
    return [
        make_interval(metadata, 0, 14, SceneTruth.LIVE),
        make_interval(metadata, 14, 16, SceneTruth.LIVE_NON_STANDARD),
        make_interval(metadata, 16, 18, SceneTruth.REPLAY),
        make_interval(metadata, 18, 20, SceneTruth.CUTAWAY),
        make_interval(metadata, 20, 22, SceneTruth.OTHER),
        make_interval(metadata, 22, 100, SceneTruth.LIVE),
    ]


def _scored_record():
    record = _valid_record()
    assert record.observed_sampling is not None
    return replace(
        record,
        shard=replace(record.shard, start_frame=10, end_frame=22),
        observed_sampling=replace(record.observed_sampling, sampled_source_frames=(10, 21)),
        segments=(
            _segment(10, 15, SceneTruth.LIVE),
            _segment(15, 17, SceneTruth.LIVE_NON_STANDARD),
            _segment(17, 19, SceneTruth.REPLAY),
            _segment(19, 21, SceneTruth.CUTAWAY),
            _segment(21, 22, SceneTruth.OTHER),
        ),
    )


def test_frame_metrics_match_hand_calculation() -> None:
    summary = score_run_record(_scored_record(), _truth())
    accuracy = summary["accuracy"]

    assert summary["deployment_gate"] == {"passed": True, "failures": []}
    assert accuracy["confusion_matrix"] == [
        [4, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1],
    ]
    assert accuracy["correct_frames"] == 8
    assert accuracy["accuracy"] == pytest.approx(8 / 12)
    assert accuracy["per_class"]["live"]["f1"] == pytest.approx(8 / 9)
    assert accuracy["per_class"]["other"]["f1"] == pytest.approx(2 / 3)
    assert accuracy["macro_f1"] == pytest.approx(11 / 18)
    assert accuracy["live_non_standard_confusion"] == {
        "truth_live_predicted_live_non_standard": 0,
        "truth_live_non_standard_predicted_live": 1,
    }


def test_runtime_gate_hides_accuracy_for_cpu_offload() -> None:
    record = _scored_record()
    record = replace(record, runtime=replace(record.runtime, cpu_offload=True))

    summary = score_run_record(record, _truth())

    assert summary["deployment_gate"]["passed"] is False
    assert summary["deployment_gate"]["failures"] == ["CPU offload is prohibited by the benchmark"]
    assert summary["accuracy"] is None
    assert summary["boundaries"] is None


def test_runtime_gate_rejects_two_endpoint_frames_for_full_shard() -> None:
    record = _valid_record()
    assert record.observed_sampling is not None
    shard = replace(record.shard, frame_count=100_000, start_frame=18_419, end_frame=63_419)
    observed = replace(record.observed_sampling, sampled_source_frames=(18_419, 63_418))
    record = replace(
        record,
        shard=shard,
        observed_sampling=observed,
        segments=(_segment(18_419, 63_419),),
    )

    summary = score_run_record(record, _truth())

    assert summary["deployment_gate"]["passed"] is False
    assert any("expected about 1800" in failure for failure in summary["deployment_gate"]["failures"])
    assert any("requested sampling cadence" in failure for failure in summary["deployment_gate"]["failures"])
    assert summary["accuracy"] is None


def test_runtime_gate_verifies_uniform_grid_claim() -> None:
    record = _valid_record()
    assert record.observed_sampling is not None
    observed = replace(record.observed_sampling, sampled_source_frames=(10, 20, 59), uniform_frame_grid=True)
    record = replace(record, observed_sampling=observed)

    summary = score_run_record(record, _truth())

    assert "observed frame gaps contradict a uniform frame grid" in summary["deployment_gate"]["failures"]
    assert summary["accuracy"] is None


def test_runtime_gate_rejects_processor_resolution_change() -> None:
    record = _valid_record()
    assert record.observed_sampling is not None
    observed = replace(record.observed_sampling, width=480, height=288)
    record = replace(record, observed_sampling=observed)

    summary = score_run_record(record, _truth())

    assert summary["deployment_gate"]["failures"] == [
        "observed resolution 480x288 differs from requested 512x288"
    ]
    assert summary["accuracy"] is None


def test_failed_backend_record_returns_deployment_evidence() -> None:
    record = _scored_record()
    record = replace(
        record,
        outcome=RunOutcome.FAILED,
        observed_sampling=None,
        failure_reason="CUDA out of memory",
        segments=(),
    )

    summary = score_run_record(record, _truth())

    assert summary["deployment_gate"] == {
        "passed": False,
        "failures": ["backend failed: CUDA out of memory"],
    }
    assert summary["accuracy"] is None


def test_boundary_matching_does_not_reuse_one_prediction() -> None:
    metrics = boundary_metrics([10, 12], [11], 25.0)

    within_one_second = metrics["matches"]["one_second"]
    assert within_one_second["matched"] == 1
    assert within_one_second["precision"] == 1.0
    assert within_one_second["recall"] == 0.5
    assert within_one_second["f1"] == pytest.approx(2 / 3)


def test_boundary_errors_are_reported_in_both_directions() -> None:
    summary = score_run_record(_scored_record(), _truth())
    boundaries = summary["boundaries"]

    assert boundaries["truth_boundary_count"] == 4
    assert boundaries["prediction_boundary_count"] == 4
    assert boundaries["truth_to_prediction_error"] == {
        "reference_boundaries": 4,
        "median_frames": 1.0,
        "p95_frames": 1,
        "max_frames": 1,
    }
    assert boundaries["matches"]["5_frames"]["matched"] == 4


def test_score_cli_writes_reloadable_json_and_signals_gate_failure(tmp_path: Path) -> None:
    from annotator.vlm_scene_benchmark.contracts import write_run_record

    record_path = tmp_path / "run.json"
    truth_path = tmp_path / "truth.csv.gz"
    output_path = tmp_path / "score.json"
    write_run_record(record_path, _scored_record())
    write_label_csv(truth_path, _truth(), VideoMetadata("sset_15", 25.0, 100))

    assert score_main([str(record_path), str(truth_path), "--out", str(output_path)]) == 0
    value = json.loads(output_path.read_text(encoding="utf-8"))
    assert value["accuracy"]["macro_f1"] == pytest.approx(11 / 18)

    failed = replace(_scored_record(), runtime=replace(_scored_record().runtime, cpu_offload=True))
    write_run_record(record_path, failed)
    assert score_main([str(record_path), str(truth_path), "--out", str(output_path)]) == 3
    assert json.loads(output_path.read_text(encoding="utf-8"))["accuracy"] is None
