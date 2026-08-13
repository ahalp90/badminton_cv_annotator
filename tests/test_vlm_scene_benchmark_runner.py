"""Fake-backend end-to-end tests for the VLM benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annotator.vlm_scene_benchmark.backends import GenerationEvidence
from annotator.vlm_scene_benchmark.contracts import RunOutcome, ShardSpec, read_run_record
from annotator.vlm_scene_benchmark.prepare import PreparedShardManifest, VideoFileSpec
from annotator.vlm_scene_benchmark.run_cli import run_benchmark
from annotator.vlm_scene_benchmark.scoring import deployment_failures


SHA256 = "a" * 64


def _segment(start: int, end: int) -> dict[str, object]:
    return {
        "start_frame": start,
        "end_frame": end,
        "scene_label": "live",
        "broadcast_phase": "live_rally",
        "view": "full_court",
        "playback": "real_time",
        "continuity_from_previous": "same_rally",
        "data_use": "usable_standard",
        "confidence": 0.8,
        "evidence_frames": [start],
        "reason": "Standard live court view.",
    }


def _valid_response() -> str:
    return json.dumps({"segments": [_segment(10, 60)]})


def _manifest() -> PreparedShardManifest:
    model = VideoFileSpec("model.mp4", SHA256, 1.0, 2, 512, 288)
    return PreparedShardManifest(
        shard=ShardSpec(
            "sset_15", "source.mp4", "b" * 64, model.file_name, model.sha256, 25.0, 100, 10, 60
        ),
        original_source=VideoFileSpec("source.mp4", "b" * 64, 25.0, 100, 640, 360),
        reference_video=VideoFileSpec("reference.mp4", "c" * 64, 25.0, 50, 512, 288),
        model_video=model,
        sampled_source_frames=(10, 35),
        cut_frames=(30,),
        content_threshold=27.0,
        min_scene_len=15,
        ffmpeg_version="ffmpeg version test",
    )


class FakeMonitor:
    device_name = "NVIDIA L40"
    peak_used_memory_mib = 12345.0
    error: str | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


class FakeBackend:
    backend_version = "test-backend"
    cpu_offload = False
    cache_dtype = "bfloat16"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate(self, _video_path: Path, prompt: str, **_kwargs) -> GenerationEvidence:
        self.prompts.append(prompt)
        return GenerationEvidence(
            raw_response=self.responses.pop(0),
            sampled_input_frames=(0, 1),
            width=512,
            height=288,
            visual_tokens=216,
            total_input_tokens=300,
        )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    backend: FakeBackend,
) -> Path:
    manifest_path = tmp_path / "manifest.json"
    video_path = tmp_path / "model.mp4"
    video_path.write_bytes(b"fake")
    monkeypatch.setattr("annotator.vlm_scene_benchmark.run_cli.read_manifest", lambda _path: _manifest())
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.run_cli.resolve_model_video",
        lambda _path, _manifest_value: video_path,
    )
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.run_cli.load_backend",
        lambda _name, expected_input_frames, max_model_len: backend,
    )
    monkeypatch.setattr("annotator.vlm_scene_benchmark.run_cli.GpuMemoryMonitor", FakeMonitor)
    return manifest_path


def test_runner_writes_success_record_and_raw_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([_valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    output_path = tmp_path / "run.json"

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        output_path,
        run_id="internvideo3-test",
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 1
    assert record.first_attempt_valid_json is True
    assert record.first_attempt_valid_prediction is True
    assert record.attempt_response_sha256s == (record.raw_response_sha256,)
    assert record.observed_sampling is not None
    assert record.observed_sampling.sampled_source_frames == (10, 35)
    assert read_run_record(output_path) == record
    assert (tmp_path / "run.attempt-1.txt").read_text(encoding="utf-8") == _valid_response()


def test_runner_passes_explicit_qwen_context_to_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([_valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    calls: list[tuple[int, int | None]] = []

    def load(_name: str, *, expected_input_frames: int, max_model_len: int | None) -> FakeBackend:
        calls.append((expected_input_frames, max_model_len))
        return backend

    monkeypatch.setattr("annotator.vlm_scene_benchmark.run_cli.load_backend", load)

    record = run_benchmark(
        "qwen3-vl",
        manifest_path,
        tmp_path / "qwen.json",
        run_id="qwen-fine-test",
        max_new_tokens=4_096,
        max_model_len=16_384,
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert calls == [(2, 16_384)]


def test_runner_rejects_context_that_cannot_hold_requested_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must exceed maximum new tokens"):
        run_benchmark(
            "qwen3-vl",
            tmp_path / "unused-manifest.json",
            tmp_path / "unused-result.json",
            run_id="qwen-invalid-context",
            max_new_tokens=4_096,
            max_model_len=4_096,
        )


def test_runner_accepts_fenced_json_without_retry_and_retains_raw_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fenced_response = f"```json\n{_valid_response()}\n```"
    backend = FakeBackend([fenced_response])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    output_path = tmp_path / "fenced.json"

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        output_path,
        run_id="internvideo3-fenced",
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 1
    assert record.first_attempt_valid_json is False
    assert record.first_attempt_valid_prediction is True
    assert (tmp_path / "fenced.attempt-1.txt").read_text(encoding="utf-8") == fenced_response


def test_runner_accepts_complete_frame_code_prefix_without_claiming_valid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = '{"frames":["LLRFSS9R","LBRFRS8B","unterminated'
    backend = FakeBackend([response])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        tmp_path / "prefix.json",
        run_id="internvideo3-prefix",
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 1
    assert record.first_attempt_valid_json is False
    assert record.first_attempt_valid_prediction is True


def test_runner_retries_once_with_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(["not JSON", _valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        tmp_path / "retry.json",
        run_id="internvideo3-retry",
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 2
    assert record.first_attempt_valid_json is False
    assert record.first_attempt_valid_prediction is False
    assert record.attempt_response_sha256s is not None
    assert len(record.attempt_response_sha256s) == 2
    assert "failed strict validation" in backend.prompts[1]
    assert (tmp_path / "retry.attempt-1.txt").read_text(encoding="utf-8") == "not JSON"
    assert (tmp_path / "retry.attempt-2.txt").read_text(encoding="utf-8") == _valid_response()


def test_runner_retries_strict_json_that_fails_prediction_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(['{"frames":[]}', _valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    output_path = tmp_path / "schema-retry.json"

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        output_path,
        run_id="internvideo3-schema-retry",
    )

    assert record.outcome is RunOutcome.SUCCEEDED
    assert record.attempt_count == 2
    assert record.first_attempt_valid_json is True
    assert record.first_attempt_valid_prediction is False
    assert read_run_record(output_path) == record


def test_runner_retains_failed_second_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend(["not JSON", "still not JSON"])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    output_path = tmp_path / "failed.json"

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        output_path,
        run_id="internvideo3-failed",
    )

    assert record.outcome is RunOutcome.FAILED
    assert record.attempt_count == 2
    assert "invalid JSON" in str(record.failure_reason)
    assert read_run_record(output_path) == record
    assert (tmp_path / "failed.attempt-2.txt").read_text(encoding="utf-8") == "still not JSON"


def test_runner_refuses_to_overwrite_retained_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([_valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)
    output_path = tmp_path / "run.json"
    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_benchmark(
            "internvideo3",
            manifest_path,
            output_path,
            run_id="internvideo3-test",
        )


def test_runner_does_not_claim_cache_dtype_when_backend_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([_valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)

    def fail_load(
        _name: str,
        *,
        expected_input_frames: int,
        max_model_len: int | None,
    ) -> FakeBackend:
        raise RuntimeError(f"load failed for {expected_input_frames} frames")

    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.run_cli.load_backend",
        fail_load,
    )

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        tmp_path / "load-failed.json",
        run_id="internvideo3-load-failed",
    )

    assert record.outcome is RunOutcome.FAILED
    assert record.runtime.cache_dtype is None
    assert record.attempt_count == 0


def test_runner_fails_when_gpu_monitoring_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([_valid_response()])
    manifest_path = _patch_runtime(monkeypatch, tmp_path, backend)

    class FailedMonitor(FakeMonitor):
        error = "nvidia-smi failed after an initial sample"

    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.run_cli.GpuMemoryMonitor",
        FailedMonitor,
    )

    record = run_benchmark(
        "internvideo3",
        manifest_path,
        tmp_path / "monitor-failed.json",
        run_id="internvideo3-monitor-failed",
    )

    assert record.outcome is RunOutcome.FAILED
    assert record.runtime.peak_vram_mib == 12345.0
    assert record.failure_reason == (
        "GPU monitoring failed: nvidia-smi failed after an initial sample"
    )
    assert deployment_failures(record)[0].startswith("backend failed: GPU monitoring failed")
