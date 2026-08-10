"""Strict response and runtime-evidence tests for the VLM benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import threading

import pytest

from annotator.vlm_scene_benchmark.contracts import ShardSpec
from annotator.vlm_scene_benchmark.backends.internvideo3 import _prepare_chat_inputs
from annotator.vlm_scene_benchmark.backends.qwen3_vl import (
    SPEC as QWEN_SPEC,
    _configure_vllm_environment,
    _engine_config,
    _metadata_frame_indices,
    _resolve_model_snapshot,
    _video_content,
)
from annotator.vlm_scene_benchmark.backends import require_complete_frame_grid
from annotator.vlm_scene_benchmark.runtime import (
    GpuMemoryMonitor,
    GpuSnapshot,
    parse_prediction_response,
    query_nvidia_gpu,
    sha256_bytes,
    write_raw_response,
)


SHA256 = "a" * 64


def _shard() -> ShardSpec:
    return ShardSpec("sset_15", "source.mp4", "b" * 64, "input.mp4", SHA256, 25.0, 100, 10, 60)


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


def test_prediction_response_requires_exact_complete_json() -> None:
    encoded = json.dumps({"segments": [_segment(10, 30), _segment(30, 60)]})

    segments = parse_prediction_response(encoded, _shard())

    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [(10, 30), (30, 60)]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("```json\n{}\n```", "invalid JSON"),
        ('{"segments": [], "extra": true}', "keys differ"),
        (json.dumps({"segments": [_segment(10, 59)]}), "ends at"),
        ('{"segments": [], "segments": []}', "duplicate JSON key"),
    ],
)
def test_prediction_response_rejects_non_strict_output(response: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_prediction_response(response, _shard())


def test_raw_response_is_retained_byte_for_byte(tmp_path: Path) -> None:
    response = '{"segments": []}\n'
    path = tmp_path / "attempt.txt"

    digest = write_raw_response(path, response)

    assert path.read_bytes() == response.encode("utf-8")
    assert digest == sha256_bytes(response.encode("utf-8"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_nvidia_query_parses_name_with_spaces_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    completed = SimpleNamespace(returncode=0, stdout="NVIDIA L40, 12345\n", stderr="")
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.runtime.subprocess.run",
        lambda *args, **kwargs: completed,
    )

    snapshot = query_nvidia_gpu()

    assert snapshot.device_name == "NVIDIA L40"
    assert snapshot.used_memory_mib == 12345.0


def test_nvidia_query_rejects_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def time_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired("nvidia-smi", 5.0)

    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.runtime.subprocess.run",
        time_out,
    )

    with pytest.raises(RuntimeError, match="timed out after 5.0 seconds"):
        query_nvidia_gpu()


def test_gpu_monitor_rejects_thread_that_does_not_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_query() -> GpuSnapshot:
        entered.set()
        release.wait(timeout=1.0)
        return GpuSnapshot("NVIDIA L40", 12345.0)

    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.runtime.query_nvidia_gpu",
        blocked_query,
    )
    monkeypatch.setattr(
        "annotator.vlm_scene_benchmark.runtime.GPU_MONITOR_STOP_TIMEOUT_SECONDS",
        0.01,
    )
    monitor = GpuMemoryMonitor(interval_seconds=0.01)
    monitor.start()
    assert entered.wait(timeout=1.0)

    monitor.stop()

    assert monitor.error == "GPU monitor did not stop within 0.01 seconds"
    release.set()
    monitor.stop()


def test_qwen_metadata_frame_indices_accepts_pinned_utility_mapping() -> None:
    assert _metadata_frame_indices({"frames_indices": [0, 1, 2]}) == (0, 1, 2)

    with pytest.raises(RuntimeError, match="omitted frame indices"):
        _metadata_frame_indices({})


def test_qwen_video_request_overrides_default_768_frame_cap(tmp_path: Path) -> None:
    content = _video_content(
        tmp_path / "full.mp4",
        requested_fps=1.0,
        width=512,
        height=288,
        expected_input_frames=1_800,
    )

    assert content["min_frames"] == 1_800
    assert content["max_frames"] == 1_800
    assert content["total_pixels"] == 1_800 * 512 * 288
    require_complete_frame_grid("Qwen", tuple(range(1_800)), 1_800)
    with pytest.raises(RuntimeError, match="Qwen processor sampled 768 unexpected frames"):
        require_complete_frame_grid("Qwen", tuple(range(768)), 1_800)


def test_qwen_engine_uses_pinned_local_snapshot_and_available_l40_memory(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / QWEN_SPEC.model_revision
    config = _engine_config(model_path)

    assert config["model"] == str(model_path)
    assert config["tokenizer"] == str(model_path)
    assert "revision" not in config
    assert "tokenizer_revision" not in config
    assert config["gpu_memory_utilization"] == 0.90
    assert config["max_model_len"] == 262_144
    assert config["kv_cache_dtype"] == "auto"
    assert QWEN_SPEC.cache_dtype == "bfloat16"
    assert config["tensor_parallel_size"] == 1
    assert config["cpu_offload_gb"] == 0
    assert config["swap_space"] == 0


def test_qwen_resolves_the_exact_model_revision(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / QWEN_SPEC.model_revision
    snapshot.mkdir()
    calls: list[tuple[str, str]] = []

    def fake_snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.append((repo_id, revision))
        return str(snapshot)

    assert _resolve_model_snapshot(fake_snapshot_download) == snapshot.resolve()
    assert calls == [(QWEN_SPEC.model_id, QWEN_SPEC.model_revision)]


def test_qwen_disables_vllm_usage_reporting_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    monkeypatch.delenv("VLLM_NO_USAGE_STATS", raising=False)

    _configure_vllm_environment()

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert os.environ["VLLM_NO_USAGE_STATS"] == "1"


def test_internvideo3_removes_metadata_before_tensor_conversion() -> None:
    metadata = SimpleNamespace(frames_indices=[0, 1])

    class FakeBatch(dict[str, object]):
        tensor_type: str | None = None

        def convert_to_tensors(self, tensor_type: str) -> None:
            assert "video_metadata" not in self
            self.tensor_type = tensor_type

    class FakeProcessor:
        def apply_chat_template(self, messages: object, **kwargs: object) -> FakeBatch:
            assert messages == [{"role": "user"}]
            assert kwargs == {
                "tokenize": True,
                "add_generation_prompt": True,
                "return_dict": True,
                "fps": 1.0,
                "return_metadata": True,
                "padding": True,
            }
            assert "return_tensors" not in kwargs
            return FakeBatch(video_metadata=[metadata], input_ids=[[1, 2]])

    converted, actual_metadata = _prepare_chat_inputs(
        FakeProcessor(),
        [{"role": "user"}],
        1.0,
    )

    assert converted.tensor_type == "pt"
    assert actual_metadata is metadata
