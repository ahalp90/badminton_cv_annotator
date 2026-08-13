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
from annotator.vlm_scene_benchmark.backends.internvideo3 import (
    SPEC as INTERNVIDEO3_SPEC,
    _prepare_chat_inputs,
)
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
    is_strict_json_response,
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


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ('{"segments":[]}', True),
        ('{"segments":[],"segments":[]}', False),
        ('```json\n{"segments":[]}\n```', False),
        ('{"frames":["LLRFSS9R"', False),
    ],
)
def test_strict_json_response_reports_raw_syntax(response: str, expected: bool) -> None:
    assert is_strict_json_response(response) is expected


def _compact_segment(start: int, end: int) -> list[object]:
    return [
        start,
        end,
        "live",
        "live_rally",
        "full_court",
        "real_time",
        "same_rally",
        "usable_standard",
        0.8,
        [start],
        "Standard live court view.",
    ]


def test_prediction_response_requires_exact_complete_json() -> None:
    encoded = json.dumps({"segments": [_segment(10, 30), _segment(30, 60)]})

    segments = parse_prediction_response(encoded, _shard())

    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [(10, 30), (30, 60)]


def test_prediction_response_expands_compact_segment_arrays() -> None:
    encoded = json.dumps({"segments": [_compact_segment(10, 30), _compact_segment(30, 60)]})

    segments = parse_prediction_response(encoded, _shard())

    assert [segment.to_json() for segment in segments] == [_segment(10, 30), _segment(30, 60)]


def test_prediction_response_expands_and_merges_frame_codes() -> None:
    encoded = json.dumps({"frames": ["LLRFSS9R", "LLRFSS9R", "LBRFRS8B"]})

    segments = parse_prediction_response(encoded, _shard(), (10, 30, 50))

    assert [segment.to_json() for segment in segments] == [
        {
            **_segment(10, 50),
            "confidence": 0.9,
            "evidence_frames": [10, 30],
            "reason": "Active rally is visible.",
        },
        {
            **_segment(50, 60),
            "broadcast_phase": "between_rallies",
            "continuity_from_previous": "new_rally",
            "confidence": 0.8,
            "evidence_frames": [50],
            "reason": "Players are between rallies or preparing.",
        },
    ]


def test_prediction_response_frame_codes_cover_full_twenty_minute_grid() -> None:
    shard = ShardSpec(
        "sset_15",
        "source.mp4",
        "b" * 64,
        "input.mp4",
        SHA256,
        25.0,
        50_000,
        18_419,
        48_419,
    )
    frame_grid = tuple(range(shard.start_frame, shard.end_frame, 25))
    encoded = json.dumps({"frames": ["LLRFSS9R"] * len(frame_grid)}, separators=(",", ":"))

    segments = parse_prediction_response(encoded, shard, frame_grid)

    assert len(frame_grid) == 1_200
    assert len(encoded.encode("utf-8")) < 16_000
    assert len(segments) == 1
    assert (segments[0].start_frame, segments[0].end_frame) == (18_419, 48_419)
    assert segments[0].evidence_frames == (18_419, 18_444, 18_469)


def test_prediction_response_accepts_required_prefix_before_truncated_continuation() -> None:
    response = '{"frames":["LLRFSS9R","LBRFRS8B","unterminated'

    segments = parse_prediction_response(response, _shard(), (10, 30))

    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [(10, 30), (30, 60)]


def test_prediction_response_rejects_truncation_before_complete_coverage() -> None:
    response = '{"frames":["LLRFSS9R","unterminated'

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_prediction_response(response, _shard(), (10, 30))


def test_prediction_response_ignores_valid_continuation_after_complete_coverage() -> None:
    encoded = json.dumps({"frames": ["LLRFSS9R", "LBRFRS8B", "LBRFRS8B"]})

    segments = parse_prediction_response(encoded, _shard(), (10, 30))

    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [(10, 30), (30, 60)]


def test_prediction_response_accepts_one_whole_json_fence() -> None:
    encoded = json.dumps({"segments": [_segment(10, 30), _segment(30, 60)]})

    segments = parse_prediction_response(f"```json\n{encoded}\n```\n", _shard())

    assert [(segment.start_frame, segment.end_frame) for segment in segments] == [(10, 30), (30, 60)]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("commentary\n```json\n{}\n```", "invalid JSON"),
        ("```python\n{}\n```", "invalid JSON"),
        ("```json\n{}\n```\ncommentary", "invalid JSON"),
        ('{"segments": [], "extra": true}', "keys differ"),
        (json.dumps({"segments": [_segment(10, 59)]}), "ends at"),
        (json.dumps({"segments": [[10, 60]]}), "must contain 11 values"),
        (json.dumps({"frames": ["LLRFSS9R"]}), "sampled source frames are required"),
        (json.dumps({"frames": ["LLRFSS9R"]}), "contains 1 frame codes"),
        (json.dumps({"frames": ["LLRFSS9"]}), "8-character string"),
        (json.dumps({"frames": ["XLRFSS9R"]}), "unknown code"),
        ('{"segments": [], "segments": []}', "duplicate JSON key"),
    ],
)
def test_prediction_response_rejects_non_strict_output(
    response: str,
    message: str,
) -> None:
    frame_grid = None
    if "contains 1 frame codes" in message:
        frame_grid = (10, 30)
    elif "8-character" in message or "unknown code" in message:
        frame_grid = (10,)
    with pytest.raises(ValueError, match=message):
        parse_prediction_response(response, _shard(), frame_grid)


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
    config = _engine_config(model_path, 16_384)

    assert config["model"] == str(model_path)
    assert config["tokenizer"] == str(model_path)
    assert "revision" not in config
    assert "tokenizer_revision" not in config
    assert config["gpu_memory_utilization"] == 0.90
    assert config["max_model_len"] == 16_384
    assert config["kv_cache_dtype"] == "auto"
    assert QWEN_SPEC.cache_dtype == "bfloat16"
    assert config["tensor_parallel_size"] == 1
    assert config["cpu_offload_gb"] == 0
    assert config["swap_space"] == 0

    with pytest.raises(ValueError, match="between 4,096 and 262,144"):
        _engine_config(model_path, 2_048)


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


def test_internvideo3_records_the_required_torchcodec_decoder() -> None:
    assert "torchcodec" in INTERNVIDEO3_SPEC.package_names
