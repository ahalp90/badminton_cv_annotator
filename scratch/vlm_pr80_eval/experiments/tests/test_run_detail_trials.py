from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import experiments.run_detail_trials as runner
import pytest
from experiments.backends import BackendSpec, GenerationEvidence
from experiments.detail_schema import (
    DETAIL_MANIFEST_SCHEMA,
    DetailArm,
    load_detail_manifest,
)
from experiments.multiscale_prompts import DetailPromptMode, build_detail_prompt


def _write_manifests(tmp_path: Path) -> dict[DetailArm, Path]:
    clip = tmp_path / "clips" / "detail.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"same rendered detail clip")
    clip_sha256 = hashlib.sha256(clip.read_bytes()).hexdigest()
    paths: dict[DetailArm, Path] = {}
    for arm in DetailArm:
        arm_dir = tmp_path / arm.value
        arm_dir.mkdir()
        is_short = arm is DetailArm.SHORT_ONLY
        is_broad = arm is DetailArm.BROAD_FACTS
        payload = {
            "schema": DETAIL_MANIFEST_SCHEMA,
            "arm": arm.value,
            "expected_frames": 6,
            "width": 32,
            "height": 24,
            "cases": [
                {
                    "case_id": "detail-case",
                    "pair_id": "context-sset_01-r007",
                    "context_case_id": "context-sset_01-r007--90",
                    "video_id": "sset_01",
                    "clip_path": "../clips/detail.mp4",
                    "clip_sha256": clip_sha256,
                    "source_start_frame": 100,
                    "source_end_frame": 106,
                    "source_frames": [100, 101, 102, 103, 104, 105],
                    "target_start_frame": 102,
                    "target_end_frame": 104,
                    "boundary_frame": 103,
                    "source_fps": 25.0,
                    "sample_fps": 25.0,
                    "target_segment_ids": ["S0001"],
                    "deterministic_facts": None
                    if is_short
                    else {"span_membership": True, "cut_ids": ["S0001"]},
                    "broad_facts": None
                    if not is_broad
                    else [
                        {
                            "segment_id": "S0001",
                            "content": "live",
                            "repeat_of": None,
                            "needs_close_check": False,
                        }
                    ],
                }
            ],
        }
        path = arm_dir / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[arm] = path
    return paths


class _Backend:
    spec = BackendSpec("test", "test/model", "rev", "test", "test", "1", "bf16", ())
    backend_version = "1"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, float, int, int, int]] = []

    def generate(
        self,
        video_path: Path,
        prompt: str,
        *,
        requested_fps: float,
        width: int,
        height: int,
        max_new_tokens: int,
    ) -> GenerationEvidence:
        self.calls.append((video_path, prompt, requested_fps, width, height, max_new_tokens))
        response = '{"target_content":"live"}'
        if "broad_facts=" in prompt:
            response = "not-json"
        return GenerationEvidence(response, tuple(range(6)), width, height, 12, 40)


def test_runner_loads_one_backend_and_writes_three_arm_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    loaded: list[_Backend] = []

    def load_backend(_name: str, *, expected_input_frames: int, max_model_len: int | None) -> _Backend:
        assert expected_input_frames == 6
        assert max_model_len is None
        backend = _Backend()
        loaded.append(backend)
        return backend

    monkeypatch.setattr(runner, "_load_backend", load_backend)

    runner.run_detail_trials("internvideo3", paths, tmp_path / "attempts", max_new_tokens=17)

    assert len(loaded) == 1
    backend = loaded[0]
    assert ["broad_facts=" in call[1] for call in backend.calls] == [False, False, True]
    assert [call[5] for call in backend.calls] == [17, 17, 17]
    for arm in DetailArm:
        attempt = json.loads(
            (tmp_path / "attempts" / "internvideo3" / arm.value / "detail-case.json").read_text(
                encoding="utf-8"
            )
        )
        assert attempt["arm"] == arm.value
        assert attempt["manifest_sha256"]
        assert attempt["sampling"]["sampled_input_frames"] == list(range(6))
        if arm is DetailArm.BROAD_FACTS:
            assert attempt["parsed_response"] is None
            assert attempt["parser_error"]
        else:
            assert attempt["parsed_response"] == {"target_content": "live"}
            assert attempt["parser_error"] is None


def test_runner_selected_short_only_writes_one_attempt_per_case(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    loaded: list[_Backend] = []

    def load_backend(_name: str, *, expected_input_frames: int, max_model_len: int | None) -> _Backend:
        backend = _Backend()
        loaded.append(backend)
        return backend

    monkeypatch.setattr(runner, "_load_backend", load_backend)

    attempts = tmp_path / "short-only-attempts"
    runner.run_detail_trials(
        "internvideo3",
        paths,
        attempts,
        selected_arms=(DetailArm.SHORT_ONLY,),
    )

    assert len(loaded) == 1
    assert len(loaded[0].calls) == 1
    assert (attempts / "internvideo3" / "short_only" / "detail-case.json").is_file()
    for arm in (DetailArm.DETERMINISTIC, DetailArm.BROAD_FACTS):
        assert not (attempts / "internvideo3" / arm.value).exists()


def test_runner_saves_conservative_replay_veto_prompt_without_schema_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    loaded: list[_Backend] = []

    def load_backend(_name: str, *, expected_input_frames: int, max_model_len: int | None) -> _Backend:
        backend = _Backend()
        loaded.append(backend)
        return backend

    monkeypatch.setattr(runner, "_load_backend", load_backend)
    attempts = tmp_path / "conservative-attempts"
    runner.run_detail_trials(
        "internvideo3",
        paths,
        attempts,
        selected_arms=(DetailArm.SHORT_ONLY,),
        prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
    )

    assert len(loaded) == 1
    attempt_path = attempts / "internvideo3" / "short_only" / "detail-case.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert set(attempt) == {
        "schema",
        "backend",
        "model",
        "arm",
        "manifest_sha256",
        "case",
        "prompt",
        "prompt_sha256",
        "raw_response",
        "parsed_response",
        "parser_error",
        "generation_error",
        "elapsed_seconds",
        "sampling",
    }
    expected_prompt = build_detail_prompt(
        load_detail_manifest(paths[DetailArm.SHORT_ONLY], verify_clip_hash=True).cases[0],
        DetailArm.SHORT_ONLY,
        prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
    )
    assert attempt["prompt"] == expected_prompt
    assert attempt["prompt_sha256"] == hashlib.sha256(expected_prompt.encode()).hexdigest()


@pytest.mark.parametrize(
    "selected_arms",
    [(), (DetailArm.SHORT_ONLY, DetailArm.SHORT_ONLY)],
)
def test_runner_rejects_empty_or_duplicate_arm_selection(
    tmp_path: Path,
    selected_arms: tuple[DetailArm, ...],
) -> None:
    with pytest.raises(ValueError, match="selected_arms"):
        runner.run_detail_trials(
            "internvideo3",
            _write_manifests(tmp_path),
            tmp_path / "attempts",
            selected_arms=selected_arms,
        )


@pytest.mark.parametrize(
    "selected_arms",
    [(DetailArm.DETERMINISTIC,), (DetailArm.SHORT_ONLY, DetailArm.DETERMINISTIC)],
)
def test_runner_rejects_conservative_prompt_for_other_or_multiple_arms(
    tmp_path: Path,
    selected_arms: tuple[DetailArm, ...],
) -> None:
    with pytest.raises(ValueError, match="requires exactly the short_only arm"):
        runner.run_detail_trials(
            "internvideo3",
            _write_manifests(tmp_path),
            tmp_path / "attempts",
            selected_arms=selected_arms,
            prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
        )


@pytest.mark.parametrize("backend_name", ["qwen3-vl", "qwen3-8"])
def test_runner_passes_qwen_length_and_rejects_incomplete_arm_paths(
    tmp_path: Path,
    monkeypatch,
    backend_name: str,
) -> None:
    paths = _write_manifests(tmp_path)
    captured = SimpleNamespace(max_model_len=None)

    def load_backend(_name: str, *, expected_input_frames: int, max_model_len: int | None) -> _Backend:
        captured.max_model_len = max_model_len
        return _Backend()

    monkeypatch.setattr(runner, "_load_backend", load_backend)
    runner.run_detail_trials(
        backend_name,
        {arm.value: path for arm, path in paths.items()},
        tmp_path / "qwen-attempts",
        limit=1,
        qwen_max_model_len=8_192,
    )
    assert captured.max_model_len == 8_192
