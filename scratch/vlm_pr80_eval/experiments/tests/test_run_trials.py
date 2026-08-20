from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from experiments.trial_schema import MANIFEST_SCHEMA, TrialArm

from experiments import run_trials as runner


@dataclass(frozen=True)
class _Identity:
    name: str


class _Spec:
    @staticmethod
    def identity(_backend_version: str) -> _Identity:
        return _Identity(name="failing-test-backend")


class _FailingBackend:
    spec = _Spec()
    backend_version = "test"

    @staticmethod
    def generate(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("deliberate generation failure")


def test_generation_failure_writes_immutable_attempt_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"placeholder")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "cases": [
            {
                "case_id": "event-failure",
                "kind": "event",
                "video_id": "sset_01",
                "clip_path": str(clip),
                "source_start_frame": 0,
                "source_end_frame": 50,
                "candidate_frame": 25,
                "sample_fps": 25.0,
                "expected_frames": 50,
                "width": 512,
                "height": 288,
                "pipeline_priors": {},
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        runner, "_load_backend", lambda *_args, **_kwargs: _FailingBackend()
    )

    with pytest.raises(RuntimeError, match="deliberate generation failure"):
        runner.run_trials("qwen3-vl", manifest_path, tmp_path / "attempts")

    attempt_path = tmp_path / "attempts/qwen3-vl/event-failure--video-only.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["generation_error"] == "RuntimeError: deliberate generation failure"
    assert attempt["raw_response"] is None
    assert attempt["parsed_response"] is None


def test_run_trials_can_limit_prompt_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "cases": [
                    {
                        "case_id": "event-failure",
                        "kind": "event",
                        "video_id": "sset_01",
                        "clip_path": str(clip),
                        "source_start_frame": 0,
                        "source_end_frame": 50,
                        "candidate_frame": 25,
                        "sample_fps": 25.0,
                        "expected_frames": 50,
                        "width": 512,
                        "height": 288,
                        "pipeline_priors": {
                            "court_present": True,
                            "track_visible": True,
                            "wrist_near": True,
                            "proximity_ok": None,
                            "suppressed": False,
                            "raw_masked": False,
                            "definitive_masked": False,
                            "seconds_from_previous_raw_candidate": None,
                            "seconds_to_next_raw_candidate": None,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner, "_load_backend", lambda *_args, **_kwargs: _FailingBackend()
    )

    with pytest.raises(RuntimeError, match="deliberate generation failure"):
        runner.run_trials(
            "qwen3-vl",
            manifest_path,
            tmp_path / "attempts",
            arms=(TrialArm.PIPELINE_PRIORS,),
        )

    assert not (tmp_path / "attempts/qwen3-vl/event-failure--video-only.json").exists()
    assert (
        tmp_path / "attempts/qwen3-vl/event-failure--pipeline-priors.json"
    ).exists()
