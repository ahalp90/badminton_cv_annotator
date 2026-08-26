from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import experiments.run_multiscale_trials as runner
from experiments.backends import BackendSpec, GenerationEvidence


def _write_manifest(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    payload = {
        "schema": "vlm-multiscale-manifest-v1",
        "expected_frames": 6,
        "width": 32,
        "height": 24,
        "cases": [
            {
                "case_id": "case--90",
                "pair_id": "case",
                "video_id": "sset_01",
                "context_seconds": 90,
                "clip_path": str(clip),
                "source_start_frame": 100,
                "source_end_frame": 200,
                "target_start_frame": 130,
                "target_end_frame": 160,
                "sample_fps": 2.0,
                "source_frames": [100, 120, 130, 159, 160, 199],
                "segments": [
                    {
                        "segment_id": "S0010",
                        "source_start_frame": 100,
                        "source_end_frame": 130,
                    },
                    {
                        "segment_id": "S0011",
                        "source_start_frame": 130,
                        "source_end_frame": 160,
                    },
                    {
                        "segment_id": "S0012",
                        "source_start_frame": 160,
                        "source_end_frame": 200,
                    },
                ],
                "pipeline_priors": {},
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class _Backend:
    spec = BackendSpec("test", "test/model", "rev", "test", "test", "1", "bf16", ())
    backend_version = "1"

    def generate(self, *_args, **_kwargs) -> GenerationEvidence:
        response = json.dumps(
            {
                "segments": [
                    {
                        "segment_id": "S0010",
                        "content": "replay",
                        "repeat_of": None,
                        "needs_close_check": False,
                    },
                    {
                        "segment_id": "S0011",
                        "content": "live",
                        "repeat_of": None,
                        "needs_close_check": False,
                    },
                    {
                        "segment_id": "S0012",
                        "content": "cutaway",
                        "repeat_of": None,
                        "needs_close_check": False,
                    },
                ]
            }
        )
        return GenerationEvidence(response, tuple(range(6)), 32, 24, 12, 40)


def test_runner_uses_manifest_frame_count_and_target_only_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    captured = SimpleNamespace(expected_frames=None)

    def load_backend(_name, *, expected_input_frames, max_model_len):
        captured.expected_frames = expected_input_frames
        return _Backend()

    monkeypatch.setattr(runner, "_load_backend", load_backend)

    runner.run_multiscale_trials("internvideo3", manifest, tmp_path / "attempts")

    assert captured.expected_frames == 6
    attempt = json.loads(
        (tmp_path / "attempts/internvideo3/case--90.json").read_text(encoding="utf-8")
    )
    assert attempt["parser_error"] is None
    assert attempt["target_route"] == "routine_live"
    assert attempt["sampling"]["sampled_input_frames"] == list(range(6))
