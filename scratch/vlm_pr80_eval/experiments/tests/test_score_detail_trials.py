from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.detail_schema import DETAIL_MANIFEST_SCHEMA, DetailArm
from experiments.multiscale_prompts import DetailPromptMode
from experiments.run_detail_trials import run_detail_trials
from experiments.score_detail_trials import score_detail_attempts


def _write_manifests(
    tmp_path: Path,
    *,
    target_frames: int = 2,
) -> dict[DetailArm, Path]:
    clip = tmp_path / "clips" / "detail.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"same rendered detail clip")
    clip_sha256 = hashlib.sha256(clip.read_bytes()).hexdigest()
    expected_frames = max(6, target_frames + 4)
    source_start = 100
    source_end = source_start + expected_frames
    target_start = 102
    target_end = target_start + target_frames
    paths: dict[DetailArm, Path] = {}
    for arm in DetailArm:
        arm_dir = tmp_path / arm.value
        arm_dir.mkdir()
        is_short = arm is DetailArm.SHORT_ONLY
        is_broad = arm is DetailArm.BROAD_FACTS
        payload = {
            "schema": DETAIL_MANIFEST_SCHEMA,
            "arm": arm.value,
            "expected_frames": expected_frames,
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
                    "source_start_frame": source_start,
                    "source_end_frame": source_end,
                    "source_frames": list(range(source_start, source_end)),
                    "target_start_frame": target_start,
                    "target_end_frame": target_end,
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


def _write_truth(tmp_path: Path, *, target_end: int = 106) -> Path:
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vlm-multiscale-truth-v1",
                "cases": [
                    {
                        "pair_id": "context-sset_01-r007",
                        "video_id": "sset_01",
                        "target_start_frame": 100,
                        "target_end_frame": target_end,
                        "truth_intervals": [
                            {"source_start_frame": 100, "source_end_frame": 103, "truth": "live"},
                            {"source_start_frame": 103, "source_end_frame": target_end, "truth": "replay"},
                        ],
                    }
                ],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _fake_attempts(
    tmp_path: Path,
    paths: dict[DetailArm, Path],
    monkeypatch,
    *,
    selected_arms: tuple[DetailArm, ...] | None = None,
    prompt_mode: DetailPromptMode | str = DetailPromptMode.DEFAULT,
) -> Path:
    import experiments.run_detail_trials as runner
    from experiments.backends import BackendSpec, GenerationEvidence

    class Backend:
        spec = BackendSpec("test", "test/model", "rev", "test", "test", "1", "bf16", ())
        backend_version = "1"

        def generate(self, _video_path, prompt, **kwargs):
            content = "replay" if "deterministic_facts=" in prompt else "live"
            if "broad_facts=" in prompt:
                return GenerationEvidence("bad", tuple(range(6)), 32, 24, 12, 40)
            return GenerationEvidence(json.dumps({"target_content": content}), tuple(range(6)), 32, 24, 12, 40)

    monkeypatch.setattr(runner, "_load_backend", lambda *_args, **_kwargs: Backend())
    attempts = tmp_path / "attempts"
    run_detail_trials(
        "internvideo3",
        paths,
        attempts,
        selected_arms=selected_arms,
        prompt_mode=prompt_mode,
    )
    return attempts


def test_detail_score_intersects_truth_and_keeps_invalid_reply_in_denominator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(tmp_path, paths, monkeypatch)
    score = score_detail_attempts(paths, _write_truth(tmp_path), attempts, "internvideo3")

    assert score["by_arm"]["short_only"]["cases"] == 1
    assert score["by_arm"]["short_only"]["valid_replies"] == 1
    assert score["by_arm"]["short_only"]["mean_case_exact_scene_accuracy"] == 0.5
    assert score["by_arm"]["deterministic"]["target_frame_binary_live_nonlive_accuracy"] == 0.5
    assert score["by_arm"]["broad_facts"]["valid_replies"] == 0
    assert score["by_arm"]["broad_facts"]["mean_case_exact_scene_accuracy"] == 0.0
    assert score["by_arm"]["broad_facts"]["close_check_recall"] == 1.0
    assert score["by_arm"]["short_only"]["routine_live_precision"] == 0.0
    assert score["by_arm"]["deterministic"]["routine_live_precision"] is None
    assert score["comparison"]["broad_facts_vs_deterministic"]["mean_case_exact_scene_accuracy_delta"] == -0.5
    assert set(score["by_arm"]) == {arm.value for arm in DetailArm}
    assert set(score["comparison"]) == {
        "broad_facts_vs_deterministic",
        "broad_facts_vs_short_only",
        "deterministic_vs_short_only",
    }
    assert score["prompt_mode"] == DetailPromptMode.DEFAULT.value
    assert score["rows"][0]["minimum_material_target_frames"] == 9
    assert not score["rows"][0]["meets_material_target"]
    assert score["by_arm"]["short_only"]["cases"] == 1
    assert score["by_arm_material_target"]["short_only"]["cases"] == 0
    assert score["by_arm_material_target"]["short_only"]["excluded_cases"] == 1


def test_detail_score_reproduces_conservative_prompt_mode_and_rejects_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(
        tmp_path,
        paths,
        monkeypatch,
        selected_arms=(DetailArm.SHORT_ONLY,),
        prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
    )

    score = score_detail_attempts(
        paths,
        _write_truth(tmp_path),
        attempts,
        "internvideo3",
        selected_arms=(DetailArm.SHORT_ONLY,),
        prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
    )

    assert score["prompt_mode"] == DetailPromptMode.CONSERVATIVE_REPLAY_VETO.value
    with pytest.raises(ValueError, match="prompt differs"):
        score_detail_attempts(
            paths,
            _write_truth(tmp_path),
            attempts,
            "internvideo3",
            selected_arms=(DetailArm.SHORT_ONLY,),
        )


def test_material_target_uses_ceil_native_frames_at_25_fps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path, target_frames=9)
    attempts = _fake_attempts(tmp_path, paths, monkeypatch)

    score = score_detail_attempts(
        paths,
        _write_truth(tmp_path, target_end=111),
        attempts,
        "internvideo3",
    )

    row = score["rows"][0]
    assert row["target_frames"] == 9
    assert row["minimum_material_target_frames"] == 9
    assert row["meets_material_target"]
    assert score["by_arm"]["short_only"]["cases"] == 1
    assert score["by_arm_material_target"]["short_only"]["cases"] == 1
    assert score["by_arm_material_target"]["short_only"]["excluded_cases"] == 0


def test_detail_score_selected_short_only_uses_only_selected_attempts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(
        tmp_path,
        paths,
        monkeypatch,
        selected_arms=(DetailArm.SHORT_ONLY,),
    )

    score = score_detail_attempts(
        paths,
        _write_truth(tmp_path),
        attempts,
        "internvideo3",
        selected_arms=(DetailArm.SHORT_ONLY,),
    )

    assert set(score["by_arm"]) == {DetailArm.SHORT_ONLY.value}
    assert score["by_arm"]["short_only"]["cases"] == 1
    assert score["comparison"] == {}
    assert set(score["cases"][0]["by_arm"]) == {DetailArm.SHORT_ONLY.value}
    assert {row["arm"] for row in score["rows"]} == {DetailArm.SHORT_ONLY.value}


@pytest.mark.parametrize(
    "selected_arms",
    [(DetailArm.DETERMINISTIC,), (DetailArm.SHORT_ONLY, DetailArm.DETERMINISTIC)],
)
def test_detail_score_rejects_conservative_prompt_for_other_or_multiple_arms(
    tmp_path: Path,
    monkeypatch,
    selected_arms: tuple[DetailArm, ...],
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(tmp_path, paths, monkeypatch)

    with pytest.raises(ValueError, match="requires exactly the short_only arm"):
        score_detail_attempts(
            paths,
            _write_truth(tmp_path),
            attempts,
            "internvideo3",
            selected_arms=selected_arms,
            prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
        )


def test_detail_score_recovers_newly_supported_exact_reply_form(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(
        tmp_path,
        paths,
        monkeypatch,
        selected_arms=(DetailArm.SHORT_ONLY,),
    )
    attempt_path = attempts / "internvideo3" / "short_only" / "detail-case.json"
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["raw_response"] = "{target_content: live}"
    payload["parsed_response"] = None
    payload["parser_error"] = (
        "detail reply is not valid JSON: Expecting property name enclosed in double quotes"
    )
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")

    score = score_detail_attempts(
        paths,
        _write_truth(tmp_path),
        attempts,
        "internvideo3",
        selected_arms=(DetailArm.SHORT_ONLY,),
    )

    row = score["rows"][0]
    assert row["valid_reply"]
    assert row["parser_recovered"]
    assert row["predicted_content"] == "live"


@pytest.mark.parametrize(
    "selected_arms",
    [(), (DetailArm.SHORT_ONLY, DetailArm.SHORT_ONLY)],
)
def test_detail_score_rejects_empty_or_duplicate_arm_selection(
    tmp_path: Path,
    monkeypatch,
    selected_arms: tuple[DetailArm, ...],
) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(tmp_path, paths, monkeypatch)

    with pytest.raises(ValueError, match="selected_arms"):
        score_detail_attempts(
            paths,
            _write_truth(tmp_path),
            attempts,
            "internvideo3",
            selected_arms=selected_arms,
        )


def test_detail_score_rejects_changed_clip_identity(tmp_path: Path, monkeypatch) -> None:
    paths = _write_manifests(tmp_path)
    attempts = _fake_attempts(tmp_path, paths, monkeypatch)
    attempt_path = attempts / "internvideo3" / "short_only" / "detail-case.json"
    payload = json.loads(attempt_path.read_text(encoding="utf-8"))
    payload["case"]["clip_sha256"] = "0" * 64
    attempt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case identity differs"):
        score_detail_attempts(paths, _write_truth(tmp_path), attempts, "internvideo3")
