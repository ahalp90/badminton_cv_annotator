from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.score_trials import score_trials
from experiments.trial_schema import ATTEMPT_SCHEMA, MANIFEST_SCHEMA, TrialArm


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_score_keeps_unclear_separate(tmp_path: Path) -> None:
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"test clip placeholder")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "cases": [
            {
                "case_id": "event-1",
                "kind": "event",
                "video_id": "sset_01",
                "clip_path": str(clip_path),
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
    truth = {
        "schema": "vlm-cleanup-truth/0.1",
        "cases": [
            {
                "case_id": "event-1",
                "kind": "event",
                "video_id": "sset_01",
                "stratum": "positive",
                "nearest_gt_frame": 25,
                "distance_to_gt_base30": 0.0,
                "usable_at_5": True,
                "usable_at_10": True,
                "usable_at_15": True,
                "expected_actor": "top",
                "set_id": "set1",
                "rally": 1,
                "ball_round": 2,
                "event_role": "later-stroke",
            }
        ],
    }
    raw_response = json.dumps(
        {
            "contact_at_marker": "unclear",
            "evidence_kind": "unclear",
            "actor": "unclear",
            "nearby_unmarked_contact": "no",
            "visible_evidence": "The racket is blurred.",
            "uncertainty": "The shuttle is not visible.",
        }
    )
    attempt = {
        "schema": ATTEMPT_SCHEMA,
        "backend": "qwen3-vl",
        "model": {"name": "test-model"},
        "case": {
            "case_id": "event-1",
            "kind": "event",
            "video_id": "sset_01",
            "clip_path": str(clip_path),
            "clip_sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
            "source_start_frame": 0,
            "source_end_frame": 50,
            "candidate_frame": 25,
        },
        "arm": "video-only",
        "prompt": "test prompt",
        "prompt_sha256": hashlib.sha256(b"test prompt").hexdigest(),
        "raw_response": raw_response,
        "parsed_response": json.loads(raw_response),
        "parser_error": None,
        "generation_error": None,
        "elapsed_seconds": 1.0,
        "sampling": {},
    }
    manifest_path = tmp_path / "manifest.json"
    truth_path = tmp_path / "truth.json"
    _write(manifest_path, manifest)
    _write(truth_path, truth)
    _write(tmp_path / "attempts" / "attempt.json", attempt)

    score = score_trials(manifest_path, truth_path, tmp_path / "attempts")

    metrics = score["results"][0]["metrics"]["contact_at_10"]
    assert metrics["unclear"] == 1
    assert metrics["invalid"] == 0
    assert metrics["yes"] == 0
    assert metrics["yes_recall"] == 0.0
    assert score["complete"] is False
    assert score["inference_complete"] is False
    assert score["parse_complete"] is True

    one_arm_score = score_trials(
        manifest_path,
        truth_path,
        tmp_path / "attempts",
        expected_backends=("qwen3-vl",),
        expected_arms=(TrialArm.VIDEO_ONLY,),
    )
    assert one_arm_score["complete"] is True
    assert one_arm_score["expected_arms"] == ["video-only"]

    prior_attempt = {**attempt, "arm": "pipeline-priors"}
    _write(tmp_path / "attempts" / "prior-attempt.json", prior_attempt)
    one_backend_score = score_trials(
        manifest_path,
        truth_path,
        tmp_path / "attempts",
        expected_backends=("qwen3-vl",),
    )
    assert one_backend_score["complete"] is True
    assert one_backend_score["missing_attempts"] == []

    contradictory_response = json.dumps(
        {
            "contact_at_marker": "no",
            "evidence_kind": "no-contact",
            "actor": "unclear",
            "nearby_unmarked_contact": "no",
            "visible_evidence": "No contact is visible.",
            "uncertainty": "The actor field is inconsistent.",
        }
    )
    invalid_attempt = {
        **prior_attempt,
        "raw_response": contradictory_response,
        "parsed_response": None,
        "parser_error": "a rejected contact must have actor no-contact",
    }
    _write(tmp_path / "attempts" / "prior-attempt.json", invalid_attempt)
    invalid_score = score_trials(
        manifest_path,
        truth_path,
        tmp_path / "attempts",
        expected_backends=("qwen3-vl",),
    )
    prior_result = next(
        result
        for result in invalid_score["results"]
        if result["arm"] == "pipeline-priors"
    )
    assert prior_result["attempted_cases"] == 1
    assert prior_result["parsed_cases"] == 0
    assert prior_result["metrics"]["contact_at_10"]["invalid"] == 1
    assert prior_result["metrics"]["contact_at_10"]["yes_recall"] == 0.0
    assert prior_result["metrics"]["contact_at_10"]["attempt_accuracy"] == 0.0
    assert invalid_score["parse_complete"] is False

    with pytest.raises(ValueError, match="unknown requested case IDs"):
        score_trials(
            manifest_path,
            truth_path,
            tmp_path / "attempts",
            case_ids={"missing-case"},
        )
