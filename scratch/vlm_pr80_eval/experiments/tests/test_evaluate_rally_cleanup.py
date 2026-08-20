import hashlib
from pathlib import Path

import pytest
from experiments.evaluate_rally_cleanup import (
    _validate_attempt_identity,
    select_rule_frames,
)
from experiments.prompts import build_prompt
from experiments.trial_schema import TrialArm, TrialCase, TrialKind


def test_cleanup_rules_reject_invalid_and_can_preserve_first_candidate() -> None:
    decisions = {
        10: None,
        20: {
            "contact_at_marker": "yes",
            "evidence_kind": "inferred-contact",
        },
        30: {
            "contact_at_marker": "yes",
            "evidence_kind": "visible-contact",
        },
        40: {
            "contact_at_marker": "no",
            "evidence_kind": "no-contact",
        },
    }

    assert select_rule_frames([10, 20, 30, 40], decisions, "natural") == [
        10,
        20,
        30,
        40,
    ]
    assert select_rule_frames([10, 20, 30, 40], decisions, "vlm-yes") == [20, 30]
    assert select_rule_frames([10, 20, 30, 40], decisions, "vlm-visible") == [30]
    assert select_rule_frames(
        [10, 20, 30, 40], decisions, "vlm-yes-preserve-first"
    ) == [10, 20, 30]


def test_rally_evaluator_rejects_attempt_from_different_clip(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"current clip")
    case = TrialCase(
        case_id="event-one",
        kind=TrialKind.EVENT,
        video_id="sset_01",
        clip_path=clip,
        source_start_frame=0,
        source_end_frame=50,
        candidate_frame=25,
        sample_fps=25.0,
        pipeline_priors={},
    )
    prompt = build_prompt(case, TrialArm.VIDEO_ONLY)
    attempt = {
        "case": {
            "case_id": case.case_id,
            "kind": case.kind.value,
            "video_id": case.video_id,
            "clip_path": str(case.clip_path),
            "clip_sha256": hashlib.sha256(b"stale clip").hexdigest(),
            "source_start_frame": case.source_start_frame,
            "source_end_frame": case.source_end_frame,
            "candidate_frame": case.candidate_frame,
        },
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }

    with pytest.raises(ValueError, match="wrong clip hash"):
        _validate_attempt_identity(case, attempt, TrialArm.VIDEO_ONLY)
