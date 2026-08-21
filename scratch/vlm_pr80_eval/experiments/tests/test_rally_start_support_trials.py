from __future__ import annotations

import pytest
from experiments.rally_start_support_trials import (
    SupportArm,
    _observation_sentences,
    _one_second_window,
    _proposal_sentences,
    build_parser,
    build_prompt,
)
from experiments.rally_start_trials import PROMPT, _reject_truth_keys


def _support_case() -> dict[str, object]:
    return {
        "observations": ["A compact observation."],
        "proposals": ["A fallible proposal."],
    }


def test_support_prompt_preserves_the_frozen_base_prompt() -> None:
    prompt = build_prompt(_support_case(), SupportArm.OBSERVATIONS)

    assert prompt.startswith(PROMPT + "\n\n")
    assert prompt.removeprefix(PROMPT).startswith("\n\nAutomated video analysis")
    assert "A compact observation." in prompt
    assert "A fallible proposal." not in prompt


def test_proposal_arm_only_adds_the_fallible_proposals() -> None:
    prompt = build_prompt(
        _support_case(), SupportArm.OBSERVATIONS_PLUS_PROPOSALS
    )

    assert "A compact observation." in prompt
    assert "A fallible proposal." in prompt


def test_support_window_is_one_second_and_clipped_to_the_case() -> None:
    assert _one_second_window(140, 100, 220, 30.0) == (125, 155)
    assert _one_second_window(104, 100, 220, 25.0) == (100, 125)
    assert _one_second_window(217, 100, 220, 25.0) == (195, 220)


def test_observations_use_plain_counts_and_boolean_wrist_evidence() -> None:
    contact = {"wrist_near": True, "proximity_ok": None}

    sentences = _observation_sentences(
        anchor_clip_frame=40,
        cut_clip_frame=40,
        court_count=24,
        player_count=20,
        shuttle_count=18,
        window_frames=25,
        contact=contact,
    )

    joined = " ".join(sentences)
    assert "24 of 25 frames" in joined
    assert "20 of 25 frames" in joined
    assert "18 of 25 frames" in joined
    assert "close to a player's wrist" in joined
    assert "composition" not in joined
    assert "mask" not in joined


def test_unresolved_current_rally_does_not_receive_a_proposal() -> None:
    assert _proposal_sentences({}, None, None) == [
        "The current pipeline did not resolve a server or physical-contact proposal for this clip."
    ]


def test_truth_keys_are_rejected_from_support() -> None:
    with pytest.raises(ValueError, match="forbidden key"):
        _reject_truth_keys(
            {"cases": [{"case_id": "case-1", "expected_server": "top"}]},
            "support manifest",
        )


def test_score_output_option_uses_the_shared_destination() -> None:
    args = build_parser().parse_args(
        [
            "score",
            "--base-manifest",
            "manifest.json",
            "--truth",
            "truth.json",
            "--support",
            "support.json",
            "--arm",
            "observations",
            "--backend",
            "internvideo3",
            "--attempts",
            "attempts",
            "--out",
            "score.json",
        ]
    )

    assert args.output.name == "score.json"
