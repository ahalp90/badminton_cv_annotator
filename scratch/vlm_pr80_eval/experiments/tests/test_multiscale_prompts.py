from pathlib import Path

import pytest
from experiments.detail_schema import DetailArm, DetailBroadFact, DetailCase
from experiments.multiscale_prompts import (
    DetailPromptMode,
    build_broad_prompt,
    build_detail_prompt,
)
from experiments.multiscale_schema import MultiscaleCase, Segment


def test_broad_prompt_names_target_segments_without_truth() -> None:
    case = MultiscaleCase(
        case_id="case--90",
        pair_id="case",
        video_id="sset_01",
        context_seconds=90,
        clip_path=Path("unused.mp4"),
        source_start_frame=0,
        source_end_frame=2_250,
        target_start_frame=1_000,
        target_end_frame=1_200,
        sample_fps=8.0,
        expected_frames=96,
        width=512,
        height=288,
        source_frames=tuple(range(96)),
        segments=(
            Segment("S0010", 0, 1_000),
            Segment("S0011", 1_000, 1_200),
            Segment("S0012", 1_200, 2_250),
        ),
        pipeline_priors={},
    )

    prompt = build_broad_prompt(case)

    assert "It overlaps: S0011" in prompt
    assert "fallible" in prompt
    assert "ground truth" not in prompt.lower()
    assert "human" not in prompt.lower()
    assert "S0001" not in prompt
    assert "live|replay" not in prompt
    assert "first output" in prompt
    assert "Do not return\nthat list on its own" in prompt
    assert "``` fences" in prompt
    assert "current rally" in prompt
    assert "active-looking play might be a replay" in prompt


def _detail_case(
    *,
    deterministic_facts: dict | None = None,
    broad_facts: tuple[DetailBroadFact, ...] | None = None,
) -> DetailCase:
    return DetailCase(
        case_id="detail-case",
        pair_id="pair",
        context_case_id="context-case--90",
        video_id="sset_01",
        clip_path=Path("detail.mp4"),
        clip_sha256="0" * 64,
        source_start_frame=100,
        source_end_frame=106,
        source_frames=(100, 101, 102, 103, 104, 105),
        target_start_frame=102,
        target_end_frame=104,
        boundary_frame=103,
        source_fps=25.0,
        sample_fps=25.0,
        expected_frames=6,
        width=512,
        height=288,
        target_segment_ids=("S0001",),
        deterministic_facts=deterministic_facts,
        broad_facts=broad_facts,
    )


def test_detail_prompt_short_only_has_no_context_facts() -> None:
    prompt = build_detail_prompt(_detail_case(), DetailArm.SHORT_ONLY)

    assert "target_content" in prompt
    assert "first\noutput character" in prompt
    assert "deterministic_facts" not in prompt
    assert "broad_facts" not in prompt
    assert "span_membership" not in prompt


def test_detail_prompt_default_mode_keeps_existing_identity() -> None:
    case = _detail_case()

    assert build_detail_prompt(case, DetailArm.SHORT_ONLY) == build_detail_prompt(
        case,
        DetailArm.SHORT_ONLY,
        prompt_mode=DetailPromptMode.DEFAULT,
    )


def test_detail_prompt_conservative_replay_veto_is_short_only_and_evidence_led() -> None:
    prompt = build_detail_prompt(
        _detail_case(),
        DetailArm.SHORT_ONLY,
        prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
    )

    assert "No broad context is available" in prompt
    assert "clear evidence this is the current rally" in prompt
    assert "visible replay cues" in prompt
    assert "active badminton could be a replay" in prompt
    for label in ("live", "replay", "cutaway", "other", "unclear"):
        assert f"`{label}`" in prompt
    assert "only field is `target_content`" in prompt
    assert "deterministic_facts" not in prompt
    assert "broad_facts" not in prompt

    with pytest.raises(ValueError, match="only for short_only"):
        build_detail_prompt(
            _detail_case(deterministic_facts={"span_membership": True}),
            DetailArm.DETERMINISTIC,
            prompt_mode=DetailPromptMode.CONSERVATIVE_REPLAY_VETO,
        )


def test_detail_prompt_broad_facts_keeps_deterministic_prefix_and_parsed_fields() -> None:
    deterministic = {"span_membership": True, "cut_ids": ["S0001"]}
    broad = (DetailBroadFact("S0001", "live", None, False),)
    deterministic_prompt = build_detail_prompt(
        _detail_case(deterministic_facts=deterministic),
        DetailArm.DETERMINISTIC,
    )
    broad_prompt = build_detail_prompt(
        _detail_case(deterministic_facts=deterministic, broad_facts=broad),
        DetailArm.BROAD_FACTS,
    )

    assert 'deterministic_facts={"cut_ids":["S0001"],"span_membership":true}' in deterministic_prompt
    assert 'deterministic_facts={"cut_ids":["S0001"],"span_membership":true}' in broad_prompt
    assert 'broad_facts=[{"content":"live"' in broad_prompt
    assert "raw_response" not in broad_prompt


def test_detail_prompt_rejects_facts_for_short_only() -> None:
    with pytest.raises(ValueError, match="short_only"):
        build_detail_prompt(
            _detail_case(),
            DetailArm.SHORT_ONLY,
            deterministic_facts={"span_membership": True},
        )
