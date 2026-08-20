"""Frozen-prompt tests for the Issue 38 VLM benchmark."""

from __future__ import annotations

import pytest

from annotator.vlm_scene_benchmark.contracts import ShardSpec
from annotator.vlm_scene_benchmark.prompts import (
    PROMPT_VERSION,
    build_correction_prompt,
    build_scene_prompt,
)


SHA256 = "a" * 64


def _shard() -> ShardSpec:
    return ShardSpec("sset_15", "source.mp4", "b" * 64, "input.mp4", SHA256, 25.0, 100, 10, 60)


def test_scene_prompt_contains_exact_grid_cuts_and_partition() -> None:
    prompt = build_scene_prompt(_shard(), (10, 35, 59), (20, 40))

    assert f"Prompt version: {PROMPT_VERSION}" in prompt
    assert "ordered source-frame grid: 10,35,59" in prompt
    assert "candidate hard-cut source frames: 20,40" in prompt
    assert "complete partition of [10, 60)" in prompt
    assert "live-non-standard" in prompt
    assert "Return JSON only" in prompt
    assert 'one top-level key named "frames"' in prompt
    assert "array of exactly 3 strings" in prompt
    assert "exactly eight characters" in prompt
    assert '"frames":["LBRFRS9B","LLRFRS9R"]' in prompt


def test_scene_prompt_rejects_unordered_or_outside_evidence() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        build_scene_prompt(_shard(), (10, 10), ())
    with pytest.raises(ValueError, match="outside"):
        build_scene_prompt(_shard(), (9, 35), ())
    with pytest.raises(ValueError, match="internal"):
        build_scene_prompt(_shard(), (10, 35), (60,))


def test_correction_prompt_includes_error_without_repeating_response() -> None:
    correction = build_correction_prompt("initial", "gap at frame 30")

    assert "initial" in correction
    assert "gap at frame 30" in correction
    assert "Do not copy or continue it" in correction
    assert "eight-character frame codes" in correction
    assert "raw invalid response" not in correction
