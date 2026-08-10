"""Frozen prompts for the Issue 38 whole-shard scene benchmark."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import ShardSpec


PROMPT_VERSION = "issue38-whole-shard-v1"


def _cut_text(cut_frames: Sequence[int]) -> str:
    return "none" if not cut_frames else ",".join(str(frame) for frame in cut_frames)


def _frame_text(sampled_source_frames: Sequence[int]) -> str:
    return ",".join(str(frame) for frame in sampled_source_frames)


def build_scene_prompt(
    shard: ShardSpec,
    sampled_source_frames: Sequence[int],
    cut_frames: Sequence[int],
) -> str:
    """Build the one frozen inference prompt without human labels."""
    if not sampled_source_frames:
        raise ValueError("the scene prompt requires sampled source frames")
    if any(right <= left for left, right in zip(sampled_source_frames, sampled_source_frames[1:])):
        raise ValueError("sampled source frames must be strictly increasing")
    if sampled_source_frames[0] < shard.start_frame or sampled_source_frames[-1] >= shard.end_frame:
        raise ValueError("sampled source frames are outside the benchmark shard")
    if any(not shard.start_frame < frame < shard.end_frame for frame in cut_frames):
        raise ValueError("candidate cut frames must be internal to the benchmark shard")

    return f"""You are labelling one complete badminton broadcast shard for later human review.

Prompt version: {PROMPT_VERSION}
Source metadata:
- video_id: {shard.video_id}
- source fps: {shard.fps:.12g}
- full source frame count: {shard.frame_count}
- shard: [{shard.start_frame}, {shard.end_frame}) in zero-based, half-open source frames
- supplied video frames: {len(sampled_source_frames)} uniformly sampled frames spanning the shard
- sample mapping: supplied video frame i maps to the i-th entry in the ordered source-frame grid
- ordered source-frame grid: {_frame_text(sampled_source_frames)}
- candidate hard-cut source frames: {_cut_text(cut_frames)}

Use exactly these scene labels:
- live: standard court-showing live footage
- live-non-standard: actual live action or warm-up from an unusual camera view
- replay: repeated, slow-motion, or freeze-frame footage of earlier play
- cutaway: player close-up, audience, ceremony, or another non-play broadcast shot
- other: graphics, broadcast stings, transitions, adverts, or footage outside those classes

Candidate cuts are mechanical hints. A class may continue across a cut or change inside a detected scene. A side-on service setup is cutaway until actual play begins. If actual play begins before that shot ends, label the whole shot live-non-standard.

Return JSON only, with exactly one top-level key named "segments". Its value must be an ordered array of objects with exactly these keys:
- start_frame: integer absolute source frame, inclusive
- end_frame: integer absolute source frame, exclusive
- scene_label: live, live-non-standard, replay, cutaway, or other
- broadcast_phase: live_rally, between_rallies, replay, cutaway, other, or unknown
- view: full_court, partial_court, side_on, close_up, crowd, graphic, other, or unknown
- playback: real_time, slow_motion, freeze_frame, or unknown
- continuity_from_previous: same_rally, new_rally, not_applicable, or unknown
- data_use: usable_standard, usable_alternate_view, exclude, or review
- confidence: number from 0 to 1
- evidence_frames: array of integer absolute source frames inside this segment
- reason: one short sentence based only on visible broadcast evidence

The segments must form a complete partition of [{shard.start_frame}, {shard.end_frame}). The first start_frame must be {shard.start_frame}. Every next start_frame must equal the preceding end_frame. The final end_frame must be {shard.end_frame}. Do not add markdown fences, commentary, or keys outside the schema."""


def build_correction_prompt(initial_prompt: str, invalid_response: str, validation_error: str) -> str:
    """Ask once for a full replacement after strict validation fails."""
    if not invalid_response:
        invalid_response = "<empty response>"
    return f"""{initial_prompt}

Your preceding response failed strict validation with this error:
{validation_error}

Replace it with a complete corrected response. Return JSON only. Do not explain the correction.

Preceding invalid response:
{invalid_response}"""
