"""Frozen prompts for the Issue 38 whole-shard scene benchmark."""

from __future__ import annotations

from collections.abc import Sequence

from .contracts import ShardSpec


PROMPT_VERSION = "issue38-whole-shard-v5-observed-frame-codes"


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

Return exactly one code for every supplied video frame, in the same order as the source-frame grid. The JSON must have exactly one top-level key named "frames". Its value must be an array of exactly {len(sampled_source_frames)} strings. Every string must contain exactly eight characters with no spaces:
0. scene: L=live, N=live-non-standard, R=replay, C=cutaway, O=other
1. phase: L=live_rally, B=between_rallies, R=replay, C=cutaway, O=other, U=unknown
2. playback: R=real_time, S=slow_motion, F=freeze_frame, U=unknown
3. view: F=full_court, P=partial_court, S=side_on, C=close_up, D=crowd, G=graphic, O=other, U=unknown
4. continuity: S=same_rally, R=new_rally or reset, A=not_applicable, U=unknown
5. data use: S=usable_standard, A=usable_alternate_view, E=exclude, R=review
6. confidence: 0 through 9 mean 0.0 through 0.9, A means 1.0
7. visible reason: R=active rally, B=between rallies or preparing, P=replay, C=cutaway, G=graphic or other, U=unclear

Example shape for two frames only: {{"frames":["LBRFRS9B","LLRFRS9R"]}}

Do not group frames, omit frames, repeat frames, include frame numbers, or use segment objects. The parser maps each code to its source-frame interval, merges adjacent identical states, and reconstructs a complete partition of [{shard.start_frame}, {shard.end_frame}). Return JSON only, as one bare object. Do not add markdown fences, commentary, or other keys."""


def build_correction_prompt(initial_prompt: str, validation_error: str) -> str:
    """Ask once for a fixed-width replacement after strict validation fails."""
    return f"""{initial_prompt}

Your preceding response failed strict validation with this error:
{validation_error}

Re-evaluate the video and replace the preceding response. Do not copy or continue it. Return exactly the requested number of eight-character frame codes in one bare JSON object."""
