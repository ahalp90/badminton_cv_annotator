"""Small prompt contracts for the cut-aware multiscale trials."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import StrEnum
from typing import Any

from .detail_schema import (
    DetailArm,
    DetailBroadFact,
    DetailCase,
    DetailContent,
)
from .multiscale_schema import MultiscaleCase, reject_truth_keys


class DetailPromptMode(StrEnum):
    """Prompt contract used by one paired detail trial."""

    DEFAULT = "default"
    CONSERVATIVE_REPLAY_VETO = "conservative_replay_veto"


def build_broad_prompt(case: MultiscaleCase) -> str:
    """Ask for one terse record per supplied cut-bounded segment."""
    segment_lines = []
    for segment in case.segments:
        segment_lines.append(
            f"- {segment.segment_id}: frames {segment.source_start_frame} to "
            f"{segment.source_end_frame - 1}"
        )
    segments = "\n".join(segment_lines)
    target_ids = ", ".join(case.target_segment_ids)
    return f"""You are checking the order of scenes in a badminton broadcast.

The video is a time-ordered storyboard from a {case.context_seconds}-second source window.
Each frame shows its source time and segment ID. Segment boundaries are
fallible results from an automatic cut detector and may be slightly wrong.

TARGET is source frames {case.target_start_frame} to {case.target_end_frame - 1}. It overlaps: {target_ids}.

Segments:
{segments}

For every segment, choose one content value:
- live: the current rally, including serve preparation that leads directly
  into it and unusual camera views of that action;
- replay: earlier action shown again, even though the pictured play looks real;
- cutaway: footage between rallies, including a player waiting, entering,
  celebrating, or preparing for a later point, plus crowd, coach, umpire, or
  venue shots;
- other: graphics, adverts, blank frames, or content outside those groups;
- unclear: the supplied frames do not support a safe choice.

`repeat_of` names an earlier segment only when a replay visibly repeats it.
Otherwise use null. Set `needs_close_check` true when TARGET or a doubtful
boundary in that segment needs denser frames. Also set it true when a segment
mixes broadcast roles or when active-looking play might be a replay. Judge the
broadcast role, not merely whether a player or court is visible.

Return one bare JSON object whose only field is `segments`. Your first output
character must be {{ and your last output character must be }}. The `segments`
value is a list with one record for every supplied segment ID. Do not return
that list on its own. Every record has exactly four fields: `segment_id`,
`content`, `repeat_of`, and `needs_close_check`. `needs_close_check` is a JSON
boolean. Do not add IDs or fields. Do not use Markdown or ``` fences."""


_DETAIL_BASE_PROMPT = """You are checking a marked short badminton broadcast clip.

The gold border marks TARGET. Judge the underlying pixels and motion inside that
marked interval. Automatic cuts and any supplied context are fallible. If the
marked interval is mixed, hidden, or does not support a safe decision, choose
unclear.

Return one bare JSON object whose only field is `target_content`. The first
output character must be { and the last must be }. Its value must be exactly
one of `live`, `replay`, `cutaway`, `other`, or `unclear`. Use `live`
for the current rally, including serve preparation that leads directly into it
and unusual camera views of that action. Use `replay` for earlier action shown
again, even when the pictured play looks real. Use `cutaway` for footage
between rallies, including a player waiting, entering, celebrating, or
preparing for a later point, plus crowd, coach, umpire, or venue footage. Use
`other` for graphics, adverts, blank frames, or unrelated content. Do not add
fields, prose, or Markdown fences."""


_CONSERVATIVE_REPLAY_VETO_PROMPT = """You are checking a marked short badminton broadcast
clip in an evidence-led child trial.

Use all frames in this short clip as local visual evidence, but judge the marked
TARGET. No broad context is available: do not invent broadcast order from
automatic cut results or metadata. Return `live` only when the short pixels give
clear evidence this is the current rally. Positive live evidence can include
normal-speed action from the usual broadcast view or serve preparation that
leads directly into play. Return `replay` when visible replay cues show earlier
action again. Replay cues can include slow motion, a replay transition, repeated
or reframed action within the clip, or a camera style used for replays. The lack
of a replay logo is not positive live evidence. Return `unclear` when
active badminton could be a replay but the short clip lacks enough broadcast-order evidence. Use
`cutaway` for footage between rallies, including a player waiting,
entering, celebrating, or preparing for a later point, plus crowd, coach,
umpire, or venue footage. Use `other` for graphics, adverts, blank frames, or
unrelated content.

Return one bare JSON object whose only field is `target_content`. The first
output character must be { and the last must be }. Its value must be exactly
one of `live`, `replay`, `cutaway`, `other`, or `unclear`. Do not add fields,
prose, or Markdown fences."""


def normalise_prompt_mode(raw_mode: DetailPromptMode | str) -> DetailPromptMode:
    try:
        return raw_mode if isinstance(raw_mode, DetailPromptMode) else DetailPromptMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown detail prompt mode {raw_mode!r}") from exc


def validate_detail_prompt_mode(
    raw_mode: DetailPromptMode | str,
    selected_arms: Sequence[DetailArm],
) -> DetailPromptMode:
    """Validate a prompt mode against the arms selected for one trial."""
    prompt_mode = normalise_prompt_mode(raw_mode)
    if prompt_mode is not DetailPromptMode.DEFAULT and tuple(selected_arms) != (
        DetailArm.SHORT_ONLY,
    ):
        raise ValueError("conservative_replay_veto requires exactly the short_only arm")
    return prompt_mode


def _json_facts(value: Mapping[str, Any], location: str) -> str:
    payload = dict(value)
    reject_truth_keys(payload, location)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _broad_fact_payload(
    facts: Sequence[DetailBroadFact | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_keys = {"segment_id", "content", "repeat_of", "needs_close_check"}
    payload: list[dict[str, Any]] = []
    for index, fact in enumerate(facts):
        raw = asdict(fact) if isinstance(fact, DetailBroadFact) else dict(fact)
        if set(raw) != expected_keys:
            raise ValueError(
                f"broad fact {index} keys differ from {sorted(expected_keys)}"
            )
        if not isinstance(raw["segment_id"], str) or not raw["segment_id"]:
            raise ValueError(f"broad fact {index} has an invalid segment_id")
        if not isinstance(raw["content"], (str, DetailContent)):
            raise TypeError(f"broad fact {index} has an invalid content")
        try:
            raw["content"] = DetailContent(raw["content"])
        except ValueError as exc:
            raise ValueError(f"broad fact {index} has an unsupported content") from exc
        if raw["repeat_of"] is not None and not isinstance(raw["repeat_of"], str):
            raise TypeError(f"broad fact {index} has an invalid repeat_of")
        if not isinstance(raw["needs_close_check"], bool):
            raise TypeError(f"broad fact {index} has an invalid needs_close_check")
        payload.append(raw)
    reject_truth_keys(payload, "detail prompt broad_facts")
    return payload


def build_detail_prompt(
    case: DetailCase,
    arm: DetailArm | str,
    *,
    deterministic_facts: Mapping[str, Any] | None = None,
    broad_facts: Sequence[DetailBroadFact | Mapping[str, Any]] | None = None,
    prompt_mode: DetailPromptMode | str = DetailPromptMode.DEFAULT,
) -> str:
    """Build one narrow detail prompt while enforcing arm fact isolation."""
    arm = DetailArm(arm)
    prompt_mode = normalise_prompt_mode(prompt_mode)
    if deterministic_facts is None:
        deterministic_facts = case.deterministic_facts
    if broad_facts is None:
        broad_facts = case.broad_facts

    if arm is DetailArm.SHORT_ONLY:
        if deterministic_facts is not None or broad_facts is not None:
            raise ValueError("short_only cannot receive deterministic or broad facts")
        if prompt_mode is DetailPromptMode.CONSERVATIVE_REPLAY_VETO:
            return _CONSERVATIVE_REPLAY_VETO_PROMPT
        return _DETAIL_BASE_PROMPT
    if prompt_mode is not DetailPromptMode.DEFAULT:
        raise ValueError("conservative_replay_veto is valid only for short_only")
    if deterministic_facts is None:
        raise ValueError(f"{arm.value} requires deterministic facts")
    if broad_facts is not None and arm is DetailArm.DETERMINISTIC:
        raise ValueError("deterministic cannot receive broad facts")

    prompt = (
        f"{_DETAIL_BASE_PROMPT}\n\nFallible automatic context follows. It is a hint, not a label. "
        "Resolve any conflict using the pixels.\n"
        f"deterministic_facts={_json_facts(deterministic_facts, 'detail prompt deterministic_facts')}"
    )
    if arm is DetailArm.BROAD_FACTS:
        if broad_facts is None:
            return prompt + (
                "\nNo usable broad-pass fact record is available. Judge the clip independently."
            )
        parsed_facts = _broad_fact_payload(broad_facts)
        prompt += (
            "\nA previous broad pass supplied these parsed, fallible facts. They may be wrong; "
            "resolve any conflict using the pixels.\n"
            f"broad_facts={json.dumps(parsed_facts, sort_keys=True, separators=(',', ':'))}"
        )
    return prompt
