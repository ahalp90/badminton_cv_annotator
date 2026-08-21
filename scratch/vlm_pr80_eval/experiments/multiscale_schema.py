"""Versioned truth-blind contracts for multiscale VLM trials."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "vlm-multiscale-manifest-v1"
_FORBIDDEN_INFERENCE_KEYS = {
    "truth",
    "ground_truth",
    "gt_frame",
    "gt_rally",
    "distance_to_gt",
    "expected_answer",
    "target_label",
    "stratum",
    "nearest_gt_frame",
    "usable_at_5",
    "usable_at_10",
    "usable_at_15",
    "expected_actor",
    "set_id",
    "rally",
    "ball_round",
    "event_role",
    "valid_rally",
    "dominant_scene_truth",
    "scene_fractions",
    "boundary_class",
    "mapped_set_id",
    "mapped_rally",
    "tracker_real",
    "truth_source",
    "sample_id",
    "human_label",
}


class BroadContent(StrEnum):
    """Coarse broadcast content visible inside one detected segment."""

    LIVE = "live"
    REPLAY = "replay"
    CUTAWAY = "cutaway"
    OTHER = "other"
    UNCLEAR = "unclear"


class TargetRoute(StrEnum):
    """Whether broad evidence permits a target to bypass close inspection."""

    ROUTINE_LIVE = "routine_live"
    CLOSE_CHECK = "close_check"


@dataclass(frozen=True)
class Segment:
    """One source-global interval bounded by fallible detected cuts."""

    segment_id: str
    source_start_frame: int
    source_end_frame: int

    def overlaps(self, start_frame: int, end_frame: int) -> bool:
        """Return whether this half-open segment overlaps a half-open interval."""
        return self.source_start_frame < end_frame and start_frame < self.source_end_frame


@dataclass(frozen=True)
class MultiscaleCase:
    """One broad context view, with no human truth attached."""

    case_id: str
    pair_id: str
    video_id: str
    context_seconds: int
    clip_path: Path
    source_start_frame: int
    source_end_frame: int
    target_start_frame: int
    target_end_frame: int
    sample_fps: float
    expected_frames: int
    width: int
    height: int
    source_frames: tuple[int, ...]
    segments: tuple[Segment, ...]
    pipeline_priors: dict[str, bool | int | float | str | None]

    @property
    def target_segment_ids(self) -> tuple[str, ...]:
        """Return source-global segment IDs that overlap `TARGET`."""
        return tuple(
            segment.segment_id
            for segment in self.segments
            if segment.overlaps(self.target_start_frame, self.target_end_frame)
        )


@dataclass(frozen=True)
class BroadSegmentReply:
    """One strictly parsed broad-pass answer."""

    segment_id: str
    content: BroadContent
    repeat_of: str | None
    needs_close_check: bool


def reject_truth_keys(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"{location} contains forbidden truth key {key!r}")
            reject_truth_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_truth_keys(child, f"{location}[{index}]")


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    found = set(value)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")


def _load_segment(raw: Any, case_id: str, index: int) -> Segment:
    if not isinstance(raw, dict):
        raise TypeError(f"{case_id}: segment {index} must be an object")
    _exact_keys(
        raw,
        {"segment_id", "source_start_frame", "source_end_frame"},
        f"{case_id}: segment {index}",
    )
    segment = Segment(
        segment_id=str(raw["segment_id"]),
        source_start_frame=int(raw["source_start_frame"]),
        source_end_frame=int(raw["source_end_frame"]),
    )
    if not segment.segment_id:
        raise ValueError(f"{case_id}: segment {index} has an empty ID")
    if segment.source_start_frame >= segment.source_end_frame:
        raise ValueError(f"{case_id}: segment {segment.segment_id} is empty")
    return segment


def _load_case(
    raw: Any,
    *,
    manifest_dir: Path,
    expected_frames: int,
    width: int,
    height: int,
) -> MultiscaleCase:
    if not isinstance(raw, dict):
        raise TypeError("manifest cases must be objects")
    expected_keys = {
        "case_id",
        "pair_id",
        "video_id",
        "context_seconds",
        "clip_path",
        "source_start_frame",
        "source_end_frame",
        "target_start_frame",
        "target_end_frame",
        "sample_fps",
        "source_frames",
        "segments",
        "pipeline_priors",
    }
    _exact_keys(raw, expected_keys, "manifest case")
    case_id = str(raw["case_id"])
    pair_id = str(raw["pair_id"])
    if not case_id or not pair_id:
        raise ValueError("case_id and pair_id must be non-empty")
    source_start = int(raw["source_start_frame"])
    source_end = int(raw["source_end_frame"])
    target_start = int(raw["target_start_frame"])
    target_end = int(raw["target_end_frame"])
    if not source_start <= target_start < target_end <= source_end:
        raise ValueError(f"{case_id}: TARGET must sit inside the source window")
    source_frames = tuple(int(frame) for frame in raw["source_frames"])
    if len(source_frames) != expected_frames:
        raise ValueError(
            f"{case_id}: source frame map has {len(source_frames)} frames, expected {expected_frames}"
        )
    if source_frames != tuple(sorted(set(source_frames))):
        raise ValueError(f"{case_id}: source frames must be unique and increasing")
    if source_frames[0] < source_start or source_frames[-1] >= source_end:
        raise ValueError(f"{case_id}: source frame map leaves its source window")
    segments = tuple(
        _load_segment(segment, case_id, index)
        for index, segment in enumerate(raw["segments"])
    )
    if not segments:
        raise ValueError(f"{case_id}: at least one segment is required")
    segment_ids = [segment.segment_id for segment in segments]
    if len(set(segment_ids)) != len(segment_ids):
        raise ValueError(f"{case_id}: segment IDs must be unique")
    if segments != tuple(sorted(segments, key=lambda segment: segment.source_start_frame)):
        raise ValueError(f"{case_id}: segments must be in source order")
    previous_end = source_start
    for segment in segments:
        if segment.source_start_frame < source_start or segment.source_end_frame > source_end:
            raise ValueError(f"{case_id}: segment {segment.segment_id} leaves its source window")
        if segment.source_start_frame < previous_end:
            raise ValueError(f"{case_id}: segments overlap")
        previous_end = segment.source_end_frame
    priors = raw["pipeline_priors"]
    if not isinstance(priors, dict):
        raise TypeError(f"{case_id}: pipeline_priors must be an object")
    clip_path = Path(raw["clip_path"])
    if not clip_path.is_absolute():
        clip_path = manifest_dir / clip_path
    case = MultiscaleCase(
        case_id=case_id,
        pair_id=pair_id,
        video_id=str(raw["video_id"]),
        context_seconds=int(raw["context_seconds"]),
        clip_path=clip_path,
        source_start_frame=source_start,
        source_end_frame=source_end,
        target_start_frame=target_start,
        target_end_frame=target_end,
        sample_fps=float(raw["sample_fps"]),
        expected_frames=expected_frames,
        width=width,
        height=height,
        source_frames=source_frames,
        segments=segments,
        pipeline_priors=priors,
    )
    if not case.target_segment_ids:
        raise ValueError(f"{case_id}: no supplied segment overlaps TARGET")
    return case


def load_manifest(path: Path, *, require_clips: bool = True) -> tuple[MultiscaleCase, ...]:
    """Load a multiscale manifest and reject leakage or structural drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("multiscale manifest must be an object")
    reject_truth_keys(payload)
    _exact_keys(payload, {"schema", "expected_frames", "width", "height", "cases"}, "manifest")
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported multiscale manifest schema {payload['schema']!r}")
    expected_frames = int(payload["expected_frames"])
    width = int(payload["width"])
    height = int(payload["height"])
    if expected_frames < 2 or width < 1 or height < 1:
        raise ValueError("manifest frame count and geometry must be positive")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("multiscale manifest cases must be a non-empty list")
    cases = tuple(
        _load_case(
            raw,
            manifest_dir=path.parent,
            expected_frames=expected_frames,
            width=width,
            height=height,
        )
        for raw in payload["cases"]
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("multiscale manifest case IDs must be unique")
    if require_clips:
        for case in cases:
            if not case.clip_path.is_file():
                raise FileNotFoundError(f"{case.case_id}: clip is missing: {case.clip_path}")
    return cases


def validate_context_pairs(cases: tuple[MultiscaleCase, ...]) -> None:
    """Require fair 90/120 pairs with the same target identity and frame budget."""
    by_pair: dict[str, list[MultiscaleCase]] = {}
    for case in cases:
        by_pair.setdefault(case.pair_id, []).append(case)
    for pair_id, pair in by_pair.items():
        durations = {case.context_seconds for case in pair}
        if len(pair) != 2 or durations != {90, 120}:
            raise ValueError(f"{pair_id}: expected exactly one 90-second and one 120-second case")
        first, second = pair
        fixed_values = {
            (case.video_id, case.target_start_frame, case.target_end_frame, case.expected_frames, case.width, case.height)
            for case in pair
        }
        if len(fixed_values) != 1:
            raise ValueError(f"{pair_id}: paired cases differ in target identity or input budget")
        if first.target_segment_ids != second.target_segment_ids:
            raise ValueError(f"{pair_id}: TARGET segment IDs differ across durations")


def parse_broad_reply(case: MultiscaleCase, raw_response: str) -> tuple[BroadSegmentReply, ...]:
    """Parse a complete segment reply and reject unknown or ambiguous IDs."""
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("broad reply must be an object")
    _exact_keys(payload, {"segments"}, "broad reply")
    if not isinstance(payload["segments"], list):
        raise TypeError("broad reply segments must be a list")
    known_ids = [segment.segment_id for segment in case.segments]
    known_set = set(known_ids)
    replies: list[BroadSegmentReply] = []
    for index, raw in enumerate(payload["segments"]):
        if not isinstance(raw, dict):
            raise TypeError(f"broad reply segment {index} must be an object")
        _exact_keys(
            raw,
            {"segment_id", "content", "repeat_of", "needs_close_check"},
            f"broad reply segment {index}",
        )
        segment_id = str(raw["segment_id"])
        if segment_id not in known_set:
            raise ValueError(f"broad reply contains unknown segment ID {segment_id!r}")
        if not isinstance(raw["needs_close_check"], bool):
            raise TypeError(f"{segment_id}: needs_close_check must be boolean")
        repeat_of = raw["repeat_of"]
        if repeat_of is not None and not isinstance(repeat_of, str):
            raise TypeError(f"{segment_id}: repeat_of must be a segment ID or null")
        if repeat_of is not None:
            if repeat_of not in known_set:
                raise ValueError(f"{segment_id}: repeat_of names unknown segment {repeat_of!r}")
            if known_ids.index(repeat_of) >= known_ids.index(segment_id):
                raise ValueError(f"{segment_id}: repeat_of must name an earlier segment")
        replies.append(
            BroadSegmentReply(
                segment_id=segment_id,
                content=BroadContent(raw["content"]),
                repeat_of=repeat_of,
                needs_close_check=raw["needs_close_check"],
            )
        )
    reply_ids = [reply.segment_id for reply in replies]
    if len(set(reply_ids)) != len(reply_ids):
        raise ValueError("broad reply contains duplicate segment IDs")
    if set(reply_ids) != known_set:
        missing = sorted(known_set - set(reply_ids))
        raise ValueError(f"broad reply omits segment IDs: {missing}")
    return tuple(replies)


def target_route(
    case: MultiscaleCase,
    replies: tuple[BroadSegmentReply, ...] | None,
) -> TargetRoute:
    """Reduce broad evidence to a target-only route; parse failure requests a check."""
    if replies is None:
        return TargetRoute.CLOSE_CHECK
    by_id = {reply.segment_id: reply for reply in replies}
    for segment_id in case.target_segment_ids:
        reply = by_id[segment_id]
        if reply.content is not BroadContent.LIVE or reply.needs_close_check:
            return TargetRoute.CLOSE_CHECK
    return TargetRoute.ROUTINE_LIVE
