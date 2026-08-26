"""Strict truth-blind contracts for paired multiscale detail checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .multiscale_schema import BroadContent, reject_truth_keys

DETAIL_MANIFEST_SCHEMA = "vlm-multiscale-detail-manifest-v1"


class DetailArm(StrEnum):
    """Prompt information supplied to one paired detail arm."""

    SHORT_ONLY = "short_only"
    DETERMINISTIC = "deterministic"
    BROAD_FACTS = "broad_facts"


# Detail scene values deliberately match the broad-pass vocabulary.
DetailContent = BroadContent


@dataclass(frozen=True)
class DetailBroadFact:
    """One compact, parsed fact copied from a broad-pass reply."""

    segment_id: str
    content: DetailContent
    repeat_of: str | None
    needs_close_check: bool


@dataclass(frozen=True)
class DetailCase:
    """One detail clip and its truth-blind prompt inputs."""

    case_id: str
    pair_id: str
    context_case_id: str
    video_id: str
    clip_path: Path
    clip_sha256: str
    source_start_frame: int
    source_end_frame: int
    source_frames: tuple[int, ...]
    target_start_frame: int
    target_end_frame: int
    boundary_frame: int | None
    source_fps: float
    sample_fps: float
    expected_frames: int
    width: int
    height: int
    target_segment_ids: tuple[str, ...]
    deterministic_facts: dict[str, Any] | None
    broad_facts: tuple[DetailBroadFact, ...] | None


@dataclass(frozen=True)
class DetailManifest:
    """One versioned manifest for one of the three paired arms."""

    arm: DetailArm
    expected_frames: int
    width: int
    height: int
    cases: tuple[DetailCase, ...]


@dataclass(frozen=True)
class DetailReply:
    """One strictly parsed detail response."""

    target_content: DetailContent


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")


def _required_string(raw: Any, location: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{location} must be a non-empty string")
    return raw


def _required_int(raw: Any, location: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{location} must be an integer")
    return raw


def _required_float(raw: Any, location: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise TypeError(f"{location} must be a number")
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{location} must be finite and positive")
    return value


def _load_broad_fact(raw: Any, case_id: str, index: int) -> DetailBroadFact:
    location = f"{case_id}: broad fact {index}"
    if not isinstance(raw, dict):
        raise TypeError(f"{location} must be an object")
    _exact_keys(
        raw,
        {"segment_id", "content", "repeat_of", "needs_close_check"},
        location,
    )
    segment_id = _required_string(raw["segment_id"], f"{location}.segment_id")
    content = raw["content"]
    if not isinstance(content, str):
        raise TypeError(f"{location}.content must be a string")
    try:
        parsed_content = DetailContent(content)
    except ValueError as exc:
        raise ValueError(f"{location}.content has an unsupported value") from exc
    repeat_of = raw["repeat_of"]
    if repeat_of is not None:
        repeat_of = _required_string(repeat_of, f"{location}.repeat_of")
    needs_close_check = raw["needs_close_check"]
    if not isinstance(needs_close_check, bool):
        raise TypeError(f"{location}.needs_close_check must be a boolean")
    return DetailBroadFact(segment_id, parsed_content, repeat_of, needs_close_check)


def _load_case(
    raw: Any,
    *,
    manifest_dir: Path,
    arm: DetailArm,
    expected_frames: int,
    width: int,
    height: int,
    verify_clip_hash: bool,
) -> DetailCase:
    if not isinstance(raw, dict):
        raise TypeError("detail manifest cases must be objects")
    expected_keys = {
        "case_id",
        "pair_id",
        "context_case_id",
        "video_id",
        "clip_path",
        "clip_sha256",
        "source_start_frame",
        "source_end_frame",
        "source_frames",
        "target_start_frame",
        "target_end_frame",
        "boundary_frame",
        "source_fps",
        "sample_fps",
        "target_segment_ids",
        "deterministic_facts",
        "broad_facts",
    }
    _exact_keys(raw, expected_keys, "detail manifest case")

    case_id = _required_string(raw["case_id"], "case_id")
    pair_id = _required_string(raw["pair_id"], f"{case_id}.pair_id")
    context_case_id = _required_string(
        raw["context_case_id"], f"{case_id}.context_case_id"
    )
    video_id = _required_string(raw["video_id"], f"{case_id}.video_id")
    clip_sha256 = _required_string(raw["clip_sha256"], f"{case_id}.clip_sha256")
    if len(clip_sha256) != 64 or any(character not in "0123456789abcdef" for character in clip_sha256.lower()):
        raise ValueError(f"{case_id}.clip_sha256 must be a SHA-256 hex digest")

    source_start = _required_int(raw["source_start_frame"], f"{case_id}.source_start_frame")
    source_end = _required_int(raw["source_end_frame"], f"{case_id}.source_end_frame")
    if source_start >= source_end:
        raise ValueError(f"{case_id}: source window is empty")
    source_frames_raw = raw["source_frames"]
    if not isinstance(source_frames_raw, list):
        raise TypeError(f"{case_id}.source_frames must be a list")
    source_frames = tuple(
        _required_int(frame, f"{case_id}.source_frames[{index}]")
        for index, frame in enumerate(source_frames_raw)
    )
    if len(source_frames) != expected_frames:
        raise ValueError(
            f"{case_id}: source frame map has {len(source_frames)} frames, expected {expected_frames}"
        )
    if source_frames != tuple(range(source_start, source_end)):
        raise ValueError(f"{case_id}: source frames must be consecutive and source-global")

    target_start = _required_int(raw["target_start_frame"], f"{case_id}.target_start_frame")
    target_end = _required_int(raw["target_end_frame"], f"{case_id}.target_end_frame")
    if not source_start <= target_start < target_end <= source_end:
        raise ValueError(f"{case_id}: detail TARGET must sit inside the source window")
    boundary = raw["boundary_frame"]
    if boundary is not None:
        boundary = _required_int(boundary, f"{case_id}.boundary_frame")
        if not source_start <= boundary <= source_end:
            raise ValueError(f"{case_id}: boundary frame leaves the source window")

    source_fps = _required_float(raw["source_fps"], f"{case_id}.source_fps")
    sample_fps = _required_float(raw["sample_fps"], f"{case_id}.sample_fps")
    segment_ids_raw = raw["target_segment_ids"]
    if not isinstance(segment_ids_raw, list) or not segment_ids_raw:
        raise ValueError(f"{case_id}.target_segment_ids must be a non-empty list")
    target_segment_ids = tuple(
        _required_string(segment_id, f"{case_id}.target_segment_ids[{index}]")
        for index, segment_id in enumerate(segment_ids_raw)
    )
    if len(set(target_segment_ids)) != len(target_segment_ids):
        raise ValueError(f"{case_id}: target segment IDs must be unique")

    deterministic_raw = raw["deterministic_facts"]
    if deterministic_raw is not None and not isinstance(deterministic_raw, dict):
        raise TypeError(f"{case_id}.deterministic_facts must be an object or null")
    if arm is DetailArm.SHORT_ONLY and deterministic_raw is not None:
        raise ValueError(f"{case_id}: short_only cannot contain deterministic facts")
    if arm is not DetailArm.SHORT_ONLY and deterministic_raw is None:
        raise ValueError(f"{case_id}: {arm.value} requires deterministic facts")
    if deterministic_raw is not None:
        reject_truth_keys(deterministic_raw, f"{case_id}.deterministic_facts")

    broad_raw = raw["broad_facts"]
    if broad_raw is not None and not isinstance(broad_raw, list):
        raise TypeError(f"{case_id}.broad_facts must be a list or null")
    if arm is not DetailArm.BROAD_FACTS and broad_raw is not None:
        raise ValueError(f"{case_id}: only broad_facts can contain broad facts")
    broad_facts = None
    if broad_raw is not None:
        broad_facts = tuple(
            _load_broad_fact(fact, case_id, index)
            for index, fact in enumerate(broad_raw)
        )
        broad_ids = [fact.segment_id for fact in broad_facts]
        if len(set(broad_ids)) != len(broad_ids):
            raise ValueError(f"{case_id}: broad fact segment IDs must be unique")
        unknown_ids = set(broad_ids) - set(target_segment_ids)
        if unknown_ids:
            raise ValueError(
                f"{case_id}: broad facts contain non-target segment IDs {sorted(unknown_ids)}"
            )
    if arm is DetailArm.BROAD_FACTS and broad_raw is not None:
        reject_truth_keys(broad_raw, f"{case_id}.broad_facts")

    clip_path = Path(_required_string(raw["clip_path"], f"{case_id}.clip_path"))
    if not clip_path.is_absolute():
        clip_path = manifest_dir / clip_path
    if verify_clip_hash and clip_path.is_file():
        observed_hash = hashlib.sha256(clip_path.read_bytes()).hexdigest()
        if observed_hash != clip_sha256:
            raise ValueError(f"{case_id}: clip SHA-256 does not match manifest")

    return DetailCase(
        case_id=case_id,
        pair_id=pair_id,
        context_case_id=context_case_id,
        video_id=video_id,
        clip_path=clip_path,
        clip_sha256=clip_sha256,
        source_start_frame=source_start,
        source_end_frame=source_end,
        source_frames=source_frames,
        target_start_frame=target_start,
        target_end_frame=target_end,
        boundary_frame=boundary,
        source_fps=source_fps,
        sample_fps=sample_fps,
        expected_frames=expected_frames,
        width=width,
        height=height,
        target_segment_ids=target_segment_ids,
        deterministic_facts=deterministic_raw,
        broad_facts=broad_facts,
    )


def load_detail_manifest(
    path: Path,
    *,
    require_clips: bool = True,
    verify_clip_hash: bool = False,
) -> DetailManifest:
    """Load one strict arm manifest and reject truth leakage or drift."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("detail manifest must be an object")
    reject_truth_keys(payload)
    _exact_keys(
        payload,
        {"schema", "arm", "expected_frames", "width", "height", "cases"},
        "detail manifest",
    )
    if payload["schema"] != DETAIL_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported detail manifest schema {payload['schema']!r}")
    try:
        arm = DetailArm(payload["arm"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported detail arm {payload['arm']!r}") from exc
    expected_frames = _required_int(payload["expected_frames"], "expected_frames")
    width = _required_int(payload["width"], "width")
    height = _required_int(payload["height"], "height")
    if expected_frames < 2 or width < 1 or height < 1:
        raise ValueError("detail frame count and geometry must be positive")
    cases_raw = payload["cases"]
    if not isinstance(cases_raw, list) or not cases_raw:
        raise ValueError("detail manifest cases must be a non-empty list")
    cases = tuple(
        _load_case(
            raw,
            manifest_dir=path.parent,
            arm=arm,
            expected_frames=expected_frames,
            width=width,
            height=height,
            verify_clip_hash=verify_clip_hash,
        )
        for raw in cases_raw
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("detail manifest case IDs must be unique")
    if require_clips:
        for case in cases:
            if not case.clip_path.is_file():
                raise FileNotFoundError(f"{case.case_id}: clip is missing: {case.clip_path}")
    return DetailManifest(arm, expected_frames, width, height, cases)


def _case_identity(case: DetailCase) -> tuple[Any, ...]:
    return (
        case.case_id,
        case.pair_id,
        case.context_case_id,
        case.video_id,
        case.clip_sha256,
        case.source_start_frame,
        case.source_end_frame,
        case.source_frames,
        case.target_start_frame,
        case.target_end_frame,
        case.boundary_frame,
        case.source_fps,
        case.sample_fps,
        case.expected_frames,
        case.width,
        case.height,
        case.target_segment_ids,
    )


def validate_detail_arms(manifests: Iterable[DetailManifest]) -> None:
    """Require exactly three arms over identical pixels and case identities."""
    observed = list(manifests)
    if len(observed) != len(DetailArm):
        raise ValueError(f"expected exactly {len(DetailArm)} detail arm manifests")
    by_arm: dict[DetailArm, DetailManifest] = {}
    for manifest in observed:
        if manifest.arm in by_arm:
            raise ValueError(f"duplicate detail arm {manifest.arm.value}")
        by_arm[manifest.arm] = manifest
    if set(by_arm) != set(DetailArm):
        raise ValueError(f"detail arms must be {sorted(arm.value for arm in DetailArm)}")
    baseline = by_arm[DetailArm.SHORT_ONLY]
    baseline_cases = {case.case_id: _case_identity(case) for case in baseline.cases}
    for arm in (DetailArm.DETERMINISTIC, DetailArm.BROAD_FACTS):
        manifest = by_arm[arm]
        if (manifest.expected_frames, manifest.width, manifest.height) != (
            baseline.expected_frames,
            baseline.width,
            baseline.height,
        ):
            raise ValueError(f"{arm.value}: detail geometry differs across arms")
        identities = {case.case_id: _case_identity(case) for case in manifest.cases}
        if identities != baseline_cases:
            raise ValueError(f"{arm.value}: case identity or clip pixels differ")


def load_detail_arms(
    paths: Mapping[DetailArm | str, Path],
    *,
    require_clips: bool = True,
    verify_clip_hash: bool = False,
) -> dict[DetailArm, DetailManifest]:
    """Load and cross-check the three arm manifests from explicit paths."""
    normalised: dict[DetailArm, Path] = {}
    for raw_arm, path in paths.items():
        try:
            arm = raw_arm if isinstance(raw_arm, DetailArm) else DetailArm(raw_arm)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported detail arm path key {raw_arm!r}") from exc
        if arm in normalised:
            raise ValueError(f"duplicate detail arm path {arm.value}")
        normalised[arm] = path
    if set(normalised) != set(DetailArm):
        raise ValueError("paths must contain short_only, deterministic, and broad_facts")
    loaded = {
        arm: load_detail_manifest(
            path,
            require_clips=require_clips,
            verify_clip_hash=verify_clip_hash,
        )
        for arm, path in normalised.items()
    }
    validate_detail_arms(loaded.values())
    return loaded


def parse_detail_reply(raw_response: str) -> DetailReply:
    """Parse a detail answer after narrow, unambiguous format repair."""
    if not isinstance(raw_response, str):
        raise TypeError("detail reply must be text")
    response = raw_response.strip()
    fenced = re.fullmatch(r"```json\s*\n(?P<body>.*?)\n```", response, re.DOTALL)
    if fenced is not None:
        response = fenced.group("body").strip()
    shorthand = re.fullmatch(
        r"\{\s*(live|replay|cutaway|other|unclear)\s*\}",
        response,
    )
    if shorthand is not None:
        return DetailReply(DetailContent(shorthand.group(1)))
    unquoted_field = re.fullmatch(
        r"\{\s*target_content\s*:\s*(live|replay|cutaway|other|unclear)\s*\}",
        response,
    )
    if unquoted_field is not None:
        return DetailReply(DetailContent(unquoted_field.group(1)))
    bare_label = re.fullmatch(r"live|replay|cutaway|other|unclear", response)
    if bare_label is not None:
        return DetailReply(DetailContent(bare_label.group(0)))
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"detail reply is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("detail reply must be an object")
    _exact_keys(payload, {"target_content"}, "detail reply")
    value = payload["target_content"]
    if not isinstance(value, str):
        raise TypeError("target_content must be a string")
    try:
        content = DetailContent(value)
    except ValueError as exc:
        raise ValueError("target_content has an unsupported value") from exc
    return DetailReply(content)


parse_detail_response = parse_detail_reply


__all__ = [
    "DETAIL_MANIFEST_SCHEMA",
    "DetailArm",
    "DetailBroadFact",
    "DetailCase",
    "DetailContent",
    "DetailManifest",
    "DetailReply",
    "load_detail_arms",
    "load_detail_manifest",
    "parse_detail_reply",
    "parse_detail_response",
    "reject_truth_keys",
    "validate_detail_arms",
]
