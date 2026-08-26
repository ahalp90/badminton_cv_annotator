"""Build, run, and score a truth-blind replay-pair VLM check.

The experiment gives the model a short target clip and a nearby earlier
automatic rally candidate.  The candidate is selected from the annotator
spans, without looking at scene truth or an earlier model answer.  This keeps
the comparison useful as a test of whether visual continuity can catch replay
that looks live in isolation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any

from .detail_schema import DetailArm, DetailCase, load_detail_manifest
from .multiscale_schema import reject_truth_keys

PAIR_MANIFEST_SCHEMA = "vlm-replay-pair-manifest-v1"
PAIR_PROVENANCE_SCHEMA = "vlm-replay-pair-provenance-v1"
PAIR_ATTEMPT_SCHEMA = "vlm-replay-pair-attempt-v1"
PAIR_SCORE_SCHEMA = "vlm-replay-pair-score-v1"
TRUTH_SCHEMA = "vlm-multiscale-truth-v1"

REFERENCE_FRAMES = 120
TARGET_FRAMES = 120
EXPECTED_FRAMES = REFERENCE_FRAMES + TARGET_FRAMES
WIDTH = 512
HEIGHT = 288
OUTPUT_FPS = 8.0
MAX_NEW_TOKENS = 128
QWEN_MAX_MODEL_LEN = 16_384
MAX_CANDIDATE_GAP_SECONDS = 90.0
MATERIAL_TARGET_BASE_30_FRAMES = 10

GOLD = (30, 190, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (220, 200, 20)


class TargetRelation(StrEnum):
    """Visual relationship between the target and the earlier candidate."""

    REPEATED_ACTION = "repeated_action"
    DIFFERENT_ACTION = "different_action"
    NO_COMPARABLE_ACTION = "no_comparable_action"
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class ReplayPairCase:
    """One rendered reference/target pair and its truth-blind provenance."""

    case_id: str
    detail_case_id: str
    pair_id: str
    context_case_id: str
    video_id: str
    clip_path: Path
    clip_sha256: str
    reference_case_id: str
    reference_clip_sha256: str
    source_video_sha256: str
    source_fps: float
    output_fps: float
    reference_source_frames: tuple[int, ...]
    target_source_frames: tuple[int, ...]
    target_start_frame: int
    target_end_frame: int
    target_segment_ids: tuple[str, ...]
    candidate_span_id: int
    candidate_start_frame: int
    candidate_end_frame: int
    candidate_filtered_contact_count: int
    candidate_track_visible_fraction: float
    candidate_gap_frames: int
    expected_frames: int
    width: int
    height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def parse_source_video(value: str) -> tuple[str, Path]:
    """Parse one explicit ``VIDEO_ID=PATH`` source-video mapping."""
    video_id, separator, raw_path = value.partition("=")
    if not separator or not video_id or not raw_path:
        raise ValueError("--source-video must have the form VIDEO_ID=PATH")
    return video_id, Path(raw_path)


def _source_info(path: Path) -> tuple[float, int]:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"source video is missing: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {path}")
        return float(capture.get(cv2.CAP_PROP_FPS)), round(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )
    finally:
        capture.release()


def _candidate_key(record: Mapping[str, Any]) -> tuple[int, int, float, int]:
    """Return the frozen ordering for eligible earlier spans."""
    return (
        int(record["gap_frames"]),
        -int(record["filtered_contact_count"]),
        -float(record["track_visible_fraction"]),
        int(record["span_id"]),
    )


def _eligible_candidates(
    candidate_spans: Sequence[Mapping[str, Any]],
    *,
    target_start_frame: int,
    source_fps: float,
) -> list[dict[str, Any]]:
    """Filter frozen deterministic span facts for one target start."""
    candidates: list[dict[str, Any]] = []
    max_gap_frames = source_fps * MAX_CANDIDATE_GAP_SECONDS
    for span in candidate_spans:
        span_id = int(span["span_id"])
        start = int(span["start_frame"])
        end = int(span["end_frame"])
        if start < 0 or start >= end:
            raise ValueError(f"span {span_id} must be a non-empty frame interval")
        if end > target_start_frame:
            continue
        gap_frames = target_start_frame - end
        if gap_frames > max_gap_frames:
            continue
        filtered_count = int(span["filtered_contact_count"])
        if filtered_count < 1:
            continue
        track_visible_fraction = float(span["track_visible_fraction"])
        candidates.append(
            {
                "span_id": span_id,
                "start_frame": start,
                "end_frame": end,
                "gap_frames": gap_frames,
                "filtered_contact_count": filtered_count,
                "track_visible_fraction": track_visible_fraction,
            }
        )
    return sorted(candidates, key=_candidate_key)


def select_reference_span(
    candidate_spans: Sequence[Mapping[str, Any]],
    *,
    target_start_frame: int,
    source_fps: float,
) -> dict[str, Any] | None:
    """Choose the nearest eligible earlier automatic span."""
    candidates = _eligible_candidates(
        candidate_spans,
        target_start_frame=target_start_frame,
        source_fps=source_fps,
    )
    return None if not candidates else candidates[0]


def _detail_span_id(case: DetailCase) -> int | None:
    """Read the automatic span ID from frozen deterministic facts."""
    facts = case.deterministic_facts
    if not isinstance(facts, dict):
        return None
    priors = facts.get("pipeline_priors")
    if not isinstance(priors, dict):
        return None
    value = priors.get("span_id")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _deterministic_span_fact(case: DetailCase) -> dict[str, Any]:
    """Extract the automatic span facts allowed in the detail manifest."""
    facts = case.deterministic_facts
    if not isinstance(facts, dict):
        raise TypeError(f"{case.case_id}: deterministic facts are missing")
    reject_truth_keys(facts, f"{case.case_id}.deterministic_facts")
    proposed_span = facts.get("proposed_span")
    priors = facts.get("pipeline_priors")
    if not isinstance(proposed_span, dict) or not isinstance(priors, dict):
        raise TypeError(f"{case.case_id}: deterministic span facts are incomplete")
    start = proposed_span.get("source_start_frame")
    end = proposed_span.get("source_end_frame")
    span_id = priors.get("span_id")
    filtered_count = priors.get("filtered_contact_count")
    visible = priors.get("track_visible_fraction")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (start, end, span_id, filtered_count)
    ):
        raise TypeError(f"{case.case_id}: deterministic span frame/count facts must be integers")
    if not isinstance(visible, (int, float)) or isinstance(visible, bool):
        raise TypeError(f"{case.case_id}: track_visible_fraction must be numeric")
    visible = float(visible)
    if start < 0 or start >= end or span_id < 0 or filtered_count < 0:
        raise ValueError(f"{case.case_id}: deterministic span facts are out of range")
    if not math.isfinite(visible) or not 0 <= visible <= 1:
        raise ValueError(f"{case.case_id}: track_visible_fraction must be in [0, 1]")
    return {
        "span_id": span_id,
        "start_frame": start,
        "end_frame": end,
        "filtered_contact_count": filtered_count,
        "track_visible_fraction": visible,
    }


def _valid_reference_case(
    case: DetailCase,
    span_id: int,
    *,
    span_start_frame: int | None = None,
    span_end_frame: int | None = None,
) -> bool:
    """Check that a detail case is a usable dense reference for one span."""
    if _detail_span_id(case) != span_id:
        return False
    if case.expected_frames != TARGET_FRAMES or (case.width, case.height) != (WIDTH, HEIGHT):
        return False
    if len(case.source_frames) != TARGET_FRAMES:
        return False
    if case.source_frames != tuple(range(case.source_frames[0], case.source_frames[-1] + 1)):
        return False
    if span_start_frame is not None and case.target_start_frame < span_start_frame:
        return False
    return span_end_frame is None or case.target_end_frame <= span_end_frame


def _choose_reference_case(
    cases: Sequence[DetailCase],
    span_id: int,
    *,
    span_start_frame: int | None = None,
    span_end_frame: int | None = None,
) -> DetailCase | None:
    """Choose the longest gold-marked detail case within one candidate span."""
    valid = [
        case
        for case in cases
        if _valid_reference_case(
            case,
            span_id,
            span_start_frame=span_start_frame,
            span_end_frame=span_end_frame,
        )
    ]
    if not valid:
        return None
    return min(
        valid,
        key=lambda case: (
            -(case.target_end_frame - case.target_start_frame),
            -case.target_start_frame,
            case.case_id,
        ),
    )


def _case_payload(case: ReplayPairCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "detail_case_id": case.detail_case_id,
        "pair_id": case.pair_id,
        "context_case_id": case.context_case_id,
        "video_id": case.video_id,
        "clip_path": str(Path("clips") / case.clip_path.name),
        "clip_sha256": case.clip_sha256,
        "reference_case_id": case.reference_case_id,
        "reference_clip_sha256": case.reference_clip_sha256,
        "source_video_sha256": case.source_video_sha256,
        "source_fps": case.source_fps,
        "output_fps": case.output_fps,
        "reference_source_frames": list(case.reference_source_frames),
        "target_source_frames": list(case.target_source_frames),
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "target_segment_ids": list(case.target_segment_ids),
        "candidate_span_id": case.candidate_span_id,
        "candidate_start_frame": case.candidate_start_frame,
        "candidate_end_frame": case.candidate_end_frame,
        "candidate_filtered_contact_count": case.candidate_filtered_contact_count,
        "candidate_track_visible_fraction": case.candidate_track_visible_fraction,
        "candidate_gap_frames": case.candidate_gap_frames,
    }


def _valid_digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{location} must be a SHA-256 hex digest")
    return value


def _load_case(
    raw: Any,
    manifest_dir: Path,
    *,
    verify_clip_hash: bool,
) -> ReplayPairCase:
    if not isinstance(raw, dict):
        raise TypeError("replay-pair manifest cases must be objects")
    reject_truth_keys(raw, "replay-pair manifest case")
    expected_keys = {
        "case_id",
        "detail_case_id",
        "pair_id",
        "context_case_id",
        "video_id",
        "clip_path",
        "clip_sha256",
        "reference_case_id",
        "reference_clip_sha256",
        "source_video_sha256",
        "source_fps",
        "output_fps",
        "reference_source_frames",
        "target_source_frames",
        "target_start_frame",
        "target_end_frame",
        "target_segment_ids",
        "candidate_span_id",
        "candidate_start_frame",
        "candidate_end_frame",
        "candidate_filtered_contact_count",
        "candidate_track_visible_fraction",
        "candidate_gap_frames",
    }
    if set(raw) != expected_keys:
        raise ValueError("replay-pair manifest case keys differ")
    case_id = str(raw["case_id"])
    if not case_id:
        raise ValueError("case_id must be non-empty")
    clip_path = Path(str(raw["clip_path"]))
    if not clip_path.is_absolute():
        clip_path = manifest_dir / clip_path
    clip_sha256 = _valid_digest(raw["clip_sha256"], f"{case_id}.clip_sha256")
    reference_sha256 = _valid_digest(
        raw["reference_clip_sha256"],
        f"{case_id}.reference_clip_sha256",
    )
    source_sha256 = _valid_digest(
        raw["source_video_sha256"],
        f"{case_id}.source_video_sha256",
    )
    if verify_clip_hash and clip_path.is_file() and _sha256(clip_path) != clip_sha256:
        raise ValueError(f"{case_id}: clip hash differs")
    source_fps = float(raw["source_fps"])
    output_fps = float(raw["output_fps"])
    if not math.isfinite(source_fps) or source_fps <= 0 or output_fps != OUTPUT_FPS:
        raise ValueError(f"{case_id}: invalid source/output FPS")

    def frame_map(name: str, expected_length: int) -> tuple[int, ...]:
        values = raw[name]
        if not isinstance(values, list) or len(values) != expected_length:
            raise ValueError(f"{case_id}.{name} must contain {expected_length} frames")
        frames = tuple(int(value) for value in values)
        if frames != tuple(sorted(set(frames))):
            raise ValueError(f"{case_id}.{name} must be sorted and unique")
        if frames != tuple(range(frames[0], frames[0] + expected_length)):
            raise ValueError(f"{case_id}.{name} must be consecutive source frames")
        return frames

    reference_frames = frame_map("reference_source_frames", REFERENCE_FRAMES)
    target_frames = frame_map("target_source_frames", TARGET_FRAMES)
    target_start = int(raw["target_start_frame"])
    target_end = int(raw["target_end_frame"])
    if (
        target_start >= target_end
        or target_frames[0] > target_start
        or target_end > target_frames[-1] + 1
    ):
        raise ValueError(f"{case_id}: target frame map does not cover TARGET")
    candidate_start = int(raw["candidate_start_frame"])
    candidate_end = int(raw["candidate_end_frame"])
    candidate_gap = int(raw["candidate_gap_frames"])
    if candidate_start >= candidate_end or candidate_end > target_start:
        raise ValueError(f"{case_id}: candidate span is not earlier than TARGET")
    if candidate_gap != target_start - candidate_end:
        raise ValueError(f"{case_id}: candidate gap does not reproduce")
    if int(raw["candidate_filtered_contact_count"]) < 1:
        raise ValueError(f"{case_id}: candidate has no filtered contacts")
    visible = float(raw["candidate_track_visible_fraction"])
    if not 0 <= visible <= 1 or not math.isfinite(visible):
        raise ValueError(f"{case_id}: candidate visibility must be in [0, 1]")
    segment_ids_raw = raw["target_segment_ids"]
    if not isinstance(segment_ids_raw, list) or not segment_ids_raw:
        raise ValueError(f"{case_id}: target segment IDs are missing")
    return ReplayPairCase(
        case_id=case_id,
        detail_case_id=str(raw["detail_case_id"]),
        pair_id=str(raw["pair_id"]),
        context_case_id=str(raw["context_case_id"]),
        video_id=str(raw["video_id"]),
        clip_path=clip_path,
        clip_sha256=clip_sha256,
        reference_case_id=str(raw["reference_case_id"]),
        reference_clip_sha256=reference_sha256,
        source_video_sha256=source_sha256,
        source_fps=source_fps,
        output_fps=output_fps,
        reference_source_frames=reference_frames,
        target_source_frames=target_frames,
        target_start_frame=target_start,
        target_end_frame=target_end,
        target_segment_ids=tuple(str(value) for value in segment_ids_raw),
        candidate_span_id=int(raw["candidate_span_id"]),
        candidate_start_frame=candidate_start,
        candidate_end_frame=candidate_end,
        candidate_filtered_contact_count=int(raw["candidate_filtered_contact_count"]),
        candidate_track_visible_fraction=visible,
        candidate_gap_frames=candidate_gap,
        expected_frames=EXPECTED_FRAMES,
        width=WIDTH,
        height=HEIGHT,
    )


def load_replay_pair_manifest(
    path: Path,
    *,
    require_clips: bool = True,
    verify_clip_hash: bool = False,
) -> tuple[ReplayPairCase, ...]:
    """Load and validate a truth-blind replay-pair manifest."""
    payload = _load_json(path)
    reject_truth_keys(payload, "replay-pair manifest")
    expected_keys = {"schema", "expected_frames", "width", "height", "output_fps", "cases"}
    if set(payload) != expected_keys:
        raise ValueError("replay-pair manifest keys differ")
    if payload["schema"] != PAIR_MANIFEST_SCHEMA:
        raise ValueError("unexpected replay-pair manifest schema")
    if (
        payload["expected_frames"],
        payload["width"],
        payload["height"],
        payload["output_fps"],
    ) != (EXPECTED_FRAMES, WIDTH, HEIGHT, OUTPUT_FPS):
        raise ValueError("replay-pair manifest geometry differs")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("replay-pair manifest cases must be a non-empty list")
    cases = tuple(
        _load_case(raw, path.parent, verify_clip_hash=verify_clip_hash)
        for raw in raw_cases
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("replay-pair manifest has duplicate case IDs")
    if require_clips:
        for case in cases:
            if not case.clip_path.is_file():
                raise FileNotFoundError(f"{case.case_id}: clip is missing: {case.clip_path}")
    return cases


def _read_case_ids(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = {value for value in values if value}
    if len(ids) != len([value for value in values if value]):
        raise ValueError(f"{path}: duplicate case IDs")
    if not ids:
        raise ValueError(f"{path}: case ID file is empty")
    return ids


def _render_pair_clip(source_path: Path, case: ReplayPairCase) -> None:
    import cv2

    from .build_trials import _letterbox, _open_writer

    capture = cv2.VideoCapture(str(source_path))
    writer = _open_writer(case.clip_path, case.output_fps)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {source_path}")
        for block_name, source_frames in (
            ("REFERENCE", case.reference_source_frames),
            ("TARGET", case.target_source_frames),
        ):
            previous_frame = None
            for source_frame in source_frames:
                if previous_frame is None or source_frame != previous_frame + 1:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"failed to read source frame {source_frame}")
                composed, _, _, _ = _letterbox(frame, WIDTH, HEIGHT)
                in_target = (
                    block_name == "TARGET"
                    and case.target_start_frame <= source_frame < case.target_end_frame
                )
                label = f"{block_name}  {source_frame / case.source_fps:07.2f}s"
                if in_target:
                    label += "  TARGET"
                    cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
                cv2.rectangle(composed, (0, 0), (WIDTH, 28), BLACK, -1)
                cv2.putText(
                    composed,
                    label,
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    CYAN if in_target else WHITE,
                    1,
                    cv2.LINE_AA,
                )
                writer.write(composed)
                previous_frame = source_frame
    finally:
        writer.release()
        capture.release()
    check = cv2.VideoCapture(str(case.clip_path))
    try:
        if not check.isOpened():
            raise RuntimeError(f"could not reopen pair clip {case.clip_path}")
        observed = (
            round(check.get(cv2.CAP_PROP_FRAME_COUNT)),
            round(check.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(check.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if observed != (EXPECTED_FRAMES, WIDTH, HEIGHT):
            raise ValueError(f"{case.clip_path}: clip geometry is {observed}")
        if abs(check.get(cv2.CAP_PROP_FPS) - OUTPUT_FPS) > 0.01:
            raise ValueError(f"{case.clip_path}: clip FPS differs from {OUTPUT_FPS}")
    finally:
        check.release()


def build_replay_pair_cases(
    detail_manifest: Path,
    output_dir: Path,
    *,
    source_videos: Mapping[str, Path],
    case_ids: set[str] | None = None,
) -> None:
    """Build replay-pair clips from the frozen deterministic detail manifest."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    detail = load_detail_manifest(detail_manifest, require_clips=True, verify_clip_hash=True)
    if detail.arm is not DetailArm.DETERMINISTIC:
        raise ValueError("replay-pair builder requires the deterministic detail manifest")
    if (detail.expected_frames, detail.width, detail.height) != (TARGET_FRAMES, WIDTH, HEIGHT):
        raise ValueError("deterministic detail manifest must contain 120 512x288 frames")
    available_cases = {case.case_id: case for case in detail.cases}
    requested_ids = set(available_cases) if case_ids is None else set(case_ids)
    if not requested_ids:
        raise ValueError("no case IDs selected")
    missing_detail = requested_ids - set(available_cases)
    selected = [
        case
        for case in detail.cases
        if case.case_id in requested_ids
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "inference" / "clips"
    scoring_dir = output_dir / "scoring"
    clips_dir.mkdir(parents=True)
    scoring_dir.mkdir()

    video_info: dict[str, tuple[float, int, str]] = {}
    detail_cases_by_video_span: dict[str, dict[int, list[DetailCase]]] = {}
    span_facts_by_video: dict[str, dict[int, dict[str, Any]]] = {}
    for detail_case in detail.cases:
        fact = _deterministic_span_fact(detail_case)
        span_id = int(fact["span_id"])
        video_facts = span_facts_by_video.setdefault(detail_case.video_id, {})
        previous = video_facts.get(span_id)
        if previous is not None and previous != fact:
            raise ValueError(
                f"{detail_case.video_id} span {span_id}: deterministic facts disagree"
            )
        video_facts[span_id] = fact
        detail_cases_by_video_span.setdefault(detail_case.video_id, {}).setdefault(
            span_id, []
        ).append(detail_case)
    unavailable: dict[str, str] = {
        case_id: "case ID is absent from the deterministic detail manifest"
        for case_id in sorted(missing_detail)
    }
    manifest_cases: list[dict[str, Any]] = []
    selected_records: list[ReplayPairCase] = []
    for detail_case in selected:
        if detail_case.video_id not in source_videos:
            unavailable[detail_case.case_id] = "source video mapping is missing"
            continue
        if detail_case.video_id not in video_info:
            source_path = source_videos[detail_case.video_id]
            source_fps, source_count = _source_info(source_path)
            if abs(source_fps - detail_case.source_fps) > 0.01:
                raise ValueError(f"{detail_case.case_id}: source FPS differs from detail manifest")
            video_info[detail_case.video_id] = (
                source_fps,
                source_count,
                _sha256(source_path),
            )
        source_fps, source_count, source_sha256 = video_info[detail_case.video_id]
        if detail_case.source_frames[-1] >= source_count:
            unavailable[detail_case.case_id] = "detail target leaves source video"
            continue
        span_facts = list(span_facts_by_video[detail_case.video_id].values())
        candidates = _eligible_candidates(
            span_facts,
            target_start_frame=detail_case.target_start_frame,
            source_fps=source_fps,
        )
        candidate = None
        reference_case = None
        for possible in candidates:
            possible_reference = _choose_reference_case(
                detail_cases_by_video_span.get(detail_case.video_id, {}).get(
                    int(possible["span_id"]), []
                ),
                int(possible["span_id"]),
                span_start_frame=int(possible["start_frame"]),
                span_end_frame=int(possible["end_frame"]),
            )
            if possible_reference is not None:
                candidate = possible
                reference_case = possible_reference
                break
        if candidate is None:
            unavailable[detail_case.case_id] = (
                "no eligible earlier automatic span with a valid detail reference"
            )
            continue
        assert reference_case is not None
        case = ReplayPairCase(
            case_id=detail_case.case_id,
            detail_case_id=detail_case.case_id,
            pair_id=detail_case.pair_id,
            context_case_id=detail_case.context_case_id,
            video_id=detail_case.video_id,
            clip_path=clips_dir / f"{detail_case.case_id}.mp4",
            clip_sha256="0" * 64,
            reference_case_id=reference_case.case_id,
            reference_clip_sha256=reference_case.clip_sha256,
            source_video_sha256=source_sha256,
            source_fps=source_fps,
            output_fps=OUTPUT_FPS,
            reference_source_frames=reference_case.source_frames,
            target_source_frames=detail_case.source_frames,
            target_start_frame=detail_case.target_start_frame,
            target_end_frame=detail_case.target_end_frame,
            target_segment_ids=detail_case.target_segment_ids,
            candidate_span_id=int(candidate["span_id"]),
            candidate_start_frame=int(candidate["start_frame"]),
            candidate_end_frame=int(candidate["end_frame"]),
            candidate_filtered_contact_count=int(candidate["filtered_contact_count"]),
            candidate_track_visible_fraction=float(candidate["track_visible_fraction"]),
            candidate_gap_frames=int(candidate["gap_frames"]),
            expected_frames=EXPECTED_FRAMES,
            width=WIDTH,
            height=HEIGHT,
        )
        _render_pair_clip(source_videos[case.video_id], case)
        case = ReplayPairCase(**{**asdict(case), "clip_sha256": _sha256(case.clip_path)})
        selected_records.append(case)
        manifest_cases.append(_case_payload(case))

    if not selected_records:
        provenance = {
            "schema": PAIR_PROVENANCE_SCHEMA,
            "detail_manifest_sha256": _sha256(detail_manifest),
            "requested_case_ids": sorted(requested_ids),
            "available_case_ids": [],
            "unavailable_cases": unavailable,
            "pairing_rule": {
                "same_video": True,
                "candidate_ends_before_target": True,
                "maximum_gap_seconds": MAX_CANDIDATE_GAP_SECONDS,
                "minimum_filtered_contact_count": 1,
                "minimum_candidate_detail_cases": 1,
                "ordering": [
                    "smallest_non_negative_frame_gap",
                    "highest_filtered_contact_count",
                    "highest_track_visible_fraction",
                    "lowest_span_id",
                ],
            },
        }
        _write_new_json(scoring_dir / "provenance.json", provenance)
        raise ValueError("no requested detail cases had an eligible replay candidate")

    manifest = {
        "schema": PAIR_MANIFEST_SCHEMA,
        "expected_frames": EXPECTED_FRAMES,
        "width": WIDTH,
        "height": HEIGHT,
        "output_fps": OUTPUT_FPS,
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "inference" / "manifest.json"
    _write_new_json(manifest_path, manifest)
    load_replay_pair_manifest(manifest_path, require_clips=True, verify_clip_hash=True)
    provenance = {
        "schema": PAIR_PROVENANCE_SCHEMA,
        "detail_manifest_sha256": _sha256(detail_manifest),
        "requested_case_ids": sorted(requested_ids),
        "available_case_ids": [case.case_id for case in selected_records],
        "unavailable_cases": unavailable,
        "pairing_rule": {
            "same_video": True,
            "candidate_ends_before_target": True,
            "maximum_gap_seconds": MAX_CANDIDATE_GAP_SECONDS,
            "minimum_filtered_contact_count": 1,
            "minimum_candidate_detail_cases": 1,
            "ordering": [
                "smallest_non_negative_frame_gap",
                "highest_filtered_contact_count",
                "highest_track_visible_fraction",
                "lowest_span_id",
            ],
        },
        "settings": {
            "reference_frames": REFERENCE_FRAMES,
            "target_frames": TARGET_FRAMES,
            "output_fps": OUTPUT_FPS,
            "width": WIDTH,
            "height": HEIGHT,
        },
        "candidate_gap_frames": [case.candidate_gap_frames for case in selected_records],
        "reference_clips": [
            {
                "case_id": case.case_id,
                "reference_case_id": case.reference_case_id,
                "reference_clip_sha256": case.reference_clip_sha256,
            }
            for case in selected_records
        ],
    }
    _write_new_json(scoring_dir / "provenance.json", provenance)


def build_replay_pair_prompt(case: ReplayPairCase) -> str:
    """Return the fixed comparison prompt."""
    return f"""You are checking whether one marked badminton broadcast action is a repeat of an earlier action.

Frames 0-{REFERENCE_FRAMES - 1} are a separate REFERENCE clip. It is an automatic earlier rally candidate from the same video. Frames {REFERENCE_FRAMES}-{EXPECTED_FRAMES - 1} are a separate TARGET clip. Judge the TARGET block only, and compare it with the REFERENCE block.

The same players, court, camera, or general broadcast layout alone is not evidence of a repeat. Compare body positions, stroke order, shuttle movement, camera movement, and the sequence of action. A slow-motion replay, crop, or different camera angle can still be a replay of the reference action. The automatic reference candidate may itself be imperfect.

Return one bare JSON object with exactly one field, target_relation. Its value must be exactly one of repeated_action, different_action, no_comparable_action, or unclear. Use no_comparable_action when the reference does not show a comparable action. Use unclear when the pixels do not support a safe decision. Do not add prose, Markdown fences, or other fields."""


def parse_relation_reply(raw_response: str) -> TargetRelation:
    """Parse only the required JSON object, with one narrow fence tolerance."""
    if not isinstance(raw_response, str):
        raise TypeError("replay-pair reply must be text")
    response = raw_response.strip()
    try:
        return TargetRelation(response)
    except ValueError:
        pass
    fenced = re.fullmatch(r"```json\s*\n(?P<body>.*?)\n```", response, re.DOTALL)
    if fenced is not None:
        response = fenced.group("body").strip()
    unquoted_field = re.fullmatch(
        r'\{\s*target_relation\s*:\s*'
        r'(repeated_action|different_action|no_comparable_action|unclear)\s*\}',
        response,
    )
    if unquoted_field is not None:
        return TargetRelation(unquoted_field.group(1))
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"replay-pair reply is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or set(payload) != {"target_relation"}:
        raise ValueError("replay-pair reply must contain only target_relation")
    value = payload["target_relation"]
    if not isinstance(value, str):
        raise TypeError("target_relation must be a string")
    try:
        return TargetRelation(value)
    except ValueError as exc:
        raise ValueError(f"unsupported target_relation {value!r}") from exc


def _load_backend(backend_name: str) -> Any:
    from .backends import load_backend

    return load_backend(
        backend_name,
        expected_input_frames=EXPECTED_FRAMES,
        max_model_len=QWEN_MAX_MODEL_LEN if backend_name == "qwen3-vl" else None,
    )


def _attempt_case_payload(case: ReplayPairCase) -> dict[str, Any]:
    return _case_payload(case)


def run_replay_pair_trials(
    manifest_path: Path,
    backend_name: str,
    output_dir: Path,
    *,
    case_ids: set[str] | None = None,
    limit: int | None = None,
) -> None:
    """Run one resident backend over selected replay-pair cases."""
    cases = list(load_replay_pair_manifest(manifest_path, require_clips=True, verify_clip_hash=True))
    if case_ids is not None:
        cases = [case for case in cases if case.case_id in case_ids]
        missing = case_ids - {case.case_id for case in cases}
        if missing:
            raise ValueError(f"unknown requested case IDs: {sorted(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        cases = cases[:limit]
    if not cases:
        raise ValueError("no replay-pair cases selected")
    backend = _load_backend(backend_name)
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    manifest_sha256 = _sha256(manifest_path)
    for case in cases:
        prompt = build_replay_pair_prompt(case)
        base = {
            "schema": PAIR_ATTEMPT_SCHEMA,
            "backend": backend_name,
            "model": model_identity,
            "manifest_sha256": manifest_sha256,
            "case": _attempt_case_payload(case),
            "prompt": prompt,
            "prompt_sha256": _sha256_text(prompt),
        }
        started = perf_counter()
        try:
            evidence = backend.generate(
                case.clip_path,
                prompt,
                requested_fps=case.output_fps,
                width=case.width,
                height=case.height,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        except Exception as exc:
            _write_new_json(
                output_dir / backend_name / f"{case.case_id}.json",
                {
                    **base,
                    "raw_response": None,
                    "parsed_response": None,
                    "parser_error": None,
                    "generation_error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": perf_counter() - started,
                    "sampling": {
                        "expected_input_frames": EXPECTED_FRAMES,
                        "sampled_input_frames": None,
                        "requested_fps": case.output_fps,
                        "requested_width": case.width,
                        "requested_height": case.height,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                },
            )
            raise
        parsed_response = None
        parser_error = None
        try:
            parsed_response = {"target_relation": parse_relation_reply(evidence.raw_response).value}
        except (TypeError, ValueError) as exc:
            parser_error = str(exc)
        payload = {
            **base,
            "raw_response": evidence.raw_response,
            "parsed_response": parsed_response,
            "parser_error": parser_error,
            "generation_error": None,
            "elapsed_seconds": perf_counter() - started,
            "sampling": {
                "expected_input_frames": EXPECTED_FRAMES,
                "sampled_input_frames": evidence.sampled_input_frames,
                "requested_fps": case.output_fps,
                "requested_width": case.width,
                "requested_height": case.height,
                "max_new_tokens": MAX_NEW_TOKENS,
                "width": evidence.width,
                "height": evidence.height,
                "visual_tokens": evidence.visual_tokens,
                "total_input_tokens": evidence.total_input_tokens,
            },
        }
        _write_new_json(output_dir / backend_name / f"{case.case_id}.json", payload)
        print(output_dir / backend_name / f"{case.case_id}.json", flush=True)


def _load_attempt(
    attempts_root: Path,
    backend: str,
    case: ReplayPairCase,
    manifest_sha256: str,
) -> dict[str, Any]:
    path = attempts_root / backend / f"{case.case_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing replay-pair attempt: {path}")
    attempt = _load_json(path)
    expected_keys = {
        "schema",
        "backend",
        "model",
        "manifest_sha256",
        "case",
        "prompt",
        "prompt_sha256",
        "raw_response",
        "parsed_response",
        "parser_error",
        "generation_error",
        "elapsed_seconds",
        "sampling",
    }
    if set(attempt) != expected_keys:
        raise ValueError(f"{path}: attempt keys differ")
    if attempt["schema"] != PAIR_ATTEMPT_SCHEMA or attempt["backend"] != backend:
        raise ValueError(f"{path}: attempt schema/backend differs")
    if attempt["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"{path}: manifest identity differs")
    observed_case = attempt["case"]
    expected_case = _attempt_case_payload(case)
    if not isinstance(observed_case, dict) or set(observed_case) != set(expected_case):
        raise ValueError(f"{path}: case identity differs")
    if any(
        observed_case[field] != expected_case[field]
        for field in expected_case
        if field != "clip_path"
    ):
        raise ValueError(f"{path}: case identity differs")
    prompt = attempt["prompt"]
    if not isinstance(prompt, str) or _sha256_text(prompt) != attempt["prompt_sha256"]:
        raise ValueError(f"{path}: prompt hash does not reproduce")
    if prompt != build_replay_pair_prompt(case):
        raise ValueError(f"{path}: prompt differs from manifest facts")
    sampling = attempt["sampling"]
    if not isinstance(sampling, dict):
        raise TypeError(f"{path}: sampling must be an object")
    if sampling.get("expected_input_frames") != EXPECTED_FRAMES:
        raise ValueError(f"{path}: expected frame count differs")
    generation_error = attempt["generation_error"]
    parser_error = attempt["parser_error"]
    parsed = attempt["parsed_response"]
    if generation_error is not None:
        if any(attempt[field] is not None for field in ("raw_response", "parsed_response", "parser_error")):
            raise ValueError(f"{path}: failed generation contains response data")
        return {**attempt, "_relation": None}
    sampled = sampling.get("sampled_input_frames")
    if sampled is None or tuple(sampled) != tuple(range(EXPECTED_FRAMES)):
        raise ValueError(f"{path}: backend did not consume all 240 frames")
    if sampling.get("width") != WIDTH or sampling.get("height") != HEIGHT:
        raise ValueError(f"{path}: backend input geometry differs")
    if not isinstance(attempt["raw_response"], str):
        raise TypeError(f"{path}: successful generation must contain raw text")
    reparsed = None
    reproduced_error = None
    try:
        reparsed = {"target_relation": parse_relation_reply(attempt["raw_response"]).value}
    except (TypeError, ValueError) as exc:
        reproduced_error = str(exc)
    if reparsed != parsed or reproduced_error != parser_error:
        raise ValueError(f"{path}: stored parser result does not reproduce")
    if (parsed is None) == (parser_error is None):
        raise ValueError(f"{path}: parsed response and parser error are inconsistent")
    return {
        **attempt,
        "_relation": None if parsed is None else TargetRelation(parsed["target_relation"]),
    }


def _truth_route(
    case: ReplayPairCase,
    parent_row: Mapping[str, Any],
    truth_by_pair: Mapping[str, Mapping[str, Any]],
) -> str:
    candidates = [case.pair_id, case.context_case_id]
    if "--" in case.context_case_id:
        candidates.append(case.context_case_id.rsplit("--", 1)[0])
    truth = next((truth_by_pair.get(candidate) for candidate in candidates if truth_by_pair.get(candidate) is not None), None)
    if truth is None:
        raise ValueError(f"{case.case_id}: no truth row matches parent detail case")
    if truth.get("video_id") != case.video_id:
        raise ValueError(f"{case.case_id}: truth video differs")
    intervals = truth.get("truth_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(f"{case.case_id}: truth intervals are missing")
    covered = case.target_start_frame
    all_live = True
    for interval in intervals:
        if not isinstance(interval, dict):
            raise TypeError(f"{case.case_id}: truth interval is not an object")
        start = max(case.target_start_frame, int(interval["source_start_frame"]))
        end = min(case.target_end_frame, int(interval["source_end_frame"]))
        if start < end:
            if start != covered:
                raise ValueError(f"{case.case_id}: truth does not cover target")
            covered = end
            if interval.get("truth") not in {"live", "live-non-standard"}:
                all_live = False
    if covered != case.target_end_frame:
        raise ValueError(f"{case.case_id}: truth does not cover target")
    observed_parent_route = parent_row.get("truth_route")
    route = "routine_live" if all_live else "close_check"
    if observed_parent_route is not None and observed_parent_route != route:
        raise ValueError(f"{case.case_id}: parent score truth route differs")
    return route


def _route_aggregate(rows: Sequence[Mapping[str, Any]], *, excluded_cases: int = 0) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "valid_replies": 0,
            "invalid_replies": 0,
            "route_accuracy": None,
            "routine_live_recall": None,
            "routine_live_precision": None,
            "close_check_recall": None,
            "close_check_precision": None,
            "excluded_cases": excluded_cases,
        }
    truth_live = [row for row in rows if row["truth_route"] == "routine_live"]
    truth_close = [row for row in rows if row["truth_route"] == "close_check"]
    predicted_live = [row for row in rows if row["predicted_route"] == "routine_live"]
    predicted_close = [row for row in rows if row["predicted_route"] == "close_check"]
    return {
        "cases": len(rows),
        "valid_replies": sum(row["relation"] is not None for row in rows),
        "invalid_replies": sum(row["relation"] is None for row in rows),
        "route_accuracy": sum(row["predicted_route"] == row["truth_route"] for row in rows) / len(rows),
        "routine_live_recall": (
            sum(row["predicted_route"] == "routine_live" for row in truth_live) / len(truth_live)
            if truth_live
            else None
        ),
        "routine_live_precision": (
            sum(row["truth_route"] == "routine_live" for row in predicted_live) / len(predicted_live)
            if predicted_live
            else None
        ),
        "close_check_recall": (
            sum(row["predicted_route"] == "close_check" for row in truth_close) / len(truth_close)
            if truth_close
            else None
        ),
        "close_check_precision": (
            sum(row["truth_route"] == "close_check" for row in predicted_close) / len(predicted_close)
            if predicted_close
            else None
        ),
        "excluded_cases": excluded_cases,
    }


def _parent_rows(parent_score: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = parent_score.get("rows")
    if not isinstance(rows, list):
        raise TypeError("parent detail score rows must be a list")
    selected = {
        str(row["case_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("arm") == DetailArm.SHORT_ONLY.value
    }
    if len(selected) != sum(
        isinstance(row, dict) and row.get("arm") == DetailArm.SHORT_ONLY.value
        for row in rows
    ):
        raise ValueError("parent detail score has duplicate short_only case IDs")
    return selected


def _material_target(case: ReplayPairCase) -> bool:
    target_frames = case.target_end_frame - case.target_start_frame
    minimum = math.ceil(case.source_fps * MATERIAL_TARGET_BASE_30_FRAMES / 30)
    return target_frames >= minimum


def score_replay_pair_trials(
    manifest_path: Path,
    attempts_root: Path,
    backend: str,
    parent_score_path: Path,
    truth_path: Path,
) -> dict[str, Any]:
    """Score replay veto rules while retaining the parent short-only metrics."""
    cases = load_replay_pair_manifest(manifest_path, require_clips=True, verify_clip_hash=True)
    parent_score = _load_json(parent_score_path)
    parent_rows = _parent_rows(parent_score)
    truth_payload = _load_json(truth_path)
    if truth_payload.get("schema") != TRUTH_SCHEMA:
        raise ValueError("unexpected truth schema")
    truth_cases = truth_payload.get("cases")
    if not isinstance(truth_cases, list):
        raise TypeError("truth cases must be a list")
    truth_by_pair: dict[str, Mapping[str, Any]] = {}
    for row in truth_cases:
        if not isinstance(row, dict) or not isinstance(row.get("pair_id"), str):
            raise TypeError("truth rows must have pair IDs")
        if row["pair_id"] in truth_by_pair:
            raise ValueError(f"duplicate truth pair ID {row['pair_id']!r}")
        truth_by_pair[row["pair_id"]] = row
    manifest_sha256 = _sha256(manifest_path)
    rows: list[dict[str, Any]] = []
    unavailable: list[str] = []
    for case in cases:
        parent = parent_rows.get(case.detail_case_id)
        if parent is None:
            unavailable.append(case.case_id)
            continue
        if parent.get("video_id") != case.video_id or parent.get("pair_id") != case.pair_id:
            raise ValueError(f"{case.case_id}: parent score identity differs")
        truth_route = _truth_route(case, parent, truth_by_pair)
        attempt = _load_attempt(attempts_root, backend, case, manifest_sha256)
        relation = attempt["_relation"]
        parent_route = parent.get("predicted_route")
        if parent_route not in {"routine_live", "close_check"}:
            raise ValueError(f"{case.case_id}: parent route is invalid")
        relation_value = None if relation is None else relation.value
        route_all = (
            "routine_live"
            if parent_route == "routine_live" and relation is TargetRelation.DIFFERENT_ACTION
            else "close_check"
        )
        route_repeat_only = (
            "close_check"
            if parent_route == "routine_live" and relation is TargetRelation.REPEATED_ACTION
            else parent_route
        )
        rows.append(
            {
                "case_id": case.case_id,
                "video_id": case.video_id,
                "pair_id": case.pair_id,
                "truth_route": truth_route,
                "parent_route": parent_route,
                "relation": relation_value,
                "predicted_route_all_relations": route_all,
                "predicted_route_repeat_only": route_repeat_only,
                "target_frames": case.target_end_frame - case.target_start_frame,
                "meets_material_target": _material_target(case),
                "candidate_gap_frames": case.candidate_gap_frames,
                "candidate_span_id": case.candidate_span_id,
                "candidate_filtered_contact_count": case.candidate_filtered_contact_count,
                "candidate_track_visible_fraction": case.candidate_track_visible_fraction,
            }
        )
    if not rows:
        raise ValueError("no replay-pair cases could be scored")
    rule_rows: dict[str, list[dict[str, Any]]] = {}
    for rule in ("all_relations", "repeat_only"):
        rule_rows[rule] = [
            {
                **row,
                "predicted_route": row[f"predicted_route_{rule}"],
            }
            for row in rows
        ]
    by_rule = {rule: _route_aggregate(rule_rows[rule]) for rule in rule_rows}
    by_rule_material = {
        rule: _route_aggregate(
            [row for row in rule_rows[rule] if row["meets_material_target"]],
            excluded_cases=sum(not row["meets_material_target"] for row in rule_rows[rule]),
        )
        for rule in rule_rows
    }
    parent_global_strict = parent_score.get("by_arm", {}).get(DetailArm.SHORT_ONLY.value)
    parent_global_material = parent_score.get("by_arm_material_target", {}).get(DetailArm.SHORT_ONLY.value)
    if not isinstance(parent_global_strict, dict) or not isinstance(parent_global_material, dict):
        raise TypeError("parent score lacks short_only strict/material metrics")
    parent_subset_rows = [
        {
            **row,
            "predicted_route": row["parent_route"],
            "relation": "parent_valid" if parent_rows[row["case_id"]].get("valid_reply") else None,
        }
        for row in rows
    ]
    parent_subset_strict = _route_aggregate(parent_subset_rows)
    parent_subset_material_rows = [
        row for row in parent_subset_rows if row["meets_material_target"]
    ]
    parent_subset_material = _route_aggregate(
        parent_subset_material_rows,
        excluded_cases=len(parent_subset_rows) - len(parent_subset_material_rows),
    )
    delta_fields = (
        "route_accuracy",
        "routine_live_recall",
        "routine_live_precision",
        "close_check_recall",
        "close_check_precision",
    )
    deltas: dict[str, dict[str, Any]] = {}
    for rule in rule_rows:
        scoped_deltas: dict[str, dict[str, float | None]] = {}
        for scope, parent_aggregate, child_aggregate in (
            ("strict", parent_subset_strict, by_rule[rule]),
            ("material_target", parent_subset_material, by_rule_material[rule]),
        ):
            scoped_deltas[scope] = {
                field: (
                    None
                    if parent_aggregate.get(field) is None
                    or child_aggregate.get(field) is None
                    else float(child_aggregate[field]) - float(parent_aggregate[field])
                )
                for field in delta_fields
            }
        deltas[rule] = {
            **scoped_deltas["strict"],
            **scoped_deltas,
        }
    changed = {
        rule: sorted(
            row["case_id"]
            for row in rule_rows[rule]
            if row["parent_route"] != row["predicted_route"]
        )
        for rule in rule_rows
    }
    relation_counts = Counter(
        row["relation"] if row["relation"] is not None else "invalid"
        for row in rows
    )
    gap_values = [int(row["candidate_gap_frames"]) for row in rows]
    return {
        "schema": PAIR_SCORE_SCHEMA,
        "backend": backend,
        "manifest_sha256": manifest_sha256,
        "parent_score_sha256": _sha256(parent_score_path),
        "truth_sha256": _sha256(truth_path),
        "parent_metrics": {
            "strict": parent_subset_strict,
            "material_target": parent_subset_material,
            "global_parent_score": {
                "strict": parent_global_strict,
                "material_target": parent_global_material,
            },
        },
        "by_rule": by_rule,
        "by_rule_material_target": by_rule_material,
        "delta_vs_parent": deltas,
        "changed_case_ids": changed,
        "relation_counts": dict(sorted(relation_counts.items())),
        "candidate_availability": {
            "manifest_cases": len(cases),
            "scored_cases": len(rows),
            "unavailable_case_ids": sorted(unavailable),
        },
        "candidate_gap_frames": {
            "cases": len(gap_values),
            "minimum": min(gap_values),
            "maximum": max(gap_values),
            "mean": mean(gap_values),
        },
        "rows": rows,
    }


def _parse_source_videos(values: Sequence[str]) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for raw in values:
        video_id, path = parse_source_video(raw)
        if video_id in mappings:
            raise ValueError(f"duplicate source-video mapping for {video_id}")
        mappings[video_id] = path
    return mappings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--detail-manifest", type=Path, required=True)
    build.add_argument("--source-video", action="append", dest="source_videos", required=True)
    build.add_argument("--case-id-file", type=Path)
    build.add_argument("--out", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--case-id", action="append", dest="case_ids")
    run.add_argument("--limit", type=int)

    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--attempts", type=Path, required=True)
    score.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    score.add_argument(
        "--parent-score",
        "--parent-detail-score",
        dest="parent_score",
        type=Path,
        required=True,
    )
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        requested = None if args.case_id_file is None else _read_case_ids(args.case_id_file)
        build_replay_pair_cases(
            args.detail_manifest,
            args.out,
            source_videos=_parse_source_videos(args.source_videos),
            case_ids=requested,
        )
    elif args.command == "run":
        run_replay_pair_trials(
            args.manifest,
            args.backend,
            args.out,
            case_ids=None if args.case_ids is None else set(args.case_ids),
            limit=args.limit,
        )
    else:
        score = score_replay_pair_trials(
            args.manifest,
            args.attempts,
            args.backend,
            args.parent_score,
            args.truth,
        )
        _write_new_json(args.out, score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
