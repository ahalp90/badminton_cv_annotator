"""Build, run, and score sparse-context plus dense-close VLM trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from .detail_schema import (
    DetailArm,
    load_detail_manifest,
    parse_detail_reply,
)
from .multiscale_sampling import (
    required_storyboard_frames,
    segment_for_frame,
    storyboard_source_frames,
)
from .multiscale_schema import Segment, load_manifest, reject_truth_keys
from .score_detail_trials import (
    DETAIL_SCORE_SCHEMA,
    TRUTH_SCHEMA,
    _aggregate,
    _case_score,
    _truth_for_case,
)

COMBINED_MANIFEST_SCHEMA = "vlm-multiscale-combined-visual-manifest-v1"
COMBINED_ATTEMPT_SCHEMA = "vlm-multiscale-combined-visual-attempt-v1"
COMBINED_SCORE_SCHEMA = "vlm-multiscale-combined-visual-score-v1"
COMBINED_PROVENANCE_SCHEMA = "vlm-multiscale-combined-visual-provenance-v1"
BROAD_FRAMES = 80
DENSE_FRAMES = 120
EXPECTED_FRAMES = BROAD_FRAMES + DENSE_FRAMES
MAX_NEW_TOKENS = 128
QWEN_MAX_MODEL_LEN = 16_384
STORYBOARD_FPS = 8.0
WIDTH = 512
HEIGHT = 288
GOLD = (30, 190, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CONTEXT_BLUE = (210, 120, 30)
CYAN = (220, 200, 20)


@dataclass(frozen=True)
class CombinedVisualCase:
    """One truth-blind sparse-context and dense-close visual input."""

    case_id: str
    pair_id: str
    context_case_id: str
    video_id: str
    clip_path: Path
    clip_sha256: str
    dense_reference_clip_sha256: str
    source_video_sha256: str
    source_fps: float
    output_fps: float
    broad_source_start_frame: int
    broad_source_end_frame: int
    broad_target_start_frame: int
    broad_target_end_frame: int
    broad_source_frames: tuple[int, ...]
    segments: tuple[Segment, ...]
    dense_source_frames: tuple[int, ...]
    target_start_frame: int
    target_end_frame: int
    target_segment_ids: tuple[str, ...]
    expected_frames: int
    width: int
    height: int


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _exact_keys(payload: Mapping[str, Any], expected: set[str], location: str) -> None:
    if set(payload) != expected:
        raise ValueError(f"{location} keys differ")


def _required_string(raw: Any, location: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{location} must be a non-empty string")
    return raw


def _required_int(raw: Any, location: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{location} must be an integer")
    return raw


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_digest(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _case_payload(case: CombinedVisualCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "context_case_id": case.context_case_id,
        "video_id": case.video_id,
        "clip_path": str(Path("clips") / case.clip_path.name),
        "clip_sha256": case.clip_sha256,
        "dense_reference_clip_sha256": case.dense_reference_clip_sha256,
        "source_video_sha256": case.source_video_sha256,
        "source_fps": case.source_fps,
        "output_fps": case.output_fps,
        "broad_source_start_frame": case.broad_source_start_frame,
        "broad_source_end_frame": case.broad_source_end_frame,
        "broad_target_start_frame": case.broad_target_start_frame,
        "broad_target_end_frame": case.broad_target_end_frame,
        "broad_source_frames": list(case.broad_source_frames),
        "segments": [asdict(segment) for segment in case.segments],
        "dense_source_frames": list(case.dense_source_frames),
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "target_segment_ids": list(case.target_segment_ids),
    }


def _load_case(
    raw: Any, manifest_dir: Path, *, verify_clip_hash: bool
) -> CombinedVisualCase:
    if not isinstance(raw, dict):
        raise TypeError("combined manifest cases must be objects")
    reject_truth_keys(raw, "combined visual manifest case")
    expected_keys = {
        "case_id",
        "pair_id",
        "context_case_id",
        "video_id",
        "clip_path",
        "clip_sha256",
        "dense_reference_clip_sha256",
        "source_video_sha256",
        "source_fps",
        "output_fps",
        "broad_source_start_frame",
        "broad_source_end_frame",
        "broad_target_start_frame",
        "broad_target_end_frame",
        "broad_source_frames",
        "segments",
        "dense_source_frames",
        "target_start_frame",
        "target_end_frame",
        "target_segment_ids",
    }
    _exact_keys(raw, expected_keys, "combined visual manifest case")
    case_id = _required_string(raw["case_id"], "case_id")
    clip_path = Path(_required_string(raw["clip_path"], f"{case_id}.clip_path"))
    if not clip_path.is_absolute():
        clip_path = manifest_dir / clip_path
    digests = {
        "clip_sha256": _required_string(raw["clip_sha256"], f"{case_id}.clip_sha256"),
        "dense_reference_clip_sha256": _required_string(
            raw["dense_reference_clip_sha256"],
            f"{case_id}.dense_reference_clip_sha256",
        ),
        "source_video_sha256": _required_string(
            raw["source_video_sha256"], f"{case_id}.source_video_sha256"
        ),
    }
    if any(not _valid_digest(digest) for digest in digests.values()):
        raise ValueError(f"{case_id}: hashes must be lowercase SHA-256 digests")
    if verify_clip_hash:
        if not clip_path.is_file():
            raise FileNotFoundError(f"missing combined clip: {clip_path}")
        if _sha256(clip_path) != digests["clip_sha256"]:
            raise ValueError(f"{case_id}: combined clip hash differs")

    segments_raw = raw["segments"]
    if not isinstance(segments_raw, list) or not segments_raw:
        raise ValueError(f"{case_id}.segments must be a non-empty list")
    parsed_segments = []
    for index, segment in enumerate(segments_raw):
        if not isinstance(segment, dict):
            raise TypeError(f"{case_id}.segments[{index}] must be an object")
        _exact_keys(
            segment,
            {"segment_id", "source_start_frame", "source_end_frame"},
            f"{case_id}.segments[{index}]",
        )
        parsed_segments.append(
            Segment(
                _required_string(
                    segment["segment_id"],
                    f"{case_id}.segments[{index}].segment_id",
                ),
                _required_int(
                    segment["source_start_frame"],
                    f"{case_id}.segments[{index}].start",
                ),
                _required_int(
                    segment["source_end_frame"],
                    f"{case_id}.segments[{index}].end",
                ),
            )
        )
    segments = tuple(parsed_segments)
    broad_start = _required_int(
        raw["broad_source_start_frame"], f"{case_id}.broad_start"
    )
    broad_end = _required_int(raw["broad_source_end_frame"], f"{case_id}.broad_end")
    broad_target_start = _required_int(
        raw["broad_target_start_frame"], f"{case_id}.broad_target_start"
    )
    broad_target_end = _required_int(
        raw["broad_target_end_frame"], f"{case_id}.broad_target_end"
    )
    broad_frames_raw = raw["broad_source_frames"]
    if not isinstance(broad_frames_raw, list):
        raise TypeError(f"{case_id}.broad_source_frames must be a list")
    broad_frames = tuple(
        _required_int(frame, f"{case_id}.broad_source_frames[{index}]")
        for index, frame in enumerate(broad_frames_raw)
    )
    if len(broad_frames) != BROAD_FRAMES or broad_frames != tuple(
        sorted(set(broad_frames))
    ):
        raise ValueError(
            f"{case_id}: broad source map must contain 80 sorted unique frames"
        )
    required = required_storyboard_frames(
        segments,
        broad_start,
        broad_end,
        broad_target_start,
        broad_target_end,
    )
    if not required <= set(broad_frames):
        raise ValueError(f"{case_id}: broad source map dropped a required frame")

    dense_frames_raw = raw["dense_source_frames"]
    if not isinstance(dense_frames_raw, list):
        raise TypeError(f"{case_id}.dense_source_frames must be a list")
    dense_frames = tuple(
        _required_int(frame, f"{case_id}.dense_source_frames[{index}]")
        for index, frame in enumerate(dense_frames_raw)
    )
    if len(dense_frames) != DENSE_FRAMES or dense_frames != tuple(
        range(dense_frames[0], dense_frames[0] + DENSE_FRAMES)
    ):
        raise ValueError(
            f"{case_id}: dense source map must contain 120 consecutive frames"
        )
    target_start = _required_int(raw["target_start_frame"], f"{case_id}.target_start")
    target_end = _required_int(raw["target_end_frame"], f"{case_id}.target_end")
    if not dense_frames[0] <= target_start < target_end <= dense_frames[-1] + 1:
        raise ValueError(f"{case_id}: close TARGET leaves the dense source map")
    target_ids_raw = raw["target_segment_ids"]
    if not isinstance(target_ids_raw, list) or len(target_ids_raw) != 1:
        raise ValueError(f"{case_id}: exactly one close target segment is required")
    if target_ids_raw[0] not in {segment.segment_id for segment in segments}:
        raise ValueError(
            f"{case_id}: close target segment is absent from broad context"
        )

    source_fps = float(raw["source_fps"])
    output_fps = float(raw["output_fps"])
    if source_fps <= 0 or output_fps != STORYBOARD_FPS:
        raise ValueError(f"{case_id}: source/output FPS is invalid")
    return CombinedVisualCase(
        case_id=case_id,
        pair_id=_required_string(raw["pair_id"], f"{case_id}.pair_id"),
        context_case_id=_required_string(
            raw["context_case_id"], f"{case_id}.context_case_id"
        ),
        video_id=_required_string(raw["video_id"], f"{case_id}.video_id"),
        clip_path=clip_path,
        clip_sha256=digests["clip_sha256"],
        dense_reference_clip_sha256=digests["dense_reference_clip_sha256"],
        source_video_sha256=digests["source_video_sha256"],
        source_fps=source_fps,
        output_fps=output_fps,
        broad_source_start_frame=broad_start,
        broad_source_end_frame=broad_end,
        broad_target_start_frame=broad_target_start,
        broad_target_end_frame=broad_target_end,
        broad_source_frames=broad_frames,
        segments=segments,
        dense_source_frames=dense_frames,
        target_start_frame=target_start,
        target_end_frame=target_end,
        target_segment_ids=tuple(str(value) for value in target_ids_raw),
        expected_frames=EXPECTED_FRAMES,
        width=WIDTH,
        height=HEIGHT,
    )


def load_combined_manifest(
    path: Path, *, verify_clip_hash: bool = True
) -> tuple[CombinedVisualCase, ...]:
    """Load and validate one truth-blind combined visual manifest."""
    payload = _load_json(path)
    reject_truth_keys(payload, "combined visual manifest")
    _exact_keys(
        payload,
        {
            "schema",
            "expected_frames",
            "broad_frames",
            "dense_frames",
            "width",
            "height",
            "output_fps",
            "cases",
        },
        "combined visual manifest",
    )
    if payload["schema"] != COMBINED_MANIFEST_SCHEMA:
        raise ValueError("unexpected combined visual manifest schema")
    expected = (
        payload["expected_frames"],
        payload["broad_frames"],
        payload["dense_frames"],
        payload["width"],
        payload["height"],
        payload["output_fps"],
    )
    if expected != (
        EXPECTED_FRAMES,
        BROAD_FRAMES,
        DENSE_FRAMES,
        WIDTH,
        HEIGHT,
        STORYBOARD_FPS,
    ):
        raise ValueError("combined visual manifest geometry differs")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("combined visual manifest cases must be a non-empty list")
    cases = tuple(
        _load_case(raw, path.parent, verify_clip_hash=verify_clip_hash)
        for raw in raw_cases
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("combined visual manifest has duplicate case IDs")
    return cases


def parse_source_video(value: str) -> tuple[str, Path]:
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


def _annotated_frame(
    frame: Any,
    label: str,
    *,
    border_colour: tuple[int, int, int] | None,
) -> Any:
    import cv2

    from .build_trials import _letterbox

    composed, _, _, _ = _letterbox(frame, WIDTH, HEIGHT)
    if border_colour is not None:
        cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), border_colour, 3)
    cv2.rectangle(composed, (0, 0), (WIDTH, 28), BLACK, -1)
    cv2.putText(
        composed,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        CYAN if border_colour is not None else WHITE,
        1,
        cv2.LINE_AA,
    )
    return composed


def _render_combined_clip(source_path: Path, case: CombinedVisualCase) -> None:
    import cv2

    from .build_trials import _open_writer

    capture = cv2.VideoCapture(str(source_path))
    writer = _open_writer(case.clip_path, case.output_fps)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {source_path}")
        for source_frame in case.broad_source_frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read broad source frame {source_frame}")
            segment = segment_for_frame(case.segments, source_frame)
            in_span = (
                case.broad_target_start_frame
                <= source_frame
                < case.broad_target_end_frame
            )
            label = f"90s CONTEXT  {segment.segment_id}  {source_frame / case.source_fps:07.2f}s"
            if in_span:
                label += "  PROPOSED SPAN"
            writer.write(
                _annotated_frame(
                    frame,
                    label,
                    border_colour=CONTEXT_BLUE if in_span else None,
                )
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, case.dense_source_frames[0])
        for source_frame in case.dense_source_frames:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read dense source frame {source_frame}")
            in_target = case.target_start_frame <= source_frame < case.target_end_frame
            label = f"CLOSE VIEW  {case.target_segment_ids[0]}  {source_frame / case.source_fps:07.2f}s"
            if in_target:
                label += "  TARGET"
            writer.write(
                _annotated_frame(
                    frame, label, border_colour=GOLD if in_target else None
                )
            )
    finally:
        writer.release()
        capture.release()
    capture = cv2.VideoCapture(str(case.clip_path))
    try:
        observed = (
            round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if not capture.isOpened() or observed != (EXPECTED_FRAMES, WIDTH, HEIGHT):
            raise ValueError(f"{case.clip_path}: combined clip geometry is {observed}")
        if abs(capture.get(cv2.CAP_PROP_FPS) - STORYBOARD_FPS) > 0.01:
            raise ValueError(f"{case.clip_path}: combined clip FPS differs")
    finally:
        capture.release()


def build_combined_cases(
    context_manifest: Path,
    detail_manifest: Path,
    source_videos: Mapping[str, Path],
    output_dir: Path,
) -> None:
    """Build combined inputs from frozen 90-second and short-only cases."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    context_cases = load_manifest(context_manifest)
    context_by_id = {
        case.case_id: case for case in context_cases if case.context_seconds == 90
    }
    detail = load_detail_manifest(
        detail_manifest, require_clips=True, verify_clip_hash=True
    )
    if detail.arm is not DetailArm.SHORT_ONLY or detail.expected_frames != DENSE_FRAMES:
        raise ValueError(
            "combined trial requires the frozen 120-frame short_only manifest"
        )
    required_videos = {case.video_id for case in detail.cases}
    if set(source_videos) != required_videos:
        raise ValueError("source-video mappings must exactly match detail videos")
    source_info = {
        video_id: _source_info(path) for video_id, path in source_videos.items()
    }
    source_hashes = {
        video_id: _sha256(path) for video_id, path in source_videos.items()
    }

    prepared: list[tuple[Any, Any, tuple[int, ...]]] = []
    for detail_case in detail.cases:
        context_case = context_by_id.get(detail_case.context_case_id)
        if context_case is None:
            raise ValueError(f"{detail_case.case_id}: no frozen 90-second context case")
        expected_detail_pair = (
            f"{context_case.pair_id}-{detail_case.target_segment_ids[0]}"
        )
        if (
            context_case.video_id != detail_case.video_id
            or detail_case.pair_id != expected_detail_pair
        ):
            raise ValueError(
                f"{detail_case.case_id}: context and detail identity differ"
            )
        broad_frames = storyboard_source_frames(
            context_case.segments,
            context_case.source_start_frame,
            context_case.source_end_frame,
            context_case.target_start_frame,
            context_case.target_end_frame,
            BROAD_FRAMES,
        )
        if broad_frames is None:
            raise ValueError(
                f"{detail_case.case_id}: required broad frames do not fit 80"
            )
        source_fps, source_count = source_info[detail_case.video_id]
        if abs(source_fps - detail_case.source_fps) > 0.01:
            raise ValueError(
                f"{detail_case.case_id}: source FPS differs from detail case"
            )
        if (
            broad_frames[-1] >= source_count
            or detail_case.source_frames[-1] >= source_count
        ):
            raise ValueError(f"{detail_case.case_id}: source frame leaves the video")
        prepared.append((context_case, detail_case, broad_frames))

    clips_dir = output_dir / "inference/clips"
    clips_dir.mkdir(parents=True)
    manifest_cases = []
    for context_case, detail_case, broad_frames in prepared:
        clip_path = clips_dir / f"{detail_case.case_id}.mp4"
        case = CombinedVisualCase(
            case_id=detail_case.case_id,
            pair_id=detail_case.pair_id,
            context_case_id=detail_case.context_case_id,
            video_id=detail_case.video_id,
            clip_path=clip_path,
            clip_sha256="0" * 64,
            dense_reference_clip_sha256=detail_case.clip_sha256,
            source_video_sha256=source_hashes[detail_case.video_id],
            source_fps=detail_case.source_fps,
            output_fps=STORYBOARD_FPS,
            broad_source_start_frame=context_case.source_start_frame,
            broad_source_end_frame=context_case.source_end_frame,
            broad_target_start_frame=context_case.target_start_frame,
            broad_target_end_frame=context_case.target_end_frame,
            broad_source_frames=broad_frames,
            segments=context_case.segments,
            dense_source_frames=detail_case.source_frames,
            target_start_frame=detail_case.target_start_frame,
            target_end_frame=detail_case.target_end_frame,
            target_segment_ids=detail_case.target_segment_ids,
            expected_frames=EXPECTED_FRAMES,
            width=WIDTH,
            height=HEIGHT,
        )
        _render_combined_clip(source_videos[case.video_id], case)
        case = replace(case, clip_sha256=_sha256(clip_path))
        manifest_cases.append(_case_payload(case))

    manifest = {
        "schema": COMBINED_MANIFEST_SCHEMA,
        "expected_frames": EXPECTED_FRAMES,
        "broad_frames": BROAD_FRAMES,
        "dense_frames": DENSE_FRAMES,
        "width": WIDTH,
        "height": HEIGHT,
        "output_fps": STORYBOARD_FPS,
        "cases": manifest_cases,
    }
    manifest_path = output_dir / "inference/manifest.json"
    _write_new_json(manifest_path, manifest)
    load_combined_manifest(manifest_path)
    provenance = {
        "schema": COMBINED_PROVENANCE_SCHEMA,
        "context_manifest_sha256": _sha256(context_manifest),
        "detail_manifest_sha256": _sha256(detail_manifest),
        "source_video_sha256": source_hashes,
        "manifest_sha256": _sha256(manifest_path),
        "cases": len(manifest_cases),
    }
    _write_new_json(output_dir / "provenance.json", provenance)


def build_combined_prompt(case: CombinedVisualCase) -> str:
    """Describe the two visual scales without supplying a model-written label."""
    return f"""You are checking one marked part of a badminton broadcast.

Frames 0-{BROAD_FRAMES - 1} are a sparse, time-ordered storyboard covering 90 seconds.
They show source time and automatic cut-segment IDs. A blue border marks the
larger proposed span. Frames {BROAD_FRAMES}-{EXPECTED_FRAMES - 1} are 120 consecutive
source frames shown slowly as CLOSE VIEW. A gold border marks the CLOSE VIEW
TARGET. Judge only that gold-marked target. Use the earlier storyboard to work
out whether the close view is current action, a replay, or footage between
rallies. Cut boundaries and the proposed span are fallible.

Return one bare JSON object whose only field is `target_content`. Its value must
be exactly one of `live`, `replay`, `cutaway`, `other`, or `unclear`. Use `live`
for the current rally, including serve preparation that leads directly into it
and unusual views of that action. Use `replay` for earlier action shown again.
Use `cutaway` for footage between rallies, including players waiting,
celebrating, entering, or preparing for a later point. Use `other` for graphics,
adverts, blank frames, or unrelated footage. Use `unclear` when the pixels do
not support a safe answer. Do not add fields, prose, or Markdown fences."""


def _load_backend(backend_name: str) -> Any:
    from .backends import load_backend

    return load_backend(
        backend_name,
        expected_input_frames=EXPECTED_FRAMES,
        max_model_len=QWEN_MAX_MODEL_LEN if backend_name == "qwen3-vl" else None,
    )


def run_combined_trials(
    manifest_path: Path,
    backend_name: str,
    output_dir: Path,
    *,
    limit: int | None = None,
) -> None:
    """Run one resident backend across every combined visual case."""
    cases = list(load_combined_manifest(manifest_path))
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("no combined visual cases selected")
    backend = _load_backend(backend_name)
    model_identity = asdict(backend.spec.identity(backend.backend_version))
    manifest_sha256 = _sha256(manifest_path)
    for case in cases:
        prompt = build_combined_prompt(case)
        base = {
            "schema": COMBINED_ATTEMPT_SCHEMA,
            "backend": backend_name,
            "model": model_identity,
            "manifest_sha256": manifest_sha256,
            "case": _case_payload(case),
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
                    "sampling": None,
                },
            )
            raise
        parsed_response = None
        parser_error = None
        try:
            parsed_response = asdict(parse_detail_reply(evidence.raw_response))
        except (TypeError, ValueError) as exc:
            parser_error = str(exc)
        attempt = {
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
                "width": evidence.width,
                "height": evidence.height,
                "visual_tokens": evidence.visual_tokens,
                "total_input_tokens": evidence.total_input_tokens,
                "max_new_tokens": MAX_NEW_TOKENS,
                "qwen_max_model_len": QWEN_MAX_MODEL_LEN
                if backend_name == "qwen3-vl"
                else None,
            },
        }
        _write_new_json(output_dir / backend_name / f"{case.case_id}.json", attempt)


def _load_parent_rows(
    path: Path,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _load_json(path)
    if payload.get("schema") != DETAIL_SCORE_SCHEMA:
        raise ValueError("parent score has an unexpected schema")
    backend = _required_string(payload.get("backend"), "parent score backend")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise TypeError("parent score rows must be a list")
    short_rows = {
        row["case_id"]: row
        for row in rows
        if isinstance(row, dict) and row.get("arm") == DetailArm.SHORT_ONLY.value
    }
    if not short_rows:
        raise ValueError("parent score has no short_only rows")
    return backend, short_rows, payload["by_arm"][DetailArm.SHORT_ONLY.value]


def _load_attempt(
    attempts_root: Path,
    backend_name: str,
    case: CombinedVisualCase,
    manifest_sha256: str,
) -> dict[str, Any]:
    path = attempts_root / backend_name / f"{case.case_id}.json"
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
    _exact_keys(attempt, expected_keys, str(path))
    if (
        attempt["schema"] != COMBINED_ATTEMPT_SCHEMA
        or attempt["backend"] != backend_name
    ):
        raise ValueError(f"{path}: attempt identity differs")
    if attempt["manifest_sha256"] != manifest_sha256:
        raise ValueError(f"{path}: manifest hash differs")
    expected_case = _case_payload(case)
    observed_case = attempt["case"]
    if not isinstance(observed_case, dict) or any(
        observed_case.get(key) != value
        for key, value in expected_case.items()
        if key != "clip_path"
    ):
        raise ValueError(f"{path}: case identity differs")
    prompt = build_combined_prompt(case)
    if attempt["prompt"] != prompt or attempt["prompt_sha256"] != _sha256_text(prompt):
        raise ValueError(f"{path}: prompt identity differs")
    if attempt["generation_error"] is not None:
        return attempt
    raw_response = attempt["raw_response"]
    parsed = None
    parser_error = None
    try:
        parsed = asdict(parse_detail_reply(raw_response))
    except (TypeError, ValueError) as exc:
        parser_error = str(exc)
    if parsed != attempt["parsed_response"] or parser_error != attempt["parser_error"]:
        raise ValueError(f"{path}: parser result does not reproduce")
    sampling = attempt["sampling"]
    if not isinstance(sampling, dict) or sampling.get("sampled_input_frames") != list(
        range(EXPECTED_FRAMES)
    ):
        raise ValueError(f"{path}: backend did not consume the complete 200-frame grid")
    return attempt


def score_combined_trials(
    manifest_path: Path,
    truth_path: Path,
    attempts_root: Path,
    backend_name: str,
    parent_score_path: Path,
) -> dict[str, Any]:
    """Score combined inputs against the same truth and frozen short-only parent."""
    cases = load_combined_manifest(manifest_path)
    truth = _load_json(truth_path)
    if truth.get("schema") != TRUTH_SCHEMA or not isinstance(truth.get("cases"), list):
        raise ValueError("context truth has an unexpected schema")
    truth_by_pair = {row["pair_id"]: row for row in truth["cases"]}
    parent_backend, parent_rows, parent_aggregate = _load_parent_rows(parent_score_path)
    if parent_backend != backend_name:
        raise ValueError("parent and combined backends differ")
    case_ids = {case.case_id for case in cases}
    if set(parent_rows) != case_ids:
        raise ValueError("parent and combined case IDs differ")
    manifest_sha256 = _sha256(manifest_path)
    rows = []
    changed_routes = []
    for case in cases:
        attempt = _load_attempt(attempts_root, backend_name, case, manifest_sha256)
        truth_row = _truth_for_case(case, truth_by_pair)
        row = _case_score(case, truth_row, attempt, DetailArm.SHORT_ONLY)
        row["arm"] = "combined_visual"
        parent = parent_rows[case.case_id]
        for key in (
            "target_start_frame",
            "target_end_frame",
            "target_frames",
            "truth_route",
            "truth_content_frames",
        ):
            if row[key] != parent[key]:
                raise ValueError(f"{case.case_id}: parent truth identity differs")
        if row["predicted_route"] != parent["predicted_route"]:
            changed_routes.append(case.case_id)
        rows.append(row)
    aggregate = _aggregate(rows)
    delta_fields = (
        "mean_case_exact_scene_accuracy",
        "target_frame_exact_scene_accuracy",
        "mean_case_binary_live_nonlive_accuracy",
        "target_frame_binary_live_nonlive_accuracy",
        "close_check_recall",
        "routine_live_recall",
        "routine_live_precision",
    )
    deltas = {
        f"{field}_delta": (
            None
            if aggregate[field] is None or parent_aggregate[field] is None
            else aggregate[field] - parent_aggregate[field]
        )
        for field in delta_fields
    }
    return {
        "schema": COMBINED_SCORE_SCHEMA,
        "backend": backend_name,
        "manifest_sha256": manifest_sha256,
        "truth_sha256": _sha256(truth_path),
        "parent_score_sha256": _sha256(parent_score_path),
        "combined_visual": aggregate,
        "parent_short_only": parent_aggregate,
        "combined_vs_parent": deltas,
        "changed_route_case_ids": sorted(changed_routes),
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--context-manifest", type=Path, required=True)
    build.add_argument("--detail-manifest", type=Path, required=True)
    build.add_argument("--source-video", action="append", required=True)
    build.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--limit", type=int)
    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--attempts", type=Path, required=True)
    score.add_argument("--backend", choices=("qwen3-vl", "internvideo3"), required=True)
    score.add_argument("--parent-score", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        source_videos = dict(parse_source_video(value) for value in args.source_video)
        if len(source_videos) != len(args.source_video):
            raise ValueError("duplicate source-video mapping")
        build_combined_cases(
            args.context_manifest, args.detail_manifest, source_videos, args.out
        )
    elif args.command == "run":
        run_combined_trials(args.manifest, args.backend, args.out, limit=args.limit)
    else:
        result = score_combined_trials(
            args.manifest,
            args.truth,
            args.attempts,
            args.backend,
            args.parent_score,
        )
        _write_new_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
