"""Build paired short detail clips from a frozen broad-context result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from .build_trials import _letterbox, _open_writer, _sha256, _write_json, shift_window
from .detail_schema import (
    DETAIL_MANIFEST_SCHEMA,
    DetailArm,
    DetailBroadFact,
    DetailCase,
)
from .multiscale_schema import (
    MultiscaleCase,
    Segment,
    load_manifest,
    parse_broad_reply,
    validate_context_pairs,
)

ATTEMPT_SCHEMA = "vlm-multiscale-attempt-v1"
PROVENANCE_SCHEMA = "vlm-multiscale-detail-provenance-v1"
DETAIL_FRAMES = 120
WIDTH = 512
HEIGHT = 288
GOLD = (30, 190, 240)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CYAN = (220, 200, 20)


def parse_source_video(value: str) -> tuple[str, Path]:
    """Parse one explicit ``VIDEO_ID=PATH`` source-video mapping."""
    video_id, separator, raw_path = value.partition("=")
    if not separator or not video_id or not raw_path:
        raise ValueError("--source-video must have the form VIDEO_ID=PATH")
    return video_id, Path(raw_path)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _attempt_directory(root: Path, backend: str) -> Path:
    """Accept either the runner root or its backend-specific child directory."""
    backend_directory = root / backend
    if backend_directory.is_dir():
        return backend_directory
    return root


def _attempt_path(root: Path, backend: str, case_id: str) -> Path:
    directory = _attempt_directory(root, backend)
    path = directory / f"{case_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing broad attempt for {case_id}: {path}")
    return path


def _check_attempt(
    path: Path,
    case: MultiscaleCase,
    *,
    backend: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Load one attempt and reject stale or internally inconsistent records."""
    attempt = _load_json(path)
    if attempt.get("schema") != ATTEMPT_SCHEMA:
        raise ValueError(f"{path}: unexpected attempt schema")
    if attempt.get("backend") != backend:
        raise ValueError(f"{path}: backend identity differs")
    if attempt.get("manifest_sha256") != manifest_sha256:
        raise ValueError(f"{path}: manifest identity differs")

    attempt_case = attempt.get("case")
    if not isinstance(attempt_case, dict):
        raise TypeError(f"{path}: attempt case record is missing")
    if attempt_case.get("case_id") != case.case_id:
        raise ValueError(f"{path}: case identity differs")
    expected_clip_sha256 = _sha256(case.clip_path)
    if attempt_case.get("clip_sha256") != expected_clip_sha256:
        raise ValueError(f"{path}: clip identity differs")

    identity_fields = {
        "pair_id": case.pair_id,
        "video_id": case.video_id,
        "context_seconds": case.context_seconds,
        "source_start_frame": case.source_start_frame,
        "source_end_frame": case.source_end_frame,
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "target_segment_ids": list(case.target_segment_ids),
    }
    for field, expected in identity_fields.items():
        if attempt_case.get(field) != expected:
            raise ValueError(f"{path}: attempt has wrong {field}")

    parsed = attempt.get("parsed_response")
    parser_error = attempt.get("parser_error")
    generation_error = attempt.get("generation_error")
    if parsed is not None:
        if parser_error is not None or generation_error is not None:
            raise ValueError(f"{path}: parsed response conflicts with an error")
    elif parser_error is not None:
        if generation_error is not None:
            raise ValueError(f"{path}: parser and generation errors are both present")
    elif generation_error is None:
        raise ValueError(f"{path}: parsed response and error state are inconsistent")
    return attempt


def _parsed_broad_facts(
    case: MultiscaleCase,
    attempt: dict[str, Any],
) -> dict[str, DetailBroadFact] | None:
    """Return parsed facts by segment, or ``None`` for an invalid answer."""
    parsed_response = attempt["parsed_response"]
    if parsed_response is None:
        return None
    if not isinstance(parsed_response, list):
        raise TypeError("parsed_response must be a list")
    parsed = parse_broad_reply(
        case,
        json.dumps({"segments": parsed_response}, separators=(",", ":")),
    )
    return {
        reply.segment_id: DetailBroadFact(
            segment_id=reply.segment_id,
            content=reply.content,
            repeat_of=reply.repeat_of,
            needs_close_check=reply.needs_close_check,
        )
        for reply in parsed
    }


def _detail_window(
    segment: Segment,
    case: MultiscaleCase,
    source_frame_count: int,
) -> tuple[int, int, int, int, int | None]:
    """Choose a fixed detail window around the target part of one segment."""
    overlap_start = max(segment.source_start_frame, case.target_start_frame)
    overlap_end = min(segment.source_end_frame, case.target_end_frame)
    if overlap_start >= overlap_end:
        raise ValueError(f"{case.case_id}: segment {segment.segment_id} does not overlap TARGET")
    centre = (overlap_start + overlap_end) // 2
    source_start, source_end = shift_window(centre, DETAIL_FRAMES, source_frame_count)
    target_start = max(source_start, overlap_start)
    target_end = min(source_end, overlap_end)
    if target_start >= target_end:
        raise ValueError(f"{case.case_id}: shifted detail window lost TARGET")

    boundary_frame = None
    if source_start <= segment.source_start_frame <= source_end:
        boundary_frame = segment.source_start_frame
    elif source_start <= segment.source_end_frame <= source_end:
        boundary_frame = segment.source_end_frame
    return source_start, source_end, target_start, target_end, boundary_frame


def _deterministic_facts(case: MultiscaleCase, segment: Segment) -> dict[str, Any]:
    """Keep only the automatic facts allowed in the deterministic arm."""
    return {
        "inspected_segment": {
            "segment_id": segment.segment_id,
            "source_start_frame": segment.source_start_frame,
            "source_end_frame": segment.source_end_frame,
        },
        "proposed_span": {
            "source_start_frame": case.target_start_frame,
            "source_end_frame": case.target_end_frame,
        },
        "pipeline_priors": dict(case.pipeline_priors),
    }


def _render_detail_clip(
    source_path: Path,
    clip_path: Path,
    *,
    source_fps: float,
    source_start_frame: int,
    source_end_frame: int,
    target_start_frame: int,
    target_end_frame: int,
    segment_id: str,
) -> None:
    """Render one consecutive source window and verify its basic geometry."""
    capture = cv2.VideoCapture(str(source_path))
    writer = _open_writer(clip_path, source_fps)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {source_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_start_frame)
        for source_frame in range(source_start_frame, source_end_frame):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed to read source frame {source_frame}")
            composed, _, _, _ = _letterbox(frame, WIDTH, HEIGHT)
            in_target = target_start_frame <= source_frame < target_end_frame
            label = f"{segment_id}  {source_frame / source_fps:07.2f}s"
            if in_target:
                label += "  TARGET"
                cv2.rectangle(composed, (2, 2), (WIDTH - 3, HEIGHT - 3), GOLD, 3)
            cv2.rectangle(composed, (0, 0), (WIDTH, 28), BLACK, -1)
            cv2.putText(
                composed,
                label,
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                CYAN if in_target else WHITE,
                1,
                cv2.LINE_AA,
            )
            writer.write(composed)
    finally:
        writer.release()
        capture.release()

    capture = cv2.VideoCapture(str(clip_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not reopen detail clip {clip_path}")
        observed = (
            round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if observed != (DETAIL_FRAMES, WIDTH, HEIGHT):
            raise ValueError(f"{clip_path}: clip geometry/count is {observed}")
        if abs(capture.get(cv2.CAP_PROP_FPS) - source_fps) > 0.01:
            raise ValueError(f"{clip_path}: clip FPS differs from source FPS {source_fps}")
    finally:
        capture.release()


def _read_source_info(path: Path) -> tuple[float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"source video is missing: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open source video {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or frame_count < DETAIL_FRAMES:
        raise ValueError(f"{path}: invalid source geometry FPS={fps}, frames={frame_count}")
    return fps, frame_count


def _manifest_case(case: DetailCase, arm: DetailArm) -> dict[str, Any]:
    clip_path = Path(case.clip_path)
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "context_case_id": case.context_case_id,
        "video_id": case.video_id,
        "clip_path": str(Path("..") / "clips" / clip_path.name),
        "clip_sha256": case.clip_sha256,
        "source_start_frame": case.source_start_frame,
        "source_end_frame": case.source_end_frame,
        "source_frames": list(case.source_frames),
        "target_start_frame": case.target_start_frame,
        "target_end_frame": case.target_end_frame,
        "boundary_frame": case.boundary_frame,
        "source_fps": case.source_fps,
        "sample_fps": case.sample_fps,
        "target_segment_ids": list(case.target_segment_ids),
        "deterministic_facts": case.deterministic_facts
        if arm is not DetailArm.SHORT_ONLY
        else None,
        "broad_facts": None
        if arm is not DetailArm.BROAD_FACTS or case.broad_facts is None
        else [
            {
                "segment_id": fact.segment_id,
                "content": fact.content.value,
                "repeat_of": fact.repeat_of,
                "needs_close_check": fact.needs_close_check,
            }
            for fact in case.broad_facts
        ],
    }


def build_detail_from_context(
    context_cases: Path,
    context_attempts: Path | None,
    output_dir: Path,
    *,
    backend: str | None,
    context_seconds: int,
    source_videos: dict[str, Path],
) -> None:
    """Build three identical-pixel detail arms from fixed context cases."""
    if context_seconds not in {90, 120}:
        raise ValueError("context_seconds must be 90 or 120")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    manifest_path = context_cases / "inference" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"context manifest is missing: {manifest_path}")
    cases = load_manifest(manifest_path)
    validate_context_pairs(cases)
    selected_cases = sorted(
        (case for case in cases if case.context_seconds == context_seconds),
        key=lambda case: (case.video_id, case.target_start_frame, case.case_id),
    )
    if not selected_cases:
        raise ValueError(f"context manifest has no {context_seconds}-second cases")

    manifest_sha256 = _sha256(manifest_path)
    if (context_attempts is None) != (backend is None):
        raise ValueError("context_attempts and backend must be supplied together")
    attempts: dict[str, dict[str, Any]] = {}
    attempt_paths: dict[str, Path] = {}
    if context_attempts is not None and backend is not None:
        for case in selected_cases:
            attempt_path = _attempt_path(context_attempts, backend, case.case_id)
            attempts[case.case_id] = _check_attempt(
                attempt_path,
                case,
                backend=backend,
                manifest_sha256=manifest_sha256,
            )
            attempt_paths[case.case_id] = attempt_path

    video_info: dict[str, tuple[Path, float, int]] = {}
    for case in selected_cases:
        if case.video_id not in source_videos:
            raise ValueError(f"missing --source-video mapping for {case.video_id}")
        if case.video_id not in video_info:
            source_path = source_videos[case.video_id]
            source_fps, frame_count = _read_source_info(source_path)
            video_info[case.video_id] = (source_path, source_fps, frame_count)
        _, _, frame_count = video_info[case.video_id]
        if case.source_end_frame > frame_count:
            raise ValueError(f"{case.case_id}: context window leaves source video")

    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = output_dir / "inference" / "clips"
    scoring_dir = output_dir / "scoring"
    clips_dir.mkdir(parents=True)
    scoring_dir.mkdir()
    arm_dirs = {arm: output_dir / "inference" / arm.value for arm in DetailArm}
    for arm_dir in arm_dirs.values():
        arm_dir.mkdir(parents=True)

    shared_cases: list[DetailCase] = []
    source_records: dict[str, dict[str, Any]] = {}
    attempt_records: list[dict[str, Any]] = []
    for case in selected_cases:
        attempt = attempts.get(case.case_id)
        parsed_by_segment = (
            None if attempt is None else _parsed_broad_facts(case, attempt)
        )
        source_path, source_fps, frame_count = video_info[case.video_id]
        if case.video_id not in source_records:
            source_records[case.video_id] = {
                "video_id": case.video_id,
                "path": str(source_path.resolve()),
                "sha256": _sha256(source_path),
                "fps": source_fps,
                "frame_count": frame_count,
            }
        if attempt is not None:
            if attempt["parsed_response"] is not None:
                state = "parsed"
            elif attempt["parser_error"] is not None:
                state = "parser_error"
            else:
                state = "generation_error"
            attempt_records.append(
                {
                    "case_id": case.case_id,
                    "path": str(attempt_paths[case.case_id].resolve()),
                    "sha256": _sha256(attempt_paths[case.case_id]),
                    "state": state,
                }
            )
        segments_by_id = {segment.segment_id: segment for segment in case.segments}
        for segment_id in case.target_segment_ids:
            segment = segments_by_id[segment_id]
            start, end, target_start, target_end, boundary = _detail_window(
                segment,
                case,
                frame_count,
            )
            detail_case_id = f"detail-{case.pair_id}-{segment_id}"
            clip_path = clips_dir / f"{detail_case_id}.mp4"
            _render_detail_clip(
                source_path,
                clip_path,
                source_fps=source_fps,
                source_start_frame=start,
                source_end_frame=end,
                target_start_frame=target_start,
                target_end_frame=target_end,
                segment_id=segment_id,
            )
            broad_fact = None
            if parsed_by_segment is not None:
                broad_fact = (parsed_by_segment[segment_id],)
            shared_cases.append(
                DetailCase(
                    case_id=detail_case_id,
                    pair_id=f"{case.pair_id}-{segment_id}",
                    context_case_id=case.case_id,
                    video_id=case.video_id,
                    clip_path=clip_path,
                    clip_sha256=_sha256(clip_path),
                    source_start_frame=start,
                    source_end_frame=end,
                    source_frames=tuple(range(start, end)),
                    target_start_frame=target_start,
                    target_end_frame=target_end,
                    boundary_frame=boundary,
                    source_fps=source_fps,
                    sample_fps=source_fps,
                    expected_frames=DETAIL_FRAMES,
                    width=WIDTH,
                    height=HEIGHT,
                    target_segment_ids=(segment_id,),
                    deterministic_facts=_deterministic_facts(case, segment),
                    broad_facts=broad_fact,
                )
            )

    for arm, arm_dir in arm_dirs.items():
        payload = {
            "schema": DETAIL_MANIFEST_SCHEMA,
            "arm": arm.value,
            "expected_frames": DETAIL_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
            "cases": [_manifest_case(case, arm) for case in shared_cases],
        }
        _write_json(arm_dir / "manifest.json", payload)

    output_manifests = [
        {
            "arm": arm.value,
            "path": str((arm_dirs[arm] / "manifest.json").resolve()),
            "sha256": _sha256(arm_dirs[arm] / "manifest.json"),
        }
        for arm in DetailArm
    ]
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "settings": {
            "backend": backend,
            "context_seconds": context_seconds,
            "detail_frames": DETAIL_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
        },
        "context_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": manifest_sha256,
        },
        "context_cases": [case.case_id for case in selected_cases],
        "broad_attempts": attempt_records,
        "source_videos": [source_records[video_id] for video_id in sorted(source_records)],
        "manifests": output_manifests,
        "clips": [
            {
                "case_id": case.case_id,
                "path": str(case.clip_path.resolve()),
                "sha256": case.clip_sha256,
            }
            for case in shared_cases
        ],
    }
    _write_json(scoring_dir / "provenance.json", provenance)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-cases", type=Path, required=True)
    parser.add_argument("--context-attempts", type=Path)
    parser.add_argument("--context-seconds", type=int, choices=(90, 120), required=True)
    parser.add_argument("--backend", choices=("qwen3-vl", "internvideo3"))
    parser.add_argument("--without-broad-attempts", action="store_true")
    parser.add_argument("--source-video", action="append", dest="source_videos", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    supplied_broad = args.context_attempts is not None or args.backend is not None
    if args.without_broad_attempts and supplied_broad:
        raise ValueError("--without-broad-attempts cannot be combined with broad inputs")
    if not args.without_broad_attempts and (
        args.context_attempts is None or args.backend is None
    ):
        raise ValueError(
            "provide --context-attempts and --backend, or use --without-broad-attempts"
        )
    source_videos: dict[str, Path] = {}
    for raw_mapping in args.source_videos:
        video_id, source_path = parse_source_video(raw_mapping)
        if video_id in source_videos:
            raise ValueError(f"duplicate --source-video mapping for {video_id}")
        source_videos[video_id] = source_path
    build_detail_from_context(
        args.context_cases,
        args.context_attempts,
        args.out,
        backend=args.backend,
        context_seconds=args.context_seconds,
        source_videos=source_videos,
    )


if __name__ == "__main__":
    main()
