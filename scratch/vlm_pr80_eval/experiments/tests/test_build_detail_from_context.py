from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from experiments.build_detail_from_context import (
    build_detail_from_context,
    parse_source_video,
)
from experiments.detail_schema import DetailArm, load_detail_arms


def _write_source(path: Path, frame_count: int = 200) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        25.0,
        (64, 48),
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer did not open")
    try:
        for frame_number in range(frame_count):
            frame = np.full((48, 64, 3), frame_number % 255, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _context_case(context_seconds: int, clip_path: str) -> dict:
    source_start = 0
    source_end = 200
    source_frames = [int(frame) for frame in np.linspace(source_start, source_end - 1, 96, dtype=int)]
    return {
        "case_id": f"context-pair--{context_seconds}",
        "pair_id": "context-pair",
        "video_id": "sset_01",
        "context_seconds": context_seconds,
        "clip_path": clip_path,
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "target_start_frame": 90,
        "target_end_frame": 110,
        "sample_fps": 8.0,
        "source_frames": source_frames,
        "segments": [
            {
                "segment_id": "S0007",
                "source_start_frame": 80,
                "source_end_frame": 120,
            }
        ],
        "pipeline_priors": {
            "span_id": 7,
            "raw_contact_count": 4,
            "definitive_mask_fraction": 0.0,
        },
    }


def _write_context(tmp_path: Path) -> tuple[Path, Path, Path]:
    context_root = tmp_path / "context"
    inference = context_root / "inference"
    clips = inference / "clips"
    clips.mkdir(parents=True)
    (clips / "context-90.mp4").write_bytes(b"context 90")
    (clips / "context-120.mp4").write_bytes(b"context 120")
    manifest = {
        "schema": "vlm-multiscale-manifest-v1",
        "expected_frames": 96,
        "width": 512,
        "height": 288,
        "cases": [
            _context_case(90, "clips/context-90.mp4"),
            _context_case(120, "clips/context-120.mp4"),
        ],
    }
    manifest_path = inference / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    attempts = tmp_path / "attempts" / "qwen3-vl"
    attempts.mkdir(parents=True)
    return context_root, manifest_path, attempts


def _write_attempt(
    attempts: Path,
    manifest_path: Path,
    context_case: dict,
    *,
    parsed_response: list[dict] | None,
    parser_error: str | None = None,
) -> Path:
    clip_path = manifest_path.parent / context_case["clip_path"]
    payload = {
        "schema": "vlm-multiscale-attempt-v1",
        "backend": "qwen3-vl",
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "case": {
            "case_id": context_case["case_id"],
            "pair_id": context_case["pair_id"],
            "video_id": context_case["video_id"],
            "context_seconds": context_case["context_seconds"],
            "clip_path": str(clip_path),
            "clip_sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
            "source_start_frame": context_case["source_start_frame"],
            "source_end_frame": context_case["source_end_frame"],
            "target_start_frame": context_case["target_start_frame"],
            "target_end_frame": context_case["target_end_frame"],
            "target_segment_ids": ["S0007"],
        },
        "parsed_response": parsed_response,
        "parser_error": parser_error,
        "generation_error": None,
    }
    path = attempts / f"{context_case['case_id']}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_valid_attempts(tmp_path: Path) -> tuple[Path, Path, Path]:
    context_root, manifest_path, attempts = _write_context(tmp_path)
    payload = {
        "segments": [
            {
                "segment_id": "S0007",
                "content": "live",
                "repeat_of": None,
                "needs_close_check": False,
            }
        ]
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for context_case in manifest["cases"]:
        _write_attempt(
            attempts,
            manifest_path,
            context_case,
            parsed_response=payload["segments"],
        )
    source_path = tmp_path / "source.avi"
    _write_source(source_path)
    return context_root, attempts.parent, source_path


def test_parse_source_video_requires_explicit_mapping() -> None:
    assert parse_source_video("sset_01=/tmp/source.avi") == (
        "sset_01",
        Path("/tmp/source.avi"),
    )
    with pytest.raises(ValueError, match="VIDEO_ID=PATH"):
        parse_source_video("/tmp/source.avi")


def test_builder_centres_one_clip_on_each_target_segment_and_shares_arms(
    tmp_path: Path,
) -> None:
    context_root, attempts, source_path = _write_valid_attempts(tmp_path)
    output = tmp_path / "detail"

    build_detail_from_context(
        context_root,
        attempts,
        output,
        backend="qwen3-vl",
        context_seconds=90,
        source_videos={"sset_01": source_path},
    )

    paths = {
        arm: output / "inference" / arm.value / "manifest.json"
        for arm in DetailArm
    }
    loaded = load_detail_arms(paths, verify_clip_hash=True)
    assert {len(manifest.cases) for manifest in loaded.values()} == {1}
    short_case = loaded[DetailArm.SHORT_ONLY].cases[0]
    assert (short_case.source_start_frame, short_case.source_end_frame) == (40, 160)
    assert (short_case.target_start_frame, short_case.target_end_frame) == (90, 110)
    assert short_case.source_frames == tuple(range(40, 160))
    assert loaded[DetailArm.DETERMINISTIC].cases[0].deterministic_facts == {
        "inspected_segment": {
            "segment_id": "S0007",
            "source_start_frame": 80,
            "source_end_frame": 120,
        },
        "pipeline_priors": {
            "definitive_mask_fraction": 0.0,
            "raw_contact_count": 4,
            "span_id": 7,
        },
        "proposed_span": {
            "source_end_frame": 110,
            "source_start_frame": 90,
        },
    }
    assert loaded[DetailArm.BROAD_FACTS].cases[0].broad_facts is not None
    assert loaded[DetailArm.BROAD_FACTS].cases[0].broad_facts[0].content.value == "live"
    assert loaded[DetailArm.SHORT_ONLY].cases[0].clip_sha256 == (
        loaded[DetailArm.BROAD_FACTS].cases[0].clip_sha256
    )


def test_builder_retains_invalid_broad_reply_as_null_facts(tmp_path: Path) -> None:
    context_root, manifest_path, attempts = _write_context(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for context_case in manifest["cases"]:
        _write_attempt(
            attempts,
            manifest_path,
            context_case,
            parsed_response=None,
            parser_error="reply is not valid JSON",
        )
    source_path = tmp_path / "source.avi"
    _write_source(source_path)

    output = tmp_path / "detail"
    build_detail_from_context(
        context_root,
        attempts.parent,
        output,
        backend="qwen3-vl",
        context_seconds=120,
        source_videos={"sset_01": source_path},
    )

    broad_manifest = output / "inference" / "broad_facts" / "manifest.json"
    loaded = load_detail_arms(
        {
            arm: output / "inference" / arm.value / "manifest.json"
            for arm in DetailArm
        }
    )
    assert loaded[DetailArm.BROAD_FACTS].cases[0].broad_facts is None
    assert json.loads(broad_manifest.read_text(encoding="utf-8"))["cases"][0][
        "broad_facts"
    ] is None


def test_builder_can_make_short_only_cases_without_a_broad_run(tmp_path: Path) -> None:
    context_root, _manifest_path, _attempts = _write_context(tmp_path)
    source_path = tmp_path / "source.avi"
    _write_source(source_path)
    output = tmp_path / "detail"

    build_detail_from_context(
        context_root,
        None,
        output,
        backend=None,
        context_seconds=90,
        source_videos={"sset_01": source_path},
    )

    loaded = load_detail_arms(
        {
            arm: output / "inference" / arm.value / "manifest.json"
            for arm in DetailArm
        }
    )
    assert loaded[DetailArm.BROAD_FACTS].cases[0].broad_facts is None
    provenance = json.loads(
        (output / "scoring/provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["settings"]["backend"] is None
    assert provenance["broad_attempts"] == []


def test_builder_rejects_missing_or_stale_attempt(tmp_path: Path) -> None:
    context_root, attempts, source_path = _write_valid_attempts(tmp_path)
    manifest_path = context_root / "inference" / "manifest.json"
    missing_path = attempts / "qwen3-vl" / "context-pair--90.json"
    missing_path.unlink()
    with pytest.raises(FileNotFoundError, match="context-pair--90"):
        build_detail_from_context(
            context_root,
            attempts,
            tmp_path / "missing",
            backend="qwen3-vl",
            context_seconds=90,
            source_videos={"sset_01": source_path},
        )

    _write_attempt(
        attempts / "qwen3-vl",
        manifest_path,
        json.loads(manifest_path.read_text(encoding="utf-8"))["cases"][0],
        parsed_response=[
            {
                "segment_id": "S0007",
                "content": "live",
                "repeat_of": None,
                "needs_close_check": False,
            }
        ],
    )
    stale = attempts / "qwen3-vl" / "context-pair--90.json"
    payload = json.loads(stale.read_text(encoding="utf-8"))
    payload["manifest_sha256"] = "0" * 64
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest identity"):
        build_detail_from_context(
            context_root,
            attempts,
            tmp_path / "stale",
            backend="qwen3-vl",
            context_seconds=90,
            source_videos={"sset_01": source_path},
        )
