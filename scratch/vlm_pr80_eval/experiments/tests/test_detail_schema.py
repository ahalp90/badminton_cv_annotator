from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.detail_schema import (
    DETAIL_MANIFEST_SCHEMA,
    DetailArm,
    DetailContent,
    load_detail_arms,
    load_detail_manifest,
    parse_detail_reply,
    validate_detail_arms,
)


def _write_clip(tmp_path: Path) -> str:
    clip = tmp_path / "clips" / "detail.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"same rendered detail clip")
    return hashlib.sha256(clip.read_bytes()).hexdigest()


def _case_payload(tmp_path: Path, arm: str) -> dict:
    clip_sha256 = _write_clip(tmp_path)
    is_short = arm == DetailArm.SHORT_ONLY.value
    is_broad = arm == DetailArm.BROAD_FACTS.value
    return {
        "case_id": "detail-sset_01-r007",
        "pair_id": "context-sset_01-r007",
        "context_case_id": "context-sset_01-r007--90",
        "video_id": "sset_01",
        "clip_path": "../clips/detail.mp4",
        "clip_sha256": clip_sha256,
        "source_start_frame": 100,
        "source_end_frame": 106,
        "source_frames": [100, 101, 102, 103, 104, 105],
        "target_start_frame": 102,
        "target_end_frame": 104,
        "boundary_frame": 103,
        "source_fps": 25.0,
        "sample_fps": 25.0,
        "target_segment_ids": ["S0001"],
        "deterministic_facts": None
        if is_short
        else {"span_membership": True, "cut_ids": ["S0001"]},
        "broad_facts": None
        if not is_broad
        else [
            {
                "segment_id": "S0001",
                "content": "live",
                "repeat_of": None,
                "needs_close_check": False,
            }
        ],
    }


def _write_manifest(tmp_path: Path, arm: str) -> Path:
    manifest_dir = tmp_path / arm
    manifest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": DETAIL_MANIFEST_SCHEMA,
        "arm": arm,
        "expected_frames": 6,
        "width": 512,
        "height": 288,
        "cases": [_case_payload(tmp_path, arm)],
    }
    path = manifest_dir / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_detail_manifest_requires_consecutive_source_global_frames(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, DetailArm.SHORT_ONLY.value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["source_frames"][2] = 200
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="consecutive and source-global"):
        load_detail_manifest(path, require_clips=False)


def test_load_detail_manifest_rejects_nested_truth_keys(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, DetailArm.DETERMINISTIC.value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["deterministic_facts"] = {"nested": [{"HuMaN_LaBeL": "live"}]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden truth key"):
        load_detail_manifest(path, require_clips=False)


def test_load_detail_manifest_keeps_arm_fact_contracts_strict(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, DetailArm.SHORT_ONLY.value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["deterministic_facts"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="short_only cannot contain"):
        load_detail_manifest(path, require_clips=False)


def test_load_detail_manifest_rejects_broad_facts_outside_target_segments(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, DetailArm.BROAD_FACTS.value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["broad_facts"][0]["segment_id"] = "S9999"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-target segment IDs"):
        load_detail_manifest(path, require_clips=False)


@pytest.mark.parametrize(
    ("raw_response", "expected"),
    [
        ('{"target_content":"live"}', DetailContent.LIVE),
        ('  {"target_content":"unclear"}  ', DetailContent.UNCLEAR),
        ('```json\n{"target_content":"replay"}\n```', DetailContent.REPLAY),
        ("{cutaway}", DetailContent.CUTAWAY),
        ("{ other }", DetailContent.OTHER),
        ("{target_content: live}", DetailContent.LIVE),
        ("{ target_content : replay }", DetailContent.REPLAY),
        ("live", DetailContent.LIVE),
    ],
)
def test_parse_detail_reply_accepts_one_exact_content_field(
    raw_response: str,
    expected: DetailContent,
) -> None:
    assert parse_detail_reply(raw_response).target_content is expected


@pytest.mark.parametrize(
    "raw_response",
    [
        "The answer is {cutaway}",
        "```\n{\"target_content\":\"live\"}\n```",
        "```json\n{\"target_content\":\"live\"}\n``` trailing prose",
        "{maybe}",
        "live.",
        "This is live",
        "{target_content: live, extra: replay}",
        "{other_key: live}",
        '{"target_content":"live","evidence":"extra"}',
        '{"target_content":"maybe"}',
        '{"target_content":true}',
    ],
)
def test_parse_detail_reply_rejects_wrapping_extra_or_invalid_values(raw_response: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        parse_detail_reply(raw_response)


def test_validate_detail_arms_accepts_same_case_and_clip_identity(tmp_path: Path) -> None:
    paths = {
        DetailArm.SHORT_ONLY: _write_manifest(tmp_path, DetailArm.SHORT_ONLY.value),
        DetailArm.DETERMINISTIC: _write_manifest(tmp_path, DetailArm.DETERMINISTIC.value),
        DetailArm.BROAD_FACTS: _write_manifest(tmp_path, DetailArm.BROAD_FACTS.value),
    }

    loaded = load_detail_arms(paths)

    assert set(loaded) == set(DetailArm)
    validate_detail_arms(loaded.values())


def test_validate_detail_arms_rejects_different_pixels(tmp_path: Path) -> None:
    paths = {
        DetailArm.SHORT_ONLY: _write_manifest(tmp_path, DetailArm.SHORT_ONLY.value),
        DetailArm.DETERMINISTIC: _write_manifest(tmp_path, DetailArm.DETERMINISTIC.value),
        DetailArm.BROAD_FACTS: _write_manifest(tmp_path, DetailArm.BROAD_FACTS.value),
    }
    path = paths[DetailArm.BROAD_FACTS]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cases"][0]["clip_sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case identity or clip pixels differ"):
        load_detail_arms(paths, require_clips=False)
