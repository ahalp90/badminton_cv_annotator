"""Strict, truth-blind contracts for the VLM cleanup trials."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "vlm-cleanup-trials/0.1"
ATTEMPT_SCHEMA = "vlm-cleanup-attempt/0.1"
EXPECTED_FRAMES = 50
WIDTH = 512
HEIGHT = 288

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
}


class TrialKind(StrEnum):
    EVENT = "event"
    BROADCAST = "broadcast"
    TRACK = "track"


class TrialArm(StrEnum):
    VIDEO_ONLY = "video-only"
    PIPELINE_PRIORS = "pipeline-priors"


@dataclass(frozen=True)
class TrialCase:
    """One truth-blind video question supplied to both prompt arms."""

    case_id: str
    kind: TrialKind
    video_id: str
    clip_path: Path
    source_start_frame: int
    source_end_frame: int
    candidate_frame: int | None
    sample_fps: float
    pipeline_priors: dict[str, bool | int | float | str | None]


def _reject_truth_keys(value: Any, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"{location} contains forbidden inference key {key!r}")
            _reject_truth_keys(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_truth_keys(child, f"{location}[{index}]")


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")


def load_manifest(path: Path, *, require_clips: bool = True) -> tuple[TrialCase, ...]:
    """Load a manifest and reject truth leakage, schema drift, or missing clips."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("trial manifest must be a JSON object")
    _reject_truth_keys(payload)
    _exact_keys(payload, {"schema", "cases"}, "manifest")
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported trial manifest schema {payload['schema']!r}")
    if not isinstance(payload["cases"], list) or not payload["cases"]:
        raise ValueError("trial manifest cases must be a non-empty list")

    expected_case_keys = {
        "case_id",
        "kind",
        "video_id",
        "clip_path",
        "source_start_frame",
        "source_end_frame",
        "candidate_frame",
        "sample_fps",
        "expected_frames",
        "width",
        "height",
        "pipeline_priors",
    }
    cases: list[TrialCase] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["cases"]):
        if not isinstance(raw, dict):
            raise TypeError(f"case {index} must be a JSON object")
        _exact_keys(raw, expected_case_keys, f"case {index}")
        case_id = raw["case_id"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"case {index} has an invalid case_id")
        if case_id in seen:
            raise ValueError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        if raw["expected_frames"] != EXPECTED_FRAMES:
            raise ValueError(f"{case_id}: expected_frames must be {EXPECTED_FRAMES}")
        if (raw["width"], raw["height"]) != (WIDTH, HEIGHT):
            raise ValueError(f"{case_id}: clip geometry must be {WIDTH}x{HEIGHT}")
        clip_path = Path(raw["clip_path"])
        if require_clips and not clip_path.is_file():
            raise FileNotFoundError(f"{case_id}: clip is missing: {clip_path}")
        priors = raw["pipeline_priors"]
        if not isinstance(priors, dict):
            raise TypeError(f"{case_id}: pipeline_priors must be an object")
        cases.append(
            TrialCase(
                case_id=case_id,
                kind=TrialKind(raw["kind"]),
                video_id=str(raw["video_id"]),
                clip_path=clip_path,
                source_start_frame=int(raw["source_start_frame"]),
                source_end_frame=int(raw["source_end_frame"]),
                candidate_frame=None
                if raw["candidate_frame"] is None
                else int(raw["candidate_frame"]),
                sample_fps=float(raw["sample_fps"]),
                pipeline_priors=priors,
            )
        )
    return tuple(cases)


def parse_reply(kind: TrialKind, raw_response: str) -> dict[str, Any]:
    """Parse one bare JSON reply with exact keys and enumerated decisions."""
    try:
        payload = json.loads(raw_response.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply is not bare valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise TypeError("reply must be a JSON object")

    if kind is TrialKind.EVENT:
        expected = {
            "contact_at_marker",
            "evidence_kind",
            "actor",
            "nearby_unmarked_contact",
            "visible_evidence",
            "uncertainty",
        }
        _exact_keys(payload, expected, "event reply")
        if payload["contact_at_marker"] not in {"yes", "no", "unclear"}:
            raise ValueError("contact_at_marker must be yes, no, or unclear")
        if payload["evidence_kind"] not in {
            "visible-contact",
            "inferred-contact",
            "no-contact",
            "unclear",
        }:
            raise ValueError("evidence_kind has an unsupported value")
        if payload["actor"] not in {"top", "bottom", "no-contact", "unclear"}:
            raise ValueError("actor must be top, bottom, no-contact, or unclear")
        if payload["nearby_unmarked_contact"] not in {"yes", "no", "unclear"}:
            raise ValueError("nearby_unmarked_contact must be yes, no, or unclear")
        if payload["contact_at_marker"] == "yes" and payload["actor"] == "no-contact":
            raise ValueError("a confirmed contact cannot have actor no-contact")
        if payload["contact_at_marker"] == "no" and payload["actor"] != "no-contact":
            raise ValueError("a rejected contact must have actor no-contact")
        if payload["contact_at_marker"] == "yes" and payload["evidence_kind"] not in {
            "visible-contact",
            "inferred-contact",
        }:
            raise ValueError("a confirmed contact needs visible or inferred evidence")
        if (
            payload["contact_at_marker"] == "no"
            and payload["evidence_kind"] != "no-contact"
        ):
            raise ValueError("a rejected contact needs no-contact evidence")
        if (
            payload["contact_at_marker"] == "unclear"
            and payload["evidence_kind"] != "unclear"
        ):
            raise ValueError("an unclear contact needs unclear evidence")
    elif kind is TrialKind.BROADCAST:
        expected = {
            "valid_rally_evidence",
            "broadcast_content",
            "contains_camera_cut",
            "visible_evidence",
            "uncertainty",
        }
        _exact_keys(payload, expected, "broadcast reply")
        if payload["valid_rally_evidence"] not in {"yes", "no", "unclear"}:
            raise ValueError("valid_rally_evidence must be yes, no, or unclear")
        if payload["broadcast_content"] not in {
            "live-play",
            "mixed",
            "replay",
            "cutaway",
            "other",
            "unclear",
        }:
            raise ValueError("broadcast_content has an unsupported value")
        if payload["contains_camera_cut"] not in {"yes", "no", "unclear"}:
            raise ValueError("contains_camera_cut must be yes, no, or unclear")
        if payload["valid_rally_evidence"] == "yes" and payload[
            "broadcast_content"
        ] in {"replay", "cutaway", "other"}:
            raise ValueError(
                "confirmed live-rally evidence conflicts with non-live content"
            )
        if (
            payload["valid_rally_evidence"] == "no"
            and payload["broadcast_content"] == "live-play"
        ):
            raise ValueError(
                "rejected live-rally evidence conflicts with live-play content"
            )
    else:
        expected = {
            "tracked_object",
            "visible_evidence",
            "uncertainty",
        }
        _exact_keys(payload, expected, "track reply")
        if payload["tracked_object"] not in {
            "real-shuttle",
            "text-or-logo",
            "player-or-racket",
            "empty-or-unrelated",
            "unclear",
        }:
            raise ValueError("tracked_object has an unsupported value")
        payload["tracker_follows_real_shuttle"] = {
            "real-shuttle": "yes",
            "unclear": "unclear",
        }.get(payload["tracked_object"], "no")

    for key in ("visible_evidence", "uncertainty"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    return payload
