from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.trial_schema import (
    MANIFEST_SCHEMA,
    TrialKind,
    load_manifest,
    parse_reply,
)


def _case() -> dict[str, object]:
    return {
        "case_id": "event-sset01-001",
        "kind": "event",
        "video_id": "sset_01",
        "clip_path": "/not/needed.mp4",
        "source_start_frame": 100,
        "source_end_frame": 150,
        "candidate_frame": 125,
        "sample_fps": 25.0,
        "expected_frames": 50,
        "width": 512,
        "height": 288,
        "pipeline_priors": {"wrist_near": True},
    }


def test_manifest_rejects_truth_leakage(tmp_path: Path) -> None:
    case = _case()
    case["pipeline_priors"] = {"gt_frame": 125}
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "cases": [case]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="forbidden inference key 'gt_frame'"):
        load_manifest(path, require_clips=False)


def test_manifest_rejects_current_trial_truth_fields(tmp_path: Path) -> None:
    case = _case()
    case["pipeline_priors"] = {"scene_fractions": {"replay": 1.0}}
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "cases": [case]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="forbidden inference key 'scene_fractions'"):
        load_manifest(path, require_clips=False)


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = _case()
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps({"schema": MANIFEST_SCHEMA, "cases": [case, case]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_manifest(path, require_clips=False)


def test_event_reply_requires_bare_exact_json() -> None:
    response = json.dumps(
        {
            "contact_at_marker": "yes",
            "evidence_kind": "visible-contact",
            "actor": "top",
            "nearby_unmarked_contact": "no",
            "visible_evidence": "The top player swings through the shuttle.",
            "uncertainty": "The shuttle is small but visible.",
        }
    )
    parsed = parse_reply(TrialKind.EVENT, response)
    assert parsed["contact_at_marker"] == "yes"

    with pytest.raises(ValueError, match="not bare valid JSON"):
        parse_reply(TrialKind.EVENT, f"```json\n{response}\n```")


def test_broadcast_reply_rejects_unknown_label() -> None:
    response = json.dumps(
        {
            "valid_rally_evidence": "yes",
            "broadcast_content": "warmup",
            "contains_camera_cut": "no",
            "visible_evidence": "A player is hitting.",
            "uncertainty": "The context is short.",
        }
    )
    with pytest.raises(ValueError, match="unsupported value"):
        parse_reply(TrialKind.BROADCAST, response)


@pytest.mark.parametrize(
    ("contact", "actor"),
    [("yes", "no-contact"), ("no", "top"), ("no", "unclear")],
)
def test_event_reply_rejects_contradictory_actor(contact: str, actor: str) -> None:
    response = json.dumps(
        {
            "contact_at_marker": contact,
            "evidence_kind": "visible-contact" if contact == "yes" else "no-contact",
            "actor": actor,
            "nearby_unmarked_contact": "no",
            "visible_evidence": "The marker and actor answer conflict.",
            "uncertainty": "None.",
        }
    )
    with pytest.raises(ValueError, match="contact"):
        parse_reply(TrialKind.EVENT, response)


def test_event_reply_rejects_evidence_kind_conflict() -> None:
    response = json.dumps(
        {
            "contact_at_marker": "yes",
            "evidence_kind": "no-contact",
            "actor": "top",
            "nearby_unmarked_contact": "no",
            "visible_evidence": "The fields conflict.",
            "uncertainty": "None.",
        }
    )
    with pytest.raises(ValueError, match="evidence"):
        parse_reply(TrialKind.EVENT, response)


def test_broadcast_reply_rejects_live_replay_contradiction() -> None:
    response = json.dumps(
        {
            "valid_rally_evidence": "yes",
            "broadcast_content": "replay",
            "contains_camera_cut": "yes",
            "visible_evidence": "The clip is a replay.",
            "uncertainty": "None.",
        }
    )
    with pytest.raises(ValueError, match="conflicts"):
        parse_reply(TrialKind.BROADCAST, response)


def test_track_reply_requires_concrete_failure_for_rejection() -> None:
    valid = json.dumps(
        {
            "tracked_object": "text-or-logo",
            "visible_evidence": "The ring stays fixed on court lettering.",
            "uncertainty": "The lettering remains visible throughout.",
        }
    )
    parsed = parse_reply(TrialKind.TRACK, valid)
    assert parsed["tracked_object"] == "text-or-logo"
    assert parsed["tracker_follows_real_shuttle"] == "no"

    contradictory = json.dumps(
        {
            "tracked_object": "bird",
            "visible_evidence": "The fields conflict.",
            "uncertainty": "None.",
        }
    )
    with pytest.raises(ValueError, match="unsupported value"):
        parse_reply(TrialKind.TRACK, contradictory)
