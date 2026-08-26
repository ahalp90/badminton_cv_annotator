"""Focused tests for the label-blind contact evidence harness."""

# ruff: noqa: E402

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import freeze_contact_evidence as freezer
import score_contact_evidence as scorer

from annotator.types import StickyResult


def _sticky(
    distances: list[float], picks: list[int], ankle_y: list[float]
) -> StickyResult:
    ankles = np.column_stack([np.zeros(2), np.asarray(ankle_y, dtype=float)])
    return StickyResult(
        distances=np.array([np.nanmin(distances)], dtype=float),
        picks=np.array([picks], dtype=int),
        standing_count=np.ones(1, dtype=int),
        ankle_pos=np.array([ankles], dtype=float),
        bbox_height=np.ones((1, 2), dtype=float),
        distances_per_slot=np.array([distances], dtype=float),
        wrist_dist_px=np.array([distances], dtype=float),
        analysed=np.ones(1, dtype=bool),
    )


def test_ankle_rule_two_players_preserves_nearest_wrist_slot() -> None:
    sticky = _sticky([0.2, 0.5], [4, 7], [0.25, 0.75])
    assert freezer.ankle_half(0, sticky, (500.0, 600.0), 1080.0) == "Top"

    selected_bottom = _sticky([0.5, 0.2], [4, 7], [0.25, 0.75])
    assert freezer.ankle_half(0, selected_bottom, (500.0, 600.0), 1080.0) == "Bot"


@pytest.mark.parametrize(
    ("ankle_y", "expected"),
    [([0.40, np.nan], "Top"), ([0.60, np.nan], "Bot"), ([0.50, np.nan], "Top")],
)
def test_ankle_rule_one_player_uses_net_band_midpoint(
    ankle_y: list[float], expected: str
) -> None:
    sticky = _sticky([0.2, np.inf], [4, -1], ankle_y)
    assert freezer.ankle_half(0, sticky, (500.0, 600.0), 1080.0) == expected


def test_ankle_rule_tie_missing_or_no_wrist_is_none() -> None:
    assert freezer.ankle_half(
        0, _sticky([0.2, 0.5], [4, 7], [0.4, 0.4]), (500.0, 600.0), 1080.0
    ) is None
    assert freezer.ankle_half(
        0, _sticky([0.2, 0.5], [4, 7], [np.nan, 0.4]), (500.0, 600.0), 1080.0
    ) is None
    assert freezer.ankle_half(
        0, _sticky([np.nan, np.inf], [-1, -1], [0.4, 0.6]), (500.0, 600.0), 1080.0
    ) is None


def _empty_evidence() -> dict[str, object]:
    fixture_rows: list[dict[str, object]] = []
    for fixture, (video_id, fps) in freezer.FIXTURE_SPECS.items():
        fixture_rows.append(
            {
                "fixture": fixture,
                "video_id": video_id,
                "fps": fps,
                "frame_count": 20,
                "resolution": [1920.0, 1080.0],
                "tracker_segment_count": 1,
                "spans": [
                    {
                        "span_id": 0,
                        "start_frame": 0,
                        "end_frame": 20,
                        "raw_contact_count": 0,
                        "filtered_contact_count": 0,
                        "stored_striker_half": None,
                        "current_striker_half": None,
                        "geometry_striker_half": None,
                        "stored_server_half": None,
                        "current_server_half": None,
                        "geometry_server_half": None,
                        "contacts": [],
                    }
                ],
            }
        )
    return {"schema": freezer.EVIDENCE_SCHEMA, "fixtures": fixture_rows}


def _write_verified_freeze(root: Path, evidence: dict[str, object]) -> Path:
    evidence_path = root / freezer.EVIDENCE_FILENAME
    evidence_bytes = freezer.deterministic_gzip_bytes(evidence)
    evidence_path.write_bytes(evidence_bytes)
    input_rows = []
    roles = (
        "shuttle_track",
        "pose_kps",
        "pose_bboxes",
        "pose_scores",
        "pose_kp_scores",
        "pose_ndet",
        "court_evidence",
        "court_keep_vote",
        "court_present",
        "annotation",
    )
    for fixture in freezer.FIXTURE_SPECS:
        input_rows.append(
            {
                "fixture": fixture,
                "files": [
                    {
                        "role": role,
                        "stage": role.split("_")[0],
                        "filename": f"{role}.bin",
                        "size_bytes": 0,
                        "sha256": "0" * 64,
                    }
                    for role in roles
                ],
            }
        )
    manifest = {
        "schema": scorer.MANIFEST_SCHEMA,
        "labels_read": False,
        "source_commit": "abc123",
        "fixture_set": list(freezer.FIXTURE_SPECS),
        "inputs": input_rows,
        "evidence_schema": freezer.EVIDENCE_SCHEMA,
        "evidence_file": freezer.EVIDENCE_FILENAME,
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
    }
    manifest_path = root / freezer.MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_freeze_firewall_verifies_digest_and_rejects_mutation(tmp_path: Path) -> None:
    manifest_path = _write_verified_freeze(tmp_path, _empty_evidence())
    verified = scorer.verify_freeze(manifest_path)
    assert verified.manifest["labels_read"] is False
    assert verified.evidence["schema"] == freezer.EVIDENCE_SCHEMA

    with verified.evidence_path.open("ab") as target:
        target.write(b"mutation")
    with pytest.raises(ValueError, match="SHA-256"):
        scorer.verify_freeze(manifest_path)


def test_freeze_firewall_rejects_forbidden_nested_key(tmp_path: Path) -> None:
    evidence = _empty_evidence()
    evidence["fixtures"][0]["spans"][0]["ground_truth"] = None  # type: ignore[index]
    manifest_path = _write_verified_freeze(tmp_path, evidence)
    with pytest.raises(ValueError, match="forbidden"):
        scorer.verify_freeze(manifest_path)


def test_serve_nonserve_matching_and_fps_scaling() -> None:
    assert scorer.scale_base30_frames(5, 25.0) == 4
    assert scorer.scale_base30_frames(5, 30.0) == 5
    rallies = [SimpleNamespace(stroke_frames=(100, 200))]
    evidence_fixture = {"fixture": "sset_01", "spans": [{"span_id": 0, "start_frame": 0, "end_frame": 300}]}
    rows = [
        scorer.ContactRow("sset_01", 0, 0, 104, "Top", "Top", 2, 104, True),
        scorer.ContactRow("sset_01", 0, 1, 204, "Bot", "Bot", 1, 204, True),
        scorer.ContactRow("sset_01", 0, 2, 250, None, None, 0, 250, True),
    ]
    metrics, matches = scorer._match_variant(evidence_fixture, rallies, 25.0, rows, 5)
    assert [(match.gt_index, match.candidate.contact_frame) for match in matches] == [(0, 104), (1, 204)]
    assert metrics["serve"] == {"matched": 1, "total": 1, "recall": 1.0}
    assert metrics["non_serve"] == {"matched": 1, "total": 1, "recall": 1.0}
    assert metrics["candidate_count"] == 3
    assert metrics["noise_count"] == 1

    sides = {("sset_01", 100): "Top", ("sset_01", 200): "Bot"}
    half_metrics = scorer._half_metrics(matches, sides)
    assert half_metrics["all"]["ankle"]["correct"] == 2
    assert half_metrics["one_player"]["ankle"]["correct"] == 1
    assert half_metrics["two_players"]["current"]["correct"] == 1


def test_results_writer_has_stable_plain_json(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    scorer.write_results(path, {"b": 2, "a": 1})
    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    compressed = tmp_path / "results.json.gz"
    scorer.write_results(compressed, {"b": 2, "a": 1})
    with gzip.open(compressed, "rt", encoding="utf-8") as source:
        assert json.load(source) == {"a": 1, "b": 2}
