from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from annotator.calibration.shuttleset22_features import (
    _player_slot,
    _require_digest,
    _validate_identity,
    feature_summary,
    leave_one_video_out,
    load_annotation_rallies,
)


def test_consumed_artifact_identity_checks_size_and_md5(tmp_path: Path) -> None:
    artifact = tmp_path / "pose.npy.xz"
    artifact.write_bytes(b"artifact")
    identity = {
        "name": "pose_kps",
        "path": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "md5": hashlib.md5(b"artifact", usedforsecurity=False).hexdigest(),
    }

    _validate_identity(artifact, identity)
    identity["size_bytes"] = 1
    with pytest.raises(ValueError, match="size differs"):
        _validate_identity(artifact, identity)


def test_annotation_records_report_exclusions_and_reject_bad_rallies(
    tmp_path: Path,
) -> None:
    table = pd.DataFrame(
        {
            "rally": [1, 1, 1, 2, 2, 3, 4, 4],
            "ball_round": [1, 2, 3, 1, 2, 1, 1, 2],
            "frame_num": [10, 20, 30, 50, 40, 200, 60, 70],
            "flaw": [None, 1, None, None, None, None, None, None],
            "player_location_y": [10, 20, 10, 10, 20, 10, 10, 20],
            "opponent_location_y": [20, 10, 20, 20, 10, 20, 20, 10],
            "type": [
                "serve",
                "clear",
                "drop",
                "serve",
                "clear",
                "serve",
                "serve",
                "clear",
            ],
        }
    )
    table.to_csv(tmp_path / "set1.csv", index=False)

    rows, population = load_annotation_rallies(tmp_path, frame_count=100)

    assert len(rows) == 1
    assert rows[0]["contacts"]["stroke_count"] == 2
    assert rows[0]["outcomes"]["server_prediction"] == "Top"
    assert population == {
        "source_contact_rows": 8,
        "usable_contact_rows": 2,
        "excluded_flaw_rows": 1,
        "excluded_invalid_frame_rows": 1,
        "usable_rallies": 1,
        "excluded_incomplete_rallies": 2,
        "excluded_incomplete_rally_rows": 4,
        "excluded_non_monotonic_rallies": 1,
        "excluded_non_monotonic_rally_rows": 2,
    }


def test_expected_digest_must_match() -> None:
    _require_digest("source manifest", "same", "same")
    with pytest.raises(ValueError, match="source manifest SHA-256 differs"):
        _require_digest("source manifest", "actual", "expected")


def test_player_slot_requires_distinct_finite_image_y() -> None:
    assert _player_slot(pd.Series({"player_location_y": 10, "opponent_location_y": 20})) == "Top"
    assert _player_slot(pd.Series({"player_location_y": 30, "opponent_location_y": 20})) == "Bot"
    assert _player_slot(pd.Series({"player_location_y": None, "opponent_location_y": 20})) is None


def test_feature_summary_keeps_exact_numeric_populations() -> None:
    rows = [
        {
            "shots_per_rally": 3,
            "posture": [{"mad": 1.0}, {"mad": None}],
            "recovery": {"observations": [{"mean_distance": 0.2}]},
            "movement_inefficiency": [{"top": 0.3, "bottom": None}],
        },
        {
            "shots_per_rally": 5,
            "posture": [{"mad": 2.0}, {"mad": 3.0}],
            "recovery": {"observations": [{"mean_distance": None}]},
            "movement_inefficiency": [{"top": 0.5, "bottom": 0.7}],
        },
    ]

    summary = feature_summary(rows)

    assert summary["shots_per_rally"]["eligible"] == 2
    assert summary["shots_per_rally"]["median"] == 4.0
    assert summary["posture_mad"]["eligible"] == 3
    assert summary["recovery_distance"]["eligible"] == 1
    assert summary["movement_inefficiency"]["eligible"] == 3

    per_video = {
        "08": {"rallies": [rows[0]]},
        "09": {"rallies": [rows[1]]},
    }
    stability = leave_one_video_out(per_video)
    assert stability["shots_per_rally"] == {
        "runs": 2,
        "min_median": 3.0,
        "max_median": 5.0,
    }
