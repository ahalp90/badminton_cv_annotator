"""Focused tests for the observational missed-contact audit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import analyse_contact_failures as auditor
import numpy as np
import pytest
import score_tree_contact_detector as tree_scorer


def _feature_rows(rows: list[tuple[int, int, int, int, int, int, int]]) -> np.ndarray:
    dtype = np.dtype(
        [
            ("fixture", "S7"),
            ("interval_id", "<i2"),
            ("frame", "<i4"),
            ("shuttle_visible_t+0", "<f4"),
            ("pose_valid_top_t+0", "<f4"),
            ("pose_valid_bot_t+0", "<f4"),
            ("wrist_valid_top_t+0", "<f4"),
            ("wrist_valid_bot_t+0", "<f4"),
        ]
    )
    output = np.zeros(len(rows), dtype=dtype)
    for index, (frame, interval, shuttle, pose_top, pose_bot, wrist_top, wrist_bot) in enumerate(rows):
        output[index] = (b"sset_21", interval, frame, shuttle, pose_top, pose_bot, wrist_top, wrist_bot)
    return output


def _candidate_rows(
    frames: list[int], scores: list[float], decisions: list[int], *, interval_id: int = 0
) -> np.ndarray:
    rows = np.zeros(len(frames), dtype=tree_scorer.CANDIDATE_SCORE_DTYPE)
    rows["fixture"] = b"sset_21"
    rows["interval_id"] = interval_id
    rows["frame"] = frames
    rows["timing_score"] = scores
    rows["threshold"] = 0.5
    rows["decision"] = decisions
    return rows


def test_scale_tolerance_uses_source_fps() -> None:
    assert auditor.scale_tolerance(10, 25.0) == 8
    assert auditor.scale_tolerance(10, 30.0) == 10


def test_one_to_one_matching_marks_only_one_close_gt_as_found() -> None:
    assert auditor._matched_gt_indices([100, 108], [104], 5) == {0}


def test_best_peaks_keep_raw_below_threshold_alternatives_and_apply_nms() -> None:
    rows = _candidate_rows(
        [90, 94, 99, 105, 107],
        [0.90, 0.95, 0.80, 0.94, 0.70],
        [0, 2, 1, 2, 0],
    )

    peaks = auditor.best_timing_peaks(rows, 100, 10, 5)

    assert [row["frame"] for row in peaks] == [94, 105]
    assert peaks[0]["decision_name"] == "retained"
    assert peaks[1]["timing_score"] == pytest.approx(0.94)


def test_evidence_availability_is_explicit_and_uses_all_frozen_rows() -> None:
    rows = _feature_rows(
        [
            (95, 0, 1, 0, 0, 0, 0),
            (100, 0, 0, 0, 1, 0, 0),
            (105, 0, 0, 0, 0, 0, 1),
        ]
    )

    assert auditor.evidence_availability(rows, 100, 10) == {
        "frozen_row_count": 3,
        "shuttle": True,
        "pose": True,
        "wrist": True,
    }


def test_audit_fixture_reports_missed_contact_causes() -> None:
    feature_rows = _feature_rows(
        [
            (90, 0, 0, 0, 0, 0, 0),
            (100, 0, 1, 1, 1, 1, 1),
            (106, 0, 1, 0, 0, 0, 0),
            (200, 0, 0, 0, 0, 0, 0),
        ]
    )
    candidates = _candidate_rows(
        [90, 100, 106],
        [0.40, 0.92, 0.49],
        [0, 2, 0],
    )

    result = auditor.audit_fixture(
        "sset_21",
        30.0,
        [100, 200],
        {100},
        feature_rows,
        candidates,
        [200],
    )

    assert result["summary"]["matched_contacts"] == 1
    assert result["summary"]["missed_contacts"] == 1
    row = result["missed_contacts"][0]
    assert row["frame"] == 200
    assert row["contact_type"] == "exchange"
    assert row["seeded_candidate_within_tolerance"] is False
    assert row["best_timing_score"] is None
    assert row["nearby_below_threshold"] is False
    assert row["nearby_duplicate"] is False
    assert row["nearby_retained"] is False
    assert row["handcrafted_filtered_found"] is True
    assert row["evidence"] == {
        "frozen_row_count": 1,
        "shuttle": False,
        "pose": False,
        "wrist": False,
    }


def test_audit_fixture_reports_duplicate_and_below_threshold_near_miss() -> None:
    feature_rows = _feature_rows([(100, 0, 1, 1, 1, 1, 1)])
    candidates = _candidate_rows(
        [96, 101, 106],
        [0.55, 0.80, 0.70],
        [0, 1, 0],
    )

    result = auditor.audit_fixture(
        "sset_21",
        30.0,
        [100],
        {100},
        feature_rows,
        candidates,
        [],
    )

    row = result["missed_contacts"][0]
    assert row["seeded_candidate_within_tolerance"] is True
    assert row["best_timing_score"] == pytest.approx(0.8)
    assert row["fold_threshold"] == pytest.approx(0.5)
    assert row["nearby_below_threshold"] is True
    assert row["nearby_duplicate"] is True
    assert row["nearby_retained"] is False


def test_write_results_is_deterministic_and_summary_is_readable(tmp_path: Path) -> None:
    payload = {"summary": {"missed_contacts": 2}, "schema": auditor.RESULTS_SCHEMA}
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    auditor.write_results(first, payload)
    auditor.write_results(second, payload)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as source:
        assert json.load(source) == payload


def test_invalid_feature_table_fails_loudly() -> None:
    rows = np.zeros(1, dtype=[("frame", "<i4")])
    with pytest.raises(ValueError, match="missing fields"):
        auditor.evidence_availability(rows, 0, 1)
