"""Focused tests for the recording-only replay and serve measurements."""

from __future__ import annotations

import gzip
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from annotator.calibration.scoring import GtRally
from annotator.broadcast_timeline_labels import SceneTruth, VideoMetadata, make_interval


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/scraper_pipeline/broadcast_nonstandard_camera_id/measure_replay_and_serve_behaviour.py"
)
SPEC = importlib.util.spec_from_file_location("measure_replay_and_serve_behaviour", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
measurement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measurement)


def test_expand_truth_and_binary_metrics_exclude_other() -> None:
    metadata = VideoMetadata("sset_01", 25.0, 8)
    intervals = [
        make_interval(metadata, 0, 2, SceneTruth.LIVE),
        make_interval(metadata, 2, 4, SceneTruth.REPLAY),
        make_interval(metadata, 4, 6, SceneTruth.CUTAWAY),
        make_interval(metadata, 6, 8, SceneTruth.OTHER),
    ]
    truth = measurement.expand_truth(intervals, metadata)
    prediction = np.array([False, True, True, False, True, True, True, False])

    metrics = measurement.binary_metrics(prediction, truth)

    assert truth.tolist() == [
        "live", "live", "replay", "replay", "cutaway", "cutaway", "other", "other",
    ]
    assert metrics == {
        "frames": 6,
        "flagged_frames": 4,
        "flag_rate": 4 / 6,
        "tp": 3,
        "fp": 1,
        "tn": 1,
        "fn": 1,
        "precision": 3 / 4,
        "recall": 3 / 4,
    }


def test_nearest_replay_or_cutaway_distance_uses_both_classes() -> None:
    truth = np.array([
        "live", "replay", "replay", "live", "live", "cutaway", "other",
    ])

    assert measurement.nearest_replay_or_cutaway_distance(truth).tolist() == [1, 0, 0, 1, 1, 0, 1]


def test_slow_motion_details_reconstruct_current_signal() -> None:
    x = np.concatenate((np.arange(10) * 0.02, 0.2 + np.arange(10) * 0.003))
    track = np.column_stack((x, np.zeros(20), np.ones(20)))
    non_evidence = np.zeros(20, dtype=bool)
    baseline_exclude = np.zeros(20, dtype=bool)

    details = measurement.slow_motion_details(
        track,
        [(0, 20)],
        25.0,
        non_evidence=non_evidence,
        baseline_exclude=baseline_exclude,
    )
    expected = measurement.velocity_drop_signal(
        track,
        [(0, 20)],
        20,
        25.0,
        non_evidence=non_evidence,
        baseline_exclude=baseline_exclude,
    )

    np.testing.assert_array_equal(details.signal, expected)
    assert details.rally_median > 0
    assert details.slow_threshold == measurement.SLOWMO_SPEED_FRAC * details.rally_median


def test_gt_statuses_distinguish_target_serve_miss() -> None:
    rallies = [GtRally("set1", 1, (10, 20)), GtRally("set1", 2, (40, 50))]
    spans = [(8, 25), (38, 55)]
    accepted = {0: [20], 1: [40, 50]}

    statuses = measurement.gt_statuses(rallies, spans, accepted, tolerance=1)

    assert statuses[0]["target_miss"] is True
    assert statuses[0]["serve_matched"] is False
    assert statuses[1]["target_miss"] is False
    assert statuses[1]["serve_matched"] is True


def test_serve_candidate_rule_records_mask_interaction(monkeypatch) -> None:
    monkeypatch.setattr(
        measurement,
        "detect_contact_flags",
        lambda *_args, **_kwargs: [(5, 3.0), (7, 4.0), (9, 1.0), (10, 0.5)],
    )
    track = np.column_stack((np.linspace(0, 1, 25), np.zeros(25), np.ones(25)))
    truth = np.full(25, "live", dtype="U18")
    truth[7:9] = "replay"
    inpaint_codes = np.zeros(25, dtype=np.int8)
    raw_mask = np.zeros(25, dtype=bool)
    raw_mask[7:9] = True
    definitive = raw_mask.copy()
    distances = np.full(25, 0.5)
    sticky = SimpleNamespace(
        distances=distances,
        analysed=np.ones(25, dtype=bool),
        picks=np.zeros((25, 2), dtype=int),
    )
    rallies = [GtRally("set1", 1, (5, 20))]
    statuses = [{
        "rally_index": 0,
        "serve_frame": 5,
        "target_miss": True,
        "serve_matched": False,
    }]

    candidates, opportunities, summary = measurement.evaluate_serve_candidates(
        track=track,
        truth=truth,
        inpaint_codes=inpaint_codes,
        raw_mask=raw_mask,
        definitive_mask=definitive,
        sticky=sticky,
        spans=[(5, 20)],
        natural_raw_frames=set(),
        accepted_by_span={0: [15]},
        gt_rallies=rallies,
        statuses=statuses,
        fps=25.0,
    )

    by_frame = {row["candidate_frame"]: row for row in candidates}
    assert by_frame[7]["evidence_pass"] is True
    assert by_frame[7]["policy_pass"] is False
    assert by_frame[7]["reject_reasons"] == "definitive-mask"
    assert by_frame[9]["tolerance_frames"] == 4
    assert by_frame[9]["gt_serve_match"] is True
    assert by_frame[10]["gt_serve_match"] is False
    assert by_frame[5]["selected_policy"] is True
    assert opportunities[0]["selected_evidence_frame"] == 7
    assert opportunities[0]["selected_policy_frame"] == 5
    assert summary["current_mask_policy"]["true_positives"] == 1
    assert summary["current_mask_policy"]["false_positives"] == 0


def test_compressed_writers_are_deterministic_and_reload(tmp_path: Path) -> None:
    rows = [{"detector": "raw", "frames": 2}]
    first_csv = tmp_path / "first.csv.gz"
    second_csv = tmp_path / "second.csv.gz"
    first_json = tmp_path / "first.json.gz"
    second_json = tmp_path / "second.json.gz"

    measurement._write_csv_gz(first_csv, ("detector", "frames"), rows)
    measurement._write_csv_gz(second_csv, ("detector", "frames"), rows)
    measurement._write_json_gz(first_json, {"value": 2})
    measurement._write_json_gz(second_json, {"value": 2})

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    with gzip.open(first_csv, "rt", encoding="utf-8") as handle:
        assert handle.read() == "detector,frames\nraw,2\n"


@pytest.mark.parametrize(
    "row",
    [
        {"detector": "raw"},
        {"detector": "raw", "frames": 2, "unexpected": 3},
    ],
)
def test_compressed_csv_writer_rejects_schema_drift(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="keys differ"):
        measurement._write_csv_gz(
            tmp_path / "invalid.csv.gz",
            ("detector", "frames"),
            [row],
        )


def test_une_replacement_profile_pins_the_surviving_case() -> None:
    profile = measurement.FIXTURE_PROFILES["une-189c5af-static-stride8"]

    assert profile.source_commit == "189c5af58e45d23ae827dde516924194eb238e18"
    assert profile.fixture.digests.dead_mask == "70a2a4e9cbd7c6c02b497b468682c462"
    assert profile.fixture.digests.court_present == "65f4e28d0556c0e5422f569ad4b69fac"
    assert profile.fixture.digests.scene_rows == "7d781f33e29804ef8363bbbd1b60d772"


def test_report_uses_each_replay_mask_count() -> None:
    metric = {"flagged_frames": 0, "frames": 100, "precision": None, "recall": 0.0}
    summary = {
        "generated_at_utc": "2026-08-05T00:00:00+00:00",
        "fixture_profile": {"name": "test", "source_commit": None},
        "replay_mask": {
            "fresh_union_pinned_raw_diff_frames": 3,
            "detectors": {
                "court_absence": {**metric, "flagged_frames": 5},
                "perspective_shift": {**metric, "flagged_frames": 4},
                "velocity_drop": {**metric, "flagged_frames": 2},
                "raw_union": {**metric, "flagged_frames": 11},
                "duration_filtered": {
                    **metric,
                    "flagged_frames": 7,
                    "precision": 0.5,
                    "recall": 0.25,
                },
                "e2e_definitive": {**metric, "flagged_frames": 9},
            },
            "gt_extent_loss": {
                "e2e_definitive": {"masked_gt_extent_frames": 1, "gt_extent_frames": 10},
            },
        },
        "slow_motion": {
            "rally_speed_median": 0.1,
            "slow_threshold": 0.01,
            "signal_frames": 2,
        },
        "replay_duplicate_margin": {
            "status": "supported-for-follow-up",
            "eligible_preceding_live_source_relations": 1,
            "excluded_long_replay_montage": {"start_frame": 80, "end_frame": 100},
            "same_video_gt_rallies": 3,
            "retrieval_margin_limit": "Exact source-frame pairs are not annotated.",
        },
        "serve_lookback": {
            "target_serve_misses": 4,
            "evidence_only": {"selected_triggers": 1},
            "current_mask_policy": {
                "selected_triggers": 1,
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 3,
                "precision": 1.0,
                "recall": 0.25,
                "selected_by_manual_truth": {
                    "live": 1,
                    "live-non-standard": 0,
                    "replay": 0,
                    "cutaway": 0,
                    "other": 0,
                },
            },
        },
    }
    summary["serve_lookback"]["evidence_only"] = summary["serve_lookback"][
        "current_mask_policy"
    ].copy()

    report = measurement._build_report(summary)

    assert "raw union flags 11 of 100 scored frames" in report
    assert "Duration filtering leaves 7 flagged frames" in report
    assert "e2e court-invalid union flags 9 scored frames" in report


def test_duplicate_study_records_supported_interval_relations() -> None:
    metadata = VideoMetadata("sset_01", 25.0, 20)
    intervals = [
        make_interval(metadata, 0, 5, SceneTruth.LIVE),
        make_interval(metadata, 5, 8, SceneTruth.REPLAY),
        make_interval(metadata, 8, 15, SceneTruth.LIVE),
        make_interval(metadata, 15, 20, SceneTruth.REPLAY),
    ]

    result = measurement.replay_duplicate_feasibility(
        intervals,
        3,
        excluded_long_montage=(15, 20),
    )

    assert result["status"] == "supported-for-follow-up"
    assert result["replay_intervals"] == 2
    assert result["eligible_preceding_live_source_relations"] == 1
    assert result["same_video_gt_rallies"] == 3
    assert result["retrieval_margin_status"] == "unmeasured"
