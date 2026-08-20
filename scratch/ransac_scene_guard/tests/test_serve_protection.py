from __future__ import annotations

# Import the standalone experiments after adding their folder to sys.path.
# ruff: noqa: E402

import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import freeze_serve_qualification as freeze
import run_serve_protection_e2e as e2e
import score_serve_protection as score


@pytest.mark.parametrize(
    "result_name",
    [
        "rally_ender_counterfactual.json.gz",
        "rally_ender_e2e.json.gz",
        "scene_aware_ransac.json.gz",
        "serve_protection_all_qualified.json.gz",
        "serve_protection_e2e_all_qualified.json.gz",
        "serve_protection_e2e_first_per_region.json.gz",
        "serve_protection_first_per_region.json.gz",
        "serve_qualification.json.gz",
    ],
)
def test_retained_results_do_not_expose_absolute_local_paths(
    result_name: str,
) -> None:
    with gzip.open(
        EXPERIMENT_ROOT / "results" / result_name,
        "rt",
        encoding="utf-8",
    ) as result_file:
        result_text = result_file.read()

    assert "/home/" not in result_text
    assert "/scratch/" not in result_text


def test_qualify_burst_rows_keeps_region_order_and_all_bursts() -> None:
    fast_runs = [(3, 6), (12, 15), (18, 21), (40, 43)]
    regions = [(0, 30), (35, 50), (60, 70)]

    rows, counts = freeze.qualify_burst_rows(
        fast_runs,
        regions,
        lambda burst: burst in {12, 40},
    )

    assert [row["burst_start"] for row in rows] == [3, 12, 18, 40]
    assert [row["qualified"] for row in rows] == [False, True, False, True]
    assert counts == [1, 1]


def test_protection_window_is_inclusive_and_clips_video_edges() -> None:
    mask = score.protection_window_mask(12, [0, 10], radius=2)

    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 8, 9, 10, 11]


def test_protection_window_rejects_out_of_range_burst() -> None:
    with pytest.raises(ValueError, match="outside"):
        score.protection_window_mask(10, [10], radius=2)


def test_first_qualified_policy_matches_close_none_region_consumption() -> None:
    frozen = {
        "bursts": [
            {"region_index": 0, "burst_start": 10, "qualified": False},
            {"region_index": 0, "burst_start": 20, "qualified": True},
            {"region_index": 0, "burst_start": 30, "qualified": True},
            {"region_index": 1, "burst_start": 50, "qualified": True},
        ]
    }

    assert score.qualified_burst_frames(
        frozen,
        score.ALL_QUALIFIED_BURSTS,
    ) == [20, 30, 50]
    assert score.qualified_burst_frames(
        frozen,
        score.FIRST_QUALIFIED_PER_REGION,
    ) == [20, 50]


def test_subtract_protection_never_adds_candidates() -> None:
    baseline = np.asarray([True, False, True, True, False])
    window = np.asarray([False, True, True, False, True])

    protected = score.subtract_protection(baseline, window)

    assert protected.tolist() == [True, False, False, True, False]
    assert not np.any(protected & ~baseline)


def test_primary_decision_requires_span_contact_and_enrichment_gates() -> None:
    baseline = {
        "selected_frames": 100,
        "visual_positive_spans_hit": 7,
        "contact_tolerances": {"base30_10": {"contacts_with_candidate": 40}},
    }
    protected = {
        "selected_frames": 95,
        "visual_positive_spans_hit": 7,
        "contact_tolerances": {"base30_10": {"contacts_with_candidate": 35}},
    }
    first_baseline = {
        "exact": 8,
        "tolerances": {"base30_10": {"contacts_with_candidate": 12}},
    }
    first_protected = {
        "exact": 7,
        "tolerances": {"base30_10": {"contacts_with_candidate": 11}},
    }

    decision = score.primary_decision(
        baseline,
        protected,
        first_baseline,
        first_protected,
    )

    assert decision["passes_mask_screen"] is True
    assert decision["first_exact_conflicts_rescued"] == 1
    assert decision["contact_reduction_at_least_twice_frame_reduction"] is True


@pytest.mark.parametrize(
    ("protected_spans", "first_after", "risk_after"),
    [
        (6, 7, 35),
        (7, 8, 35),
        (7, 7, 37),
    ],
)
def test_primary_decision_rejects_each_failed_gate(
    protected_spans: int,
    first_after: int,
    risk_after: int,
) -> None:
    baseline = {
        "selected_frames": 100,
        "visual_positive_spans_hit": 7,
        "contact_tolerances": {"base30_10": {"contacts_with_candidate": 40}},
    }
    protected = {
        "selected_frames": 95,
        "visual_positive_spans_hit": protected_spans,
        "contact_tolerances": {"base30_10": {"contacts_with_candidate": risk_after}},
    }
    first_baseline = {
        "exact": 8,
        "tolerances": {"base30_10": {"contacts_with_candidate": 12}},
    }
    first_protected = {
        "exact": first_after,
        "tolerances": {"base30_10": {"contacts_with_candidate": 12}},
    }

    decision = score.primary_decision(
        baseline,
        protected,
        first_baseline,
        first_protected,
    )

    assert decision["passes_mask_screen"] is False


def test_verify_sha256_rejects_changed_input(tmp_path: Path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256"):
        score.verify_sha256(path, "0" * 64)


def test_strict_metrics_counts_one_to_one_rows() -> None:
    rows = [
        {
            "tolerance_base30": 10,
            "tolerance_frames": 8,
            "row_kind": "matched",
            "offset_frames": -4,
        },
        {
            "tolerance_base30": 10,
            "tolerance_frames": 8,
            "row_kind": "unmatched_gt",
            "offset_frames": None,
        },
        {
            "tolerance_base30": 10,
            "tolerance_frames": 8,
            "row_kind": "unmatched_candidate",
            "offset_frames": None,
        },
    ]

    metrics = e2e.strict_metrics(rows, tolerance_base30=10, fps=25.0)

    assert metrics["matched_count"] == 1
    assert metrics["gt_count"] == 2
    assert metrics["candidate_count"] == 2
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["mean_abs_offset_frames"] == 4.0


def test_safety_failures_flag_lost_correct_and_new_wrong_outputs() -> None:
    fixtures = {
        "sset_01": {
            "comparison": {
                "changed_gt_rallies": [
                    {
                        "baseline": {
                            "landing_correct": True,
                            "landing_pred": "Top",
                            "getpoint_correct": False,
                            "getpoint_pred": "Bot",
                        },
                        "protected": {
                            "landing_correct": False,
                            "landing_pred": "Bot",
                            "getpoint_correct": True,
                            "getpoint_pred": "Top",
                        },
                    },
                    {
                        "baseline": {
                            "landing_correct": False,
                            "landing_pred": None,
                            "getpoint_correct": False,
                            "getpoint_pred": "Bot",
                        },
                        "protected": {
                            "landing_correct": True,
                            "landing_pred": "Top",
                            "getpoint_correct": True,
                            "getpoint_pred": "Top",
                        },
                    },
                ]
            }
        }
    }

    failures = e2e.safety_failures(fixtures)

    assert len(failures) == 1
    assert failures[0]["fixture"] == "sset_01"
    assert failures[0]["reasons"] == [
        "lost_correct_landing",
        "new_or_changed_wrong_or_unscored_landing",
    ]


def test_safety_failures_flag_new_wrong_prediction_from_no_prediction() -> None:
    fixtures = {
        "sset_21": {
            "comparison": {
                "changed_gt_rallies": [
                    {
                        "baseline": {
                            "landing_correct": False,
                            "landing_pred": None,
                            "getpoint_correct": False,
                            "getpoint_pred": "Bot",
                        },
                        "protected": {
                            "landing_correct": False,
                            "landing_pred": "Top",
                            "getpoint_correct": False,
                            "getpoint_pred": "Bot",
                        },
                    }
                ]
            }
        }
    }

    failures = e2e.safety_failures(fixtures)

    assert failures[0]["reasons"] == [
        "new_or_changed_wrong_or_unscored_landing"
    ]


def test_safety_failures_flag_changed_unscored_winner() -> None:
    fixtures = {
        "sset_15": {
            "comparison": {
                "changed_gt_rallies": [
                    {
                        "baseline": {
                            "landing_correct": True,
                            "landing_pred": "Top",
                            "getpoint_correct": None,
                            "getpoint_pred": None,
                        },
                        "protected": {
                            "landing_correct": True,
                            "landing_pred": "Top",
                            "getpoint_correct": None,
                            "getpoint_pred": "Bot",
                        },
                    }
                ]
            }
        }
    }

    failures = e2e.safety_failures(fixtures)

    assert failures[0]["reasons"] == [
        "new_or_changed_wrong_or_unscored_winner"
    ]


def test_rally_ender_counterfactual_cannot_become_deployable_evidence() -> None:
    decision = e2e.replay_decision(
        fixed_upstream=True,
        added_rejections=0,
        failures=[],
        unscored_output_changes=0,
        protection_source=e2e.RALLY_ENDER_PROTECTION,
    )

    assert decision["passes_observed_output_screen"] is True
    assert decision["diagnostic_only"] is True
    assert decision["deployable_evidence"] is False
    assert decision["passes_e2e_safety_screen"] is False


def test_changed_signatures_keeps_added_removed_and_changed_rallies() -> None:
    changes = e2e.changed_signatures(
        {"1": {"frame": 10}, "2": {"frame": 20}},
        {"1": {"frame": 11}, "3": {"frame": 30}},
    )

    assert changes == [
        {
            "rally_id": 1,
            "baseline": {"frame": 10},
            "protected": {"frame": 11},
        },
        {"rally_id": 2, "baseline": {"frame": 20}, "protected": None},
        {"rally_id": 3, "baseline": None, "protected": {"frame": 30}},
    ]
