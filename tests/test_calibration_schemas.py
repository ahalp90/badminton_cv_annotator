"""Pinned output contracts for the Stage 6 sweep runner."""

from __future__ import annotations

from enum import Enum

from annotator.calibration.scoring import CONTACT_TOLERANCES_BASE30
from annotator.calibration.schemas import (
    ALIGNMENT_OWN_COVERED_COLUMNS,
    ALIGNMENT_SHARED_COLUMNS,
    BEST_CONFIG_COMPARISON_COLUMNS,
    BOUNDARY_SWEEP_COLUMNS,
    CONTACT_FRONTIER_COLUMNS,
    CONTACT_STABILITY_COLUMNS,
    CONTACT_SWEEP_COLUMNS,
    CSV_COLUMNS_BY_FILENAME,
    FROZEN_ROW_COLUMNS,
    SPLIT_LOG_COLUMNS,
    WINNER_JSON_TOLERANCES_BASE30,
    winner_document,
    winner_spec,
)


EXPECTED_FROZEN_ROW_COLUMNS = (
    "label", "rest_speed", "rest_window", "end_rest_frames", "start_speed",
    "start_min_frames", "smooth_window", "min_contact_speed",
    "n_spans", "covered", "covered_fraction", "split", "missed", "merged_spans",
    "spurious_spans", "start_alignment_mean", "start_alignment_median",
    "count_gate_covered_fraction", "count_gate_unmerged_fraction", "recall_1",
    "precision_1", "f1_1", "recall_2", "precision_2", "f1_2", "recall_5",
    "precision_5", "f1_5", "recall_10", "precision_10", "f1_10",
    "total_candidates", "precision_raw_1", "precision_raw_2", "precision_raw_5",
    "precision_raw_10",
)


def test_winner_schema_uses_the_shared_contact_tolerances() -> None:
    assert WINNER_JSON_TOLERANCES_BASE30 is CONTACT_TOLERANCES_BASE30


def test_csv_column_tuples_are_exact() -> None:
    boundary = EXPECTED_FROZEN_ROW_COLUMNS + (
        "clean_covered", "swallowed_rallies", "max_rallies_in_one_span",
        "strict_align_median", "strict_align_p90", "strict_f1", "tolerance_frames_1",
        "tolerance_frames_2", "tolerance_frames_5", "tolerance_frames_10",
    )
    contact = EXPECTED_FROZEN_ROW_COLUMNS + (
        "impulse_floor_half_window_frames", "contact_dedup_radius_frames",
        "contact_impulse_multiple", "f1_raw_5", "tolerance_frames_1",
        "tolerance_frames_2", "tolerance_frames_5", "tolerance_frames_10",
    )
    assert CSV_COLUMNS_BY_FILENAME == {
        "boundary_sweep.csv": boundary,
        "contact_sweep.csv": contact,
        "best_config_comparison.csv": (
            "rule", *boundary, "coverage_gap_from_best", "needed_allowance",
        ),
        "contact_frontier.csv": contact,
        "contact_stability.csv": ("rule", *contact, "same_winner_as_live"),
        "split_log.csv": (
            "video_id", "gt_rally_index", "gt_start", "gt_end", "piece_count", "piece_spans",
        ),
        "alignment_own_covered.csv": (
            "rule", "n_rallies", "median_abs_start_offset", "p90_abs_start_offset",
        ),
        "alignment_shared.csv": (
            "rule", "n_rallies", "median_abs_start_offset", "p90_abs_start_offset",
        ),
    }
    assert BOUNDARY_SWEEP_COLUMNS == boundary
    assert CONTACT_SWEEP_COLUMNS == contact
    assert BEST_CONFIG_COMPARISON_COLUMNS == ("rule", *boundary, "coverage_gap_from_best", "needed_allowance")
    assert CONTACT_FRONTIER_COLUMNS == contact
    assert CONTACT_STABILITY_COLUMNS == ("rule", *contact, "same_winner_as_live")
    assert SPLIT_LOG_COLUMNS == ("video_id", "gt_rally_index", "gt_start", "gt_end", "piece_count", "piece_spans")
    assert ALIGNMENT_OWN_COVERED_COLUMNS == ("rule", "n_rallies", "median_abs_start_offset", "p90_abs_start_offset")
    assert ALIGNMENT_SHARED_COLUMNS == ("rule", "n_rallies", "median_abs_start_offset", "p90_abs_start_offset")


def test_sweep_schemas_preserve_frozen_rows_in_order() -> None:
    assert FROZEN_ROW_COLUMNS == EXPECTED_FROZEN_ROW_COLUMNS
    assert BOUNDARY_SWEEP_COLUMNS[:len(EXPECTED_FROZEN_ROW_COLUMNS)] == EXPECTED_FROZEN_ROW_COLUMNS
    assert CONTACT_SWEEP_COLUMNS[:len(EXPECTED_FROZEN_ROW_COLUMNS)] == EXPECTED_FROZEN_ROW_COLUMNS


class Strategy(Enum):
    BACK_FILL = 1


def test_winner_json_helpers_preserve_the_pinned_shape() -> None:
    boundary = winner_spec({"rest_window": 5}, {"fill": Strategy.BACK_FILL})
    assert boundary == {
        "overrides_base30": {"rest_window": 5},
        "strategies": {"fill": "BACK_FILL"},
    }
    assert winner_document("fixture-a", ["boundary"], boundary=boundary) == {
        "meta": {
            "fixture": "fixture-a",
            "phases_run": ["boundary"],
            "verdict": "issued",
            "tolerances_base30": [1, 2, 5, 10],
        },
        "boundary": boundary,
    }
