"""Synthetic tests for pure calibration score-row selection."""

from __future__ import annotations

from annotator.calibration.selection import (
    best_config_clears_quality_floor,
    boundary_live_key_rally_id_f1,
    boundary_report_key_coverage_first,
    boundary_report_key_fewest_swallowed_rallies,
    boundary_report_key_tightest_start,
    contact_live_key_raw_f1,
    contact_meets_floors,
    coverage_allowance_rows,
    f1_raw_5,
    select_contact_live_winner,
)


def make_row(**changes: object) -> dict[str, object]:
    """Build a complete synthetic score row with one deterministic default."""
    row: dict[str, object] = {
        "label": "grid",
        "changed_from_defaults": 0,
        "settings": (5, 0.2),
        "clean_covered": 8,
        "swallowed_rallies": 0,
        "max_rallies_in_one_span": 1,
        "strict_align_median": 2.0,
        "strict_align_p90": 3.0,
        "covered": 8,
        "covered_fraction": 0.8,
        "split": 1,
        "missed": 2,
        "spurious_spans": 1,
        "start_alignment_median": 2.0,
        "recall_5": 0.8,
        "precision_raw_5": 0.8,
    }
    row.update(changes)
    return row


def test_boundary_live_key_prefers_strict_f1_then_alignment() -> None:
    lower_f1 = make_row(clean_covered=7, strict_align_median=0.0)
    tighter_alignment = make_row(strict_align_median=1.0, settings=(7, 0.2))
    winner = min(
        [lower_f1, tighter_alignment],
        key=lambda row: boundary_live_key_rally_id_f1(row, 10),
    )
    assert winner is tighter_alignment


def test_boundary_live_key_none_alignment_sorts_last() -> None:
    missing_alignment = make_row(strict_align_median=None, strict_align_p90=None)
    measured_alignment = make_row(strict_align_median=10.0, settings=(7, 0.2))
    winner = min(
        [missing_alignment, measured_alignment],
        key=lambda row: boundary_live_key_rally_id_f1(row, 10),
    )
    assert winner is measured_alignment


def test_boundary_report_keys_follow_their_exact_ordering() -> None:
    fewest_swallowed_rallies = make_row(
        swallowed_rallies=0, split=9, missed=9, spurious_spans=9,
    )
    lower_split = make_row(swallowed_rallies=1, split=0, missed=0, spurious_spans=0)
    assert (
        min(
            [lower_split, fewest_swallowed_rallies],
            key=boundary_report_key_fewest_swallowed_rallies,
        )
        is fewest_swallowed_rallies
    )

    coverage = make_row(covered_fraction=0.9, split=9, missed=9, spurious_spans=9)
    lower_split = make_row(covered_fraction=0.8, split=0, missed=0, spurious_spans=0)
    assert min([lower_split, coverage], key=boundary_report_key_coverage_first) is coverage

    tight_start = make_row(start_alignment_median=-1.0, swallowed_rallies=9)
    fewer_merges = make_row(start_alignment_median=2.0, swallowed_rallies=0)
    assert min([fewer_merges, tight_start], key=boundary_report_key_tightest_start) is tight_start


def test_report_keys_handle_missing_measurements() -> None:
    missing_coverage = make_row(covered_fraction=None)
    valid_coverage = make_row(covered_fraction=0.0, settings=(7, 0.2))
    assert min([missing_coverage, valid_coverage], key=boundary_report_key_coverage_first) is valid_coverage

    missing_start = make_row(start_alignment_median=None)
    measured_start = make_row(start_alignment_median=99.0, settings=(7, 0.2))
    assert min([missing_start, measured_start], key=boundary_report_key_tightest_start) is measured_start


def test_standard_tail_deterministically_settles_key_ties() -> None:
    more_changed = make_row(changed_from_defaults=1, settings=(1, 0.1))
    fewer_changed = make_row(changed_from_defaults=0, settings=(9, 0.9))
    assert (
        min([more_changed, fewer_changed], key=boundary_report_key_fewest_swallowed_rallies)
        is fewer_changed
    )

    later_settings = make_row(settings=(9, 0.9))
    earlier_settings = make_row(settings=(1, 0.1))
    assert (
        min([later_settings, earlier_settings], key=boundary_report_key_fewest_swallowed_rallies)
        is earlier_settings
    )


def test_raw_f1_propagates_missing_metrics_and_uses_raw_precision() -> None:
    assert f1_raw_5(make_row(recall_5=None)) is None
    assert f1_raw_5(make_row(precision_raw_5=None)) is None
    assert f1_raw_5(make_row(recall_5=0.0, precision_raw_5=0.0)) == 0.0
    assert f1_raw_5(make_row(recall_5=0.5, precision_raw_5=0.25)) == 1 / 3


def test_contact_floors_fail_closed_and_accept_exact_threshold() -> None:
    missing_precision = make_row(precision_raw_5=None)
    assert not contact_meets_floors(missing_precision, minimum_precision=0.5)
    assert not contact_meets_floors(make_row(recall_5=None), minimum_recall=0.5)
    assert contact_meets_floors(make_row(recall_5=0.5, precision_raw_5=0.5), 0.5, 0.5)


def test_contact_live_key_uses_raw_f1_after_floor_filter() -> None:
    high_f1 = make_row(recall_5=0.8, precision_raw_5=0.8, settings=(7, 0.2))
    low_f1 = make_row(recall_5=0.9, precision_raw_5=0.4)
    reference = make_row(label="reference", recall_5=1.0, precision_raw_5=1.0)
    assert min([low_f1, high_f1], key=contact_live_key_raw_f1) is high_f1
    assert select_contact_live_winner([reference, low_f1, high_f1], 0.5) is high_f1
    assert select_contact_live_winner([low_f1], minimum_precision=0.5) is None


def test_coverage_allowance_uses_fractional_arithmetic_and_excludes_references() -> None:
    best = make_row(covered=10)
    within_fractional_allowance = make_row(covered=9, settings=(7, 0.2))
    outside_fractional_allowance = make_row(covered=8, settings=(9, 0.2))
    reference = make_row(label="reference", covered=99)
    eligible = coverage_allowance_rows(
        [best, within_fractional_allowance, outside_fractional_allowance, reference],
        n_gt_rallies=3,
        coverage_allowance_fraction=0.5,
    )
    assert eligible == [best, within_fractional_allowance]


def test_quality_floor_fails_closed() -> None:
    assert best_config_clears_quality_floor(make_row(covered_fraction=0.5), 0.5)
    assert not best_config_clears_quality_floor(make_row(covered_fraction=0.49), 0.5)
    assert not best_config_clears_quality_floor(make_row(covered_fraction=None), 0.0)


def test_live_key_pins_exact_f1_formula_and_tail() -> None:
    # TP=6, FN=10-6=4, FP=2 -> F1 = 12/18.
    row = make_row(clean_covered=6, spurious_spans=2)
    assert boundary_live_key_rally_id_f1(row, 10)[0] == -(12 / 18)

    tied_metrics = make_row(changed_from_defaults=1, settings=(1, 0.1))
    fewer_changed = make_row(changed_from_defaults=0, settings=(9, 0.9))
    winner = min(
        [tied_metrics, fewer_changed],
        key=lambda row: boundary_live_key_rally_id_f1(row, 10),
    )
    assert winner is fewer_changed


def test_fewest_swallowed_rallies_key_orders_every_position() -> None:
    lower_split = make_row(split=0, missed=9, spurious_spans=9)
    higher_split = make_row(split=1, missed=0, spurious_spans=0)
    assert (
        min([higher_split, lower_split], key=boundary_report_key_fewest_swallowed_rallies)
        is lower_split
    )

    lower_missed = make_row(missed=0, spurious_spans=9)
    higher_missed = make_row(missed=1, spurious_spans=0)
    assert (
        min([higher_missed, lower_missed], key=boundary_report_key_fewest_swallowed_rallies)
        is lower_missed
    )

    lower_spurious = make_row(spurious_spans=0, settings=(9, 0.9))
    higher_spurious = make_row(spurious_spans=1, settings=(1, 0.1))
    assert (
        min([higher_spurious, lower_spurious], key=boundary_report_key_fewest_swallowed_rallies)
        is lower_spurious
    )


def test_contact_tail_settles_equal_f1() -> None:
    later_settings = make_row(settings=(9, 0.9))
    earlier_settings = make_row(settings=(1, 0.1))
    assert min([later_settings, earlier_settings], key=contact_live_key_raw_f1) is earlier_settings


def test_contact_missing_f1_rows_are_ineligible_without_floors() -> None:
    unmeasured = make_row(recall_5=None)
    measured = make_row(settings=(7, 0.2))
    assert select_contact_live_winner([unmeasured, measured]) is measured
    assert select_contact_live_winner([unmeasured]) is None


def test_coverage_allowance_boundary_equality_passes() -> None:
    best = make_row(covered=10)
    at_boundary = make_row(covered=8, settings=(7, 0.2))
    eligible = coverage_allowance_rows(
        [best, at_boundary], n_gt_rallies=4, coverage_allowance_fraction=0.5,
    )
    assert eligible == [best, at_boundary]
