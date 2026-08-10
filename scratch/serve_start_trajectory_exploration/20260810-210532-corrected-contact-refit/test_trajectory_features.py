"""Synthetic checks for corrected serve-trajectory measurements."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from trajectory_features import (
    IncomingMotion,
    RobustDistanceTrend,
    align_anchor_to_gt,
    classify_anchor_frame,
    closest_pre_contact_run,
    decide_fixed_motion_rules,
    first_player_from_final_half,
    fit_path,
    fit_robust_distance_trend,
    measure_incoming_motion,
    summarise_unmatched_anchor_sequence,
)

from annotator.point_winner import Half, fit_alternation


def _motion(distances: list[float], points: list[tuple[float, float]]) -> IncomingMotion:
    frame_count = len(distances)
    return measure_incoming_motion(
        np.asarray(distances, dtype=float),
        np.asarray(points, dtype=float),
        np.full(frame_count, 10.0),
        (100.0, 100.0),
    )


def test_closest_run_ends_immediately_before_contact() -> None:
    usable = np.array([False, True, True, False, True, True, True, False])

    run = closest_pre_contact_run(usable, contact_frame=7, lookback_frames=7)

    assert run == (4, 7, 1)


def test_closest_run_chooses_latest_of_multiple_runs() -> None:
    usable = np.array([True, True, False, True, False, True, True, False])

    run = closest_pre_contact_run(usable, contact_frame=8, lookback_frames=8)

    assert run == (5, 7, 2)


def test_run_does_not_search_before_maximum_lookback() -> None:
    usable = np.zeros(12, dtype=bool)
    usable[1:3] = True
    usable[6:8] = True

    run = closest_pre_contact_run(usable, contact_frame=10, lookback_frames=4)

    assert run == (6, 8, 3)


def test_contact_frame_is_excluded_even_when_marked_usable() -> None:
    usable = np.zeros(8, dtype=bool)
    usable[7] = True

    assert closest_pre_contact_run(usable, contact_frame=7, lookback_frames=8) is None


def test_run_gap_counts_frames_between_run_and_contact() -> None:
    usable = np.zeros(10, dtype=bool)
    usable[3:6] = True

    run = closest_pre_contact_run(usable, contact_frame=9, lookback_frames=8)

    assert run is not None
    assert run.start == 3
    assert run.end == 6
    assert run.frames_to_contact == 4


def test_scene_mask_splits_a_usable_run() -> None:
    usable = np.zeros(8, dtype=bool)
    usable[1:7] = True
    same_scene = np.zeros(8, dtype=bool)
    same_scene[1:4] = True

    run = closest_pre_contact_run(usable, 7, 8, same_scene)

    assert run == (1, 4, 4)


def test_incoming_motion_closes_consistently() -> None:
    motion = _motion([3.0, 2.0, 1.0], [(0.0, 0.0), (0.1, 0.0), (0.2, 0.0)])

    assert motion.n_frames == 3
    assert motion.start_distance_bh == 3.0
    assert motion.end_distance_bh == 1.0
    assert motion.net_closure_bh == 2.0
    assert motion.closing_fraction == 1.0
    assert motion.total_movement_bh == 2.0
    assert motion.largest_step_ratio == 1.0


def test_outgoing_motion_has_negative_closure_and_no_closing_steps() -> None:
    motion = _motion([1.0, 2.0, 3.0], [(0.2, 0.0), (0.1, 0.0), (0.0, 0.0)])

    assert motion.net_closure_bh == -2.0
    assert motion.closing_fraction == 0.0


def test_mixed_motion_counts_only_strictly_closing_changes() -> None:
    motion = _motion([3.0, 2.0, 2.0, 1.0], [(0.0, 0.0)] * 4)

    assert motion.closing_fraction == pytest.approx(2 / 3)


def test_stationary_motion_has_zero_movement_and_jump_ratio() -> None:
    motion = _motion([1.0, 1.0, 1.0], [(0.4, 0.4)] * 3)

    assert motion.total_movement_bh == 0.0
    assert motion.largest_step_ratio == 0.0


def test_wild_jump_is_large_relative_to_typical_steps() -> None:
    motion = _motion(
        [3.0, 2.0, 1.0, 0.5],
        [(0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (0.42, 0.0)],
    )

    assert motion.total_movement_bh == pytest.approx(4.2)
    assert motion.largest_step_ratio == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("distances", "shuttle", "heights", "resolution"),
    [
        (np.ones((2, 1)), np.zeros((2, 2)), np.ones(2), (100.0, 100.0)),
        (np.ones(2), np.zeros((3, 2)), np.ones(2), (100.0, 100.0)),
        (np.array([1.0, np.nan]), np.zeros((2, 2)), np.ones(2), (100.0, 100.0)),
        (np.ones(2), np.array([[0.0, 0.0], [np.inf, 0.0]]), np.ones(2), (100.0, 100.0)),
        (np.ones(2), np.zeros((2, 2)), np.array([1.0, np.inf]), (100.0, 100.0)),
        (np.ones(2), np.zeros((2, 2)), np.ones(2), (100.0, np.nan)),
    ],
)
def test_motion_rejects_shape_and_finite_input_failures(
    distances: np.ndarray,
    shuttle: np.ndarray,
    heights: np.ndarray,
    resolution: tuple[float, float],
) -> None:
    with pytest.raises(ValueError):
        measure_incoming_motion(distances, shuttle, heights, resolution)


def test_motion_requires_at_least_two_frames() -> None:
    with pytest.raises(ValueError):
        measure_incoming_motion(np.array([1.0]), np.zeros((1, 2)), np.ones(1), (100.0, 100.0))


def test_line_path_has_no_quadratic_claim_and_curve_prefers_quadratic() -> None:
    frame_numbers = np.arange(8, dtype=float)
    line_fit = fit_path(np.column_stack((0.1 * frame_numbers, 0.2 + 0.3 * frame_numbers)))
    curve_fit = fit_path(np.column_stack((0.1 * frame_numbers, 0.03 * frame_numbers**2 + 0.2)))
    short_fit = fit_path(np.column_stack((frame_numbers[:4], frame_numbers[:4] ** 2)))

    assert line_fit.linear_rmse < 1e-12
    assert line_fit.quadratic_rmse < 1e-12
    assert line_fit.quadratic_improvement < 1e-10
    assert curve_fit.quadratic_rmse < 1e-12
    assert curve_fit.quadratic_improvement > 0.99
    assert np.isnan(short_fit.quadratic_rmse)
    assert np.isnan(short_fit.quadratic_improvement)


def test_anchor_categories_cover_unique_first_second_later_ambiguous_and_unmatched() -> None:
    gt_frames = np.array([100, 110, 120])

    assert classify_anchor_frame(100, gt_frames, 0) == "contact_1"
    assert classify_anchor_frame(109, gt_frames, 1) == "contact_2"
    assert classify_anchor_frame(120, gt_frames, 0) == "later"
    assert classify_anchor_frame(105, gt_frames, 5) == "ambiguous"
    assert classify_anchor_frame(130, gt_frames, 2) == "unmatched"


def test_alignment_keeps_nearest_stroke_when_tolerance_changes() -> None:
    gt_frames = [100, 112, 136]

    strict = align_anchor_to_gt(106, gt_frames, fps=30.0, tolerance_base30=5)
    primary = align_anchor_to_gt(106, gt_frames, fps=30.0, tolerance_base30=10)
    sanity = align_anchor_to_gt(106, gt_frames, fps=30.0, tolerance_base30=30)

    assert strict == (1, 6.0, 6.0, 0, False, "unmatched")
    assert primary == (1, 6.0, 6.0, 2, True, "contact_1")
    assert sanity == (1, 6.0, 6.0, 3, True, "contact_1")


@pytest.mark.parametrize(("fps", "source_offset"), [(25.0, 8), (30.0, 10), (60.0, 20)])
def test_alignment_uses_inclusive_half_up_scaled_tolerance(
    fps: float, source_offset: int
) -> None:
    on_boundary = align_anchor_to_gt(
        100 + source_offset,
        [100],
        fps=fps,
        tolerance_base30=10,
    )
    outside = align_anchor_to_gt(
        101 + source_offset,
        [100],
        fps=fps,
        tolerance_base30=10,
    )

    assert on_boundary.label == "contact_1"
    assert on_boundary.signed_offset_base30 > 0
    assert outside.label == "unmatched"
    assert outside.nearest_gt_ordinal == 1


def test_alignment_signed_offset_is_anchor_minus_gt_frame() -> None:
    alignment = align_anchor_to_gt(94, [100, 130], fps=30.0, tolerance_base30=10)

    assert alignment.nearest_gt_ordinal == 1
    assert alignment.signed_offset_base30 == -6.0
    assert alignment.absolute_offset_base30 == 6.0


@pytest.mark.parametrize(
    "gt_frames",
    [[], [100, 100], [110, 100], np.array([[100, 110]])],
)
def test_alignment_rejects_invalid_gt_frames(gt_frames: object) -> None:
    with pytest.raises(ValueError):
        align_anchor_to_gt(100, gt_frames, fps=30.0, tolerance_base30=10)  # type: ignore[arg-type]


def test_unmatched_anchor_sequence_reports_later_serve_and_return() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80, 100, 110],
        [100, 110, 140],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary == (2, True, True, 2, 1, False, False)


def test_unmatched_anchor_sequence_reports_return_without_serve() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80, 130],
        [100, 130],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary.later_serve_within_tolerance is False
    assert summary.later_first_return_within_tolerance is True
    assert summary.first_gt_match_rank == 2
    assert summary.first_gt_match_ordinal == 2


def test_unmatched_anchor_sequence_flags_reused_gt_ordinal() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80, 99, 101],
        [100, 130],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary.first_gt_match_rank == 2
    assert summary.reused_gt_ordinal is True


def test_unmatched_anchor_sequence_keeps_multiple_flag_on_first_match() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80, 105],
        [100, 110],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary.first_gt_match_rank == 2
    assert summary.first_gt_match_ordinal == 1
    assert summary.first_gt_match_multiple is True


def test_unmatched_anchor_sequence_keeps_no_later_match_explicit() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80, 90],
        [110, 130],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary.later_contacts_checked == 1
    assert summary.first_gt_match_rank is None
    assert summary.first_gt_match_ordinal is None


def test_unmatched_anchor_sequence_accepts_no_later_contacts() -> None:
    summary = summarise_unmatched_anchor_sequence(
        [80],
        [100, 130],
        fps=30.0,
        tolerance_base30=5,
    )

    assert summary == (0, False, False, None, None, False, False)


@pytest.mark.parametrize("accepted", [[], [80, 80], [90, 80]])
def test_unmatched_anchor_sequence_rejects_missing_or_unordered_anchor(
    accepted: list[int],
) -> None:
    with pytest.raises(ValueError):
        summarise_unmatched_anchor_sequence(
            accepted,
            [100, 130],
            fps=30.0,
            tolerance_base30=5,
        )


def test_unmatched_anchor_sequence_rejects_a_matched_anchor() -> None:
    with pytest.raises(ValueError, match="first accepted contact must be unmatched"):
        summarise_unmatched_anchor_sequence(
            [100, 130],
            [100, 130],
            fps=30.0,
            tolerance_base30=5,
        )


def test_robust_distance_trend_handles_inward_outward_and_constant_paths() -> None:
    inward = fit_robust_distance_trend([5.0, 4.0, 3.0, 2.0, 1.0])
    outward = fit_robust_distance_trend([1.0, 2.0, 3.0, 4.0, 5.0])
    constant = fit_robust_distance_trend([2.0] * 5)

    assert inward.fitted_decrease_bh == 4.0
    assert np.isposinf(inward.trend_to_jitter)
    assert outward.fitted_decrease_bh == -4.0
    assert np.isneginf(outward.trend_to_jitter)
    assert constant.fitted_decrease_bh == 0.0
    assert constant.residual_rms_bh == 0.0
    assert constant.trend_to_jitter == 0.0


def test_robust_distance_trend_resists_one_bad_endpoint() -> None:
    trend = fit_robust_distance_trend([5.0, 4.0, 3.0, 2.0, 10.0])

    assert trend.fitted_decrease_bh > 0
    assert trend.residual_rms_bh > 0


def test_robust_distance_trend_accepts_two_samples() -> None:
    trend = fit_robust_distance_trend([1.0, 0.9])

    assert trend.slope_bh_per_path == pytest.approx(-0.1)
    assert trend.fitted_decrease_bh == pytest.approx(0.1)
    assert trend.residual_rms_bh == 0.0
    assert np.isposinf(trend.trend_to_jitter)


def test_robust_distance_trend_reports_noise_without_using_it_for_the_fit() -> None:
    trend = fit_robust_distance_trend([2.0, 1.90, 1.82, 1.70, 1.62])

    assert trend.fitted_decrease_bh > 0.05
    assert trend.residual_rms_bh > 0
    assert np.isfinite(trend.trend_to_jitter)


def test_robust_distance_trend_ratio_is_invariant_to_constant_rescaling() -> None:
    distances = np.array([2.0, 1.90, 1.82, 1.70, 1.62])

    original = fit_robust_distance_trend(distances)
    scaled = fit_robust_distance_trend(3.0 * distances)

    assert scaled.fitted_decrease_bh == pytest.approx(3.0 * original.fitted_decrease_bh)
    assert scaled.residual_rms_bh == pytest.approx(3.0 * original.residual_rms_bh)
    assert scaled.trend_to_jitter == pytest.approx(original.trend_to_jitter)


def test_robust_rule_does_not_inherit_historical_total_movement_floor() -> None:
    distances = [1.0, 0.975, 0.95, 0.925, 0.9]
    motion = _motion(
        distances,
        [(0.0, 0.0), (0.00375, 0.0), (0.0075, 0.0), (0.01125, 0.0), (0.015, 0.0)],
    )
    trend = fit_robust_distance_trend(distances)

    decisions = decide_fixed_motion_rules(motion, trend, 1, 2)

    assert motion.total_movement_bh == pytest.approx(0.15)
    assert decisions.common_path_eligible is True
    assert decisions.historical_path_eligible is False
    assert decisions.historical_incoming is False
    assert decisions.robust_trend_incoming is True


def test_gross_jump_gate_is_common_to_both_rules() -> None:
    motion = IncomingMotion(5, 1.0, 0.8, 0.2, 1.0, 0.3, 4.5)
    trend = RobustDistanceTrend(-0.2, 1.0, 0.2, 0.01, 20.0)

    decisions = decide_fixed_motion_rules(motion, trend, 1, 2)

    assert decisions.common_path_eligible is False
    assert decisions.historical_incoming is False
    assert decisions.robust_trend_incoming is False


def test_fixed_rule_boundaries_are_inclusive() -> None:
    motion = IncomingMotion(5, 1.0, 0.75, 0.25, 0.55, 0.25, 4.0)
    trend = RobustDistanceTrend(-0.05, 1.0, 0.05, 0.01, 5.0)

    decisions = decide_fixed_motion_rules(motion, trend, 2, 2)

    assert decisions.historical_incoming is True
    assert decisions.robust_trend_incoming is True


@pytest.mark.parametrize("distances", [[], [1.0], [1.0, np.nan], [[1.0, 0.5]]])
def test_robust_distance_trend_rejects_invalid_input(distances: object) -> None:
    with pytest.raises(ValueError):
        fit_robust_distance_trend(distances)  # type: ignore[arg-type]


def test_first_player_reuses_fitted_final_half_phase() -> None:
    assert first_player_from_final_half(Half.TOP, 3) == Half.TOP
    assert first_player_from_final_half(Half.TOP, 4) == Half.BOT
    assert first_player_from_final_half(None, 3) is None


def test_none_prepend_preserves_alternation_fit() -> None:
    guesses = [Half.TOP, Half.BOT, Half.TOP, Half.BOT]

    assert fit_alternation([None, *guesses]) == fit_alternation(guesses)


def test_labelled_prepend_can_resolve_tie_or_turn_one_vote_into_tie() -> None:
    tie = [Half.TOP, Half.TOP]
    one_vote = [Half.TOP]

    assert fit_alternation(tie) is None
    assert fit_alternation([Half.TOP, *tie]) == Half.TOP
    assert fit_alternation(one_vote) == Half.TOP
    assert fit_alternation([Half.TOP, *one_vote]) is None


def test_one_labelled_prepend_cannot_jump_between_resolved_winners() -> None:
    for guess_tuple in product((Half.TOP, Half.BOT), repeat=5):
        guesses = list(guess_tuple)
        original = fit_alternation(guesses)
        for label in (Half.TOP, Half.BOT):
            labelled = fit_alternation([label, *guesses])
            if original is not None and labelled is not None:
                assert labelled == original
