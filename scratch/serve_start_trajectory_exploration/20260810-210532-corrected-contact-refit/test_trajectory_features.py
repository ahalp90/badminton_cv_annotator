"""Synthetic checks for corrected serve-trajectory measurements."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from trajectory_features import (
    classify_anchor_frame,
    closest_pre_contact_run,
    first_player_from_final_half,
    fit_path,
    measure_incoming_motion,
)

from annotator.point_winner import Half, fit_alternation


def _motion(distances: list[float], points: list[tuple[float, float]]) -> object:
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
    assert run.gap_frames == 4


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
