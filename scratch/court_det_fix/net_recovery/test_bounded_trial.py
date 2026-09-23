"""Focused checks for the frozen post-base scoring rule."""

from __future__ import annotations

import numpy as np
from bounded_trial import choose, post_features, reward, supported

POST = np.asarray([[10.0, 20.0], [10.0, -3.0]])
SIZE = (30, 30)


def segment(first_y: float, last_y: float) -> np.ndarray:
    return np.asarray([[10.0, first_y, 10.0, last_y]])


def test_lower_samples_are_base_first_and_union_fragment_ids() -> None:
    fragments = np.asarray([[10.0, 20.0, 10.0, 18.0], [10.0, 17.0, 10.0, 15.0]])
    feature = post_features(POST, fragments, SIZE)
    assert np.allclose(np.asarray(feature["samples_working_px"])[:, 1],
                       20 - np.arange(6))
    assert feature["covering_ids"] == [0, 1]
    assert feature["visible_count"] == 6
    assert feature["covered_count"] == 6


def test_endpoint_reversal_and_below_base_veto() -> None:
    forward = post_features(POST, segment(18, 24), SIZE)
    reversed_fragment = post_features(POST, segment(24, 18), SIZE)
    assert forward["covering_ids"] == reversed_fragment["covering_ids"] == [0]
    assert forward["lowest_endpoint_offset_working_px"] == reversed_fragment["lowest_endpoint_offset_working_px"]
    assert not supported(forward, 2)
    assert supported(forward, 4)


def test_off_image_base_and_no_fragments_are_neutral() -> None:
    off_image = post_features(POST + [25, 0], segment(18, 24) + [25, 0, 25, 0], SIZE)
    empty = post_features(POST, np.empty((0, 4)), SIZE)
    assert off_image["covering_ids"] == []
    assert not supported(off_image, 8)
    assert empty["covering_ids"] == []
    assert reward({"post_left": off_image, "post_right": empty}, 4) == 0


def test_two_posts_yield_half_step_rewards() -> None:
    good = post_features(POST, segment(20, 15), SIZE)
    absent = post_features(POST, np.empty((0, 4)), SIZE)
    assert reward({"post_left": absent, "post_right": absent}, 4) == 0
    assert reward({"post_left": good, "post_right": absent}, 4) == 0.5
    assert reward({"post_left": good, "post_right": good}, 4) == 1


def test_rank_ties_and_zero_weight_keep_baseline() -> None:
    good = post_features(POST, segment(20, 15), SIZE)
    absent = post_features(POST, np.empty((0, 4)), SIZE)
    rows = [
        {"origin_key": "first", "historical_fullcourt": True, "full_court_rank": 1,
         "paint_score": 0.4, "net_state": "measured", "posts": {"post_left": absent, "post_right": absent}},
        {"origin_key": "second", "historical_fullcourt": True, "full_court_rank": 2,
         "paint_score": 0.38, "net_state": "measured", "posts": {"post_left": good, "post_right": absent}},
    ]
    assert choose(rows, 0, 4)[0] == "first"
    assert choose(rows, 0.04, 4)[0] == "first"
    assert choose(rows, 0.08, 4)[0] == "second"
    assert all(0 <= row["bonus"] <= 0.08 for row in choose(rows, 0.08, 4)[1])
