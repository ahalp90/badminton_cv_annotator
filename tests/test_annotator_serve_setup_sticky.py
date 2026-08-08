"""Sticky-sourced serve-setup coverage."""
from __future__ import annotations

import numpy as np
import pytest

from annotator.rally.serve import _sticky_serve_setup_before
from annotator.rally_segmentation import (
    ServeSetupInputs,
    ServeStartMode,
    ServeStartOptions,
    StickyResult,
    build_serve_setup_inputs,
    find_rally_spans,
)
from annotator.types import Slot


@pytest.fixture
def make_setup(serve_setup_defaults):
    def build(count: float, *, n: int = 4) -> ServeSetupInputs:
        return ServeSetupInputs(
            count=np.full(n, count), wrist_dist=np.full((n, 2), (0.02, 0.08)),
            **serve_setup_defaults(n),
        )

    return build


@pytest.mark.parametrize('count', (0.0, 0.5, 1.0, 1.5, 2.0))
def test_sticky_lanes_route_each_bound_median(make_setup, count: float) -> None:
    setup = make_setup(count)
    expected = count >= 1.0
    assert _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None) is expected


def test_sticky_coverage_fails_closed_and_stillness_can_be_off(make_setup) -> None:
    setup = make_setup(2.0)
    setup = setup._replace(analysed=np.array([True, False, True, True]))
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)
    assert _sticky_serve_setup_before(make_setup(2.0), 3, 0.3, 4, None, None)


def test_partial_lane_rejects_alternating_cross_slot_minimum(make_setup) -> None:
    setup = make_setup(1.0)
    distances = np.array([[0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    setup = setup._replace(wrist_dist=distances)
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)


def test_standard_lane_accepts_either_slot_when_its_ratio_passes(make_setup) -> None:
    setup = make_setup(2.0)._replace(wrist_dist=np.full((4, 2), (0.02, 0.8)))
    assert _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)


def test_standard_lane_pairs_each_distance_with_its_own_height(make_setup) -> None:
    setup = make_setup(2.0)._replace(
        wrist_dist=np.full((4, 2), 0.4),
        top_height=np.full(4, 0.4), bot_height=np.full(4, 0.8),
    )
    # Top's ratio is 1.0 and bottom's is 0.5. The bottom slot passes at 0.55;
    # pooled distance and height evidence would incorrectly fail at 0.4 / 0.6.
    assert _sticky_serve_setup_before(setup, 3, 0.55, 4, None, None)


def test_sticky_gate_ignores_invisible_corner_garbage(make_setup) -> None:
    setup = make_setup(1.0, n=6)
    wrist_dist = np.full((6, 2), np.nan)
    wrist_dist[4] = 0.12  # one visible setup frame; the first four are invisible
    new_setup = setup._replace(wrist_dist=wrist_dist)
    old_corner_setup = setup._replace(
        wrist_dist=np.where(np.isfinite(wrist_dist), wrist_dist, 0.02),
    )

    assert not _sticky_serve_setup_before(new_setup, 5, 0.3, 6, None, None)
    # The old cache's finite corner distances make the same window pass.
    assert _sticky_serve_setup_before(old_corner_setup, 5, 0.3, 6, None, None)


def test_sticky_distance_window_excludes_burst_frame(make_setup) -> None:
    setup = make_setup(1.0)
    wrist_dist = np.full((4, 2), 0.4)
    wrist_dist[3] = 0.01
    assert not _sticky_serve_setup_before(
        setup._replace(wrist_dist=wrist_dist), 3, 0.3, 4, None, None,
    )


def test_burst_frame_count_cannot_change_lane_selection(make_setup) -> None:
    setup = make_setup(1.0, n=3)._replace(
        count=np.array([1.0, 2.0, 2.0]),
        wrist_dist=np.array([[0.02, np.nan], [0.02, np.nan], [0.02, np.nan]]),
    )
    # The burst row would route the inclusive window into the >=2 lane, where the
    # absent bottom slot fails the presence floor. The exclusive setup window is >=1.
    assert _sticky_serve_setup_before(setup, 2, 0.3, 2, None, None)


def test_burst_analysed_row_is_ignored_without_stillness_and_required_with_it(make_setup) -> None:
    setup = make_setup(2.0)._replace(
        analysed=np.array([True, True, True, False]),
        wrist_dist=np.full((4, 2), 0.02),
    )
    assert _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, 0.5, 1)


def test_empty_setup_window_returns_before_downstream_medians(
    make_setup, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def poisoned_median(*_args: object, **_kwargs: object) -> None:
        pytest.fail('empty setup window must return before median work')

    monkeypatch.setattr(np, 'median', poisoned_median)
    assert not _sticky_serve_setup_before(make_setup(1.0), 0, 0.3, 4, None, None)


def test_standard_lane_presence_floor_keeps_absent_slot_from_being_rescued(make_setup) -> None:
    setup = make_setup(2.0)._replace(wrist_dist=np.column_stack([np.full(4, 0.02), np.full(4, np.nan)]))
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)


def test_standard_lane_stillness_rejects_a_player_even_when_distance_passes(make_setup) -> None:
    bot_ankles = np.full((4, 2), (0.7, 0.3))
    bot_ankles[3] = (1.0, 0.3)
    setup = make_setup(2.0)._replace(
        wrist_dist=np.full((4, 2), 0.02), bot_ankles=bot_ankles,
    )
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, 0.1, 4)


def test_builder_converts_and_preserves_sentinels() -> None:
    # Frame 0: both slots picked (real ankles, positive heights). Frame 1: analysed,
    # no picks (all-NaN row). Frame 2: outside segments (+inf distance sentinel).
    sticky = StickyResult(
        distances=np.zeros(3), picks=np.full((3, 2), -1), standing_count=np.array([2, 0, 0]),
        ankle_pos=np.array([
            [[0.1, 0.2], [0.2, 0.3]],
            [[np.nan, np.nan], [np.nan, np.nan]],
            [[np.nan, np.nan], [np.nan, np.nan]],
        ]),
        bbox_height=np.array([[100.0, 300.0], [np.nan, np.nan], [np.nan, np.nan]]),
        distances_per_slot=np.array([[0.1, 0.3], [np.nan, np.nan], [np.inf, np.inf]]),
        wrist_dist_px=np.array([[50.0, 200.0], [np.nan, np.nan], [np.inf, np.inf]]),
        analysed=np.array([True, True, False]),
    )
    inputs = build_serve_setup_inputs(sticky, (1000.0, 500.0))
    assert not hasattr(inputs, 'distances')
    np.testing.assert_allclose(inputs.top_ankles[0], [0.1, 0.2])
    np.testing.assert_allclose(inputs.bot_ankles[0], [0.2, 0.3])
    assert inputs.top_height[0] == pytest.approx(0.2)
    assert inputs.bot_height[0] == pytest.approx(0.6)
    assert inputs.wrist_dist[0, Slot.TOP] == pytest.approx(0.1)
    assert inputs.wrist_dist[0, Slot.BOTTOM] == pytest.approx(0.4)
    assert np.isnan(inputs.wrist_dist[1]).all()
    assert np.isnan(inputs.top_height[1]) and np.isnan(inputs.bot_height[1])
    assert np.isposinf(inputs.wrist_dist[2]).all()
    assert inputs.analysed.tolist() == [True, True, False]


@pytest.mark.parametrize('resolution', [(0.0, 10.0), (10.0, np.nan), (10.0,), [10.0, 10.0]])
def test_builder_rejects_bad_resolution(resolution: object) -> None:
    sticky = StickyResult(
        np.zeros(1), np.full((1, 2), -1), np.zeros(1, dtype=int),
        np.full((1, 2, 2), np.nan), np.full((1, 2), np.nan),
        np.full((1, 2), np.nan), np.full((1, 2), np.nan), np.zeros(1, dtype=bool),
    )
    with pytest.raises(ValueError):
        build_serve_setup_inputs(sticky, resolution)  # type: ignore[arg-type]


def test_dispatch_validates_options_cross_fields(make_setup) -> None:
    track = np.zeros((6, 3))
    track[:, 2] = 1
    setup = make_setup(2.0, n=6)

    def options(**overrides: object) -> ServeStartOptions:
        fields = dict(dist=None, threshold=0.3, mode=ServeStartMode.TRIM, setup=setup,
                      lookback_frames=4)
        fields.update(overrides)
        return ServeStartOptions(**fields)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='setup must be supplied'):
        find_rally_spans(track, serve_start=options(setup=None))
    with pytest.raises(ValueError, match='legacy serve-start dist'):
        find_rally_spans(track, serve_start=options(dist=np.zeros(6)))
    with pytest.raises(ValueError, match='lookback_frames'):
        find_rally_spans(track, serve_start=options(lookback_frames=None))
    with pytest.raises(ValueError, match='stillness_window_frames'):
        find_rally_spans(track, serve_start=options(stillness_threshold_bh=0.2))
    with pytest.raises(ValueError, match='stillness_window_frames'):
        find_rally_spans(track, serve_start=options(stillness_window_frames=-3))
    with pytest.raises(ValueError, match='threshold'):
        find_rally_spans(track, serve_start=options(threshold=-1.0))


@pytest.mark.parametrize('count', (2.0, 1.0))
def test_one_row_clipped_window_fails_closed_in_both_lanes(make_setup, count: float) -> None:
    # Claimed frame 0 clips the window to a single row: below the primitive's
    # two-detection floor, so both lanes fail even with the stillness gate off.
    setup = make_setup(count)
    assert _sticky_serve_setup_before(setup, 0, 0.3, 4, None, None) is False
