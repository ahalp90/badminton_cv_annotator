"""Stage 5 B2 sticky-sourced serve setup coverage."""
from __future__ import annotations

import numpy as np
import pytest

from annotator.rally_segmentation import (
    ServeSetupInputs,
    ServeStartMode,
    ServeStartOptions,
    StickyResult,
    _sticky_serve_setup_before,
    build_serve_setup_inputs,
    find_rally_spans,
)
from annotator.types import Slot


def _setup(count: float, *, n: int = 4) -> ServeSetupInputs:
    return ServeSetupInputs(
        count=np.full(n, count), distances=np.full((n, 2), (0.2, 0.8)),
        analysed=np.ones(n, dtype=bool),
        top_ankles=np.full((n, 2), (0.2, 0.3)),
        bot_ankles=np.full((n, 2), (0.7, 0.3)),
        top_height=np.full(n, 0.2), bot_height=np.full(n, 0.2),
    )


@pytest.mark.parametrize('count', (0.0, 0.5, 1.0, 1.5, 2.0))
def test_sticky_lanes_route_each_bound_median(count: float) -> None:
    setup = _setup(count)
    expected = count >= 1.0
    assert _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None) is expected


def test_sticky_coverage_fails_closed_and_stillness_can_be_off() -> None:
    setup = _setup(2.0)
    setup = setup._replace(analysed=np.array([True, False, True, True]))
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)
    assert _sticky_serve_setup_before(_setup(2.0), 3, 0.3, 4, None, None)


def test_partial_lane_requires_one_coherent_slot() -> None:
    setup = _setup(1.0)
    distances = np.array([[0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2]])
    setup = setup._replace(distances=distances)
    assert not _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)


def test_standard_lane_uses_nearest_of_two_distances() -> None:
    setup = _setup(2.0)._replace(distances=np.array([
        [0.2, 0.8], [0.8, 0.2], [0.2, 0.8], [0.8, 0.2],
    ]))
    assert _sticky_serve_setup_before(setup, 3, 0.3, 4, None, None)


def test_builder_enforces_paired_zero_and_height_fill_rule() -> None:
    sticky = StickyResult(
        distances=np.zeros(2), picks=np.full((2, 2), -1), standing_count=np.array([1, 2]),
        ankle_pos=np.array([[[0.0, 0.0], [0.2, 0.3]], [[0.1, 0.2], [0.4, 0.5]]]),
        bbox_height=np.array([[100.0, 0.0], [200.0, 300.0]]),
        distances_per_slot=np.array([[0.1, np.nan], [0.2, 0.3]]),
        analysed=np.array([True, False]),
    )
    inputs = build_serve_setup_inputs(sticky, (1000.0, 500.0))
    assert np.isnan(inputs.top_ankles[0]).all()
    assert np.isnan(inputs.top_height[0])
    assert np.isnan(inputs.bot_ankles[0]).all()
    assert np.isnan(inputs.bot_height[0])
    assert inputs.top_height[1] == pytest.approx(0.4)
    assert inputs.bot_height[1] == pytest.approx(0.6)
    assert np.isnan(inputs.distances[0, Slot.BOTTOM])
    assert inputs.distances[1, Slot.TOP] == pytest.approx(0.2)
    assert inputs.analysed.tolist() == [True, False]


@pytest.mark.parametrize('resolution', [(0.0, 10.0), (10.0, np.nan), (10.0,), [10.0, 10.0]])
def test_builder_rejects_bad_resolution(resolution: object) -> None:
    sticky = StickyResult(
        np.zeros(1), np.full((1, 2), -1), np.zeros(1, dtype=int),
        np.full((1, 2, 2), np.nan), np.full((1, 2), np.nan),
        np.full((1, 2), np.nan), np.zeros(1, dtype=bool),
    )
    with pytest.raises(ValueError):
        build_serve_setup_inputs(sticky, resolution)  # type: ignore[arg-type]


def test_dispatch_validates_options_cross_fields() -> None:
    track = np.zeros((6, 3))
    track[:, 2] = 1
    setup = _setup(2.0, n=6)

    def options(**overrides: object) -> ServeStartOptions:
        fields = dict(dist=None, threshold=0.3, mode=ServeStartMode.TRIM, setup=setup,
                      lookback_frames=4)
        fields.update(overrides)
        return ServeStartOptions(**fields)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='exactly one'):
        find_rally_spans(track, serve_start=options(setup=None))
    with pytest.raises(ValueError, match='exactly one'):
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
def test_one_row_clipped_window_fails_closed_in_both_lanes(count: float) -> None:
    # Claimed frame 0 clips the window to a single row: below the primitive's
    # two-detection floor, so both lanes fail even with the stillness gate off.
    setup = _setup(count)
    assert _sticky_serve_setup_before(setup, 0, 0.3, 4, None, None) is False
