"""Stage 5 B1 serve-gate primitives."""
from __future__ import annotations

import numpy as np
import pytest

from annotator.rally_segmentation import ServeSetupInputs, series_drift, serve_setup_still
from annotator.types import Slot


def _inputs(
    n: int = 4,
    *,
    top_ankles: np.ndarray | None = None,
    bot_ankles: np.ndarray | None = None,
    top_height: np.ndarray | None = None,
    bot_height: np.ndarray | None = None,
) -> ServeSetupInputs:
    shape = (n, 2)
    return ServeSetupInputs(
        count=np.ones(n, dtype=int),
        wrist_dist=np.ones(shape, dtype=float),
        analysed=np.ones(n, dtype=bool),
        top_ankles=np.full(shape, (0.2, 0.3), dtype=float) if top_ankles is None else top_ankles,
        bot_ankles=np.full(shape, (0.7, 0.3), dtype=float) if bot_ankles is None else bot_ankles,
        top_height=np.full(n, 0.2, dtype=float) if top_height is None else top_height,
        bot_height=np.full(n, 0.2, dtype=float) if bot_height is None else bot_height,
    )


def test_series_drift_sentinel_and_partial_zero_detection() -> None:
    points = np.array([[0, 0], [0, 2], [3, 0], [np.nan, 1]], dtype=float)
    drift, count = series_drift(points)
    assert count == 2
    assert drift == pytest.approx(np.sqrt(13.0))


def test_series_drift_odd_split_assigns_extra_sample_to_first_half() -> None:
    # Asymmetric spacing: the ceiling split {0, 10 | 30} gives 25; the wrong
    # floor split {0 | 10, 30} would give 20.
    points = np.array([[0.0, 1.0], [10.0, 1.0], [30.0, 1.0]])
    drift, count = series_drift(points)
    assert count == 3
    assert drift == pytest.approx(25.0)


@pytest.mark.parametrize(
    'points,expected_count',
    [
        (np.array([[0, 0], [np.nan, 1], [2, np.nan]], dtype=float), 0),
        (np.array([[0, 0], [np.nan, 1], [2.0, 3.0]], dtype=float), 1),
    ],
)
def test_series_drift_below_two_detected_returns_nan_and_count(
    points: np.ndarray, expected_count: int,
) -> None:
    drift, count = series_drift(points)
    assert np.isnan(drift)
    assert count == expected_count


@pytest.mark.parametrize('points', [np.ones(2), np.ones((2, 3)), np.ones((1, 2, 1)), np.array([['x', 'y']])])
def test_series_drift_rejects_shape_rank_and_dtype(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        series_drift(points)


def test_serve_setup_still_requires_each_player_and_fails_closed_on_nan() -> None:
    inputs = _inputs(top_ankles=np.full((4, 2), np.nan))
    assert not serve_setup_still(inputs, 3, 4, 1.0, (Slot.TOP, Slot.BOTTOM))


def test_serve_setup_still_claimed_frame_is_inclusive() -> None:
    ankles = np.full((3, 2), (0.2, 0.3), dtype=float)
    ankles[2] = (0.4, 0.3)
    inputs = _inputs(3, top_ankles=ankles)
    assert serve_setup_still(inputs, 1, 2, 0.1, (Slot.TOP,))
    assert not serve_setup_still(inputs, 2, 3, 0.1, (Slot.TOP,))


def test_serve_setup_still_clips_window_at_frame_zero() -> None:
    assert not serve_setup_still(_inputs(3), 0, 20, 1.0, (Slot.TOP,))


def test_serve_setup_still_nonpositive_body_unit_fails_closed() -> None:
    inputs = _inputs(top_height=np.zeros(4))
    assert not serve_setup_still(inputs, 3, 4, 1.0, (Slot.TOP,))


def test_serve_setup_still_fails_when_one_player_is_over_threshold() -> None:
    bot = np.full((4, 2), (0.7, 0.3), dtype=float)
    bot[3] = (1.0, 0.3)
    assert not serve_setup_still(_inputs(bot_ankles=bot), 3, 4, 0.5, (Slot.TOP, Slot.BOTTOM))


@pytest.mark.parametrize('window', [0, -1, 1.5, True])
def test_serve_setup_still_rejects_invalid_window(window: object) -> None:
    with pytest.raises(ValueError):
        serve_setup_still(_inputs(), 2, window, 1.0, (Slot.TOP,))  # type: ignore[arg-type]


@pytest.mark.parametrize('frame', [-1, 4, 1.5, True])
def test_serve_setup_still_rejects_invalid_claimed_frame(frame: object) -> None:
    with pytest.raises(ValueError):
        serve_setup_still(_inputs(), frame, 2, 1.0, (Slot.TOP,))  # type: ignore[arg-type]


@pytest.mark.parametrize('threshold', [-1.0, np.nan, np.inf])
def test_serve_setup_still_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        serve_setup_still(_inputs(), 2, 2, threshold, (Slot.TOP,))


@pytest.mark.parametrize('slots', [(), (Slot.TOP, Slot.TOP), (0,), (Slot.TOP, 1), [Slot.TOP], ([],)])
def test_serve_setup_still_rejects_invalid_slots(slots: object) -> None:
    with pytest.raises(ValueError):
        serve_setup_still(_inputs(), 2, 2, 1.0, slots)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'field,value',
    [
        ('count', np.ones(4, dtype=bool)),
        ('count', np.array([1.0, np.nan, 1.0, 1.0])),
        ('count', np.array([1.0, np.inf, 1.0, 1.0])),
        ('count', np.array([1.0, -1.0, 1.0, 1.0])),
        ('count', np.array([1.0, 1.5, 1.0, 1.0])),
        ('analysed', np.ones(4, dtype=int)),
        ('top_height', np.ones(4, dtype=int)),
    ],
)
def test_serve_setup_inputs_validate_rejects_bad_dtypes_and_counts(field: str, value: np.ndarray) -> None:
    values = _inputs()._asdict()
    values[field] = value
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()


def test_serve_setup_inputs_validate_rejects_wrong_shapes_and_lengths() -> None:
    values = _inputs()._asdict()
    values['wrist_dist'] = np.ones(4, dtype=float)
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()
    values = _inputs()._asdict()
    values['bot_height'] = np.ones(3, dtype=float)
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()
