"""Serve movement and input validation."""
from __future__ import annotations

import numpy as np
import pytest

from annotator.rally.serve import ServeSetupInputs, player_is_still, series_drift


@pytest.fixture
def make_inputs(serve_setup_defaults):
    def build(
        n: int = 4,
        *,
        top_ankles: np.ndarray | None = None,
        bot_ankles: np.ndarray | None = None,
        top_height: np.ndarray | None = None,
        bot_height: np.ndarray | None = None,
    ) -> ServeSetupInputs:
        shape = (n, 2)
        defaults = serve_setup_defaults(n)
        defaults.update({
            'top_ankles': defaults['top_ankles'] if top_ankles is None else top_ankles,
            'bot_ankles': defaults['bot_ankles'] if bot_ankles is None else bot_ankles,
            'top_height': defaults['top_height'] if top_height is None else top_height,
            'bot_height': defaults['bot_height'] if bot_height is None else bot_height,
        })
        return ServeSetupInputs(
            count=np.ones(n, dtype=int),
            wrist_dist=np.ones(shape, dtype=float),
            **defaults,
        )

    return build


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


def test_player_stillness_rejects_missing_positions(make_inputs) -> None:
    inputs = make_inputs(top_ankles=np.full((4, 2), np.nan))
    assert not player_is_still(inputs.top_ankles, inputs.top_height, 1.0)


@pytest.mark.parametrize('body_height', (0.0, -1.0, np.nan))
def test_player_stillness_requires_positive_body_height(make_inputs, body_height: float) -> None:
    inputs = make_inputs(top_height=np.full(4, body_height))
    assert not player_is_still(inputs.top_ankles, inputs.top_height, 1.0)


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
def test_serve_setup_inputs_validate_rejects_bad_dtypes_and_counts(
    make_inputs, field: str, value: np.ndarray,
) -> None:
    values = make_inputs()._asdict()
    values[field] = value
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()


def test_serve_setup_inputs_validate_rejects_wrong_shapes_and_lengths(make_inputs) -> None:
    values = make_inputs()._asdict()
    values['wrist_dist'] = np.ones(4, dtype=float)
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()
    values = make_inputs()._asdict()
    values['bot_height'] = np.ones(3, dtype=float)
    with pytest.raises(ValueError):
        ServeSetupInputs(**values).validate()
