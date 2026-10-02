"""Public line-support behaviour at the compiled scorer boundary."""

import numpy as np
import pytest

from court_detector.candidate_geometry import continuous_support


@pytest.mark.parametrize("samples", [16, 64])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_support_counts_the_split_centre_as_one_marking(samples: int, dtype: type) -> None:
    homographies = np.asarray([[[10, 0, 20], [0, 10, 20], [0, 0, 1]]], dtype=dtype)
    maps = np.zeros((2, 200, 200), dtype=np.float32)
    maps[0] = 2

    scores = continuous_support(homographies, maps, (200, 200), samples)

    # Five lengthwise markings include the two centre intervals; six run cross-court.
    expected = (5 * np.exp(-0.5) + 6) / 11
    assert scores.dtype == np.float32
    np.testing.assert_allclose(scores, [expected], rtol=1e-6)


def test_support_preserves_empty_input_shape_and_dtype() -> None:
    scores = continuous_support(np.empty((0, 3, 3), dtype=np.float32),
                                np.zeros((2, 200, 200), dtype=np.float32), (200, 200))
    assert scores.shape == (0,)
    assert scores.dtype == np.float32


def test_support_returns_zero_when_no_intervals_are_visible() -> None:
    homographies = np.asarray([[[10, 0, 1000], [0, 10, 1000], [0, 0, 1]]], dtype=np.float32)
    scores = continuous_support(homographies, np.zeros((2, 200, 200), dtype=np.float32), (200, 200))
    np.testing.assert_array_equal(scores, [0])


def test_support_hides_nonfinite_projection_before_map_lookup() -> None:
    homographies = np.zeros((1, 3, 3), dtype=np.float32)
    scores = continuous_support(homographies, np.zeros((2, 200, 200), dtype=np.float32), (200, 200))
    np.testing.assert_array_equal(scores, [0])
