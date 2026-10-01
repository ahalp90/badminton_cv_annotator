"""Fixed contact inputs, candidate filtering and suppression behaviour."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.contacts.features import REGION_FIELDS
from annotator.contacts.model import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    contact_feature_matrix,
    remove_nearby_contacts,
    score_contact_features,
)


def _rows() -> np.ndarray:
    dtype = [('frame', 'i4'), ('interval_id', 'i4')]
    dtype.extend((name, 'u1') for name in REGION_FIELDS)
    dtype.extend((name, 'f4') for name in CONTACT_FEATURE_NAMES)
    rows = np.zeros(4, dtype=dtype)
    rows['frame'] = [10, 16, 23, 50]
    rows['region_current_raw'][:3] = 1
    for index, name in enumerate(CONTACT_FEATURE_NAMES):
        rows[name] = index
    rows[CONTACT_FEATURE_NAMES[0]][1] = np.nan
    return rows


def test_matrix_order_and_missing_values() -> None:
    matrix = contact_feature_matrix(_rows())
    assert len(CONTACT_FEATURE_NAMES) == 85
    assert matrix.shape == (4, 85)
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], np.arange(85))
    assert np.isnan(matrix[1, 0])


def test_suppression_ties_boundaries_and_separate_intervals() -> None:
    frames = np.array([16, 10, 22, 10, 40])
    intervals = np.array([0, 0, 0, 1, 0])
    scores = np.array([0.95, 0.95, 0.95, 0.95, 0.89])
    kept = remove_nearby_contacts(frames, intervals, scores, 0.9, 6)
    np.testing.assert_array_equal(kept, [1, 3, 2])


@pytest.mark.parametrize(('fps', 'expected'), [(25.0, [10, 16, 23]), (30.0, [10, 23]), (60.0, [10, 23])])
def test_region_filter_cutoff_and_fps_scaling(fps: float, expected: list[int]) -> None:
    observed = []
    def predict(matrix: np.ndarray) -> np.ndarray:
        observed.append(matrix)
        return np.tile([0.1, 0.9], (len(matrix), 1))
    model = SimpleNamespace(predict_proba=predict, feature_names_in_=CONTACT_FEATURE_NAMES, n_features_in_=85)
    result = score_contact_features(_rows(), model, fps)
    np.testing.assert_array_equal(result.frames, expected)
    assert len(result.probabilities) == 3
    assert observed[0].shape == (3, 85)
    assert np.isnan(observed[0][1, 0])


@pytest.mark.parametrize('probability', [np.nan, np.inf, 1.1, -0.1])
def test_invalid_probabilities_fail(probability: float) -> None:
    model = SimpleNamespace(predict_proba=lambda matrix: np.tile([0, probability], (len(matrix), 1)))
    with pytest.raises(ValueError, match='probabilities'):
        score_contact_features(_rows(), model, 30.0)


def test_model_schema_and_empty_candidates() -> None:
    with pytest.raises(ValueError, match='feature names'):
        score_contact_features(_rows(), SimpleNamespace(feature_names_in_=CONTACT_FEATURE_NAMES[::-1]), 30.0)
    rows = _rows()
    for name in REGION_FIELDS:
        rows[name] = 0
    result = score_contact_features(rows, SimpleNamespace(), 30.0, ContactModelConfig())
    assert len(result.frames) == 0
    assert result.probabilities.dtype == np.float64
