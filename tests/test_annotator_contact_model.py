"""Fixed contact inputs, candidate filtering and suppression behaviour."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.contacts.build_features import ContactFeatures, build_contact_features
from annotator.contacts.score_candidates import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    contact_feature_matrix,
    positive_probabilities,
    remove_nearby_contacts,
    score_contact_features,
)
from annotator.training.trees import TreeFitConfig, fit_binary_tree
from annotator.types import TrackedPlayers, WristGaps


def _rows() -> np.ndarray:
    dtype = [('frame', 'i4'), ('interval_id', 'i4')]
    dtype.extend((name, 'f4') for name in CONTACT_FEATURE_NAMES)
    rows = np.zeros(4, dtype=dtype)
    rows['frame'] = [10, 16, 23, 50]
    for index, name in enumerate(CONTACT_FEATURE_NAMES):
        rows[name] = index
    rows[CONTACT_FEATURE_NAMES[0]][1] = np.nan
    return rows


def test_matrix_order_and_missing_values() -> None:
    matrix = contact_feature_matrix(_rows())
    assert matrix.shape == (4, len(CONTACT_FEATURE_NAMES))
    assert matrix.dtype == np.float32
    np.testing.assert_array_equal(matrix[0], np.arange(len(CONTACT_FEATURE_NAMES)))
    assert np.isnan(matrix[1, 0])


def test_suppression_ties_boundaries_and_separate_intervals() -> None:
    frames = np.array([16, 10, 22, 10, 40])
    intervals = np.array([0, 0, 0, 1, 0])
    scores = np.array([0.95, 0.95, 0.95, 0.95, 0.89])
    kept = remove_nearby_contacts(frames, intervals, scores, 0.9, 6)
    np.testing.assert_array_equal(kept, [1, 3, 2])


@pytest.mark.parametrize(('fps', 'expected'), [(25.0, [10, 16, 23]), (30.0, [10, 23]), (60.0, [10, 23])])
def test_candidate_cutoff_and_fps_scaling(fps: float, expected: list[int]) -> None:
    observed = []
    def predict(matrix: np.ndarray) -> np.ndarray:
        observed.append(matrix)
        return np.tile([0.1, 0.9], (len(matrix), 1))
    model = SimpleNamespace(predict_proba=predict)
    result = score_contact_features(_rows()[:3], model, fps)
    np.testing.assert_array_equal(result.frames, expected)
    assert len(result.probabilities) == 3
    assert result.feature_matrix is observed[0]
    assert result.feature_matrix.shape == (3, len(CONTACT_FEATURE_NAMES))
    assert np.isnan(observed[0][1, 0])


@pytest.mark.parametrize('probability', [np.nan, np.inf, 1.1, -0.1])
def test_invalid_probabilities_fail(probability: float) -> None:
    model = SimpleNamespace(predict_proba=lambda matrix: np.tile([0, probability], (len(matrix), 1)))
    with pytest.raises(ValueError, match='probabilities'):
        score_contact_features(_rows(), model, 30.0)


def test_binary_tree_scoring_keeps_model_precision() -> None:
    predictions = np.array([[0.1, 0.9], [0.4, 0.6]], dtype=np.float32)
    model = SimpleNamespace(predict_proba=lambda _matrix: predictions)

    probabilities = positive_probabilities(model, np.zeros((2, len(CONTACT_FEATURE_NAMES)), dtype=np.float32))

    assert probabilities.dtype == np.float32
    np.testing.assert_allclose(probabilities, [0.9, 0.6])


def test_contact_cutoff_compares_model_values_in_float64() -> None:
    predictions = np.array([[0.1, 0.9], [0.05, 0.95]], dtype=np.float32)
    model = SimpleNamespace(predict_proba=lambda _matrix: predictions)

    result = score_contact_features(_rows()[[0, 2]], model, 30.0)

    # The float32 representation of 0.9 is below the configured float64 cutoff.
    assert result.probabilities.dtype == np.float64
    assert result.probabilities[0] < 0.9
    np.testing.assert_array_equal(result.frames, [23])


def test_wrong_model_input_count_fails_during_prediction() -> None:
    model = fit_binary_tree(
        np.array([[0.0], [1.0]], dtype=np.float32), np.array([0, 1]),
        TreeFitConfig(max_iter=1, min_child_weight=0), balanced=False,
    )
    with pytest.raises(ValueError, match='shape mismatch'):
        score_contact_features(_rows(), model, 30.0)


def test_empty_candidates_skip_prediction() -> None:
    rows = _rows()[:0]
    result = score_contact_features(rows, SimpleNamespace(), 30.0, ContactModelConfig())
    assert len(result.frames) == 0
    assert result.feature_matrix.shape == (0, len(CONTACT_FEATURE_NAMES))
    assert result.probabilities.dtype == np.float64


def _nomination_features(fps: float, nomination_frame: int) -> ContactFeatures:
    frame_count = 180
    track = np.column_stack((np.full(frame_count, .5), np.full(frame_count, .5), np.ones(frame_count)))
    picks = np.full((frame_count, 2), -1)
    picks[nomination_frame, 0] = 0
    gaps = np.full((frame_count, 2), np.nan)
    players = TrackedPlayers(
        picks, np.full((frame_count, 2, 2), np.nan), np.full((frame_count, 2), np.nan), np.full(frame_count, np.nan),
    )
    wrist_gaps = WristGaps(np.full(frame_count, np.nan), gaps)
    return build_contact_features(
        track, np.zeros((frame_count, 2, 17, 2)), players, wrist_gaps, [(100, 150)],
        np.zeros(frame_count, dtype=bool), [(100, 150)], [70], [(100, 150)], fps, (1280.0, 720.0),
    )
