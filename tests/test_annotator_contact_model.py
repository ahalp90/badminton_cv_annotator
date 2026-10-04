"""Fixed contact inputs, candidate filtering and suppression behaviour."""

from types import SimpleNamespace

import numpy as np
import pytest

from annotator.contacts.features import (
    REGION_FIELDS,
    WINDOW_OFFSETS_BASE30,
    ContactFeatures,
    build_contact_features,
)
from annotator.contacts.model import (
    CONTACT_FEATURE_NAMES,
    ContactModelConfig,
    contact_feature_matrix,
    remove_nearby_contacts,
    score_contact_features,
)
from annotator.types import StickyResult


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


@pytest.mark.parametrize('side', ['top', 'bot'])
@pytest.mark.parametrize('offset', WINDOW_OFFSETS_BASE30)
def test_masked_rejection_requires_missing_nominated_players(side: str, offset: int) -> None:
    rows = _rows()
    for name in CONTACT_FEATURE_NAMES:
        rows[name] = np.nan
    rows[f'pose_valid_{side}_t{offset:+d}'][1] = 1.0
    # Impulse and wrist evidence do not substitute for a nominated player.
    rows['shuttle_impulse_ratio_t+0'] = 100.0
    rows['wrist_valid_top_t+0'] = 1.0
    mask = np.zeros(51, dtype=bool)
    mask[[10, 16]] = True
    model = SimpleNamespace(predict_proba=lambda matrix: np.tile([.05, .95], (len(matrix), 1)))
    result = score_contact_features(
        rows, model, 30.0, ContactModelConfig(reject_masked_without_player=True),
        shuttle_hallucination_mask=mask,
    )
    np.testing.assert_array_equal(result.candidates['frame'], [16, 23])
    np.testing.assert_array_equal(result.frames, [16, 23])
    baseline = score_contact_features(rows, model, 30.0, shuttle_hallucination_mask=mask)
    np.testing.assert_array_equal(baseline.candidates['frame'], [10, 16, 23])


def test_masked_rejection_promotes_neighbour_before_suppression() -> None:
    rows = _rows()
    for side in ('top', 'bot'):
        for offset in WINDOW_OFFSETS_BASE30:
            rows[f'pose_valid_{side}_t{offset:+d}'] = 0.0
    rows['shuttle_speed_t+0'] = [.99, .95, .8, 0.0]
    index = CONTACT_FEATURE_NAMES.index('shuttle_speed_t+0')
    observed = []

    def predict(matrix: np.ndarray) -> np.ndarray:
        observed.append(matrix)
        probability = matrix[:, index]
        return np.column_stack((1.0 - probability, probability))

    mask = np.zeros(51, dtype=bool)
    mask[10] = True
    result = score_contact_features(
        rows, SimpleNamespace(predict_proba=predict), 30.0,
        ContactModelConfig(reject_masked_without_player=True), shuttle_hallucination_mask=mask,
    )
    np.testing.assert_array_equal(result.frames, [16])
    np.testing.assert_array_equal(result.candidates['frame'], [16, 23])
    assert observed[0].shape == (2, 85)


def test_masked_policy_requires_mask_and_handles_all_rejected_rows() -> None:
    rows = _rows()
    for side in ('top', 'bot'):
        for offset in WINDOW_OFFSETS_BASE30:
            rows[f'pose_valid_{side}_t{offset:+d}'] = 0.0
    config = ContactModelConfig(reject_masked_without_player=True)
    with pytest.raises(ValueError, match='requires the shuttle hallucination mask'):
        score_contact_features(rows, SimpleNamespace(), 30.0, config)
    result = score_contact_features(
        rows, SimpleNamespace(), 30.0, config, shuttle_hallucination_mask=np.ones(51, dtype=bool),
    )
    assert len(result.candidates) == 0
    assert len(result.frames) == 0


@pytest.mark.parametrize(('fps', 'nomination_frame', 'survives'), [
    (30.0, 75, True), (60.0, 75, False), (30.0, 90, False), (60.0, 90, True),
])
def test_masked_policy_reuses_fps_scaled_feature_offsets(
    fps: float, nomination_frame: int, survives: bool,
) -> None:
    features = _nomination_features(fps, nomination_frame)
    rows = features.rows[features.rows['frame'] == 70]
    rows['region_current_raw'] = 1
    model = SimpleNamespace(predict_proba=lambda matrix: np.tile([.05, .95], (len(matrix), 1)))
    result = score_contact_features(
        rows, model, fps, ContactModelConfig(reject_masked_without_player=True),
        shuttle_hallucination_mask=np.ones(180, dtype=bool),
    )
    assert bool(len(result.frames)) is survives


@pytest.mark.parametrize('fps', [30.0, 60.0])
@pytest.mark.parametrize('boundary', ['start', 'end'])
def test_masked_policy_cannot_borrow_player_from_outside_search_interval(fps: float, boundary: str) -> None:
    features = _nomination_features(fps, 0)
    start, end = features.search_intervals[0]
    offset = 5 if fps == 30.0 else 10
    target, nominee = (start, start - offset) if boundary == 'start' else (end - 1, end - 1 + offset)
    features = _nomination_features(fps, nominee)
    rows = features.rows[features.rows['frame'] == target]
    rows['region_current_raw'] = 1
    result = score_contact_features(
        rows, SimpleNamespace(), fps, ContactModelConfig(reject_masked_without_player=True),
        shuttle_hallucination_mask=np.ones(180, dtype=bool),
    )
    assert len(result.frames) == 0


def _nomination_features(fps: float, nomination_frame: int) -> ContactFeatures:
    frame_count = 180
    track = np.column_stack((np.full(frame_count, .5), np.full(frame_count, .5), np.ones(frame_count)))
    picks = np.full((frame_count, 2), -1)
    picks[nomination_frame, 0] = 0
    gaps = np.full((frame_count, 2), np.nan)
    sticky = StickyResult(
        np.full(frame_count, np.nan), picks, np.zeros(frame_count), np.full((frame_count, 2, 2), np.nan),
        np.full((frame_count, 2), np.nan), gaps, gaps.copy(), np.ones(frame_count, dtype=bool),
    )
    return build_contact_features(
        track, np.zeros((frame_count, 2, 17, 2)), sticky, [(100, 150)],
        np.zeros(frame_count, dtype=bool), [(100, 150)], [70], [(100, 150)], fps, (1280.0, 720.0),
    )
