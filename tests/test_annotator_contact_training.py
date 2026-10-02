"""Retuning selection preserves the fitted contact model's sampling policy."""

import gzip
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from annotator.contacts.features import REGION_FIELDS
from annotator.contacts.model import (
    CONTACT_FEATURE_NAMES,
    contact_feature_matrix,
    score_contact_features,
)
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
    select_contact_training_rows,
)


def training_video(identity: str, fps: float, contacts: tuple[int, ...] = (100, 400)) -> ContactTrainingVideo:
    dtype = [('fixture', 'S12'), ('frame', 'i4'), ('interval_id', 'i4')]
    dtype.extend((name, 'u1') for name in REGION_FIELDS)
    dtype.extend((name, 'f4') for name in CONTACT_FEATURE_NAMES)
    rows = np.zeros(800, dtype=dtype)
    rows['fixture'] = identity.encode()
    rows['frame'] = np.arange(800)
    rows['region_wrist'] = 1
    rows['region_wrist'][::7] = 0
    for name in CONTACT_FEATURE_NAMES:
        rows[name] = np.sin(np.arange(800) / 10)
    rows[CONTACT_FEATURE_NAMES[0]][::11] = np.nan
    return ContactTrainingVideo(identity, rows, np.array(contacts, dtype=np.int32), fps)


REFERENCE_DIR = Path(__file__).parent / 'fixtures' / 'annotator_reference'


def test_sampling_matches_reference_for_multiple_fps_and_video_order() -> None:
    videos = [training_video('first', 25.0), training_video('second', 30.0), training_video('third', 60.0)]
    candidates = np.concatenate([video.features[video.features['region_wrist'].astype(bool)] for video in videos])
    with np.load(REFERENCE_DIR / 'contact_sampling.npz') as reference:
        expected_rows = candidates[reference['selected_indices']]
        expected_labels = reference['labels']
    expected_counts = json.loads(gzip.decompress((REFERENCE_DIR / 'contact_sampling_counts.json.gz').read_bytes()))
    actual = select_contact_training_rows(videos)
    np.testing.assert_array_equal(actual.labels, expected_labels)
    for name in candidates.dtype.names:
        np.testing.assert_array_equal(actual.rows[name], expected_rows[name], err_msg=name)
    assert {name: asdict(counts) for name, counts in actual.video_counts.items()} == expected_counts


def test_small_fit_scores_only_supplied_training_video() -> None:
    train = training_video('train', 30.0)
    validation = training_video('validation', 30.0, (200,))
    config = replace(ContactFitConfig(), max_iter=8, min_samples_leaf=2)
    with threadpool_limits(limits=1):
        fitted = fit_contact_model([train], config)
        scored = score_contact_features(validation.features, fitted.model, validation.fps)
    assert fitted.selection.video_counts.keys() == {'train'}
    assert set(fitted.selection.rows['fixture']) == {b'train'}
    assert fitted.model.get_params()['class_weight'] == 'balanced'
    assert fitted.model.n_features_in_ == 85
    assert len(scored.probabilities) > 0
    assert np.isfinite(scored.probabilities).all()
    assert np.all((scored.probabilities >= 0) & (scored.probabilities <= 1))


def test_nearby_negatives_survive_without_positive_rows() -> None:
    main = training_video('main', 30.0)
    no_positive = training_video('nearby', 30.0, (100,))
    no_positive.features['region_wrist'] = 0
    no_positive.features['region_wrist'][105:116] = 1
    selection = select_contact_training_rows([main, no_positive])
    counts = selection.video_counts['nearby']
    assert counts.positive == 0
    assert counts.nearby_negative == 11
    assert counts.sampled_other_negative == 0
    assert counts.selected == 11


def test_unlabelled_video_contributes_no_distant_negatives() -> None:
    selection = select_contact_training_rows([training_video('main', 30.0), training_video('empty', 30.0, ())])
    assert selection.video_counts['empty'].selected == 0


def test_invalid_training_inputs_fail() -> None:
    video = training_video('same', 30.0)
    with pytest.raises(ValueError, match='more than once'):
        select_contact_training_rows([video, video])
    with pytest.raises(ValueError, match='sorted'):
        select_contact_training_rows([replace(video, contact_frames=np.array([200, 100]))])
    with pytest.raises(ValueError, match='at least one'):
        select_contact_training_rows([])
    with pytest.raises(ValueError, match='contacts and non-contacts'):
        select_contact_training_rows([training_video('empty', 30.0, ())])


@pytest.mark.parametrize('fit_order', [None, ['third', 'first', 'empty', 'second'], ['first', 'second', 'third', 'empty']])
def test_fit_order_reorders_selected_chunks_without_changing_sampling(monkeypatch, fit_order) -> None:
    videos = [
        training_video('first', 25.0), training_video('second', 30.0),
        training_video('third', 60.0), training_video('empty', 30.0, ()),
    ]
    selected = select_contact_training_rows(videos)
    expected_order = list(selected.video_counts) if fit_order is None else fit_order
    expected_chunks = []
    label_chunks = []
    for identity in expected_order:
        video_rows = selected.rows['fixture'] == identity.encode()
        expected_chunks.append(selected.rows[video_rows])
        label_chunks.append(selected.labels[video_rows])
    expected_rows = np.concatenate(expected_chunks)
    expected_labels = np.concatenate(label_chunks)
    observed = []

    def fit(model, matrix: np.ndarray, labels: np.ndarray):
        observed.append((matrix, labels))
        return model

    monkeypatch.setattr(HistGradientBoostingClassifier, 'fit', fit)
    fitted = fit_contact_model(videos, fit_video_order=fit_order)
    assert list(fitted.selection.video_counts) == expected_order
    assert fitted.selection.video_counts == selected.video_counts
    for name in expected_rows.dtype.names:
        np.testing.assert_array_equal(fitted.selection.rows[name], expected_rows[name])
    np.testing.assert_array_equal(fitted.selection.labels, expected_labels)
    np.testing.assert_array_equal(observed[0][0], contact_feature_matrix(expected_rows))
    np.testing.assert_array_equal(observed[0][1], expected_labels)


@pytest.mark.parametrize('fit_order', [[], ['first'], ['first', 'first'], ['first', 'other'], ['first', 'second', 'other']])
def test_fit_order_must_permute_training_video_ids(monkeypatch, fit_order: list[str]) -> None:
    def unexpected_fit(*_args, **_kwargs):
        pytest.fail('invalid fit order reached the estimator')

    monkeypatch.setattr(HistGradientBoostingClassifier, 'fit', unexpected_fit)
    videos = [training_video('first', 30.0), training_video('second', 30.0)]
    with pytest.raises(ValueError, match='each training video ID exactly once'):
        fit_contact_model(videos, fit_video_order=fit_order)
