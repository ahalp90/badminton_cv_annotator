"""Retuning selection preserves the fitted contact model's sampling policy."""

import importlib
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from annotator.contact_features import REGION_FIELDS
from annotator.contact_model import CONTACT_FEATURE_NAMES, score_contact_features
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


def test_sampling_matches_original_for_multiple_fps_and_video_order(monkeypatch: pytest.MonkeyPatch) -> None:
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo))
    baseline = importlib.import_module('scratch.contact_det_full_ds_fit.scripts.score_contact_baseline')
    videos = [training_video('first', 25.0), training_video('second', 30.0), training_video('third', 60.0)]
    chunks = []
    ranges = {}
    row_start = 0
    for video in videos:
        rows = video.features[video.features['region_wrist'].astype(bool)]
        chunks.append(rows)
        ranges[video.identity] = (row_start, row_start + len(rows))
        row_start += len(rows)
    candidates = baseline.CandidateRows(np.concatenate(chunks), ranges)
    config = SimpleNamespace(
        random_seed=20260824, positive_radius_at_30_fps=1,
        ignored_radius_at_30_fps=4, hard_negative_radius_at_30_fps=15,
    )
    labels = baseline.ContactLabels({video.identity: video.contact_frames for video in videos}, {}, {})
    split = SimpleNamespace(
        training_videos=[SimpleNamespace(fixture=video.identity, fps=video.fps) for video in videos],
    )
    expected = baseline.choose_training_rows(candidates, split, labels, config, SimpleNamespace(negative_limit=24))
    actual = select_contact_training_rows(videos)
    np.testing.assert_array_equal(actual.labels, expected.labels[expected.selected])
    for name in candidates.rows.dtype.names:
        np.testing.assert_array_equal(actual.rows[name], candidates.rows[name][expected.selected], err_msg=name)
    assert {name: asdict(counts) for name, counts in actual.video_counts.items()} == expected.video_counts


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
