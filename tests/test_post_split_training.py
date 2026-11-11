"""Confidence training excludes each judged group through every contributing fit."""

from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from annotator.contacts.score_candidates import ContactModelConfig
from annotator.rallies.refine_rallies import RallyReviewFeatures
from annotator.rallies.split_and_correct_rallies import CorrectedRallies
from annotator.training import followup
from annotator.training.rally import RallyFitConfig
from annotator.training.sequences import LabelledRally
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf, FinishedRally


class ImmediateExecutor:
    """Run submitted test jobs in order so their dependencies can be inspected."""

    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def submit(self, function, *args):
        future = Future()
        future.set_result(function(*args))
        return future


def save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path)
    return path


@pytest.mark.parametrize('groups', [('C', 'A', 'B'), ('D', 'B', 'A', 'C')])
def test_final_confidence_group_is_absent_from_all_upstream_fits(tmp_path, monkeypatch, groups):
    videos = [followup.PreparedTrainingVideo(f'video-{index}', group, tmp_path / f'{group}.joblib', ())
              for index, group in enumerate(groups)]
    held_contacts = {group: save(set(groups) - {group}, tmp_path / f'contact-{group}.joblib') for group in groups}
    ordinary_paths = {}
    for video in videos:
        ordinary_paths[video.video_id] = save(
            {'group': video.group, 'dependencies': set(groups) - {video.group}},
            tmp_path / 'ordinary' / f'{video.video_id}.joblib',
        )
    fits, judged = [], []

    def prepare(video, contact_path, _contact_settings, _settings, destination):
        dependencies = joblib.load(contact_path)
        assert video.group in dependencies  # remaining-group contacts deliberately use the fixed model
        return save({'group': video.group, 'dependencies': dependencies}, destination)

    def fit(ordinary, pieces, _settings, destination):
        rows = [joblib.load(path) for path in ordinary]
        labels = {row['group'] for row in rows}
        dependencies = set(labels)
        for row in rows:
            dependencies.update(row['dependencies'])
        for path in pieces:
            dependencies.update(joblib.load(path)['dependencies'])
        fitted = SimpleNamespace(groups=labels, dependencies=dependencies, has_pieces=bool(pieces))
        fits.append((destination, fitted, list(ordinary), list(pieces)))
        return save(fitted, destination)

    def pieces(video, ordinary, first_path, _settings, destination):
        first = joblib.load(first_path)
        assert video.group not in first.groups
        dependencies = first.dependencies | joblib.load(ordinary)['dependencies'] | {video.group}
        return save({'dependencies': dependencies}, destination)

    def held_result(video, ordinary, first_path, followup_path, _settings, pieces_path, confidence_path):
        first, second = joblib.load(first_path), joblib.load(followup_path)
        assert video.group not in first.dependencies | second.dependencies | joblib.load(ordinary)['dependencies']
        assert second.has_pieces
        judged.append(video.video_id)
        save({'dependencies': first.dependencies | {video.group}}, pieces_path)
        save((np.asarray([[ord(video.group)]]), np.asarray([1])), confidence_path)

    def review(matrix, labels, _settings):
        assert matrix[:, 0].tolist() == [ord(group) for group in sorted(groups)]
        assert labels.tolist() == [1] * len(videos)
        return 'confidence'

    monkeypatch.setattr(followup, 'ProcessPoolExecutor', ImmediateExecutor)
    monkeypatch.setattr(followup, 'ThreadPoolExecutor', ImmediateExecutor)
    monkeypatch.setattr(followup, 'prepare_training_data', prepare)
    monkeypatch.setattr(followup, 'fit_selector_from_files', fit)
    monkeypatch.setattr(followup, 'prepare_piece_training_data', pieces)
    monkeypatch.setattr(followup, 'prepare_held_group_results', held_result)
    monkeypatch.setattr(followup, 'fit_review_model', review)
    selector, confidence = followup.fit_followup_models(
        videos, ordinary_paths, held_contacts, ContactModelConfig(), RallyFitConfig(), TreeFitConfig(), tmp_path / 'fit',
    )
    assert selector.groups == set(groups) and confidence == 'confidence'
    assert set(judged) == set(ordinary_paths)
    assert len(fits) == len(groups) * (len(groups) + 1) + 1
    assert fits[-1][2] == list(ordinary_paths.values())
    assert [path.stem for path in fits[-1][3]] == [video.video_id for video in videos]


def test_followup_requires_three_groups_before_loading_files(tmp_path):
    videos = [followup.PreparedTrainingVideo(group, group, tmp_path / group, ()) for group in 'AB']
    with pytest.raises(ValueError, match='at least three'):
        followup.fit_followup_models(videos, {}, {}, ContactModelConfig(), RallyFitConfig(), TreeFitConfig(), tmp_path)


@pytest.mark.parametrize(('fps', 'shift'), [(25.0, 4), (30.0, 5), (60.0, 10)])
def test_confidence_rows_follow_final_span_order_and_configured_tolerance(tmp_path, monkeypatch, fps, shift):
    first = ContactSequence(8, 20, 80, (ContactEvent(40, .9, CourtHalf.TOP),))
    second = ContactSequence(3, 90, 140, (ContactEvent(100, .9, CourtHalf.TOP),))
    rallies = (FinishedRally(first, CourtHalf.TOP), FinishedRally(second, CourtHalf.TOP))
    features = RallyReviewFeatures((3, 8), np.asarray([[3.0], [8.0]]))
    corrected = CorrectedRallies(rallies, (), (0, 1), features, None)
    labels = (LabelledRally('first', (40 + shift,), (CourtHalf.TOP,)),
              LabelledRally('second', (100,), (CourtHalf.TOP,)))
    inputs = save(SimpleNamespace(evidence=SimpleNamespace(fps=fps)), tmp_path / 'inputs.joblib')
    video = followup.PreparedTrainingVideo('video', 'A', inputs, labels)
    ordinary = save(SimpleNamespace(pool=SimpleNamespace(options=None), contacts=None), tmp_path / 'ordinary.joblib')
    selector = save('selector', tmp_path / 'selector.joblib')
    monkeypatch.setattr(followup, 'predict_rally_options', lambda *_args: None)
    monkeypatch.setattr(followup, 'split_and_correct_rallies', lambda *_args: corrected)
    for tolerance, expected in ((3, [1, 0]), (6, [1, 1])):
        settings = replace(RallyFitConfig(), review_tolerance_at_30_fps=tolerance)
        piece_path = tmp_path / f'pieces-{tolerance}.joblib'
        confidence_path = tmp_path / f'confidence-{tolerance}.joblib'
        followup.prepare_held_group_results(video, ordinary, selector, selector, settings, piece_path, confidence_path)
        matrix, targets = joblib.load(confidence_path)
        np.testing.assert_array_equal(matrix, features.matrix)
        np.testing.assert_array_equal(targets, expected)
        assert joblib.load(piece_path) is None
