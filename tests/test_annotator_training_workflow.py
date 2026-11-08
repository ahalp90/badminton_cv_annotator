"""Manifest contracts, held-out isolation and the runnable retune workflow."""

import json
import weakref
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from annotator.contacts.build_features import ContactFeatures
from annotator.courts.prepare_courts import CourtEvidenceResult, CourtInputs
from annotator.saved_inputs import SavedVideoInputs
from annotator.training import workflow
from annotator.training.sequences import LabelledRally
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from dataset_builder.vision import CourtVision, PoseArrays, load_json_gz
from shared.video_metadata import VideoMetadata


def label_file(directory: Path, name: str) -> Path:
    path = directory / f'{name}.csv'
    path.write_text('rally_id,frame,side\nfirst,100,Top\nsecond,300,Bot\n')
    return path


def manifest_file(directory: Path) -> Path:
    videos = [{'id': name, 'group': group, 'split': 'train', 'labels': label_file(directory, name).name}
              for name, group in [('one', 'east'), ('two', 'west'), ('three', 'north')]]
    videos.extend([
        {'id': 'held', 'group': 'validation_group', 'split': 'validation',
         'labels': label_file(directory, 'held').name},
        {'id': 'test', 'group': 'test_group', 'split': 'test', 'labels': label_file(directory, 'test').name},
    ])
    path = directory / 'manifest.json'
    path.write_text(json.dumps({'run_dir': 'extracted', 'videos': videos}))
    return path


def final_sequences() -> tuple[ContactSequence, ...]:
    return (
        ContactSequence(0, 80, 150, (ContactEvent(100, .9, CourtHalf.TOP),)),
        ContactSequence(1, 280, 370, (ContactEvent(340, .7, CourtHalf.BOT),)),
    )


def test_manifest_relative_paths_and_disjoint_groups(tmp_path: Path) -> None:
    path = manifest_file(tmp_path)
    manifest = workflow.load_manifest(path)
    assert manifest.run_dir == tmp_path / 'extracted'
    assert manifest.videos[0].labels == tmp_path / 'one.csv'
    payload = json.loads(path.read_text())
    payload['videos'][-1]['group'] = 'east'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='more than one split'):
        workflow.load_manifest(path)
    payload['videos'][-1]['group'] = 'test_group'
    payload['videos'][-1]['id'] = 'one'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='duplicate video'):
        workflow.load_manifest(path)


@pytest.mark.parametrize('body', [
    'rally_id,frame,side\nfirst,100,Near\n',
    'rally_id,frame,side\nfirst,400,Top\n',
    'rally_id,frame,side\nfirst,100,Top\nfirst,99,Bot\n',
    'rally_id,frame,side\nfirst,100,Top\nsecond,200,Bot\nfirst,300,Top\n',
    'rally_id,frame,side,side\nfirst,100,Top,Top\n',
])
def test_bad_labels_fail_clearly(tmp_path: Path, body: str) -> None:
    path = tmp_path / 'bad.csv'
    path.write_text(body)
    with pytest.raises(ValueError, match='bad.csv'):
        workflow.load_labels(path, 400)


def test_labels_keep_unknown_sides(tmp_path: Path) -> None:
    path = tmp_path / 'unknown.csv'
    path.write_text('rally_id,frame,side\nfirst,100,Top\nfirst,130,\n')
    assert workflow.load_labels(path, 200) == (LabelledRally('first', (100, 130), (CourtHalf.TOP, None)),)


def test_fit_excludes_contact_groups_and_reuses_held_group_rally_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = manifest_file(tmp_path)
    payload = json.loads(path.read_text())
    payload['videos'].insert(2, {
        'id': 'four', 'group': 'east', 'split': 'train', 'labels': label_file(tmp_path, 'four').name,
    })
    path.write_text(json.dumps(payload))
    manifest = workflow.load_manifest(path)
    training_order = ['one', 'two', 'four', 'three']
    group_by_video = {video.video_id: video.group for video in manifest.videos}
    seen = []
    contact_model, sequence_models, review_model = object(), object(), object()
    contact_training_identities = {}
    pool_models = {}
    saved = []
    prepared_arrays = []

    def evidence(_run: Path, video_id: str, _settings: Any) -> Any:
        seen.append(video_id)
        track = np.zeros((400, 3))
        prepared_arrays.append(weakref.ref(track))
        return SimpleNamespace(video_id=video_id, fps=30.0, track=track)

    def features(_evidence: Any) -> ContactFeatures:
        rows = np.zeros(1)
        prepared_arrays.append(weakref.ref(rows))
        return ContactFeatures(rows, [], [])

    def contact_fit(videos: list, _settings: Any) -> Any:
        identities = [video.video_id for video in videos]
        assert identities == [video_id for video_id in training_order if video_id in identities]
        assert all(video.contact_frames.tolist() == [100, 300] for video in videos)
        model = contact_model if identities == training_order else object()
        contact_training_identities[model] = identities
        return SimpleNamespace(model=model)

    def sequence_input(video_id: str, group: str, captured: Any, _features: Any, rallies: Any,
                       model: Any, _settings: Any) -> Any:
        expected = [video_id for video_id in training_order if group_by_video[video_id] != group]
        assert contact_training_identities[model] == expected
        assert model is not contact_model
        pool_models[video_id] = model
        return SimpleNamespace(video_id=video_id, group=group, context=SimpleNamespace(fps=captured.fps), rallies=rallies)

    def sequence_fit(videos: list, _settings: Any) -> Any:
        assert all(array() is None for array in prepared_arrays)
        assert [video.video_id for video in videos] == training_order
        predictions = {
            video.video_id: SimpleNamespace(video_id=video.video_id, sequences=final_sequences()) for video in videos
        }
        return SimpleNamespace(selector=sequence_models, held_group_predictions=predictions)

    def review_fit(videos: list, predictions: dict, _settings: Any, *, tolerance_at_30_fps: int) -> Any:
        assert [video.video_id for video in videos] == training_order
        assert tolerance_at_30_fps == 6
        assert set(predictions) == set(training_order)
        for video in videos:
            assert predictions[video.video_id].video_id == video.video_id
        return review_model

    monkeypatch.setattr(workflow, 'prepare_saved_video_inputs', evidence)
    monkeypatch.setattr(workflow, 'contact_features_from_inputs', features)
    monkeypatch.setattr(workflow, 'fit_contact_model', contact_fit)
    monkeypatch.setattr(workflow, 'sequence_training_video', sequence_input)
    monkeypatch.setattr(workflow, 'fit_rally_models', sequence_fit)
    monkeypatch.setattr(workflow, 'fit_rally_review_model', review_fit)
    monkeypatch.setattr(workflow, 'save_models', lambda models, output: saved.append((models, output)))
    settings = replace(
        workflow.DEFAULT_TRAINING_SETTINGS,
        rally=replace(workflow.DEFAULT_TRAINING_SETTINGS.rally, review_tolerance_at_30_fps=6),
    )
    result = workflow.fit_from_manifest(manifest, tmp_path / 'bundle', settings)
    assert seen == training_order
    assert list(pool_models) == training_order
    assert pool_models['one'] is pool_models['four']
    assert list(contact_training_identities.values()) == [
        training_order, ['two', 'three'], ['one', 'four', 'three'], ['one', 'two', 'four'],
    ]
    assert result.contact_model is contact_model and result.rally_selector is sequence_models
    assert saved == [(result, tmp_path / 'bundle')]
    fit_settings = load_json_gz(tmp_path / 'bundle' / 'fit-settings.json.gz')
    assert fit_settings['contact']['positive_radius_at_30_fps'] == 1
    assert fit_settings['rally']['preferred_tolerance_at_30_fps'] == 5
    assert fit_settings['rally']['review_tolerance_at_30_fps'] == 6


def test_too_few_training_groups_fail_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = workflow.load_manifest(manifest_file(tmp_path))
    manifest = workflow.TrainingManifest(manifest.run_dir, manifest.videos[:2])

    def forbidden_load(*_args: Any) -> Any:
        pytest.fail('group validation must happen before loading video artefacts')

    monkeypatch.setattr(workflow, 'prepare_saved_video_inputs', forbidden_load)
    with pytest.raises(ValueError, match='three distinct training groups'):
        workflow.fit_from_manifest(manifest, tmp_path / 'bundle')


def test_evaluation_reads_only_requested_split_and_never_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = workflow.load_manifest(manifest_file(tmp_path))
    seen = []

    def evidence(_run: Path, video_id: str, _settings: Any) -> Any:
        seen.append(video_id)
        return SimpleNamespace(fps=30.0, track=np.zeros((400, 3)))

    def forbidden_fit(*_args: Any) -> None:
        raise AssertionError('evaluation entered training')

    sequences = final_sequences()
    events = tuple(event for span in sequences for event in span.events)
    prediction = SimpleNamespace(rally_prediction=SimpleNamespace(sequences=sequences, events=events))
    monkeypatch.setattr(workflow, 'prepare_saved_video_inputs', evidence)
    monkeypatch.setattr(workflow, 'predict_rallies_and_contacts', lambda *_args: prediction)
    for name in ('fit_contact_model', 'fit_rally_models', 'fit_rally_review_model'):
        monkeypatch.setattr(workflow, name, forbidden_fit)
    output = tmp_path / 'evaluation.json.gz'
    models = SimpleNamespace(preprocessing=object())
    report = workflow.evaluate_manifest(manifest, models, workflow.Split.TEST, output)
    assert seen == ['test']
    assert report['total']['matched_contacts'] == 1
    assert report['total']['contact_precision'] == .5
    assert report['total']['contact_recall'] == .5
    assert report['total']['whole_rally_judged'] == 2
    assert report['total']['whole_rally_correct'] == 1
    assert load_json_gz(output) == report
    with pytest.raises(ValueError, match='validation or test'):
        workflow.evaluate_manifest(manifest, models, workflow.Split.TRAIN, output)


def test_cli_dispatches_settings_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = manifest_file(tmp_path)
    fitted = []
    evaluated = []
    model = object()
    monkeypatch.setattr(workflow, 'fit_from_manifest', lambda *args: fitted.append(args))
    monkeypatch.setattr(workflow, 'load_models', lambda _path: model)
    monkeypatch.setattr(workflow, 'evaluate_manifest', lambda *args: evaluated.append(args))
    assert workflow.main(['fit', '--manifest', str(path), '--output', 'bundle']) == 0
    assert fitted[0][1:] == (Path('bundle'), workflow.DEFAULT_TRAINING_SETTINGS)
    defaults = fitted[0][-1]
    assert defaults.contact.device == defaults.confidence.device == defaults.rally.tree.device == 'cuda'
    assert workflow.main([
        'fit', '--manifest', str(path), '--output', 'layer-bundle', '--device', 'cpu',
    ]) == 0
    settings = fitted[1][-1]
    assert settings.rally.tree.max_iter == 400
    assert settings.rally.tree.max_leaf_nodes == 31
    assert settings.contact.device == settings.confidence.device == settings.rally.tree.device == 'cpu'
    assert workflow.main(['evaluate', '--manifest', str(path), '--models', 'bundle', '--split', 'validation',
                          '--output', 'report.json.gz']) == 0
    assert evaluated[0][1:] == (model, workflow.Split.VALIDATION, Path('report.json.gz'))
    with pytest.raises(SystemExit) as failure:
        workflow.main(['evaluate', '--manifest', str(path), '--models', 'bundle', '--split', 'train', '--output', 'x'])
    assert failure.value.code == 2


def test_saved_inputs_pass_to_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = VideoMetadata(tmp_path / 'source.mp4', Fraction(30), 2, 1920, 1080)
    pose = PoseArrays(
        kps=np.zeros((2, 1, 17, 2)), bboxes=np.zeros((2, 1, 4)), scores=np.ones((2, 1)),
        kp_scores=np.ones((2, 1, 17)), ndet=np.ones(2, dtype=int),
    )
    court_inputs = CourtInputs((1920, 1080), pd.DataFrame())
    court = CourtVision(((0, 2),), CourtEvidenceResult(
        court_inputs, (), np.ones(2, bool), np.ones(2, bool),
    ))
    saved = SavedVideoInputs(
        metadata, np.array([[.2, .3, 1], [.3, .4, 1]]), np.zeros(2, dtype=np.uint8), pose, court,
    )
    captured = object()
    preprocessing = workflow.TrainingSettings().preprocessing

    def load(run_dir: Path, video_id: str) -> SavedVideoInputs:
        assert (run_dir, video_id) == (tmp_path, 'video')
        return saved

    def prepare(track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
                ndet: np.ndarray, **kwargs: Any) -> object:
        assert track is saved.track and bboxes is pose.bboxes and scores is pose.scores
        assert kps is pose.kps and ndet is pose.ndet
        assert kwargs['video_config'] == workflow.scale_config_to_fps(preprocessing, 30.0)
        assert kwargs['shuttle_grades'] is saved.shuttle_grades
        assert kwargs['homography_rows'] is court_inputs.homography_rows
        return captured

    monkeypatch.setattr(workflow, 'load_saved_video_inputs', load)
    monkeypatch.setattr(workflow, 'prepare_annotation_inputs', prepare)
    assert workflow.prepare_saved_video_inputs(tmp_path, 'video', preprocessing) is captured


@pytest.mark.parametrize(('frames', 'sides', 'expected'), [
    ((100, 130), (CourtHalf.TOP, CourtHalf.BOT), 1),
    ((100,), (CourtHalf.TOP,), 0),
    ((100, 115, 130), (CourtHalf.TOP, CourtHalf.TOP, CourtHalf.BOT), 0),
    ((100, 130), (CourtHalf.BOT, CourtHalf.TOP), 0),
    ((90, 140), (CourtHalf.TOP, CourtHalf.BOT), 1),
    ((100, 141), (CourtHalf.TOP, CourtHalf.BOT), 0),
    ((100, 109), (CourtHalf.TOP, CourtHalf.BOT), 0),
])
def test_whole_rally_correctness_matches_selected_judgement(
    frames: tuple, sides: tuple, expected: int,
) -> None:
    rally = LabelledRally('first', (100, 130), (CourtHalf.TOP, CourtHalf.BOT))
    events = tuple(ContactEvent(frame, .8, side) for frame, side in zip(frames, sides, strict=True))
    sequence = ContactSequence(0, 80, 150, events)
    assert workflow.whole_rally_correctness(sequence, (rally,), 30) == expected


def test_unknown_side_abstains_only_after_known_contradictions() -> None:
    rally = LabelledRally('first', (100, 130), (CourtHalf.TOP, None))
    complete = ContactSequence(0, 80, 150, (ContactEvent(100, .8, CourtHalf.TOP), ContactEvent(130, .8, CourtHalf.BOT)))
    missing = ContactSequence(0, 80, 150, (ContactEvent(100, .8, CourtHalf.TOP),))
    assert workflow.whole_rally_correctness(complete, (rally,), 30) == -1
    assert workflow.whole_rally_correctness(missing, (rally,), 30) == 0


def test_rally_target_judges_wrist_fitted_order_without_changing_raw_guesses() -> None:
    rally = LabelledRally('first', (100, 130), (CourtHalf.BOT, CourtHalf.TOP))
    sequence = ContactSequence(0, 80, 150, (
        ContactEvent(100, .9, CourtHalf.TOP, (0.1, 0.2)),
        ContactEvent(130, .8, CourtHalf.TOP, (0.1, 2.0)),
    ))
    assert workflow.whole_rally_correctness(sequence, (rally,), 30) == 1
    assert [event.side for event in sequence.events] == [CourtHalf.TOP, CourtHalf.TOP]


def test_short_inward_errors_and_neighbouring_rallies_use_shared_rule() -> None:
    rally = LabelledRally('rally', (300, 400), (CourtHalf.TOP, CourtHalf.BOT))
    sequence = ContactSequence(0, 305, 397, (
        ContactEvent(305, .9, CourtHalf.TOP), ContactEvent(395, .9, CourtHalf.BOT),
    ))
    assert workflow.whole_rally_correctness(sequence, (rally,), 30) == 1
    neighbour = LabelledRally('neighbour', (420, 440), (CourtHalf.TOP, CourtHalf.BOT))
    merged = replace(sequence, end_frame=441)
    assert workflow.whole_rally_correctness(merged, (rally, neighbour), 30) == 0


def test_timing_allowance_does_not_merge_a_neighbour_outside_the_span() -> None:
    rally = LabelledRally('rally', (100, 200, 300), (CourtHalf.TOP, CourtHalf.BOT, CourtHalf.TOP))
    neighbour = LabelledRally('neighbour', (398, 450), (CourtHalf.BOT, CourtHalf.TOP))
    events = tuple(ContactEvent(frame, .9, side) for frame, side in zip(rally.frames, rally.sides, strict=True))
    sequence = ContactSequence(0, 40, 391, events)
    for tolerance in (5, 10):
        assert workflow.whole_rally_correctness(sequence, (rally, neighbour), 30, tolerance_at_30_fps=tolerance) == 1


def test_unlimited_context_preserves_rally_and_contact_checks() -> None:
    rally = LabelledRally('rally', (300, 400, 500), (CourtHalf.TOP, CourtHalf.BOT, CourtHalf.TOP))
    events = tuple(ContactEvent(frame, .9, side) for frame, side in zip(rally.frames, rally.sides, strict=True))
    sequence = ContactSequence(0, 0, 1000, events)
    assert workflow.whole_rally_correctness(sequence, (rally,), 30) == 1
    neighbour = LabelledRally('neighbour', (800, 900), (CourtHalf.BOT, CourtHalf.TOP))
    assert workflow.whole_rally_correctness(sequence, (rally, neighbour), 30) == 0
    opposite_events = []
    for event in events:
        side = CourtHalf.BOT if event.side == CourtHalf.TOP else CourtHalf.TOP
        opposite_events.append(replace(event, side=side))
    for changed in (
        replace(sequence, start_frame=350),
        replace(sequence, events=events[:-1]),
        replace(sequence, events=(*events, ContactEvent(600, .9, CourtHalf.BOT))),
        replace(sequence, events=tuple(opposite_events)),
    ):
        assert workflow.whole_rally_correctness(changed, (rally,), 30) == 0
