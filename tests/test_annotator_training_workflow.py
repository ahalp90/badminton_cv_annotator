"""Manifest contracts, held-out isolation and the runnable retune workflow."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from annotator.contacts.features import ContactFeatures
from annotator.models import SideGeometry
from annotator.outcomes.point_winner import Half
from annotator.sequence.confidence import (
    BASE_FEATURE_NAMES,
    GAP_FEATURE_NAMES,
    ConfidenceFeatures,
)
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.training import workflow
from annotator.training.sequences import LabelledRally
from annotator.video_metadata import VideoMetadata
from dataset_builder.vision import load_json_gz


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
        ContactSequence(0, 80, 150, (ContactEvent(100, .9, Half.TOP),)),
        ContactSequence(1, 280, 370, (ContactEvent(340, .7, Half.BOT),)),
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
    assert workflow.load_labels(path, 200) == (LabelledRally('first', (100, 130), (Half.TOP, None)),)


def test_fit_excludes_contact_groups_preserves_video_order_and_reuses_insertion_models(
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
    group_by_video = {video.identity: video.group for video in manifest.videos}
    seen = []
    contact_model, sequence_models, final_insertion = object(), object(), object()
    fold_insertion = {name: object() for name in training_order}
    contact_training_identities = {}
    pool_models = {}
    saved = []

    def evidence(_run: Path, identity: str, _settings: Any) -> Any:
        seen.append(identity)
        return SimpleNamespace(identity=identity, fps=30.0, track=np.zeros((400, 3)))

    def contact_fit(videos: list, _settings: Any) -> Any:
        identities = [video.identity for video in videos]
        assert identities == [identity for identity in training_order if identity in identities]
        assert all(video.contact_frames.tolist() == [100, 300] for video in videos)
        model = contact_model if identities == training_order else object()
        contact_training_identities[model] = identities
        return SimpleNamespace(model=model)

    def sequence_input(entry: Any, captured: Any, _features: Any, rallies: Any,
                       model: Any, _settings: Any, geometry: Any) -> Any:
        expected = [identity for identity in training_order if group_by_video[identity] != entry.group]
        assert contact_training_identities[model] == expected
        assert model is not contact_model and geometry is SideGeometry.SCENE
        pool_models[entry.identity] = model
        return SimpleNamespace(identity=entry.identity, group=entry.group, fps=captured.fps, rallies=rallies)

    def sequence_fit(videos: list, _settings: Any) -> Any:
        assert [video.identity for video in videos] == training_order
        predictions = {
            video.identity: SimpleNamespace(identity=video.identity, sequences=final_sequences()) for video in videos
        }
        return SimpleNamespace(models=sequence_models, cross_fitted=predictions,
                               cross_fitted_insertion_models=fold_insertion)

    def confidence_features(prediction: Any, model: Any, fps: float) -> ConfidenceFeatures:
        assert model is fold_insertion[prediction.identity] and model is not final_insertion
        assert fps == 30
        return ConfidenceFeatures((0, 1), np.zeros((2, len(BASE_FEATURE_NAMES))), np.zeros((2, len(GAP_FEATURE_NAMES))))

    def confidence_fit(features: ConfidenceFeatures, labels: np.ndarray, _settings: Any) -> Any:
        assert features.base.shape[0] == 8
        assert labels.tolist() == [1, 0] * 4
        return object()

    monkeypatch.setattr(workflow, 'load_contact_evidence', evidence)
    monkeypatch.setattr(workflow, 'features_from_evidence', lambda _evidence: ContactFeatures(np.zeros(1), [], []))
    monkeypatch.setattr(workflow, 'fit_contact_model', contact_fit)
    monkeypatch.setattr(workflow, 'sequence_training_video', sequence_input)
    monkeypatch.setattr(workflow, 'fit_sequence_models', sequence_fit)
    monkeypatch.setattr(workflow, 'build_confidence_features', confidence_features)
    monkeypatch.setattr(workflow, 'fit_confidence_model', confidence_fit)
    monkeypatch.setattr(workflow, 'save_models', lambda models, output: saved.append((models, output)))
    result = workflow.fit_from_manifest(manifest, tmp_path / 'bundle', SideGeometry.SCENE)
    assert seen == training_order
    assert list(pool_models) == training_order
    assert pool_models['one'] is pool_models['four']
    assert list(contact_training_identities.values()) == [
        training_order, ['two', 'three'], ['one', 'four', 'three'], ['one', 'two', 'four'],
    ]
    assert result.contact is contact_model and result.sequences is sequence_models
    assert result.side_geometry is SideGeometry.SCENE
    assert saved == [(result, tmp_path / 'bundle')]


def test_too_few_training_groups_fail_before_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = workflow.load_manifest(manifest_file(tmp_path))
    manifest = workflow.TrainingManifest(manifest.run_dir, manifest.videos[:2])

    def forbidden_load(*_args: Any) -> Any:
        pytest.fail('group validation must happen before loading video artefacts')

    monkeypatch.setattr(workflow, 'load_contact_evidence', forbidden_load)
    with pytest.raises(ValueError, match='three distinct training groups'):
        workflow.fit_from_manifest(manifest, tmp_path / 'bundle')


def test_evaluation_reads_only_requested_split_and_never_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = workflow.load_manifest(manifest_file(tmp_path))
    seen = []

    def evidence(_run: Path, identity: str, _settings: Any) -> Any:
        seen.append(identity)
        return SimpleNamespace(fps=30.0, track=np.zeros((400, 3)))

    def forbidden_fit(*_args: Any) -> None:
        raise AssertionError('evaluation entered training')

    sequences = final_sequences()
    events = tuple(event for span in sequences for event in span.events)
    prediction = SimpleNamespace(refined=SimpleNamespace(sequences=sequences, events=events))
    monkeypatch.setattr(workflow, 'load_contact_evidence', evidence)
    monkeypatch.setattr(workflow, 'predict_contacts', lambda *_args: prediction)
    for name in ('fit_contact_model', 'fit_sequence_models', 'fit_confidence_model'):
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


def test_cli_dispatches_geometry_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = manifest_file(tmp_path)
    fitted = []
    evaluated = []
    model = object()
    monkeypatch.setattr(workflow, 'fit_from_manifest', lambda *args: fitted.append(args))
    monkeypatch.setattr(workflow, 'load_models', lambda _path: model)
    monkeypatch.setattr(workflow, 'evaluate_manifest', lambda *args: evaluated.append(args))
    assert workflow.main(['fit', '--manifest', str(path), '--output', 'bundle', '--side-geometry', 'scene']) == 0
    assert fitted[0][1:] == (Path('bundle'), SideGeometry.SCENE)
    assert workflow.main(['evaluate', '--manifest', str(path), '--models', 'bundle', '--split', 'validation',
                          '--output', 'report.json.gz']) == 0
    assert evaluated[0][1:] == (model, workflow.Split.VALIDATION, Path('report.json.gz'))
    with pytest.raises(SystemExit) as failure:
        workflow.main(['evaluate', '--manifest', str(path), '--models', 'bundle', '--split', 'train', '--output', 'x'])
    assert failure.value.code == 2


def test_current_stage_loader_reuses_capture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = VideoMetadata(tmp_path / 'source.mp4', Fraction(30), 2, 1920, 1080)
    paths = []
    captured = object()
    inputs = SimpleNamespace(net_band=(.4, .5), resolution=(1920, 1080), court_info={}, gate_court_info={},
                             gate_resolution_table=None, homography_rows=[], landing_error_band_m=0.1)
    court = SimpleNamespace(raw_cuts=((0, 2),), evidence=SimpleNamespace(inputs=inputs, court_present=np.ones(2, bool),
                                                                     keep_vote=np.ones(2, bool)))
    pose = SimpleNamespace(
        bboxes=np.zeros((2, 1, 4)), scores=np.ones((2, 1)), kps=np.zeros((2, 1, 17, 2)), ndet=np.ones(2),
    )

    def load_array(path: Path) -> np.ndarray:
        paths.append(path)
        if path.name == 'shuttle_track.npy.xz':
            return np.array([[.2, .3, 1], [.3, .4, 1]])
        return np.zeros(2, dtype=np.uint8)

    def run(*_args: Any, **kwargs: Any) -> None:
        assert kwargs['heuristic_only'] is True
        kwargs['capture'].contact_evidence = captured

    monkeypatch.setattr(workflow, 'load_json_gz', lambda path: paths.append(path) or metadata.to_dict())
    monkeypatch.setattr(workflow, 'load_npy_xz', load_array)
    monkeypatch.setattr(workflow, 'load_pose_arrays', lambda path, _count: paths.append(path) or pose)
    monkeypatch.setattr(workflow, 'load_court_vision', lambda path, **_kwargs: paths.append(path) or court)
    monkeypatch.setattr(workflow, 'run_video', run)
    assert workflow.load_contact_evidence(tmp_path, 'video', workflow.TrainingSettings().preprocessing) is captured
    assert tmp_path / 'stages/metadata/video/video_metadata.json.gz' in paths
    assert tmp_path / 'stages/shuttle/video/shuttle_guard_codes.npy.xz' in paths
    assert tmp_path / 'stages/pose/video' in paths
    assert tmp_path / 'stages/court/video' in paths
    inputs_before = court.evidence.inputs
    court.evidence.inputs = None
    with pytest.raises(ValueError, match='no usable operational court'):
        workflow.load_contact_evidence(tmp_path, 'video', workflow.TrainingSettings().preprocessing)
    court.evidence.inputs = inputs_before


@pytest.mark.parametrize(('frames', 'sides', 'expected'), [
    ((100, 130), (Half.TOP, Half.BOT), 1),
    ((100,), (Half.TOP,), 0),
    ((100, 115, 130), (Half.TOP, Half.TOP, Half.BOT), 0),
    ((100, 130), (Half.BOT, Half.TOP), 0),
])
def test_whole_rally_correctness_matches_selected_judgement(
    frames: tuple, sides: tuple, expected: int,
) -> None:
    rally = LabelledRally('first', (100, 130), (Half.TOP, Half.BOT))
    events = tuple(ContactEvent(frame, .8, side) for frame, side in zip(frames, sides, strict=True))
    sequence = ContactSequence(0, 80, 150, events)
    assert workflow.whole_rally_correctness(sequence, (rally,), 30) == expected


def test_unknown_side_abstains_only_after_known_contradictions() -> None:
    rally = LabelledRally('first', (100, 130), (Half.TOP, None))
    complete = ContactSequence(0, 80, 150, (ContactEvent(100, .8, Half.TOP), ContactEvent(130, .8, Half.BOT)))
    missing = ContactSequence(0, 80, 150, (ContactEvent(100, .8, Half.TOP),))
    assert workflow.whole_rally_correctness(complete, (rally,), 30) == -1
    assert workflow.whole_rally_correctness(missing, (rally,), 30) == 0
