"""Prepare, fit or evaluate the paired annotator refit on supplied court files.

Default invocation and ``--check`` only validate inputs. ``--run`` performs the
two full builds in order; evaluation is a separate later action.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from annotator.contacts.features import ContactFeatures
from annotator.courts.evidence import (
    build_court_detector_evidence,
    read_detector_scenes,
)
from annotator.hybrid import features_from_evidence
from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.run_video import RunCapture
from annotator.training.contact import ContactTrainingVideo, fit_contact_model
from annotator.training.workflow import (
    DEFAULT_TRAINING_SETTINGS,
    Split,
    VideoInput,
    load_labels,
    sequence_training_video,
)
from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    ANNOTATOR_RESULT_FILENAME,
    TRACK_FILENAME,
    CourtVision,
    PoseArrays,
    annotation_result_payload,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    persist_court_vision,
    save_json_gz,
)
from experiments.annotator.old_court_regression import retrain
from experiments.annotator.old_court_regression.old_inputs import (
    TEST_FPS,
    TEST_GUARD_CODES_FILENAME,
    TEST_RESOLUTION,
    TEST_TRACK_FILENAME,
    OldVideoInputs,
    run_old_video,
    test_video_directory,
)
from experiments.annotator.old_court_regression.runner import (
    Item,
    Progress,
    run_stage,
    utc_now,
)

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path('/scratch/ahalperi/annotator-good-court-refit')
ARM_CONFIGS = {
    'base': DEFAULT_TRAINING_SETTINGS.contact_prediction,
    'veto': replace(DEFAULT_TRAINING_SETTINGS.contact_prediction, reject_masked_without_player=True),
}
ARMS = (*ARM_CONFIGS, 'base_inference_veto')
SNAPSHOT = 'input-snapshot.json'


@dataclass(frozen=True)
class RefitConfig:
    """Resolved input paths and the per-video custom court JSON files."""

    config_path: Path
    output: Path
    development_stages: Path
    test_inpaint_root: Path
    test_pose_root: Path
    shots_master: Path
    groups_file: Path
    split_file: Path
    clean_test_labels: Path
    court_json: dict[str, dict[str, Path | None]]


@dataclass(frozen=True)
class RefitContext:
    """Values passed to resumable work items."""

    config: RefitConfig
    development: retrain.Development
    resolution_by_video: dict[str, tuple[float, float]]


@dataclass(frozen=True)
class VideoSource:
    """Paths and clock values needed to build or run one saved video."""

    split: str
    identity: str
    fps: float
    frame_count: int
    resolution: tuple[float, float]
    track_path: Path
    guard_path: Path
    pose_dir: Path
    court_json: Path
    court_dir: Path


def selected_test_ids() -> tuple[int, ...]:
    """Return the fixed ShuttleSet22 population without its misaligned video 15."""
    return tuple(video_id for video_id in retrain.TEST_VIDEO_IDS if video_id != 15)


def resolve_path(value: object, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty path')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: Path, output: Path | None) -> RefitConfig:
    """Resolve extract paths beside the config and label/split paths in the repo."""
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text())
    expected = {
        'schema', 'development_stages', 'test_inpaint_root', 'test_pose_root', 'shots_master', 'groups_file',
        'split_file', 'clean_test_labels', 'court_detector_json',
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError(f'{path}: expected fields {sorted(expected)}')
    if payload['schema'] != 'good-court-refit-inputs/1':
        raise ValueError(f'{path}: unsupported input schema {payload["schema"]!r}')
    court_paths = payload['court_detector_json']
    if not isinstance(court_paths, dict) or set(court_paths) != {'development', 'test'}:
        raise ValueError(f'{path}: court_detector_json must contain development and test maps')
    resolved_courts: dict[str, dict[str, Path | None]] = {}
    for split, mapping in court_paths.items():
        if not isinstance(mapping, dict):
            raise TypeError(f'{path}: {split} court paths must be an object')
        resolved_courts[split] = {}
        for identity, value in mapping.items():
            if not isinstance(identity, str) or not identity:
                raise ValueError(f'{path}: court path keys must be non-empty strings')
            resolved_courts[split][identity] = None if value is None else resolve_path(
                value, path.parent, f'{split}.{identity} court JSON',
            )
    repo_paths = {
        name: resolve_path(payload[name], REPOSITORY, name)
        for name in ('shots_master', 'groups_file', 'split_file', 'clean_test_labels')
    }
    destination = (output or DEFAULT_OUTPUT).expanduser().resolve()
    return RefitConfig(
        path, destination,
        resolve_path(payload['development_stages'], path.parent, 'development_stages'),
        resolve_path(payload['test_inpaint_root'], path.parent, 'test_inpaint_root'),
        resolve_path(payload['test_pose_root'], path.parent, 'test_pose_root'),
        repo_paths['shots_master'], repo_paths['groups_file'], repo_paths['split_file'], repo_paths['clean_test_labels'],
        resolved_courts,
    )


def settings_for(config: RefitConfig, output: Path) -> retrain.RunSettings:
    return retrain.RunSettings(
        output=output, dev_stages=config.development_stages, test_inputs=config.test_inpaint_root,
        shots_master=config.shots_master, groups_file=config.groups_file, split_file=config.split_file,
        clean_labels=config.clean_test_labels, pr149_root=REPOSITORY / 'scratch/contact_det_closing_pass',
        contact_records=None, frozen_dev_features=None, frozen_test_features=None, shuttleset22_annotations=None,
    )


def development_resolutions(config: RefitConfig, development: retrain.Development) -> dict[str, tuple[float, float]]:
    """Read native dimensions from the fixed development split, which also pins FPS."""
    split = retrain.load_json(config.split_file)
    records = {record['fixture']: record for record in split['videos']}
    if set(records) != set(development.id_by_video):
        raise ValueError('development dimensions and fixed split do not name the same videos')
    resolutions = {}
    for identity, record in records.items():
        width, height = float(record['width']), float(record['height'])
        if not np.isfinite((width, height)).all() or width <= 0 or height <= 0:
            raise ValueError(f'{identity}: invalid native dimensions {(width, height)} in the fixed split')
        if float(record['fps']) != development.fps_by_video[identity]:
            raise ValueError(f'{identity}: FPS differs between development metadata and the fixed split')
        resolutions[identity] = (width, height)
    return resolutions


def source_for(context: RefitContext, split: str, identity: str) -> VideoSource:
    config = context.config
    court_json = config.court_json[split][identity]
    if court_json is None:
        raise ValueError(f'{split} {identity}: court detector JSON path is still blank in {config.config_path}')
    if split == 'development':
        stages = config.development_stages / 'stages'
        shuttle = stages / 'shuttle' / identity
        track_path, guard_path = shuttle / TRACK_FILENAME, shuttle / GUARD_CODES_FILENAME
        pose_dir = stages / 'pose' / identity
        frame_count = len(load_npy_xz(track_path))
        resolution = context.resolution_by_video[identity]
        fps = context.development.fps_by_video[identity]
    else:
        video_id = int(identity)
        directory = test_video_directory(config.test_inpaint_root, video_id)
        track_path = directory / TEST_TRACK_FILENAME
        guard_path = directory / TEST_GUARD_CODES_FILENAME
        pose_dir = config.test_pose_root / directory.name
        frame_count = len(load_npy_xz(track_path))
        resolution, fps = TEST_RESOLUTION, TEST_FPS
    return VideoSource(
        split, identity, fps, frame_count, resolution, track_path, guard_path, pose_dir,
        court_json, config.output / 'shared' / 'court' / split / identity,
    )


def source_files(source: VideoSource) -> tuple[Path, ...]:
    pose_files = tuple(source.pose_dir / filename for filename in (
        'pose_kps.npy.xz', 'pose_bboxes.npy.xz', 'pose_scores.npy.xz', 'pose_kp_scores.npy.xz', 'pose_ndet.npy.xz',
    ))
    return source.track_path, source.guard_path, *pose_files, source.court_json


def check_video_source(source: VideoSource) -> tuple[Path, ...]:
    missing = [path for path in source_files(source) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'{source.split} {source.identity}: missing inputs: {missing}')
    track = load_npy_xz(source.track_path)
    if len(track) != source.frame_count:
        raise ValueError(f'{source.identity}: track has {len(track)} frames, expected {source.frame_count}')
    guard = load_npy_xz(source.guard_path)
    if guard.shape != (source.frame_count,):
        raise ValueError(f'{source.identity}: guard codes have shape {guard.shape}, expected ({source.frame_count},)')
    load_pose_arrays(source.pose_dir, source.frame_count)
    try:
        read_detector_scenes(
            load_detector_json(source.court_json), frame_count=source.frame_count, native_size=source.resolution,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'{source.split} {source.identity} ({source.court_json}): {error}') from error
    return source_files(source)


def load_detector_json(path: Path) -> dict[str, object]:
    if path.name.endswith('.gz'):
        return load_json_gz(path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f'{path}: court detector result must be a JSON object')
    return payload


def read_label_sources(context: RefitContext) -> tuple[
    dict[str, list[tuple[str, list[tuple[int, Any]]]]], dict[int, list[tuple[str, list[tuple[int, Any]]]]]
]:
    settings = settings_for(context.config, context.config.output / 'shared')
    shots = pd.read_csv(settings.shots_master, usecols=['vid', 'set_id', 'rally', 'frame_num', 'player_side'])
    development_labels = {
        identity: retrain.development_label_rows(shots, video_id)
        for identity, video_id in context.development.id_by_video.items()
    }
    test_labels = retrain.trusted_test_label_rows(settings.clean_labels)
    if tuple(sorted(test_labels)) != selected_test_ids():
        raise ValueError('trusted ShuttleSet22 labels must cover the fixed 46-video population without video 15')
    return development_labels, test_labels


def expected_id_maps(context: RefitContext) -> dict[str, tuple[str, ...]]:
    return {
        'development': context.development.id_order(list(context.development.id_by_video)),
        'test': tuple(str(video_id) for video_id in selected_test_ids()),
    }


def validate_court_map(context: RefitContext) -> None:
    expected = expected_id_maps(context)
    for split, identities in expected.items():
        mapping = context.config.court_json[split]
        if set(mapping) != set(identities):
            missing, extra = sorted(set(identities) - set(mapping)), sorted(set(mapping) - set(identities))
            raise ValueError(f'{split} court map must match the pinned video IDs; missing={missing}, extra={extra}')
        unresolved = [identity for identity in identities if mapping[identity] is None]
        if unresolved:
            raise ValueError(f'{split} court detector JSON locations are unresolved for: {", ".join(unresolved)}')


def input_snapshot(context: RefitContext, files: set[Path]) -> dict[str, Any]:
    for path in (context.config.shots_master, context.config.groups_file, context.config.split_file,
                 context.config.clean_test_labels):
        if not path.is_file():
            raise FileNotFoundError(f'missing label or split input: {path}')
        files.add(path)
    entries = []
    for path in sorted(files):
        stat = path.stat()
        entries.append({'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return {
        'schema': 'good-court-refit-snapshot/1',
        'source_paths': {
            'development_stages': str(context.config.development_stages),
            'test_inpaint_root': str(context.config.test_inpaint_root),
            'test_pose_root': str(context.config.test_pose_root),
            'court_detector_json': {
                split: {identity: str(path) for identity, path in mapping.items()}
                for split, mapping in context.config.court_json.items()
            },
        },
        'files': entries,
    }


def preflight(context: RefitContext, *, compare_snapshot: bool) -> dict[str, Any]:
    validate_court_map(context)
    settings = settings_for(context.config, context.config.output / 'shared')
    development = retrain.load_development(settings)
    if set(development.group_by_video.values()) != {*retrain.TRAINING_GROUPS, retrain.VALIDATION_GROUP}:
        raise ValueError('development groups must be A-D for training and V for validation')
    files: set[Path] = set()
    for identity in expected_id_maps(context)['development']:
        files.update(check_video_source(source_for(context, 'development', identity)))
    for identity in expected_id_maps(context)['test']:
        files.update(check_video_source(source_for(context, 'test', identity)))
    read_label_sources(context)
    snapshot = input_snapshot(context, files)
    snapshot_path = context.config.output / SNAPSHOT
    if compare_snapshot and snapshot_path.is_file():
        saved = json.loads(snapshot_path.read_text())
        if saved != snapshot:
            raise ValueError(f'{snapshot_path}: inputs differ from the saved run; use a new --output directory')
    return snapshot


def court_conversion_job(context: RefitContext, split: str, identity: str) -> dict[str, Any]:
    started = time.perf_counter()
    source = source_for(context, split, identity)
    pose = load_pose_arrays(source.pose_dir, source.frame_count)
    scenes = read_detector_scenes(
        load_detector_json(source.court_json), frame_count=source.frame_count, native_size=source.resolution,
    )
    evidence = build_court_detector_evidence(
        identity, identity, source.resolution, source.resolution, scenes,
        pose.bboxes, pose.scores, pose.ndet,
    )
    raw_cuts = tuple((scene.start_frame, scene.end_frame) for scene in scenes)
    persist_court_vision(
        source.court_dir, video_id=identity, court=CourtVision(raw_cuts, evidence),
        frame_count=source.frame_count, resolution=source.resolution,
    )
    summary = {
        'video': identity, 'split': split, 'frame_count': source.frame_count, 'scenes': len(scenes),
        'accepted_scenes': int(sum(record.scene_valid for record in evidence.scene_records)),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(source.court_dir / 'summary.json.gz', summary)
    return summary


def load_video_inputs(context: RefitContext, split: str, identity: str) -> OldVideoInputs:
    source = source_for(context, split, identity)
    track = load_npy_xz(source.track_path)
    guard_codes = load_npy_xz(source.guard_path)
    pose: PoseArrays = load_pose_arrays(source.pose_dir, source.frame_count)
    court = load_court_vision(
        source.court_dir, video_id=identity, frame_count=source.frame_count, resolution=source.resolution,
    )
    if court.evidence.inputs is None:
        raise ValueError(f'{identity}: converted court JSON has no operational court inputs')
    return OldVideoInputs(
        identity, source.fps, track, guard_codes, pose, court.raw_cuts, court.evidence.inputs,
        court.evidence.keep_vote, court.evidence.court_present,
    )


def write_labels_job(context: RefitContext) -> dict[str, Any]:
    dev_labels, test_labels = read_label_sources(context)
    root = context.config.output / 'shared' / 'labels'
    counts = {'development_videos': len(dev_labels), 'test_videos': len(test_labels)}
    for identity, rows in dev_labels.items():
        retrain.write_label_csv(root / 'development' / f'{identity}.csv', rows)
    for video_id, rows in test_labels.items():
        retrain.write_label_csv(root / 'test' / f'{video_id}.csv', rows)
    summary = {**counts, 'excluded_test_video': 15,
               'labelled_test_rallies': sum(len(rows) for rows in test_labels.values())}
    save_json_gz(root / 'summary.json.gz', summary)
    return summary


def feature_job(context: RefitContext, identity: str) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = load_video_inputs(context, 'development', identity)
    evidence = retrain.heuristic_evidence(inputs)
    features: ContactFeatures = features_from_evidence(evidence)
    retrain.dump_atomically(features, context.config.output / 'shared' / 'features' / f'{identity}.joblib')
    rallies = load_labels(context.config.output / 'shared' / 'labels' / 'development' / f'{identity}.csv',
                          len(evidence.track))
    summary = {
        'video': identity, 'group': context.development.group_by_video[identity],
        'frame_count': len(evidence.track), 'feature_rows': len(features.rows),
        'labelled_rallies': len(rallies), 'labelled_contacts': sum(len(rally.frames) for rally in rallies),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(context.config.output / 'shared' / 'features' / f'{identity}.json.gz', summary)
    return summary


def contact_fit_job(context: RefitContext, name: str) -> dict[str, Any]:
    started = time.perf_counter()
    videos, fit_order = contact_fit_plan(context.development, name)
    inputs = []
    for identity in videos:
        feature = joblib.load(context.config.output / 'shared' / 'features' / f'{identity}.joblib')
        frame_count = load_json_gz(context.config.output / 'shared' / 'features' / f'{identity}.json.gz')['frame_count']
        rallies = load_labels(context.config.output / 'shared' / 'labels' / 'development' / f'{identity}.csv', frame_count)
        frames = np.asarray(sorted(frame for rally in rallies for frame in rally.frames), dtype=np.int32)
        inputs.append(ContactTrainingVideo(identity, feature.rows, frames,
                                           context.development.fps_by_video[identity]))
    fit = fit_contact_model(inputs, DEFAULT_TRAINING_SETTINGS.contact, fit_video_order=fit_order)
    root = context.config.output / 'shared' / 'contact'
    retrain.dump_atomically(fit.model, root / f'{name}.joblib')
    summary = {
        'fit': name, 'sampling_video_order': list(videos), 'fit_video_order': list(fit_order),
        'selected_rows': len(fit.selection.labels), 'positive_rows': int(fit.selection.labels.sum()),
        'iterations': int(fit.model.n_iter_), 'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(root / f'{name}.json.gz', summary)
    return summary


def contact_fit_plan(development: retrain.Development, name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Keep historical per-fit sampling order, then sort selected rows by ShuttleSet ID."""
    sampling_order = retrain.contact_fit_videos(development)[name]
    return sampling_order, development.id_order(sampling_order)


def pool_job(context: RefitContext, arm: str, identity: str) -> dict[str, Any]:
    started = time.perf_counter()
    group = context.development.group_by_video[identity]
    inputs = load_video_inputs(context, 'development', identity)
    evidence = retrain.heuristic_evidence(inputs)
    features: ContactFeatures = joblib.load(context.config.output / 'shared' / 'features' / f'{identity}.joblib')
    label_path = context.config.output / 'shared' / 'labels' / 'development' / f'{identity}.csv'
    rallies = load_labels(label_path, len(evidence.track))
    contact_model = joblib.load(context.config.output / 'shared' / 'contact' / f'held_{group}.joblib')
    video = sequence_training_video(
        VideoInput(identity, group, Split.TRAIN, label_path), evidence, features, rallies, contact_model,
        ARM_CONFIGS[arm], SideGeometry.VIDEO,
    )
    root = context.config.output / arm / 'pools'
    retrain.dump_atomically(video, root / f'{identity}.joblib')
    summary = {
        'video': identity, 'group': group, 'initial_sequences': len(video.initial),
        'events': len(video.events), 'options': len(video.pool.options),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(root / f'{identity}.json.gz', summary)
    return summary


def sequence_fit_job(context: RefitContext, arm: str) -> dict[str, Any]:
    started = time.perf_counter()
    videos = [joblib.load(context.config.output / arm / 'pools' / f'{identity}.joblib')
              for identity in context.development.videos(retrain.TRAINING_GROUPS)]
    sequence_fit, confidence, labels = retrain.fit_downstream(videos)
    root = context.config.output / arm
    for video in videos:
        prediction = sequence_fit.cross_fitted[video.identity]
        save_json_gz(root / 'dev_heldout' / f'{video.identity}.json.gz',
                     retrain.stream_payload(prediction.sequences, prediction.events))
    settings = DEFAULT_TRAINING_SETTINGS
    for bundle, contact_fit_name in (('bundle', 'final_40'), ('bundle_v32', 'v_32')):
        contact_model = joblib.load(context.config.output / 'shared' / 'contact' / f'{contact_fit_name}.joblib')
        save_models(AnnotatorModels(
            contact_model, sequence_fit.models, confidence, ARM_CONFIGS[arm], settings.preprocessing,
            SideGeometry.VIDEO,
        ), root / bundle)
    summary = {
        'training_videos': [video.identity for video in videos], 'side_geometry': SideGeometry.VIDEO.value,
        'reject_masked_without_player': ARM_CONFIGS[arm].reject_masked_without_player,
        'confidence_rows': len(labels), 'confidence_correct': int((labels == 1).sum()),
        'confidence_wrong': int((labels == 0).sum()), 'confidence_unjudgeable': int((labels == -1).sum()),
        'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(root / 'sequence' / 'summary.json.gz', summary)
    return summary


def evaluate_job(context: RefitContext, split: str, arm: str, identity: str) -> dict[str, Any]:
    started = time.perf_counter()
    bundle = 'bundle_v32' if split == 'validation' else 'bundle'
    model_arm = 'base' if arm == 'base_inference_veto' else arm
    models = load_models(context.config.output / model_arm / bundle)
    if arm == 'base_inference_veto':
        models = replace(
            models,
            contact_settings=replace(models.contact_settings, reject_masked_without_player=True),
        )
    input_split = 'development' if split == 'validation' else 'test'
    inputs = load_video_inputs(context, input_split, identity)
    capture = RunCapture()
    result = run_old_video(inputs, models, capture)
    prediction = capture.hybrid
    if prediction is None:
        raise ValueError(f'{identity}: full annotation did not capture its prediction')
    confidence = dict(zip(map(str, prediction.confidence.features.span_ids),
                          map(float, prediction.confidence.scores), strict=True))
    stream = {**retrain.stream_payload(prediction.refined.sequences, prediction.refined.events),
              'confidence': confidence}
    label_split = 'development' if split == 'validation' else 'test'
    label_path = context.config.output / 'shared' / 'labels' / label_split / f'{identity}.csv'
    rallies = load_labels(label_path, len(inputs.track))
    score = retrain.score_stream(
        prediction.refined.sequences, prediction.refined.events, rallies, inputs.fps, confidence,
    )
    output = context.config.output / arm / 'eval' / split / identity
    save_json_gz(output / ANNOTATOR_RESULT_FILENAME, annotation_result_payload(identity, result))
    save_json_gz(output / 'stream.json.gz', stream)
    summary = {
        'video': identity, 'split': split, 'arm': arm, 'frame_count': len(inputs.track), 'fps': inputs.fps,
        **score, 'seconds': round(time.perf_counter() - started, 1),
    }
    save_json_gz(output / 'scores.json.gz', summary)
    return {'video': identity, 'sections': len(prediction.refined.sequences),
            'contacts': len(prediction.refined.events), 'seconds': summary['seconds']}


def require_stage_success(stage: str, failures: list[str]) -> None:
    if failures:
        raise RuntimeError(f'{stage} failed for: {", ".join(failures)}')


def write_or_check_snapshot(config: RefitConfig, snapshot: dict[str, Any], *, write: bool) -> None:
    path = config.output / SNAPSHOT
    if path.is_file():
        if json.loads(path.read_text()) != snapshot:
            raise ValueError(f'{path}: inputs differ from the saved run; use a new --output directory')
    elif write:
        config.output.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2) + '\n')
    elif write is False:
        raise FileNotFoundError(f'{path}: no completed/prepared run exists for evaluation')


def run_pair(context: RefitContext, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    progress = Progress(output)
    jobs, threads = 8, 4
    progress.write('paired refit start; max worker budget 8 x 4 threads; base then veto')
    label_done = output / 'shared' / 'labels' / 'summary.json.gz'
    require_stage_success('labels', run_stage(
        'labels', [Item('labels', label_done, write_labels_job, (context,))], jobs, threads, output, progress,
    ))
    conversion_items = []
    for split, identities in expected_id_maps(context).items():
        for identity in identities:
            source = source_for(context, split, identity)
            conversion_items.append(Item(
                f'{split}/{identity}', source.court_dir / 'summary.json.gz', court_conversion_job,
                (context, split, identity),
            ))
    require_stage_success('court-conversion', run_stage(
        'court-conversion', conversion_items, jobs, threads, output, progress,
    ))
    feature_items = [
        Item(identity, output / 'shared' / 'features' / f'{identity}.json.gz', feature_job, (context, identity))
        for identity in context.development.id_order(list(context.development.id_by_video))
    ]
    require_stage_success('features', run_stage('features', feature_items, jobs, threads, output, progress))
    fit_items = []
    fit_names = ('final_40', 'held_A', 'held_B', 'held_C', 'held_D', 'v_32')
    for name in fit_names:
        fit_items.append(Item(name, output / 'shared' / 'contact' / f'{name}.json.gz', contact_fit_job,
                              (context, name)))
    # Six independent fits can use 30 cores; the other parallel stages use 8 x 4.
    require_stage_success('contact-fits', run_stage('contact-fits', fit_items, 6, 5, output, progress))
    for arm in ('base', 'veto'):
        progress.write(f'{arm} full build start')
        pool_items = [
            Item(identity, output / arm / 'pools' / f'{identity}.json.gz', pool_job, (context, arm, identity))
            for identity in context.development.videos(retrain.TRAINING_GROUPS)
        ]
        require_stage_success(f'{arm}-pools', run_stage('pools', pool_items, jobs, threads, output / arm, progress))
        sequence_item = Item(
            'sequence-fit', output / arm / 'sequence' / 'summary.json.gz', sequence_fit_job, (context, arm),
        )
        require_stage_success('sequence-fit', run_stage(
            'sequence-fit', [sequence_item], jobs, threads, output / arm, progress,
        ))
        progress.write(f'{arm} full build complete')
    progress.write('paired refit complete; evaluation has not been run')


def evaluate(context: RefitContext, split: str, arm: str, output: Path) -> None:
    if arm not in ARMS:
        raise ValueError(f'unknown arm {arm!r}; choose from {", ".join(ARMS)}')
    if arm == 'base_inference_veto' and split not in ('validation', 'test'):
        raise ValueError('inference-only veto evaluation needs validation or test')
    identities = (tuple(identity for identity, group in context.development.group_by_video.items()
                        if group == retrain.VALIDATION_GROUP)
                  if split == 'validation' else tuple(str(video_id) for video_id in selected_test_ids()))
    items = [
        Item(identity, output / arm / 'eval' / split / identity / 'scores.json.gz', evaluate_job,
             (context, split, arm, identity))
        for identity in identities
    ]
    progress = Progress(output)
    require_stage_success(f'{arm}-{split}-evaluation', run_stage(
        'evaluate', items, 8, 4, output / arm, progress,
    ))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('inputs.json'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true', help='validate inputs only; this is the default')
    action.add_argument('--run', action='store_true', help='fit base, then veto, using shared prepared inputs')
    action.add_argument('--evaluate', choices=('validation', 'test'), help='run full annotation and score saved models')
    parser.add_argument('--arm', choices=ARMS, help='evaluation arm required with --evaluate')
    return parser


def record_invocation(config: RefitConfig, action: str, arm: str | None) -> None:
    """Retain the environment and checkout used for each fit or evaluation call."""
    path = config.output / 'run.json'
    records = json.loads(path.read_text()) if path.is_file() else []
    versions = {name: version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'joblib')}
    versions['python'] = platform.python_version()
    if records and records[0]['versions'] != versions:
        raise ValueError(f'{path}: Python/package versions changed; restore the fit environment or use a new --output')
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, capture_output=True, text=True, check=False)
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=REPOSITORY, capture_output=True, text=True, check=False)
    revision = commit.stdout.strip() if commit.returncode == 0 else None
    if records and records[-1]['git_commit'] != revision:
        print('Code revision changed: reuse is safe only for changes that leave computed results unchanged. '
              'Use a fresh --output for implementation changes.', flush=True)
    records.append({
        'started': utc_now(), 'action': action, 'arm': arm, 'config': str(config.config_path),
        'versions': versions, 'git_commit': revision,
        'git_status': status.stdout.splitlines() if status.returncode == 0 else None,
    })
    path.write_text(json.dumps(records, indent=2) + '\n')


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.evaluate is None and arguments.arm is not None:
        parser.error('--arm is only valid with --evaluate')
    if arguments.evaluate is not None and arguments.arm is None:
        parser.error('--evaluate requires --arm')
    try:
        config = load_config(arguments.config, arguments.output)
        development = retrain.load_development(settings_for(config, config.output / 'shared'))
        validate_court_map(RefitContext(config, development, {}))
        context = RefitContext(config, development, development_resolutions(config, development))
        snapshot = preflight(context, compare_snapshot=False)
        if arguments.run or arguments.evaluate:
            write_or_check_snapshot(config, snapshot, write=arguments.run)
            record_invocation(config, 'fit' if arguments.run else arguments.evaluate, arguments.arm)
        elif (config.output / SNAPSHOT).is_file():
            write_or_check_snapshot(config, snapshot, write=False)
        if arguments.run:
            run_pair(context, config.output)
        elif arguments.evaluate:
            evaluate(context, arguments.evaluate, arguments.arm, config.output)
        else:
            print(f'Inputs are valid: {len(context.development.id_by_video)} development videos, '
                  f'{len(selected_test_ids())} ShuttleSet22 test videos; no fitting or evaluation was run.')
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
