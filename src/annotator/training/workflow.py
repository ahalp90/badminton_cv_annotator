"""Fit and evaluate the annotator using saved extraction outputs and contact labels.

The manifest assigns videos to disjoint training, validation and test groups.
Fitting reads training videos only. The full first selector learns from contact
scores made without each video's group. Temporary selector fits reuse one fixed
contact model while excluding the group whose final predictions train confidence.
"""

import argparse
import csv
import gzip
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.score_candidates import (
    ContactModelConfig,
)
from annotator.inference.learned import (
    contact_features_from_inputs,
    predict_rallies_and_contacts,
)
from annotator.inference.prepare_annotation_inputs import (
    PreparedAnnotationInputs,
    prepare_annotation_inputs,
)
from annotator.models import AnnotatorModels, load_models, save_models
from annotator.saved_inputs import load_saved_video_inputs
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from annotator.training.followup import (
    PreparedTrainingVideo,
    fit_followup_models,
    fit_selector_from_files,
    prepare_training_data,
    save_training_inputs,
)
from annotator.training.rally import RallyFitConfig
from annotator.training.sequences import (
    LabelledRally,
    match_contacts,
    whole_rally_correctness,
)
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf, SmoothingMode
from dataset_builder.vision import save_json_gz
from shared.frame_timing import frames_at_fps

EVALUATION_TOLERANCE_AT_30_FPS = 10


class Split(StrEnum):
    """Training, validation or test partition named in the manifest."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


@dataclass(frozen=True)
class VideoInput:
    """Saved video ID, contact-label file and split assignment.

    Videos sharing a group are withheld together during fitting.
    """

    video_id: str
    group: str
    split: Split
    labels: Path


@dataclass(frozen=True)
class TrainingManifest:
    """Saved extraction run and labelled video assignments.

    Each video appears once. Groups cannot cross train/validation/test splits.
    """

    run_dir: Path
    videos: tuple[VideoInput, ...]

    def __post_init__(self) -> None:
        video_ids = set()
        group_splits = {}
        for video in self.videos:
            if video.video_id in video_ids:
                raise ValueError(f'duplicate video id: {video.video_id}')
            video_ids.add(video.video_id)
            if video.group in group_splits and group_splits[video.group] != video.split:
                raise ValueError(f'group {video.group!r} occurs in more than one split')
            group_splits[video.group] = video.split


@dataclass(frozen=True)
class TrainingSettings:
    """Settings for contact scoring, both rally selectors and final confidence."""

    contact: ContactFitConfig = field(default_factory=ContactFitConfig)
    contact_prediction: ContactModelConfig = field(default_factory=ContactModelConfig)
    confidence: TreeFitConfig = field(default_factory=lambda: TreeFitConfig(max_iter=100, max_leaf_nodes=7))
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)
    rally: RallyFitConfig = field(default_factory=RallyFitConfig)


DEFAULT_TRAINING_SETTINGS = TrainingSettings()


def load_training_settings(path: Path) -> TrainingSettings:
    """Read the settings saved beside a fitted model bundle."""
    with gzip.open(path, 'rt') as handle:
        record = json.load(handle)
    preprocessing = record['preprocessing']
    preprocessing['smoothing_mode'] = SmoothingMode(preprocessing['smoothing_mode'])
    preprocessing['rejected_grades'] = frozenset(preprocessing['rejected_grades'])
    rally = record['rally']
    rally['tree'] = TreeFitConfig(**rally['tree'])
    return TrainingSettings(
        contact=ContactFitConfig(**record['contact']), contact_prediction=ContactModelConfig(**record['contact_prediction']),
        confidence=TreeFitConfig(**record['confidence']), preprocessing=BaseAnnotatorConfig(**preprocessing),
        rally=RallyFitConfig(**rally),
    )


def fit_from_manifest(
    manifest: TrainingManifest, output: Path,
    settings: TrainingSettings = DEFAULT_TRAINING_SETTINGS,
    *, workers: int = 1, gpu_fits: int = 1,
) -> AnnotatorModels:
    """Fit contact scoring, the two rally selectors and final confidence.

    Keep the first selector's ordinary training pool unchanged. Train correction
    on ordinary examples plus pieces from held-group first-selector predictions.
    Confidence judges final predictions from a pipeline that excluded each group.
    """
    entries = videos_for_split(manifest, Split.TRAIN)
    groups = tuple(dict.fromkeys(entry.group for entry in entries))
    if len(groups) < 3:
        raise ValueError('sequence training needs at least three distinct training groups')
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.training-', dir=output) as temporary:
        working = Path(temporary)
        videos, contact_inputs = prepare_training_videos(manifest, entries, settings, working)
        contact_model, held_contacts = fit_training_contacts(videos, contact_inputs, groups, settings.contact, working)
        del contact_inputs
        ordinary_paths = {}
        for video in videos:
            ordinary_paths[video.video_id] = prepare_training_data(
                video, held_contacts[video.group], settings.contact_prediction, settings.rally,
                working / 'ordinary' / f'{video.video_id}.joblib',
            )
        first_selector_path = fit_selector_from_files(
            list(ordinary_paths.values()), (), settings.rally, working / 'first-selector.joblib',
        )
        followup_selector, review_model = fit_followup_models(
            videos, ordinary_paths, held_contacts, settings.contact_prediction, settings.rally,
            settings.confidence, working / 'followup', workers=workers, gpu_fits=gpu_fits,
        )
        models = AnnotatorModels(
            contact_model=contact_model, rally_selector=joblib.load(first_selector_path),
            followup_selector=followup_selector, review_model=review_model,
            contact_settings=settings.contact_prediction, preprocessing=settings.preprocessing,
        )
        save_models(models, output)
        save_json_gz(output / 'fit-settings.json.gz', asdict(settings))
    return models


def prepare_training_videos(
    manifest: TrainingManifest, entries: Sequence[VideoInput], settings: TrainingSettings, working: Path,
) -> tuple[list[PreparedTrainingVideo], list[ContactTrainingVideo]]:
    """Prepare each video once; keep only contact rows in memory for contact fitting."""
    videos, contact_inputs = [], []
    for entry in entries:
        evidence = prepare_saved_video_inputs(manifest.run_dir, entry.video_id, settings.preprocessing)
        features = contact_features_from_inputs(evidence)
        rallies = load_labels(entry.labels, len(evidence.track))
        frames = np.asarray(sorted(frame for rally in rallies for frame in rally.frames), dtype=np.int32)
        contact_inputs.append(ContactTrainingVideo(entry.video_id, features.rows, frames, evidence.fps))
        videos.append(save_training_inputs(
            entry.video_id, entry.group, evidence, features, rallies,
            working / 'inputs' / f'{entry.video_id}.joblib',
        ))
    return videos, contact_inputs


def fit_training_contacts(
    videos: Sequence[PreparedTrainingVideo], contact_inputs: Sequence[ContactTrainingVideo],
    groups: tuple[str, ...], settings: ContactFitConfig, working: Path,
) -> tuple[Any, dict[str, Path]]:
    """Fit the full contact model and one model excluding each training group."""
    try:
        model = fit_contact_model(contact_inputs, settings).model
    except ValueError as error:
        raise ValueError(f'contact-model fitting failed: {error}') from error
    held_contacts = {}
    for held_group in groups:
        selected = [contact for video, contact in zip(videos, contact_inputs, strict=True) if video.group != held_group]
        try:
            held_model = fit_contact_model(selected, settings).model
        except ValueError as error:
            raise ValueError(f'contact-model fitting with group {held_group!r} held out failed: {error}') from error
        destination = working / f'contact-without-{held_group}.joblib'
        joblib.dump(held_model, destination, compress=3)
        held_contacts[held_group] = destination
    return model, held_contacts


def evaluate_manifest(
    manifest: TrainingManifest, models: AnnotatorModels, split: Split, output: Path,
) -> dict[str, Any]:
    """Evaluate the requested validation/test split and save per-video and total scores."""
    if split not in (Split.VALIDATION, Split.TEST):
        raise ValueError('evaluation split must be validation or test')
    if not output.name.endswith('.json.gz'):
        raise ValueError('evaluation output must end in .json.gz')
    rows = []
    count_names = ('labelled_contacts', 'predicted_contacts', 'matched_contacts',
                   'whole_rally_judged', 'whole_rally_correct', 'whole_rally_unjudgeable')
    totals: dict[str, int | float] = dict.fromkeys(count_names, 0)
    for entry in videos_for_split(manifest, split):
        evidence = prepare_saved_video_inputs(manifest.run_dir, entry.video_id, models.preprocessing)
        rallies = load_labels(entry.labels, len(evidence.track))
        prediction = predict_rallies_and_contacts(evidence, models)
        counts = evaluation_counts(prediction.events, prediction.sequences, rallies, evidence.fps)
        rows.append({'id': entry.video_id, **counts})
        for name in count_names:
            totals[name] += int(counts[name])
    predicted = totals['predicted_contacts']
    labelled = totals['labelled_contacts']
    totals['contact_precision'] = totals['matched_contacts'] / predicted if predicted else 0.0
    totals['contact_recall'] = totals['matched_contacts'] / labelled if labelled else 0.0
    report = {
        'split': split.value, 'tolerance_at_30_fps': EVALUATION_TOLERANCE_AT_30_FPS, 'videos': rows, 'total': totals,
    }
    save_json_gz(output, report)
    return report


def evaluation_counts(events: Sequence[ContactEvent], sequences: Sequence[ContactSequence],
                      rallies: Sequence[LabelledRally], fps: float) -> dict[str, int | float]:
    """Count matched contacts and correct, wrong or unjudgeable final rallies."""
    labels = sorted(frame for rally in rallies for frame in rally.frames)
    matches = match_contacts(
        labels, [event.frame for event in events], frames_at_fps(EVALUATION_TOLERANCE_AT_30_FPS, fps),
    )
    correctness = [whole_rally_correctness(
        sequence, rallies, fps, tolerance_at_30_fps=EVALUATION_TOLERANCE_AT_30_FPS,
    ) for sequence in sequences]
    return {
        'labelled_contacts': len(labels), 'predicted_contacts': len(events), 'matched_contacts': len(matches),
        'contact_precision': len(matches) / len(events) if events else 0.0,
        'contact_recall': len(matches) / len(labels) if labels else 0.0,
        'whole_rally_judged': sum(value >= 0 for value in correctness),
        'whole_rally_correct': correctness.count(1), 'whole_rally_unjudgeable': correctness.count(-1),
    }


# Prepare saved inputs and labels

def prepare_saved_video_inputs(run_dir: Path, video_id: str, preprocessing: BaseAnnotatorConfig) -> PreparedAnnotationInputs:
    """Load saved metadata, pose, court and shuttle arrays, then prepare annotation inputs."""
    try:
        saved = load_saved_video_inputs(run_dir, video_id)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise ValueError(f'{video_id}: cannot load current metadata/pose/court/shuttle artefacts: {error}') from error
    court_inputs = saved.court.evidence.inputs
    assert court_inputs is not None
    pose = saved.pose
    return prepare_annotation_inputs(
        saved.track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
        video_config=scale_config_to_fps(preprocessing, float(saved.metadata.fps)),
        resolution=court_inputs.resolution,
        court_present=saved.court.evidence.court_present, homography_rows=court_inputs.homography_rows,
        shuttle_grades=saved.shuttle_grades,
    )


def load_labels(path: Path, frame_count: int) -> tuple[LabelledRally, ...]:
    """Read rally_id,frame,side CSV rows, grouped by rally and increasing in frame.

    Empty sides remain unknown. Each rally must occupy one contiguous CSV block.
    All frames must lie inside the source video timeline.
    """
    rallies = []
    seen = set()
    current = None
    frames: list[int] = []
    sides: list[CourtHalf | None] = []
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        expected_columns = {'rally_id', 'frame', 'side'}
        if reader.fieldnames is None or len(reader.fieldnames) != 3 or set(reader.fieldnames) != expected_columns:
            raise ValueError(f'{path}: labels need rally_id,frame,side columns')
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{line}: label row has the wrong column count')
            rally_id = row['rally_id'].strip()
            if not rally_id:
                raise ValueError(f'{path}:{line}: rally_id is empty')
            if rally_id != current:
                if rally_id in seen:
                    raise ValueError(f'{path}:{line}: rally rows must form a contiguous block')
                if current is not None:
                    rallies.append(LabelledRally(current, tuple(frames), tuple(sides)))
                seen.add(rally_id)
                current, frames, sides = rally_id, [], []
            try:
                frame = int(row['frame'])
                side_text = row['side'].strip()
                side = CourtHalf(side_text) if side_text else None
            except ValueError as error:
                raise ValueError(f'{path}:{line}: frame must be an integer and side Top, Bot or empty') from error
            if not 0 <= frame < frame_count:
                raise ValueError(f'{path}:{line}: contact frame {frame} lies outside the video')
            if frames and frame <= frames[-1]:
                raise ValueError(f'{path}:{line}: rally frames must be strictly increasing')
            frames.append(frame)
            sides.append(side)
    if current is not None:
        rallies.append(LabelledRally(current, tuple(frames), tuple(sides)))
    return tuple(rallies)


# Read video assignments

def load_manifest(path: Path) -> TrainingManifest:
    """Read and validate video assignments and paths without loading videos or labels."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or set(payload) != {'run_dir', 'videos'}:
        raise ValueError('manifest must contain run_dir and videos')
    run_dir = manifest_path(payload['run_dir'], path.resolve().parent, 'run_dir')
    if not isinstance(payload['videos'], list):
        raise TypeError('manifest videos must be a list')
    videos = []
    for entry in payload['videos']:
        if not isinstance(entry, dict) or set(entry) != {'id', 'group', 'split', 'labels'}:
            raise ValueError('each manifest video must contain id, group, split and labels')
        raw_video_id = entry['id']
        if isinstance(raw_video_id, bool) or not isinstance(raw_video_id, (str, int)):
            raise TypeError('video id must be a string or integer')
        video_id = str(raw_video_id)
        if not video_id or video_id in ('.', '..') or Path(video_id).name != video_id:
            raise ValueError(f'video id must be one directory name: {video_id!r}')
        group = entry['group']
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f'{video_id}: group must be a non-empty string')
        try:
            split = Split(entry['split'])
        except ValueError as error:
            raise ValueError(f'{video_id}: split must be train, validation or test') from error
        labels = manifest_path(entry['labels'], path.resolve().parent, f'{video_id} labels')
        videos.append(VideoInput(video_id, group, split, labels))
    return TrainingManifest(run_dir, tuple(videos))


def manifest_path(value: object, directory: Path, field_name: str) -> Path:
    """Resolve a required manifest path relative to the manifest's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty path string')
    path = Path(value)
    return (path if path.is_absolute() else directory / path).resolve()


def videos_for_split(manifest: TrainingManifest, split: Split) -> tuple[VideoInput, ...]:
    """Select exactly one named partition, failing clearly when it is empty."""
    videos = tuple(video for video in manifest.videos if video.split == split)
    if not videos:
        raise ValueError(f'manifest contains no {split.value} videos')
    return videos


# Command-line entry point

def main(argv: Sequence[str] | None = None) -> int:
    """Run fitting or held-out evaluation, reporting input errors without a traceback."""
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == 'fit':
            if min(arguments.threads, arguments.workers, arguments.gpu_fits) < 1:
                raise ValueError('--threads, --workers and --gpu-fits must be positive')
            defaults = load_training_settings(arguments.settings) if arguments.settings else DEFAULT_TRAINING_SETTINGS
            settings = replace(
                defaults,
                contact=replace(defaults.contact, device=arguments.device),
                confidence=replace(defaults.confidence, device=arguments.device),
                rally=replace(
                    defaults.rally,
                    tree=replace(defaults.rally.tree, device=arguments.device),
                ),
            )
            with threadpool_limits(limits=arguments.threads):
                fit_from_manifest(manifest, arguments.output, settings,
                                  workers=arguments.workers, gpu_fits=arguments.gpu_fits)
            print(f'Saved annotator models to {arguments.output}')
        else:
            models = load_models(arguments.models)
            evaluate_manifest(manifest, models, arguments.split, arguments.output)
            print(f'Saved {arguments.split.value} evaluation to {arguments.output}')
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


def argument_parser() -> argparse.ArgumentParser:
    """Command options for fitting models and evaluating a saved bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Fit a fresh complete bundle from the train split')
    fit.add_argument('--manifest', required=True, type=Path)
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--settings', type=Path, help='Reuse fit-settings.json.gz from a fitted bundle')
    fit.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='XGBoost fitting device')
    fit.add_argument('--threads', type=int, default=2)
    fit.add_argument('--workers', type=int, default=1, help='Parallel video-preparation workers')
    fit.add_argument('--gpu-fits', type=int, default=1, help='Simultaneous temporary selector fits')
    evaluate = commands.add_parser('evaluate', help='Score a saved bundle on validation or test videos')
    evaluate.add_argument('--manifest', required=True, type=Path)
    evaluate.add_argument('--models', required=True, type=Path)
    evaluate.add_argument('--split', choices=[Split.VALIDATION, Split.TEST], required=True, type=Split)
    evaluate.add_argument('--output', required=True, type=Path, help='Compressed report path ending in .json.gz')
    return parser
