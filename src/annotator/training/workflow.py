"""Fit and evaluate the annotator using saved extraction outputs and contact labels.

The manifest assigns videos to disjoint training, validation and test groups.
Fitting reads training videos only. Contact scores for selector training use
a tree fitted without the scored video's group. Review training likewise
uses selector predictions made while withholding that video's group.
"""

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from threadpoolctl import threadpool_limits

from annotator.config import BaseAnnotatorConfig, scale_config_to_fps
from annotator.contacts.build_features import ContactFeatures
from annotator.contacts.score_candidates import (
    ContactModelConfig,
    score_contact_features,
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
from annotator.rallies.build_contact_proposals import build_rally_context
from annotator.saved_inputs import load_saved_video_inputs
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from annotator.training.rally import (
    RallyFitConfig,
    fit_rally_models,
    fit_rally_review_model,
)
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
    match_contacts,
    whole_rally_correctness,
)
from annotator.training.trees import TreeFitConfig
from annotator.types import ContactEvent, ContactSequence, CourtHalf
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
class LoadedTrainingVideo:
    """Prepared inputs reused across contact and rally training.

    ``contact_input`` pairs candidate rows with flattened contact labels.
    ``rallies`` keeps those labels grouped, with player sides, for rally fitting.
    """

    entry: VideoInput
    evidence: PreparedAnnotationInputs
    features: ContactFeatures
    rallies: tuple[LabelledRally, ...]
    contact_input: ContactTrainingVideo


@dataclass(frozen=True)
class TrainingSettings:
    """Settings for fitting all three models; the command starts from these defaults."""

    contact: ContactFitConfig = field(default_factory=ContactFitConfig)
    contact_prediction: ContactModelConfig = field(default_factory=ContactModelConfig)
    confidence: TreeFitConfig = field(default_factory=lambda: TreeFitConfig(max_iter=100, max_leaf_nodes=7))
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)
    rally: RallyFitConfig = field(default_factory=RallyFitConfig)


DEFAULT_TRAINING_SETTINGS = TrainingSettings()


def fit_from_manifest(
    manifest: TrainingManifest, output: Path,
    settings: TrainingSettings = DEFAULT_TRAINING_SETTINGS,
) -> AnnotatorModels:
    """Fit and save the three models using the manifest's training videos.

    Fit the deployed contact model on all training videos. Prepare selector
    inputs using contact models that withhold the scored video's group, then
    fit the selector and review model. Save their shared preprocessing too.
    """
    entries = videos_for_split(manifest, Split.TRAIN)
    groups = tuple(dict.fromkeys(entry.group for entry in entries))
    if len(groups) < 3:
        raise ValueError('sequence training needs at least three distinct training groups')
    loaded = []
    for entry in entries:
        evidence = prepare_saved_video_inputs(manifest.run_dir, entry.video_id, settings.preprocessing)
        features = contact_features_from_inputs(evidence)
        rallies = load_labels(entry.labels, len(evidence.track))
        frames = np.asarray(sorted(frame for rally in rallies for frame in rally.frames), dtype=np.int32)
        contact_input = ContactTrainingVideo(entry.video_id, features.rows, frames, evidence.fps)
        loaded.append(LoadedTrainingVideo(entry, evidence, features, rallies, contact_input))
    try:
        contact_model = fit_contact_model(
            [video.contact_input for video in loaded], settings.contact,
        ).model
    except ValueError as error:
        raise ValueError(f'contact-model fitting failed: {error}') from error
    contact_models_by_held_group = {}
    for group in groups:
        try:
            contact_models_by_held_group[group] = fit_contact_model(
                [video.contact_input for video in loaded if video.entry.group != group], settings.contact,
            ).model
        except ValueError as error:
            raise ValueError(f'contact-model fitting with group {group!r} held out failed: {error}') from error
    training_videos = []
    for video in loaded:
        training_videos.append(sequence_training_video(
            video.entry.video_id, video.entry.group, video.evidence, video.features, video.rallies,
            contact_models_by_held_group[video.entry.group], settings.contact_prediction,
        ))
    # Release the large contact-feature arrays before building rally-training pools.
    del loaded, video, evidence, features, contact_input
    rally_fit = fit_rally_models(training_videos, settings.rally)
    review_model = fit_rally_review_model(
        training_videos, rally_fit.held_group_predictions, settings.confidence,
        tolerance_at_30_fps=settings.rally.review_tolerance_at_30_fps,
    )
    models = AnnotatorModels(
        contact_model=contact_model, rally_selector=rally_fit.selector, review_model=review_model,
        contact_settings=settings.contact_prediction, preprocessing=settings.preprocessing,
    )
    save_models(models, output)
    save_json_gz(output / 'fit-settings.json.gz', asdict(settings))
    return models


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
        counts = evaluation_counts(prediction.rally_prediction.events, prediction.rally_prediction.sequences, rallies, evidence.fps)
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


def sequence_training_video(
    video_id: str, group: str, evidence: PreparedAnnotationInputs, features: ContactFeatures, rallies: tuple[LabelledRally, ...],
    contact_model: Any, settings: ContactModelConfig,
) -> SequenceTrainingVideo:
    """Score contacts and prepare candidate sequences for rally fitting."""
    scored = score_contact_features(
        features.rows, contact_model, evidence.fps, settings,
    )
    context = build_rally_context(evidence, scored)
    return SequenceTrainingVideo(video_id, group, context, rallies)


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
            if arguments.threads < 1:
                raise ValueError('--threads must be positive')
            defaults = DEFAULT_TRAINING_SETTINGS
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
                fit_from_manifest(manifest, arguments.output, settings)
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
    fit.add_argument('--device', choices=('cpu', 'cuda'), default='cuda', help='XGBoost fitting device')
    fit.add_argument('--threads', type=int, default=2)
    evaluate = commands.add_parser('evaluate', help='Score a saved bundle on validation or test videos')
    evaluate.add_argument('--manifest', required=True, type=Path)
    evaluate.add_argument('--models', required=True, type=Path)
    evaluate.add_argument('--split', choices=[Split.VALIDATION, Split.TEST], required=True, type=Split)
    evaluate.add_argument('--output', required=True, type=Path, help='Compressed report path ending in .json.gz')
    return parser
