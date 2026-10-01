"""Retune and evaluate the hybrid annotator from dataset-builder artefacts.

The manifest names disjoint train/validation/test groups. Fitting reads only
training videos. Sequence and confidence training use held-group downstream
predictions. Each training video's contact scores also come from a tree fitted
without its group; the saved contact tree is fitted on all training videos.
"""

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from annotator.config import BaseAnnotatorConfig
from annotator.contacts.features import ContactFeatures
from annotator.contacts.model import ContactModelConfig, score_contact_features
from annotator.hybrid import (
    ContactEvidence,
    features_from_evidence,
    predict_contacts,
    sequence_inputs,
)
from annotator.models import AnnotatorModels, SideGeometry, load_models, save_models
from annotator.outcomes.point_winner import (
    SHIPPED_LANDING_FILTER_OPTIONS,
    Half,
)
from annotator.run_video import RunCapture, run_video
from annotator.sequence.confidence import ConfidenceFeatures, build_confidence_features
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    initial_sequences,
    scale_frames,
)
from annotator.sequence.refine import build_option_pool
from annotator.sequence.sides import alternate_sides
from annotator.shuttle_track import validate_shuttle_track
from annotator.training.confidence import ConfidenceFitConfig, fit_confidence_model
from annotator.training.contact import (
    ContactFitConfig,
    ContactTrainingVideo,
    fit_contact_model,
)
from annotator.training.sequences import (
    LabelledRally,
    SequenceFitConfig,
    SequenceTrainingVideo,
    fit_sequence_models,
    match_contacts,
    overlapping_rallies,
)
from annotator.video_metadata import VideoMetadata
from dataset_builder.shuttle_evidence import GUARD_CODES_FILENAME
from dataset_builder.vision import (
    TRACK_FILENAME,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    save_json_gz,
)

METADATA_FILENAME = 'video_metadata.json.gz'
EVALUATION_TOLERANCE_AT_30_FPS = 10


class Split(StrEnum):
    """The named data partitions accepted by the retune manifest."""

    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'


@dataclass(frozen=True)
class VideoInput:
    """One dataset-builder video and its human contact-label file."""

    identity: str
    group: str
    split: Split
    labels: Path


@dataclass(frozen=True)
class TrainingManifest:
    """An existing dataset-builder run and explicit, disjoint labelled splits."""

    run_dir: Path
    videos: tuple[VideoInput, ...]

    def __post_init__(self) -> None:
        identities = set()
        group_splits = {}
        for video in self.videos:
            if video.identity in identities:
                raise ValueError(f'duplicate video id: {video.identity}')
            identities.add(video.identity)
            if video.group in group_splits and group_splits[video.group] != video.split:
                raise ValueError(f'group {video.group!r} occurs in more than one split')
            group_splits[video.group] = video.split


@dataclass(frozen=True)
class TrainingSettings:
    """Python API settings for fitting; the command uses these supported defaults."""

    contact: ContactFitConfig = field(default_factory=ContactFitConfig)
    contact_prediction: ContactModelConfig = field(default_factory=ContactModelConfig)
    sequences: SequenceFitConfig = field(default_factory=SequenceFitConfig)
    confidence: ConfidenceFitConfig = field(default_factory=ConfidenceFitConfig)
    preprocessing: BaseAnnotatorConfig = field(default_factory=BaseAnnotatorConfig)


DEFAULT_TRAINING_SETTINGS = TrainingSettings()


def manifest_path(value: object, directory: Path, field_name: str) -> Path:
    """Resolve a required manifest path relative to the manifest's directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field_name} must be a non-empty path string')
    path = Path(value)
    return (path if path.is_absolute() else directory / path).resolve()


def load_manifest(path: Path) -> TrainingManifest:
    """Read the small JSON contract without opening any video or label artefacts."""
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
        raw_identity = entry['id']
        if isinstance(raw_identity, bool) or not isinstance(raw_identity, (str, int)):
            raise TypeError('video id must be a string or integer')
        identity = str(raw_identity)
        if not identity or identity in ('.', '..') or Path(identity).name != identity:
            raise ValueError(f'video id must be one directory name: {identity!r}')
        group = entry['group']
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f'{identity}: group must be a non-empty string')
        try:
            split = Split(entry['split'])
        except ValueError as error:
            raise ValueError(f'{identity}: split must be train, validation or test') from error
        labels = manifest_path(entry['labels'], path.resolve().parent, f'{identity} labels')
        videos.append(VideoInput(identity, group, split, labels))
    return TrainingManifest(run_dir, tuple(videos))


def videos_for_split(manifest: TrainingManifest, split: Split) -> tuple[VideoInput, ...]:
    """Select exactly one named partition, failing clearly when it is empty."""
    videos = tuple(video for video in manifest.videos if video.split == split)
    if not videos:
        raise ValueError(f'manifest contains no {split.value} videos')
    return videos


def load_labels(path: Path, frame_count: int) -> tuple[LabelledRally, ...]:
    """Read rally_id,frame,side CSV rows, grouped by rally and increasing in frame.

    Empty sides remain unknown. Each rally must occupy one contiguous CSV block.
    All frames must lie inside the source video timeline.
    """
    rallies = []
    seen = set()
    current = None
    frames: list[int] = []
    sides: list[Half | None] = []
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        expected_columns = {'rally_id', 'frame', 'side'}
        if reader.fieldnames is None or len(reader.fieldnames) != 3 or set(reader.fieldnames) != expected_columns:
            raise ValueError(f'{path}: labels need rally_id,frame,side columns')
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'{path}:{line}: label row has the wrong column count')
            identity = row['rally_id'].strip()
            if not identity:
                raise ValueError(f'{path}:{line}: rally_id is empty')
            if identity != current:
                if identity in seen:
                    raise ValueError(f'{path}:{line}: rally rows must form a contiguous block')
                if current is not None:
                    rallies.append(LabelledRally(current, tuple(frames), tuple(sides)))
                seen.add(identity)
                current, frames, sides = identity, [], []
            try:
                frame = int(row['frame'])
                side_text = row['side'].strip()
                side = Half(side_text) if side_text else None
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


def load_contact_evidence(run_dir: Path, identity: str, preprocessing: BaseAnnotatorConfig) -> ContactEvidence:
    """Run the heuristic path on canonical stages and reuse its captured evidence."""
    stages = run_dir / 'stages'
    try:
        metadata = VideoMetadata.from_dict(load_json_gz(stages / 'metadata' / identity / METADATA_FILENAME))
        track = load_npy_xz(stages / 'shuttle' / identity / TRACK_FILENAME)
        validate_shuttle_track(track, metadata.frame_count)
        guard_codes = load_npy_xz(stages / 'shuttle' / identity / GUARD_CODES_FILENAME)
        pose = load_pose_arrays(stages / 'pose' / identity, metadata.frame_count)
        court = load_court_vision(
            stages / 'court' / identity, video_id=identity, frame_count=metadata.frame_count,
            resolution=(float(metadata.width), float(metadata.height)),
        )
    except (FileNotFoundError, ValueError) as error:
        raise ValueError(f'{identity}: cannot load current metadata/pose/court/shuttle artefacts: {error}') from error
    inputs = court.evidence.inputs
    if inputs is None:
        raise ValueError(f'{identity}: court extraction has no usable operational court inputs')
    capture = RunCapture()
    run_video(
        track, pose.bboxes, pose.scores, pose.kps, pose.ndet,
        fps=float(metadata.fps), base=preprocessing, heuristic_only=True,
        landing_options=SHIPPED_LANDING_FILTER_OPTIONS,
        net_band=inputs.net_band, resolution=inputs.resolution, video_id=identity,
        court_info=inputs.court_info, homo_df=None,
        gate_court_info=inputs.gate_court_info, gate_resolution_table=inputs.gate_resolution_table,
        court_present=court.evidence.court_present, homography_rows=inputs.homography_rows,
        cut_frames=[end for _start, end in court.raw_cuts[:-1]], keep_vote=court.evidence.keep_vote,
        inpaint_codes=guard_codes, court_invalid_is_excluded=True,
        landing_error_band_m=inputs.landing_error_band_m, capture=capture,
    )
    if capture.contact_evidence is None:
        raise ValueError(f'{identity}: heuristic annotation did not capture contact evidence')
    return capture.contact_evidence


def sequence_training_video(
    entry: VideoInput, evidence: ContactEvidence, features: ContactFeatures, rallies: tuple[LabelledRally, ...],
    contact_model: Any, settings: ContactModelConfig, side_geometry: SideGeometry,
) -> SequenceTrainingVideo:
    """Score the base tree and build one reusable downstream option pool."""
    scored = score_contact_features(features.rows, contact_model, evidence.fps, settings)
    scores, events, side_for_frame = sequence_inputs(evidence, scored, side_geometry)
    initial = initial_sequences(evidence.heuristic_spans, events)
    pool = build_option_pool(initial, events, scores, features.rows, features.search_intervals,
                             fps=evidence.fps, side_for_frame=side_for_frame)
    return SequenceTrainingVideo(entry.identity, entry.group, evidence.fps, initial, events, pool, rallies, len(evidence.track))


def whole_rally_correctness(
    sequence: ContactSequence, rallies: Sequence[LabelledRally], fps: float,
) -> int:
    """Return 1 correct, 0 known wrong, or -1 unjudgeable using retained labels.

    Missing contacts, merged/partial rallies, extra events and known side
    contradictions are wrong even when another human side label is unavailable.
    """
    overlaps = overlapping_rallies(sequence, rallies)
    if not overlaps:
        return -1
    if len(overlaps) > 1:
        return 0
    rally = overlaps[0]
    if not all(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames):
        return 0
    tolerance = scale_frames(EVALUATION_TOLERANCE_AT_30_FPS, fps)
    matches = match_contacts(rally.frames, [event.frame for event in sequence.events], tolerance)
    if len(matches) < len(rally.frames):
        return 0
    revised, _events = alternate_sides((sequence,), sequence.events)
    if any(rally.sides[label] is not None and revised[0].events[prediction].side != rally.sides[label]
           for label, prediction, _offset in matches):
        return 0
    if len(sequence.events) > len(rally.frames):
        return 0
    if any(side is None for side in rally.sides):
        return -1
    return 1


def fit_from_manifest(
    manifest: TrainingManifest, output: Path, side_geometry: SideGeometry = SideGeometry.VIDEO,
    settings: TrainingSettings = DEFAULT_TRAINING_SETTINGS,
) -> AnnotatorModels:
    """Fit and save all stages from train inputs only; held-out labels are never read."""
    entries = videos_for_split(manifest, Split.TRAIN)
    groups = tuple(dict.fromkeys(entry.group for entry in entries))
    if len(groups) < 3:
        raise ValueError('sequence training needs at least three distinct training groups')
    evidence_by_video = {}
    features_by_video = {}
    labels_by_video = {}
    contact_inputs = []
    for entry in entries:
        evidence = load_contact_evidence(manifest.run_dir, entry.identity, settings.preprocessing)
        features = features_from_evidence(evidence)
        rallies = load_labels(entry.labels, len(evidence.track))
        evidence_by_video[entry.identity] = evidence
        features_by_video[entry.identity] = features
        labels_by_video[entry.identity] = rallies
        frames = np.asarray(sorted(frame for rally in rallies for frame in rally.frames), dtype=np.int32)
        contact_inputs.append(ContactTrainingVideo(entry.identity, features.rows, frames, evidence.fps))
    try:
        contact_model = fit_contact_model(contact_inputs, settings.contact).model
    except ValueError as error:
        raise ValueError(f'contact-model fitting failed: {error}') from error
    contact_models_by_held_group = {}
    for group in groups:
        other_group_inputs = [
            video for entry, video in zip(entries, contact_inputs, strict=True) if entry.group != group
        ]
        try:
            contact_models_by_held_group[group] = fit_contact_model(other_group_inputs, settings.contact).model
        except ValueError as error:
            raise ValueError(f'contact-model fitting with group {group!r} held out failed: {error}') from error
    del contact_inputs
    del other_group_inputs
    sequence_inputs = []
    for entry in entries:
        evidence = evidence_by_video.pop(entry.identity)
        features = features_by_video.pop(entry.identity)
        sequence_inputs.append(sequence_training_video(
            entry, evidence, features, labels_by_video[entry.identity],
            contact_models_by_held_group[entry.group], settings.contact_prediction, side_geometry,
        ))
    del evidence, features
    try:
        sequence_fit = fit_sequence_models(sequence_inputs, settings.sequences)
    except ValueError as error:
        raise ValueError(f'grouped sequence-model fitting failed: {error}') from error
    fps_by_video = {video.identity: video.fps for video in sequence_inputs}
    base_rows, gap_rows, correctness = [], [], []
    for entry in entries:
        prediction = sequence_fit.cross_fitted[entry.identity]
        insertion_model = sequence_fit.cross_fitted_insertion_models[entry.identity]
        features = build_confidence_features(prediction, insertion_model, fps_by_video[entry.identity])
        sequences = {sequence.span_id: sequence for sequence in prediction.sequences}
        correctness.extend(whole_rally_correctness(
            sequences[span_id], labels_by_video[entry.identity], fps_by_video[entry.identity],
        ) for span_id in features.span_ids)
        base_rows.append(features.base)
        gap_rows.append(features.gap)
    base, gap = np.concatenate(base_rows), np.concatenate(gap_rows)
    confidence_features = ConfidenceFeatures(tuple(range(len(base))), base, gap)
    try:
        confidence = fit_confidence_model(
            confidence_features, np.asarray(correctness, dtype=np.int8), settings.confidence,
        )
    except ValueError as error:
        raise ValueError(f'confidence-model fitting failed: {error}') from error
    models = AnnotatorModels(contact_model, sequence_fit.models, confidence, settings.contact_prediction,
                             settings.preprocessing, side_geometry)
    save_models(models, output)
    return models


def evaluation_counts(events: Sequence[ContactEvent], sequences: Sequence[ContactSequence],
                      rallies: Sequence[LabelledRally], fps: float) -> dict[str, int | float]:
    """Count whole-video contact matches and judged final rally sequences."""
    labels = sorted(frame for rally in rallies for frame in rally.frames)
    matches = match_contacts(
        labels, [event.frame for event in events], scale_frames(EVALUATION_TOLERANCE_AT_30_FPS, fps),
    )
    correctness = [whole_rally_correctness(sequence, rallies, fps) for sequence in sequences]
    return {
        'labelled_contacts': len(labels), 'predicted_contacts': len(events), 'matched_contacts': len(matches),
        'contact_precision': len(matches) / len(events) if events else 0.0,
        'contact_recall': len(matches) / len(labels) if labels else 0.0,
        'whole_rally_judged': sum(value >= 0 for value in correctness),
        'whole_rally_correct': correctness.count(1), 'whole_rally_unjudgeable': correctness.count(-1),
    }


def evaluate_manifest(
    manifest: TrainingManifest, models: AnnotatorModels, split: Split, output: Path,
) -> dict[str, Any]:
    """Evaluate only the named held-out split and write a compact compressed report."""
    if split not in (Split.VALIDATION, Split.TEST):
        raise ValueError('evaluation split must be validation or test')
    if not output.name.endswith('.json.gz'):
        raise ValueError('evaluation output must end in .json.gz')
    rows = []
    count_names = ('labelled_contacts', 'predicted_contacts', 'matched_contacts',
                   'whole_rally_judged', 'whole_rally_correct', 'whole_rally_unjudgeable')
    totals: dict[str, int | float] = dict.fromkeys(count_names, 0)
    for entry in videos_for_split(manifest, split):
        evidence = load_contact_evidence(manifest.run_dir, entry.identity, models.preprocessing)
        rallies = load_labels(entry.labels, len(evidence.track))
        prediction = predict_contacts(evidence, models)
        counts = evaluation_counts(prediction.refined.events, prediction.refined.sequences, rallies, evidence.fps)
        rows.append({'id': entry.identity, **counts})
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


def argument_parser() -> argparse.ArgumentParser:
    """Expose the two supported operations without research or tuning switches."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    fit = commands.add_parser('fit', help='Fit a fresh complete bundle from the train split')
    fit.add_argument('--manifest', required=True, type=Path)
    fit.add_argument('--output', required=True, type=Path)
    fit.add_argument('--side-geometry', choices=list(SideGeometry), default=SideGeometry.VIDEO, type=SideGeometry)
    evaluate = commands.add_parser('evaluate', help='Score a saved bundle on validation or test videos')
    evaluate.add_argument('--manifest', required=True, type=Path)
    evaluate.add_argument('--models', required=True, type=Path)
    evaluate.add_argument('--split', choices=[Split.VALIDATION, Split.TEST], required=True, type=Split)
    evaluate.add_argument('--output', required=True, type=Path, help='Compressed report path ending in .json.gz')
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run fitting or held-out evaluation, reporting input errors without a traceback."""
    parser = argument_parser()
    arguments = parser.parse_args(argv)
    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == 'fit':
            fit_from_manifest(manifest, arguments.output, arguments.side_geometry)
            print(f'Saved annotator models to {arguments.output}')
        else:
            models = load_models(arguments.models)
            evaluate_manifest(manifest, models, arguments.split, arguments.output)
            print(f'Saved {arguments.split.value} evaluation to {arguments.output}')
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0
