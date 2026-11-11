"""The fixed development split and shared setup for annotator experiments.

Current training and MonoTrack comparisons use these video orders, settings,
prepared inputs and report helpers. Results remain under each experiment.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib

from annotator.contacts.build_features import contact_feature_dtype
from annotator.contacts.score_candidates import score_contact_features
from annotator.rallies.build_contact_proposals import build_rally_context
from annotator.training.sequences import (
    LabelledRally,
    SequenceTrainingVideo,
    overlapping_rallies,
    whole_rally_correctness,
)
from annotator.training.workflow import (
    DEFAULT_TRAINING_SETTINGS,
    TrainingSettings,
    evaluation_counts,
)
from annotator.types import ContactEvent, ContactSequence
from experiments.annotator.heuristic_only_tune.data import (
    COUNT_FIELDS,
    SearchVideo,
    add_rates,
)
from experiments.annotator.heuristic_only_tune.run import (
    read_json,
    write_json,
)
from experiments.annotator.heuristic_tuning.sweep import CandidateSpec, base_config
from experiments.annotator.tree_simplification.data import PreparedVideo

REPOSITORY = Path(__file__).resolve().parents[2]
GROUPS_FILE = REPOSITORY / 'scratch/contact_det_full_ds_fit/records/training_video_score_groups.json'


TRAINING_GROUPS = ('A', 'B', 'C', 'D')
VALIDATION_GROUP = 'V'
TEST_VIDEO_IDS = (
    8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57,
)


COUNT_NAMES = ('labelled_contacts', 'predicted_contacts', 'matched_contacts', 'labelled_rallies',
               'whole_rally_judged', 'whole_rally_correct', 'whole_rally_unjudgeable')


@dataclass(frozen=True)
class Development:
    """The 40 original development videos: groups A-D for training and V for validation."""

    group_by_video: dict[str, str]  # original group order, then original order within each group
    fps_by_video: dict[str, float]
    id_by_video: dict[str, int]

    def videos(self, groups: Sequence[str]) -> tuple[str, ...]:
        """Return the named groups' videos in original group order."""
        return tuple(video for group in groups for video, owner in self.group_by_video.items() if owner == group)

    def id_order(self, videos: Sequence[str]) -> tuple[str, ...]:
        """Return videos in ShuttleSet ID order, as the 40- and 32-video trees used."""
        return tuple(sorted(videos, key=self.id_by_video.__getitem__))


def contact_fit_videos(development: Development) -> dict[str, tuple[str, ...]]:
    """Name each contact fit and its training videos in sampling order.

    The fold trees used group order; the 40-video final tree and the 32-video
    tree that scored V used ID order (``final_contact_scores/group_V``).
    """
    training = development.videos(TRAINING_GROUPS)
    fits = {'final_40': development.id_order(development.videos((*TRAINING_GROUPS, VALIDATION_GROUP)))}
    for held_out in TRAINING_GROUPS:
        fits[f'held_{held_out}'] = tuple(
            video for video in training if development.group_by_video[video] != held_out
        )
    fits['v_32'] = development.id_order(training)
    return fits


def dump_atomically(value: object, path: Path) -> None:
    """Write a joblib file under a temporary name, then move it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    joblib.dump(value, temporary, compress=3)
    os.replace(temporary, path)


def score_stream(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent], rallies: Sequence[LabelledRally],
    fps: float, confidence: dict[str, float] | None,
) -> dict[str, Any]:
    """Score one stream with the supported evaluator and keep per-section detail."""
    counts = evaluation_counts(events, sequences, rallies, fps)
    sections = []
    correct_rallies = []
    for sequence in sequences:
        correctness = whole_rally_correctness(sequence, rallies, fps)
        if correctness == 1:
            correct_rallies.append(overlapping_rallies(sequence, rallies)[0].rally_id)
        score = None if confidence is None else confidence[str(sequence.span_id)]
        sections.append([sequence.span_id, correctness, score])
    return {**counts, 'labelled_rallies': len(rallies), 'correct_rallies': correct_rallies, 'sections': sections}


def development_for(videos: tuple[SearchVideo, ...], groups_file: Path = GROUPS_FILE) -> Development:
    document = json.loads(groups_file.read_text())
    groups = {}
    for group in document['groups']:
        for video in group['videos']:
            identity = video['fixture']
            if identity in groups:
                raise ValueError(f'duplicate group member: {identity}')
            groups[identity] = group['group']
    for identity in document['fixed_validation_videos']:
        if identity in groups:
            raise ValueError(f'validation overlaps training: {identity}')
        groups[identity] = VALIDATION_GROUP
    by_id = {video.identity: video for video in videos}
    if len(groups) != 40 or set(groups) != set(by_id):
        raise ValueError('groups must match the original 40 development videos')
    for group in (*TRAINING_GROUPS, VALIDATION_GROUP):
        if list(groups.values()).count(group) != 8:
            raise ValueError(f'group {group} must have eight videos')
    for identity, group in groups.items():
        expected = 'validation' if group == VALIDATION_GROUP else 'train'
        if by_id[identity].split != expected or by_id[identity].dataset != 'ShuttleSet':
            raise ValueError(f'{identity}: original training/validation membership differs')
    return Development(
        groups, {video.identity: video.fps for video in videos},
        {identity: int(identity.removeprefix('sset_')) for identity in groups},
    )


def preprocessing(spec: CandidateSpec) -> Any:
    return base_config(spec)


def pooled_scores(per_video: dict[str, Any]) -> dict[str, Any]:
    totals = {}
    for tolerance in ('5', '10'):
        counts = {name: 0 for name in COUNT_FIELDS}
        for scores in per_video.values():
            for name in COUNT_FIELDS:
                counts[name] += int(scores[tolerance][name])
        totals[tolerance] = add_rates(counts)
    return totals


def progress(output: Path, stage: str, **details: Any) -> None:
    payload = {'stage': stage, **details}
    write_json(output / 'progress.json.gz', payload)
    print(payload, flush=True)


def load_prepared(directory: Path, identity: str) -> PreparedVideo:
    prepared = joblib.load(directory / 'videos' / f'{identity}.joblib')
    if prepared.features.rows.dtype.names != contact_feature_dtype().names:
        raise ValueError(f'{identity}: prepared contact columns differ; rebuild the prepared inputs')
    return prepared


def require_raw_features(prepared_root: Path) -> None:
    if read_json(prepared_root / 'preparation.json.gz').get('motion_units') is not None:
        raise ValueError('Experimental motion units need their own fit/replay path; production bundles use raw units')


def training_pool(prepared: PreparedVideo, contact: Any, settings: TrainingSettings) -> SequenceTrainingVideo:
    video = prepared.video
    scored = score_contact_features(prepared.features.rows, contact, prepared.evidence.fps, settings.contact_prediction)
    context = build_rally_context(prepared.evidence, scored)
    return SequenceTrainingVideo(video.identity, prepared.group, context, video.rallies)


def settings_for(device: str, min_child_weight: float, *, seed_offset: int = 0) -> TrainingSettings:
    """Apply one seed and device to all current model fits."""
    base = DEFAULT_TRAINING_SETTINGS
    options = {'device': device, 'min_child_weight': min_child_weight}
    return replace(
        base,
        contact=replace(base.contact, **options, random_seed=base.contact.random_seed + seed_offset),
        confidence=replace(base.confidence, **options, random_seed=base.confidence.random_seed + seed_offset),
        rally=replace(
            base.rally,
            tree=replace(
                base.rally.tree, **options, random_seed=base.rally.tree.random_seed + seed_offset,
            ),
        ),
    )
