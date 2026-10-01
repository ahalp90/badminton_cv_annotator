"""Fit the contact tree on explicitly supplied training videos.

Candidate regions, positive radii and negative sampling follow the fitted
contact model's policy. Supply training videos in a stable order to reproduce
sampling. Validation and test videos belong to the caller's evaluation path.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from annotator.contacts.features import REGION_FIELDS
from annotator.contacts.model import contact_feature_matrix
from annotator.fps_constants import ScalingKind

POSITIVE_RADIUS_AT_30_FPS = 1
IGNORED_RADIUS_AT_30_FPS = 4
NEARBY_NEGATIVE_RADIUS_AT_30_FPS = 15
MAXIMUM_NEGATIVES_PER_POSITIVE = 24


@dataclass(frozen=True)
class ContactTrainingVideo:
    """One training video's feature rows and sorted contact-frame labels."""

    identity: str
    features: np.ndarray
    contact_frames: np.ndarray
    fps: float


@dataclass(frozen=True)
class ContactFitConfig:
    """Tree settings maintainers may change during a dataset retune."""

    learning_rate: float = 0.06
    max_iter: int = 180
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    l2_regularization: float = 1.0
    random_seed: int = 20260824
    early_stopping: bool | Literal["auto"] = "auto"


DEFAULT_CONTACT_FIT_CONFIG = ContactFitConfig()


@dataclass(frozen=True)
class ContactTrainingCounts:
    """Numbers of examples selected from one video's candidate rows."""

    positive: int
    nearby_negative: int
    sampled_other_negative: int
    selected: int


@dataclass(frozen=True)
class ContactTrainingSelection:
    """Selected feature rows, aligned binary labels and counts by video."""

    rows: np.ndarray
    labels: np.ndarray
    video_counts: dict[str, ContactTrainingCounts]


@dataclass(frozen=True)
class ContactModelFit:
    """Fitted contact tree and the examples already selected for its fit."""

    model: HistGradientBoostingClassifier
    selection: ContactTrainingSelection


def nearest_contact_distances(frames: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    if not len(contacts):
        return np.full(len(frames), np.iinfo(np.int32).max, dtype=np.int32)
    positions = np.searchsorted(contacts, frames)
    left_positions = np.maximum(positions - 1, 0)
    right_positions = np.minimum(positions, len(contacts) - 1)
    left_distances = np.abs(frames - contacts[left_positions])
    right_distances = np.abs(frames - contacts[right_positions])
    return np.minimum(left_distances, right_distances).astype(np.int32)


def select_contact_training_rows(
    training_videos: Sequence[ContactTrainingVideo],
    config: ContactFitConfig = DEFAULT_CONTACT_FIT_CONFIG,
) -> ContactTrainingSelection:
    """Select contacts, nearby negatives and sampled distant negatives.

    All nearby negatives are retained, even when they exceed the nominal
    budget of 24 negatives per positive. Distant negatives fill the remaining
    budget. Videos with no positive candidate retain their nearby negatives.

    :param training_videos: Explicit training inputs in deterministic sampling order.
    :param config: Fit settings; the random seed controls distant-negative sampling.
    :return: Selected rows in input-video and original row order, labels and counts.
    """
    random = np.random.default_rng(config.random_seed)
    row_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    video_counts: dict[str, ContactTrainingCounts] = {}
    for video in training_videos:
        if video.identity in video_counts:
            raise ValueError(f"training video appears more than once: {video.identity}")
        if video.contact_frames.ndim != 1 or np.any(np.diff(video.contact_frames) < 0):
            raise ValueError(f"{video.identity}: contact frames must be a sorted one-dimensional array")
        candidate_mask = np.zeros(len(video.features), dtype=bool)
        for name in REGION_FIELDS:
            candidate_mask |= video.features[name].astype(bool)
        rows = video.features[candidate_mask]
        distances = nearest_contact_distances(rows['frame'], video.contact_frames)
        positive = distances <= ScalingKind.FRAME_COUNT.scale(POSITIVE_RADIUS_AT_30_FPS, video.fps)
        ignored = (~positive) & (distances <= ScalingKind.FRAME_COUNT.scale(IGNORED_RADIUS_AT_30_FPS, video.fps))
        negative = ~positive & ~ignored
        nearby_negative = negative & (
            distances <= ScalingKind.FRAME_COUNT.scale(NEARBY_NEGATIVE_RADIUS_AT_30_FPS, video.fps)
        )
        positive_positions = np.flatnonzero(positive)
        nearby_positions = np.flatnonzero(nearby_negative)
        other_positions = np.flatnonzero(negative & ~nearby_negative)
        negative_limit = MAXIMUM_NEGATIVES_PER_POSITIVE * len(positive_positions)
        other_count = max(0, negative_limit - len(nearby_positions))
        if len(other_positions) > other_count:
            other_positions = random.choice(other_positions, size=other_count, replace=False)
        selected = np.zeros(len(rows), dtype=bool)
        selected[positive_positions] = True
        selected[nearby_positions] = True
        selected[other_positions] = True
        row_chunks.append(rows[selected])
        label_chunks.append(positive[selected].astype(np.uint8))
        video_counts[video.identity] = ContactTrainingCounts(
            len(positive_positions), len(nearby_positions), len(other_positions), int(selected.sum()),
        )
    if not row_chunks:
        raise ValueError("supply at least one training video")
    labels = np.concatenate(label_chunks)
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("training examples must include contacts and non-contacts")
    return ContactTrainingSelection(np.concatenate(row_chunks), labels, video_counts)


def fit_contact_model(
    training_videos: Sequence[ContactTrainingVideo],
    config: ContactFitConfig = DEFAULT_CONTACT_FIT_CONFIG,
) -> ContactModelFit:
    """Fit a balanced contact tree using only the named training videos.

    :param training_videos: Explicit training features and labels; omit held-out videos.
    :param config: Histogram gradient boosting settings and sampling seed.
    :return: Fitted model and its selected examples without repeating selection.
    """
    selection = select_contact_training_rows(training_videos, config)
    model = HistGradientBoostingClassifier(
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization,
        class_weight='balanced',
        random_state=config.random_seed,
        early_stopping=config.early_stopping,
    )
    model.fit(contact_feature_matrix(selection.rows), selection.labels)
    return ContactModelFit(model, selection)
