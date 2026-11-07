"""Choose labelled examples and fit the contact classifier.

Keep candidates close to labelled hits as positives, omit ambiguous nearby
frames, and sample negatives. Video order and random seed determine the
sample. The caller supplies training videos only.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from annotator.contacts.score_candidates import contact_feature_matrix
from annotator.training.trees import fit_binary_tree
from shared.frame_timing import frames_at_fps

POSITIVE_RADIUS_AT_30_FPS = 1
IGNORED_RADIUS_AT_30_FPS = 4
NEARBY_NEGATIVE_RADIUS_AT_30_FPS = 15
MAXIMUM_NEGATIVES_PER_POSITIVE = 24


@dataclass(frozen=True)
class ContactTrainingVideo:
    """One video's candidate rows, sorted contact labels and frame rate."""

    video_id: str
    features: np.ndarray
    contact_frames: np.ndarray
    fps: float


@dataclass(frozen=True)
class ContactFitConfig:
    """Contact-tree settings and distances used to label training candidates.

    Frames within positive_radius_at_30_fps of a labelled contact are positive.
    Frames outside that radius but within ignored_radius_at_30_fps are omitted.
    Both distances are scaled from 30 fps to each video's frame rate.
    """

    learning_rate: float = 0.06
    max_iter: int = 180
    max_leaf_nodes: int = 31
    l2_regularization: float = 1.0
    random_seed: int = 20260824
    early_stopping: bool | Literal["auto"] = "auto"
    device: str = 'cuda'
    min_child_weight: float = 5.0
    positive_radius_at_30_fps: int = POSITIVE_RADIUS_AT_30_FPS
    ignored_radius_at_30_fps: int = IGNORED_RADIUS_AT_30_FPS


DEFAULT_CONTACT_FIT_CONFIG = ContactFitConfig()


@dataclass(frozen=True)
class ContactTrainingCounts:
    """Selected positive and negative counts for one video, with their total."""

    positive: int
    nearby_negative: int
    sampled_other_negative: int
    selected: int


@dataclass(frozen=True)
class ContactTrainingSelection:
    """Selected examples with one binary label per row and sampling counts per video.

    Rows stay grouped by video. Use ``video_counts`` to find each video's rows
    when changing the fit order; the counts are also available for reports.
    """

    rows: np.ndarray
    labels: np.ndarray
    video_counts: dict[str, ContactTrainingCounts]


@dataclass(frozen=True)
class ContactModelFit:
    """Fitted tree and its selected examples, available without repeating sampling."""

    model: Any
    selection: ContactTrainingSelection


def fit_contact_model(
    training_videos: Sequence[ContactTrainingVideo],
    config: ContactFitConfig = DEFAULT_CONTACT_FIT_CONFIG,
    *,
    fit_video_order: Sequence[str] | None = None,
) -> ContactModelFit:
    """Select examples and fit a class-balanced contact classifier.

    :param training_videos: Training features and labels, in sampling order.
    :param config: Tree settings and example-selection distances/seed.
    :param fit_video_order: Optional video-ID order for the already selected
        rows. Sampling still follows training_videos order.
    :return: Fitted model and the selected examples.
    """
    selection = select_contact_training_rows(training_videos, config)
    if fit_video_order is not None:
        if len(fit_video_order) != len(selection.video_counts) or set(fit_video_order) != selection.video_counts.keys():
            raise ValueError('fit_video_order must contain each training video ID exactly once')
        if tuple(fit_video_order) != tuple(selection.video_counts):
            slices = {}
            position = 0
            for video_id, counts in selection.video_counts.items():
                slices[video_id] = slice(position, position + counts.selected)
                position += counts.selected
            selection = ContactTrainingSelection(
                np.concatenate([selection.rows[slices[video_id]] for video_id in fit_video_order]),
                np.concatenate([selection.labels[slices[video_id]] for video_id in fit_video_order]),
                {video_id: selection.video_counts[video_id] for video_id in fit_video_order},
            )
    model = fit_binary_tree(
        contact_feature_matrix(selection.rows), selection.labels, config,
        balanced=True, early_stopping=config.early_stopping,
    )
    return ContactModelFit(model, selection)


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
        if video.video_id in video_counts:
            raise ValueError(f"training video appears more than once: {video.video_id}")
        if video.contact_frames.ndim != 1 or np.any(np.diff(video.contact_frames) < 0):
            raise ValueError(f"{video.video_id}: contact frames must be a sorted one-dimensional array")
        rows = video.features
        distances = nearest_contact_distances(rows['frame'], video.contact_frames)
        positive = distances <= frames_at_fps(config.positive_radius_at_30_fps, video.fps)
        ignored = (~positive) & (distances <= frames_at_fps(config.ignored_radius_at_30_fps, video.fps))
        negative = ~positive & ~ignored
        nearby_negative = negative & (
            distances <= frames_at_fps(NEARBY_NEGATIVE_RADIUS_AT_30_FPS, video.fps)
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
        video_counts[video.video_id] = ContactTrainingCounts(
            len(positive_positions), len(nearby_positions), len(other_positions), int(selected.sum()),
        )
    if not row_chunks:
        raise ValueError("supply at least one training video")
    labels = np.concatenate(label_chunks)
    if not np.any(labels == 1) or not np.any(labels == 0):
        raise ValueError("training examples must include contacts and non-contacts")
    return ContactTrainingSelection(np.concatenate(row_chunks), labels, video_counts)


def nearest_contact_distances(frames: np.ndarray, contacts: np.ndarray) -> np.ndarray:
    """Distance in frames from each candidate to the nearest sorted contact label.

    With no labels, return the largest int32 value so no candidate is nearby.
    """
    if not len(contacts):
        return np.full(len(frames), np.iinfo(np.int32).max, dtype=np.int32)
    positions = np.searchsorted(contacts, frames)
    left_positions = np.maximum(positions - 1, 0)
    right_positions = np.minimum(positions, len(contacts) - 1)
    left_distances = np.abs(frames - contacts[left_positions])
    right_distances = np.abs(frames - contacts[right_positions])
    return np.minimum(left_distances, right_distances).astype(np.int32)
