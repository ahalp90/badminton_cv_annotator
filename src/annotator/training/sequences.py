"""Fit the selected serve, sequence and insertion trees on supplied option pools.

Groups isolate the downstream model fits. Contact-detector scores and feature
rows are supplied by the caller; this module does not refit the contact tree.
Training scores exclude their own group and any outer held-out group.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, scale_frames
from annotator.sequence.edits import EditKind
from annotator.sequence.features import (
    SERVE_EDIT_FEATURE_NAMES,
    positive_probabilities,
    serve_score_features,
)
from annotator.sequence.refine import (
    OptionPool,
    RefinedSequences,
    SequenceModels,
    finish_sequences,
)
from annotator.sequence.sides import alternate_sides

TARGET_TOLERANCE_AT_30_FPS = 10


@dataclass(frozen=True)
class LabelledRally:
    """Human contact frames in chronological order and their aligned court halves."""

    identity: str
    frames: tuple[int, ...]
    sides: tuple[Half | None, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.sides) or tuple(sorted(self.frames)) != self.frames:
            raise ValueError("rally frames must be sorted and have one aligned side label each")


@dataclass(frozen=True)
class SequenceTrainingVideo:
    """One named training video's precomputed option pool and human labels."""

    identity: str
    group: str
    fps: float
    initial: tuple[ContactSequence, ...]
    events: tuple[ContactEvent, ...]
    pool: OptionPool
    rallies: tuple[LabelledRally, ...]
    frame_count: int | None = None


@dataclass(frozen=True)
class SequenceTargets:
    """Row-aligned binary targets; -1 excludes an unjudgeable example."""

    serve: np.ndarray  # one per pool.serve_inputs matrix row
    sequence: np.ndarray  # one per pool option
    insertion: np.ndarray  # one per pool option; -1 without an insertion


@dataclass(frozen=True)
class TreeFitConfig:
    """Histogram gradient boosting settings for one part of the sequence stack."""

    max_iter: int = 200
    max_leaf_nodes: int = 15
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    random_seed: int = 20260905


@dataclass(frozen=True)
class SequenceFitConfig:
    """The two selected tree sizes; both use balanced weights and no early stopping."""

    serve: TreeFitConfig = TreeFitConfig(max_iter=100, max_leaf_nodes=7, random_seed=20260824)
    sequence: TreeFitConfig = TreeFitConfig()


DEFAULT_SEQUENCE_FIT_CONFIG = SequenceFitConfig()


@dataclass(frozen=True)
class SequenceModelFit:
    """Final models and held-group outputs that confidence training can reuse."""

    models: SequenceModels
    cross_fitted: dict[str, RefinedSequences]
    targets: dict[str, SequenceTargets]
    cross_fitted_insertion_models: dict[str, HistGradientBoostingClassifier]


@dataclass(frozen=True)
class UpstreamSequenceModels:
    """Serve and insertion trees feeding a chooser trained on the same group set."""

    serve_summary: HistGradientBoostingClassifier
    serve_physical: HistGradientBoostingClassifier
    insertion: HistGradientBoostingClassifier


MATCH_CONTACT = np.uint8(1)
SKIP_PREDICTION = np.uint8(2)
SKIP_LABEL = np.uint8(3)


def better_match_score(candidate: tuple[int, int], incumbent: tuple[int, int]) -> bool:
    """Return whether a score has a larger count or a lower error at that count."""
    return candidate[0] > incumbent[0] or (
        candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
    )


def match_contacts(
    gt_frames: Sequence[int],
    predicted_frames: Sequence[int],
    tolerance: int,
) -> list[tuple[int, int, int]]:
    """Match ground-truth and predicted frames in time order.

    A pair is valid when its absolute frame difference is at most ``tolerance``.
    The result maximises the number of one-to-one valid pairs, then minimises the
    sum of their absolute frame differences.  Inputs are stably sorted by
    ``(frame, original index)`` for the chronological dynamic programme and
    deterministic duplicate handling.  Equal-objective transitions prefer a
    match, then skipping a prediction, then skipping a ground-truth frame.

    :param gt_frames: Ground-truth frame numbers in their original input order.
    :param predicted_frames: Predicted frame numbers in their original input order.
    :param tolerance: Maximum allowed absolute frame difference, in source frames.
    :return: Chronological ``(gt_index, prediction_index, prediction_minus_gt)``
        triples using indices from the original input sequences.
    :raises ValueError: If ``tolerance`` is negative.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    sorted_gt = sorted((int(frame), index) for index, frame in enumerate(gt_frames))
    sorted_predictions = sorted(
        (int(frame), index) for index, frame in enumerate(predicted_frames)
    )
    ground_truth_count = len(sorted_gt)
    prediction_count = len(sorted_predictions)

    # Score storage is linear in predictions; traceback uses one byte per prefix cell.
    traceback = np.zeros(
        (ground_truth_count + 1, prediction_count + 1),
        dtype=np.uint8,
    )
    traceback[1:, 0] = SKIP_LABEL
    traceback[0, 1:] = SKIP_PREDICTION

    previous: list[tuple[int, int]] = [(0, 0)] * (prediction_count + 1)
    for gt_position, (gt_frame, _) in enumerate(sorted_gt, start=1):
        current: list[tuple[int, int]] = [(0, 0)] * (prediction_count + 1)
        current[0] = (0, 0)
        for prediction_position, (prediction_frame, _) in enumerate(
            sorted_predictions,
            start=1,
        ):
            # Start with the match so the stated transition priority wins exact
            # ties.  Invalid matches leave the two skip transitions to compare.
            distance = abs(prediction_frame - gt_frame)
            if distance <= tolerance:
                best = (
                    previous[prediction_position - 1][0] + 1,
                    previous[prediction_position - 1][1] + distance,
                )
                action = MATCH_CONTACT
            else:
                best = current[prediction_position - 1]
                action = SKIP_PREDICTION

            skip_prediction = current[prediction_position - 1]
            if better_match_score(skip_prediction, best):
                best = skip_prediction
                action = SKIP_PREDICTION

            skip_gt = previous[prediction_position]
            if better_match_score(skip_gt, best):
                best = skip_gt
                action = SKIP_LABEL

            current[prediction_position] = best
            traceback[gt_position, prediction_position] = action
        previous = current

    matches: list[tuple[int, int, int]] = []
    gt_position = ground_truth_count
    prediction_position = prediction_count
    while gt_position > 0 and prediction_position > 0:
        action = traceback[gt_position, prediction_position]
        if action == MATCH_CONTACT:
            gt_frame, gt_index = sorted_gt[gt_position - 1]
            prediction_frame, prediction_index = sorted_predictions[prediction_position - 1]
            matches.append((gt_index, prediction_index, prediction_frame - gt_frame))
            gt_position -= 1
            prediction_position -= 1
        elif action == SKIP_PREDICTION:
            prediction_position -= 1
        elif action == SKIP_LABEL:
            gt_position -= 1
        else:
            raise RuntimeError(f"invalid traceback action {int(action)}")

    matches.reverse()
    return matches


def overlapping_rallies(sequence: ContactSequence, rallies: Sequence[LabelledRally]) -> tuple[LabelledRally, ...]:
    """Find labelled rallies with any contact inside a half-open sequence."""
    return tuple(
        rally for rally in rallies
        if any(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames)
    )


def useful_serve_edit(
    before: ContactSequence, after: ContactSequence, rally: LabelledRally,
    frame: int, kind: EditKind, tolerance: int,
) -> bool:
    """Judge a first-contact repair independently of unrelated later mistakes."""
    before_frames = [event.frame for event in before.events]
    after_frames = [event.frame for event in after.events]
    before_matches = match_contacts(rally.frames, before_frames, tolerance)
    after_matches = match_contacts(rally.frames, after_frames, tolerance)
    before_labels = {label for label, _, _ in before_matches}
    after_labels = {label for label, _, _ in after_matches}
    real_before = {before_frames[prediction] for _, prediction, _ in before_matches}
    unmatched_before = len(before_frames) - len(before_matches)
    unmatched_after = len(after_frames) - len(after_matches)
    removed_unmatched = len(set(before_frames) - set(after_frames) - real_before)
    unnecessary = max(0, unmatched_after - unmatched_before + removed_unmatched)
    removed = len(real_before - set(after_frames))
    candidate_is_first = any(label == 0 and after_frames[prediction] == frame for label, prediction, _ in after_matches)
    return (
        0 not in before_labels and candidate_is_first and before_labels <= after_labels
        and unnecessary == 0 and removed == 0
        and (kind == EditKind.ADD or before.events[0].frame not in real_before)
    )


def useful_insertion(
    before: ContactSequence, after: ContactSequence, rally: LabelledRally, frame: int, tolerance: int,
) -> bool:
    """Judge whether one insertion adds a distinct match without harming old matches."""
    before_frames = [event.frame for event in before.events]
    after_frames = [event.frame for event in after.events]
    if frame in before_frames or after_frames.count(frame) != 1:
        return False
    before_matches = match_contacts(rally.frames, before_frames, tolerance)
    after_matches = match_contacts(rally.frames, after_frames, tolerance)
    before_labels = {label for label, _, _ in before_matches}
    after_labels = {label for label, _, _ in after_matches}
    candidate_matches = {label for label, prediction, _ in after_matches if after_frames[prediction] == frame}
    if not candidate_matches or not before_labels < after_labels:
        return False
    if len(after_frames) - len(after_matches) > len(before_frames) - len(before_matches):
        return False
    real_before = Counter(before_frames[prediction] for _, prediction, _ in before_matches)
    return not real_before - Counter(after_frames)


def complete_sequence(sequence: ContactSequence, rally: LabelledRally, tolerance: int) -> bool:
    """Require complete timing and correct sides after the production alternating vote."""
    if not all(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames):
        return False
    frames = [event.frame for event in sequence.events]
    if len(frames) != len(rally.frames):
        return False
    if any(
        abs(predicted - labelled) > tolerance
        for predicted, labelled in zip(sorted(frames), rally.frames, strict=True)
    ):
        return False
    matches = match_contacts(rally.frames, frames, tolerance)
    revised, _events = alternate_sides((sequence,), sequence.events)
    return len(matches) == len(frames) and all(
        rally.sides[label] is not None and revised[0].events[prediction].side == rally.sides[label]
        for label, prediction, _ in matches
    )


def labelled_rallies_by_bounds(video: SequenceTrainingVideo) -> dict[tuple[int, int], tuple[LabelledRally, ...]]:
    """Associate each distinct section boundary once, across repeated edit alternatives."""
    associations = {}
    for sequence in video.initial:
        bounds = (sequence.start_frame, sequence.end_frame)
        associations[bounds] = overlapping_rallies(sequence, video.rallies)
    for option in video.pool.options:
        bounds = (option.sequence.start_frame, option.sequence.end_frame)
        if bounds not in associations:
            associations[bounds] = overlapping_rallies(option.sequence, video.rallies)
    return associations


def sequence_targets(video: SequenceTrainingVideo) -> SequenceTargets:
    """Label serve edits, whole sequences and insertions using the selected rules."""
    initial = {sequence.span_id: sequence for sequence in video.initial}
    by_bounds = labelled_rallies_by_bounds(video)
    associations = {
        span_id: by_bounds[(sequence.start_frame, sequence.end_frame)] for span_id, sequence in initial.items()
    }
    touches = Counter(rally.identity for rallies in associations.values() for rally in rallies)
    eligible = {}
    for span_id, rallies in associations.items():
        if len(rallies) == 1 and touches[rallies[0].identity] == 1:
            eligible[span_id] = rallies[0]
    tolerance = scale_frames(TARGET_TOLERANCE_AT_30_FPS, video.fps)
    serve = np.full(len(video.pool.serve_inputs.matrix), -1, dtype=np.int8)
    sequence = np.full(len(video.pool.options), -1, dtype=np.int8)
    insertion = np.full(len(video.pool.options), -1, dtype=np.int8)
    pure_serve_edits = {}
    for row, option in enumerate(video.pool.options):
        edit = option.edit
        if edit.kind in (EditKind.ADD, EditKind.REPLACE):
            pure_serve_edits[(option.span_id, edit.serve_frame, edit.kind)] = edit.sequence
        rally = eligible.get(option.span_id)
        bounds = (option.sequence.start_frame, option.sequence.end_frame)
        if rally is None or by_bounds[bounds] != (rally,):
            continue
        if all(side is not None for side in rally.sides):
            sequence[row] = int(complete_sequence(option.sequence, rally, tolerance))
        if option.added_contact is not None:
            insertion[row] = int(useful_insertion(
                edit.sequence, option.sequence, rally, option.added_contact.frame, tolerance,
            ))
    for key, row in video.pool.serve_inputs.rows_by_key.items():
        span_id, frame, kind = key
        after = pure_serve_edits.get(key)
        rally = eligible.get(span_id)
        if after is None or rally is None:
            continue
        if by_bounds[(after.start_frame, after.end_frame)] != (rally,):
            continue
        serve[row] = int(useful_serve_edit(initial[span_id], after, rally, frame, kind, tolerance))
    return SequenceTargets(serve, sequence, insertion)


def fit_tree(matrix: np.ndarray, targets: np.ndarray, config: TreeFitConfig) -> HistGradientBoostingClassifier:
    """Fit a balanced tree on known labels; both target classes must be present."""
    included = targets >= 0
    if set(targets[included].tolist()) != {0, 1}:
        raise ValueError("sequence tree fit requires both positive and negative labelled examples")
    model = HistGradientBoostingClassifier(
        max_iter=config.max_iter, max_leaf_nodes=config.max_leaf_nodes,
        learning_rate=config.learning_rate, min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.l2_regularization, random_state=config.random_seed,
        class_weight='balanced', early_stopping=False,
    )
    model.fit(matrix[included], targets[included])
    return model


def fit_upstream_models(
    videos: Sequence[SequenceTrainingVideo], targets: dict[str, SequenceTargets],
    allowed: frozenset[str], config: SequenceFitConfig,
) -> UpstreamSequenceModels:
    """Fit serve and insertion models using only the explicitly allowed groups."""
    serve_rows = []
    serve_labels = []
    insertion_rows = []
    insertion_labels = []
    for video in videos:
        if video.group not in allowed:
            continue
        serve_rows.append(video.pool.serve_inputs.matrix)
        serve_labels.append(targets[video.identity].serve)
        added = video.pool.has_added_contact
        insertion_rows.append(video.pool.insertion_features[added])
        insertion_labels.append(targets[video.identity].insertion[added])
    matrix = np.concatenate(serve_rows)
    labels = np.concatenate(serve_labels)
    return UpstreamSequenceModels(
        fit_tree(matrix[:, :len(SERVE_EDIT_FEATURE_NAMES)], labels, config.serve),
        fit_tree(matrix, labels, config.serve),
        fit_tree(np.concatenate(insertion_rows), np.concatenate(insertion_labels), config.sequence),
    )


def upstream_columns(pool: OptionPool, models: UpstreamSequenceModels) -> tuple[np.ndarray, np.ndarray]:
    """Score existing serve and insertion rows without rebuilding any options."""
    matrix = pool.serve_inputs.matrix
    summary = positive_probabilities(models.serve_summary, matrix[:, :len(SERVE_EDIT_FEATURE_NAMES)])
    physical = positive_probabilities(models.serve_physical, matrix)
    scores = {key: (float(summary[row]), float(physical[row])) for key, row in pool.serve_inputs.rows_by_key.items()}
    added = pool.has_added_contact
    insertion = np.full(len(pool.options), np.nan)
    insertion[added] = positive_probabilities(models.insertion, pool.insertion_features[added])
    return serve_score_features(pool.options, scores), insertion


def fit_choosers(
    videos: Sequence[SequenceTrainingVideo], targets: dict[str, SequenceTargets], allowed: frozenset[str],
    cache: dict[frozenset[str], UpstreamSequenceModels], config: SequenceFitConfig,
) -> SequenceModels:
    """Fit each chooser from upstream columns that exclude each training row's group."""
    whole_rows, whole_labels = [], []
    later_rows, later_labels = [], []
    scored_rows = []
    for video in videos:
        if video.group not in allowed:
            continue
        pool = video.pool
        serve, insertion = upstream_columns(pool, cache[allowed - {video.group}])
        base = np.column_stack((pool.sequence_features, serve))
        without_insertion = ~pool.has_added_contact
        whole_rows.append(base[without_insertion])
        whole_labels.append(targets[video.identity].sequence[without_insertion])
        later_rows.append(np.column_stack((base, pool.insertion_features)))
        later_labels.append(targets[video.identity].sequence)
        scored_rows.append(np.column_stack((base, pool.insertion_features, insertion)))
    labels = np.concatenate(later_labels)
    upstream = cache[allowed]
    whole = fit_tree(np.concatenate(whole_rows), np.concatenate(whole_labels), config.sequence)
    later = fit_tree(np.concatenate(later_rows), labels, config.sequence)
    scored = fit_tree(np.concatenate(scored_rows), labels, config.sequence)
    return SequenceModels(
        whole, later, scored, upstream.insertion, upstream.serve_summary, upstream.serve_physical,
    )


def fit_sequence_models(
    training_videos: Sequence[SequenceTrainingVideo], config: SequenceFitConfig = DEFAULT_SEQUENCE_FIT_CONFIG,
) -> SequenceModelFit:
    """Fit the selected stack and return final models and held-group predictions.

    :param training_videos: Precomputed pools and labels from training videos only.
        At least three groups are needed for nested held-group predictions.
    :param config: Selected serve and sequence tree settings, optionally retuned.
    :return: Final models, reusable held-group outputs and the targets already computed.
    """
    groups = sorted({video.group for video in training_videos})
    if len(groups) < 3:
        raise ValueError("nested sequence fits require at least three training groups")
    identities = [video.identity for video in training_videos]
    if len(set(identities)) != len(identities):
        raise ValueError("training video identities must be distinct")
    targets = {video.identity: sequence_targets(video) for video in training_videos}
    all_groups = frozenset(groups)
    cache = {}
    for size in (len(groups) - 2, len(groups) - 1, len(groups)):
        for subset in combinations(groups, size):
            allowed = frozenset(subset)
            cache[allowed] = fit_upstream_models(training_videos, targets, allowed, config)
    predictions = {}
    insertion_models = {}
    for held_out in groups:
        allowed = all_groups - {held_out}
        models = fit_choosers(training_videos, targets, allowed, cache, config)
        for video in training_videos:
            if video.group == held_out:
                predictions[video.identity] = finish_sequences(
                    video.initial, video.events, video.pool, models, fps=video.fps, frame_count=video.frame_count,
                )
                insertion_models[video.identity] = models.insertion
    models = fit_choosers(training_videos, targets, all_groups, cache, config)
    return SequenceModelFit(models, predictions, targets, insertion_models)
