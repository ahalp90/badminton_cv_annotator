"""Labelled training videos, contact matching and whole-rally judgement."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, scale_frames
from annotator.sequence.rally_model import alternate_sides
from annotator.sequence.repairs import RallyRepairs


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
    """One training video's detector proposals and human contact labels."""

    identity: str
    group: str
    fps: float
    initial: tuple[ContactSequence, ...]
    events: tuple[ContactEvent, ...]
    repairs: RallyRepairs
    rallies: tuple[LabelledRally, ...]
    frame_count: int | None = None


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


def whole_rally_correctness(
    sequence: ContactSequence, rallies: Sequence[LabelledRally], fps: float,
    *, tolerance_at_30_fps: int = 10,
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
    tolerance = scale_frames(tolerance_at_30_fps, fps)
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
