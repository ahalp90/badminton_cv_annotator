"""Labelled training videos, contact matching and whole-rally judgement."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from annotator.contacts.player_attribution import fit_rally_sides
from annotator.rallies.build_contact_proposals import RallyContext
from annotator.types import ContactSequence, CourtHalf
from shared.frame_timing import frames_at_fps


@dataclass(frozen=True)
class LabelledRally:
    """Sorted labelled contact frames and matching player sides; None means unknown side."""

    rally_id: str
    frames: tuple[int, ...]
    sides: tuple[CourtHalf | None, ...]

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.sides) or tuple(sorted(self.frames)) != self.frames:
            raise ValueError("rally frames must be sorted and have one aligned side label each")


@dataclass(frozen=True)
class RallyBoundaryJudgement:
    """How a proposed interval overlaps labelled rallies.

    ``outcome`` is no_labelled_rally, merged_rallies, part_of_rally or
    boundaries_right. ``rally`` is set when exactly one rally overlaps;
    ``overlapping_rallies`` counts the matches.
    """
    outcome: str
    rally: LabelledRally | None
    overlapping_rallies: int


@dataclass(frozen=True)
class SequenceTrainingVideo:
    """Contact predictions and labels for one video's rally-selector training.

    Build proposals from context; judge them against rallies. Withhold videos
    sharing a group together.
    """

    video_id: str
    group: str
    context: RallyContext
    rallies: tuple[LabelledRally, ...]


def overlapping_rallies(sequence: ContactSequence, rallies: Sequence[LabelledRally]) -> tuple[LabelledRally, ...]:
    """Labelled rallies with any contact inside the range, incl start, excl end."""
    return tuple(
        rally for rally in rallies
        if any(sequence.start_frame <= frame < sequence.end_frame for frame in rally.frames)
    )


def whole_rally_correctness(
    sequence: ContactSequence, rallies: Sequence[LabelledRally], fps: float,
    *, tolerance_at_30_fps: int = 10,
) -> int:
    """Return 1 for correct, 0 for wrong, -1 when the labels cannot decide.

    Require one complete labelled rally, all contacts within tolerance, and
    matching known player sides after fitting alternation. Missing/extra hits,
    partial/merged rallies and wrong known sides count as wrong even if another
    side is unlabelled. Otherwise, any unlabelled side leaves it unjudgeable.
    """
    tolerance = frames_at_fps(tolerance_at_30_fps, fps)
    boundary = judge_rally_boundaries(sequence, rallies, tolerance=tolerance)
    if boundary.outcome == 'no_labelled_rally':
        return -1
    if boundary.outcome != 'boundaries_right':
        return 0
    rally = boundary.rally
    assert rally is not None
    if len(sequence.events) != len(rally.frames):
        return 0
    # A fully correct rally pairs every contact in chronological order.
    for labelled_frame, event in zip(rally.frames, sequence.events, strict=True):
        if abs(event.frame - labelled_frame) > tolerance:
            return 0
    revised = fit_rally_sides((sequence,))
    for labelled_side, event in zip(rally.sides, revised[0].sequence.events, strict=True):
        if labelled_side is not None and event.side != labelled_side:
            return 0
    if any(side is None for side in rally.sides):
        return -1
    return 1


def judge_rally_boundaries(
    sequence: ContactSequence, rallies: Sequence[LabelledRally], *, tolerance: int,
) -> RallyBoundaryJudgement:
    """Check a rally range against labelled contacts, incl start, excl end.

    Allow small inward errors up to tolerance. Extra time before/after the
    labelled rally is allowed, but including another labelled rally fails.
    If no label lies inside the range, also check the tolerance-expanded range.
    """
    start = sequence.start_frame - tolerance
    end = sequence.end_frame + tolerance
    overlaps = overlapping_rallies(sequence, rallies)
    if not overlaps:
        overlaps = tuple(rally for rally in rallies if any(start <= frame < end for frame in rally.frames))
    if not overlaps:
        return RallyBoundaryJudgement('no_labelled_rally', None, 0)
    if len(overlaps) > 1:
        return RallyBoundaryJudgement('merged_rallies', None, len(overlaps))
    rally = overlaps[0]
    first, last = rally.frames[0], rally.frames[-1]
    if not start <= first or not last < end:
        return RallyBoundaryJudgement('part_of_rally', rally, 1)
    return RallyBoundaryJudgement('boundaries_right', rally, 1)


# Contact matching

MATCH_CONTACT = np.uint8(1)
SKIP_PREDICTION = np.uint8(2)
SKIP_LABEL = np.uint8(3)


def better_match_score(candidate: tuple[int, int], incumbent: tuple[int, int]) -> bool:
    """Prefer more matched contacts; break ties by lower total timing error."""
    return candidate[0] > incumbent[0] or (
        candidate[0] == incumbent[0] and candidate[1] < incumbent[1]
    )


def match_contacts(
    gt_frames: Sequence[int],
    predicted_frames: Sequence[int],
    tolerance: int,
) -> list[tuple[int, int, int]]:
    """Pair labelled and predicted contacts in time order, within the frame tolerance.

    Each contact can be used once. Maximise the number of pairs, then minimise
    their total absolute timing error. Sort by frame and original index to
    handle duplicates consistently. For an exact tie, prefer a match, then
    skipping a prediction, then skipping a label.

    :param gt_frames: Labelled video frames in their original input order.
    :param predicted_frames: Predicted video frames in their original input order.
    :param tolerance: Maximum frame difference, including equality.
    :return: (label index, prediction index, prediction minus label frame), in
        time order. Indices refer to the original input lists.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")

    sorted_gt = sorted((int(frame), index) for index, frame in enumerate(gt_frames))
    sorted_predictions = sorted(
        (int(frame), index) for index, frame in enumerate(predicted_frames)
    )
    ground_truth_count = len(sorted_gt)
    prediction_count = len(sorted_predictions)

    # Keep only the previous score row. Store one action per label/prediction
    # pair so the chosen matches can be reconstructed afterwards.
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
            # Try matching first so it wins exact ties; then compare the two skips.
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
