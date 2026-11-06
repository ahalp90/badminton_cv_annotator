"""Physical contact evidence and rally summaries used by the fitted selectors.

Feature names, column order and NaN placement are part of saved model bundles.
"""

from collections.abc import Mapping
from itertools import pairwise

import numpy as np

from annotator.contacts.features import CONTACT_FEATURE_NAMES
from annotator.sequence.contacts import ContactSequence
from annotator.types import CourtHalf

SUMMARY_FEATURE_NAMES = (
    "predicted_event_count",
    "section_duration_seconds",
    "minimum_score",
    "median_score",
    "mean_weakest_three",
    "shortest_gap_seconds",
    "longest_gap_seconds",
    "start_to_first_seconds",
    "last_to_end_seconds",
    "unanswered_side_count",
)


SIDE_FEATURE_NAMES = (
    "fraction_known",
    "fraction_known_starting_top",
    "fraction_known_starting_bot",
    "fraction_adjacent_known_same_side",
)


def physical_or_nan(physical: Mapping[int, np.ndarray], frame: int | None) -> np.ndarray:
    if frame is None:
        return np.full(len(CONTACT_FEATURE_NAMES), np.nan)
    return physical[frame]


def sequence_summary(sequence: ContactSequence, fps: float) -> tuple[float, ...]:
    """Summarise contact count, probabilities, spacing and unanswered sides.

    A sequence without contacts uses zero for every probability and timing
    summary. One contact has zero gaps but keeps its probability.
    """
    events = sequence.events
    probabilities = tuple(float(event.probability) for event in events)
    event_count = len(events)
    minimum = median = weakest_three = start_to_first = last_to_end = 0.0
    unanswered_sides = 0
    if event_count:
        ordered = sorted(probabilities)
        minimum = ordered[0]
        median = float(np.median(np.asarray(probabilities, dtype=np.float64)))
        weakest_three = float(np.mean(ordered[:3]))
        start_to_first = (events[0].frame - sequence.start_frame) / fps
        last_to_end = (sequence.end_frame - events[-1].frame) / fps
        unanswered_sides = sum(event.side is None for event in events)
    shortest_gap = longest_gap = 0.0
    if event_count >= 2:
        gaps = [(later.frame - earlier.frame) / fps for earlier, later in pairwise(events)]
        shortest_gap = min(gaps)
        longest_gap = max(gaps)
    return (
        float(event_count),
        (sequence.end_frame - sequence.start_frame) / fps,
        minimum,
        median,
        weakest_three,
        shortest_gap,
        longest_gap,
        start_to_first,
        last_to_end,
        float(unanswered_sides),
    )


def side_agreement(sequence: ContactSequence) -> tuple[float, float, float, float]:
    """Measure how well raw side guesses fit each alternating pattern.

    The two agreement values count known guesses matching a pattern that starts
    Top or starts Bot. They are NaN when no guess is known.
    """
    guesses = [event.side for event in sequence.events]
    known_count = sum(side is not None for side in guesses)
    fraction_known = 0.0 if not guesses else known_count / len(guesses)
    top_agreement = bot_agreement = np.nan
    if known_count:
        top_matches = 0
        bot_matches = 0
        for index, side in enumerate(guesses):
            if side is None:
                continue
            top_matches += side == (CourtHalf.TOP if index % 2 == 0 else CourtHalf.BOT)
            bot_matches += side == (CourtHalf.BOT if index % 2 == 0 else CourtHalf.TOP)
        top_agreement = top_matches / known_count
        bot_agreement = bot_matches / known_count
    adjacent = [(first, second) for first, second in pairwise(guesses) if first is not None and second is not None]
    same_side = np.nan if not adjacent else sum(first == second for first, second in adjacent) / len(adjacent)
    return (fraction_known, top_agreement, bot_agreement, same_side)
