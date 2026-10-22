"""Physical contact evidence and rally summaries used by the fitted selectors.

Feature names, column order and NaN placement are part of saved model bundles.
"""

from collections.abc import Iterable, Mapping
from itertools import pairwise
from typing import Any

import numpy as np

from annotator.contacts.model import CONTACT_FEATURE_NAMES
from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactSequence

PHYSICAL_FEATURE_NAMES = CONTACT_FEATURE_NAMES


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


def positive_probabilities(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Return a binary classifier's positive-class probability for each row."""
    if not len(matrix):
        return np.empty(0, dtype=np.float64)
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("model must have exactly one positive class")
    probabilities = np.asarray(model.predict_proba(matrix))[:, int(positive[0])]
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("model probabilities must be finite and between zero and one")
    return probabilities


def physical_rows(contact_features: np.ndarray, frames: Iterable[int]) -> dict[int, np.ndarray]:
    """Index the physical contact features of each requested frame.

    :param contact_features: Structured feature rows for one video, one per frame.
    :param frames: Frames whose physical evidence the rally model uses.
    :return: One float64 row per frame, in PHYSICAL_FEATURE_NAMES order, NaNs kept.
    """
    requested = sorted(set(frames))
    feature_frames = contact_features["frame"]
    if len(np.unique(feature_frames)) != len(feature_frames):
        raise ValueError("contact feature frames repeat")
    selected = contact_features[np.isin(feature_frames, requested)]
    values = np.column_stack([selected[name].astype(np.float64) for name in PHYSICAL_FEATURE_NAMES])
    if np.isinf(values).any():
        raise ValueError("physical contact features contain infinity")
    rows = {int(frame): values[index] for index, frame in enumerate(selected["frame"])}
    missing = [frame for frame in requested if frame not in rows]
    if missing:
        raise ValueError(f"contact features are missing frames {missing[:5]}")
    return rows


def physical_or_nan(physical: Mapping[int, np.ndarray], frame: int | None) -> np.ndarray:
    if frame is None:
        return np.full(len(PHYSICAL_FEATURE_NAMES), np.nan)
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
            top_matches += side == (Half.TOP if index % 2 == 0 else Half.BOT)
            bot_matches += side == (Half.BOT if index % 2 == 0 else Half.TOP)
        top_agreement = top_matches / known_count
        bot_agreement = bot_matches / known_count
    adjacent = [(first, second) for first, second in pairwise(guesses) if first is not None and second is not None]
    same_side = np.nan if not adjacent else sum(first == second for first, second in adjacent) / len(adjacent)
    return (fraction_known, top_agreement, bot_agreement, same_side)
