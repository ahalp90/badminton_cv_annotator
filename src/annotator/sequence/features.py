"""Describe serve edits and sequence options with the fitted models' inputs.

Column order, NaN placement and every name below match the fitted chooser
models. NaN marks an input that does not apply, such as the deleted-contact
columns of an edit that deletes nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from annotator.contacts.model import CONTACT_FEATURE_NAMES
from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import SERVE_CANDIDATE_FEATURE_NAMES, ServeCandidate
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import (
    EDIT_KINDS,
    SERVE_EDIT_KINDS,
    EditKind,
    SequenceOption,
)

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
SERVE_EDIT_FEATURE_NAMES = (*SERVE_CANDIDATE_FEATURE_NAMES, "action_is_replace")
# The fitted bundles store these column names; they are part of the model contract.
SEQUENCE_FEATURE_NAMES = (
    *(f"before__{name}" for name in SUMMARY_FEATURE_NAMES),
    *(f"after__{name}" for name in SUMMARY_FEATURE_NAMES),
    *(f"action__{kind}" for kind in EDIT_KINDS),
    *(f"existing_start__{name}" for name in SERVE_EDIT_FEATURE_NAMES),
    "deleted_contact_score",
    *(f"before__{name}" for name in SIDE_FEATURE_NAMES),
    *(f"after__{name}" for name in SIDE_FEATURE_NAMES),
    *(f"original_fixed__{name}" for name in PHYSICAL_FEATURE_NAMES),
    *(f"candidate__{name}" for name in PHYSICAL_FEATURE_NAMES),
    *(f"deleted__{name}" for name in PHYSICAL_FEATURE_NAMES),
)
SERVE_SCORE_FEATURE_NAMES = (
    "chosen_summary_opening_score",
    "chosen_physical_opening_score",
    "best_summary_opening_score",
    "best_physical_opening_score",
)
INSERTION_FEATURE_NAMES = (
    "has_later_insertion",
    "later_score",
    "left_gap_seconds",
    "right_gap_seconds",
    "left_same_raw_side",
    "right_same_raw_side",
    "base_top_vote",
    "base_bot_vote",
    *(f"later__{name}" for name in PHYSICAL_FEATURE_NAMES),
)

ServeEditKey = tuple[int, int, EditKind]  # span ID, serve frame, ADD or REPLACE


@dataclass(frozen=True)
class ServeEditInputs:
    """Serve-model inputs for adding, or replacing with, each serve candidate."""

    rows_by_key: dict[ServeEditKey, int]
    # (serve edit, 10 + 2 * physical): SERVE_EDIT_FEATURE_NAMES, then the candidate's
    # physical row, then the first contact's physical row.
    matrix: np.ndarray


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
    :param frames: Frames whose physical rows the choosers read.
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


def serve_edit_inputs(
    candidates_by_span: Mapping[int, Sequence[ServeCandidate]], physical: Mapping[int, np.ndarray],
) -> ServeEditInputs:
    """Pair every serve candidate with the add and replace edits it allows."""
    rows_by_key: dict[ServeEditKey, int] = {}
    rows = []
    for candidates in candidates_by_span.values():
        for candidate in candidates:
            for kind in SERVE_EDIT_KINDS:
                rows_by_key[(candidate.span_id, candidate.event.frame, kind)] = len(rows)
                summary = np.asarray((*candidate.features, float(kind == EditKind.REPLACE)), dtype=np.float64)
                rows.append(np.concatenate(
                    (summary, physical[candidate.event.frame], physical[candidate.first_contact_frame])
                ))
    width = len(SERVE_EDIT_FEATURE_NAMES) + 2 * len(PHYSICAL_FEATURE_NAMES)
    matrix = np.vstack(rows) if rows else np.empty((0, width), dtype=np.float64)
    return ServeEditInputs(rows_by_key, matrix)


def serve_edit_key(option: SequenceOption) -> ServeEditKey | None:
    serve_kind = option.edit.serve_kind
    if serve_kind is None or option.edit.serve_frame is None:
        return None
    return (option.span_id, option.edit.serve_frame, serve_kind)


def sequence_features(
    options: Sequence[SequenceOption],
    initial_by_span: Mapping[int, ContactSequence],
    serve_inputs: ServeEditInputs,
    events_by_frame: Mapping[int, ContactEvent],
    physical: Mapping[int, np.ndarray],
    fps: float,
) -> np.ndarray:
    """Describe each option's sequence against its initial sequence.

    :param options: The shared option pool.
    :param initial_by_span: Initial sequences by span ID, the "before" view.
    :param serve_inputs: Serve-model inputs for every serve edit in the pool.
    :param events_by_frame: Full-stream contacts, for deleted-contact probabilities.
    :param physical: Physical rows for every contact and candidate frame.
    :param fps: Source frame rate.
    :return: (option, SEQUENCE_FEATURE_NAMES) matrix.
    """
    before_by_span: dict[int, tuple[tuple[float, ...], tuple[float, ...], np.ndarray]] = {}
    serve_width = len(SERVE_EDIT_FEATURE_NAMES)
    rows = []
    for option in options:
        edit = option.edit
        if option.span_id not in before_by_span:
            initial = initial_by_span[option.span_id]
            first_frame = initial.events[0].frame if initial.events else None
            before_by_span[option.span_id] = (
                sequence_summary(initial, fps), side_agreement(initial), physical_or_nan(physical, first_frame),
            )
        before_summary, before_sides, first_contact_physical = before_by_span[option.span_id]
        serve_key = serve_edit_key(option)
        serve_values = np.full(serve_width, np.nan)
        if serve_key is not None:
            serve_values = serve_inputs.matrix[serve_inputs.rows_by_key[serve_key], :serve_width]
        deleted_probability = np.nan if edit.deleted_frame is None else events_by_frame[edit.deleted_frame].probability
        rows.append(np.concatenate((
            before_summary,
            sequence_summary(option.sequence, fps),
            [float(edit.kind == kind) for kind in EDIT_KINDS],
            serve_values,
            [deleted_probability],
            before_sides,
            side_agreement(option.sequence),
            first_contact_physical,
            physical_or_nan(physical, edit.serve_frame),
            physical_or_nan(physical, edit.deleted_frame),
        )))
    return np.vstack(rows) if rows else np.empty((0, len(SEQUENCE_FEATURE_NAMES)), dtype=np.float64)


def serve_score_features(
    options: Sequence[SequenceOption], serve_scores: Mapping[ServeEditKey, tuple[float, float]],
) -> np.ndarray:
    """Join each option's serve-edit scores and its sequence's best serve scores.

    :param options: The shared option pool.
    :param serve_scores: Summary-model and physical-model probability per serve edit.
    :return: (option, SERVE_SCORE_FEATURE_NAMES) matrix; NaN where no serve edit
        applies.
    """
    keys_by_span: dict[int, set[ServeEditKey]] = {}
    for option in options:
        key = serve_edit_key(option)
        if key is not None:
            keys_by_span.setdefault(option.span_id, set()).add(key)
    best_by_span = {}
    for span_id, keys in keys_by_span.items():
        best_by_span[span_id] = (max(serve_scores[key][0] for key in keys), max(serve_scores[key][1] for key in keys))
    rows = []
    for option in options:
        key = serve_edit_key(option)
        own = (np.nan, np.nan) if key is None else serve_scores[key]
        rows.append((*own, *best_by_span.get(option.span_id, (np.nan, np.nan))))
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(SERVE_SCORE_FEATURE_NAMES))


def insertion_features(
    options: Sequence[SequenceOption], physical: Mapping[int, np.ndarray], fps: float,
) -> np.ndarray:
    """Describe each added later contact against the edited sequence it joins.

    Gaps and raw-side matches refer to the neighbouring contacts of the edit's
    sequence before the addition. Options without an added contact have a zero
    flag and NaN elsewhere.

    :return: (option, INSERTION_FEATURE_NAMES) matrix.
    """
    rows = []
    for option in options:
        added = option.added_contact
        if added is None:
            rows.append([0.0, *([np.nan] * (len(INSERTION_FEATURE_NAMES) - 1))])
            continue
        edited = option.edit.sequence
        before = [event for event in edited.events if event.frame < added.frame]
        after = [event for event in edited.events if event.frame > added.frame]
        left = before[-1] if before else None
        right = after[0] if after else None
        left_gap = np.nan if left is None else (added.frame - left.frame) / fps
        right_gap = np.nan if right is None else (right.frame - added.frame) / fps
        left_same = np.nan
        if left is not None and added.side is not None and left.side is not None:
            left_same = float(left.side == added.side)
        right_same = np.nan
        if right is not None and added.side is not None and right.side is not None:
            right_same = float(right.side == added.side)
        _known, top_vote, bot_vote, _adjacent = side_agreement(edited)
        rows.append([
            1.0, added.probability, left_gap, right_gap, left_same, right_same, top_vote, bot_vote,
            *physical[added.frame],
        ])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(INSERTION_FEATURE_NAMES))
