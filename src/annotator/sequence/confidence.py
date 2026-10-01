"""Rank finished rallies for review without changing their annotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.sequence import RefinedSequences
from annotator.sequence.contacts import ContactEvent, scale_frames
from annotator.sequence.edits import SequenceOption
from annotator.sequence.features import (
    PHYSICAL_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    insertion_features,
    positive_probabilities,
    sequence_summary,
    side_agreement,
)

BASE_FEATURE_NAMES = (
    "selected_score", "advantage_over_next_output", "best_insertion_output_score",
    "best_different_start_output_score", "discarded_candidate_count", "strongest_discarded_score",
    "strongest_discarded_left_gap_seconds", "strongest_discarded_right_gap_seconds",
    "best_discarded_side_vote_improvement", "raw_side_fraction_known", "raw_top_vote", "raw_bot_vote",
    "adjacent_same_raw_side", *(f"output__{name}" for name in SUMMARY_FEATURE_NAMES),
    *(f"strongest_discarded__{name}" for name in PHYSICAL_FEATURE_NAMES),
)
GAP_EVIDENCE_NAMES = (
    "strongest_local_score", "second_strongest_local_score", "mean_local_score", "sum_local_score",
    "strongest_timing_score", "second_strongest_timing_score", "mean_timing_score", "sum_timing_score",
    "usable_candidate_count", "distinct_gaps_with_candidates", "positive_duration_gap_count",
    "gaps_with_no_candidates", "excluded_candidate_count", "longest_gap_seconds", "last_gap_has_candidate",
)
GAP_FEATURE_NAMES = (*BASE_FEATURE_NAMES, *GAP_EVIDENCE_NAMES)


@dataclass(frozen=True)
class ConfidenceFeatures:
    """One row per span ID; gap inputs append evidence to the base inputs."""

    span_ids: tuple[int, ...]
    base: np.ndarray
    gap: np.ndarray


@dataclass(frozen=True)
class RallyConfidence:
    """Review scores and their reusable model inputs, in the same span order.

    Scores order review priority. They do not grant automatic approval or
    modify contacts, sides or rally boundaries.
    """

    features: ConfidenceFeatures
    scores: np.ndarray


def discarded_contact_features(
    selected: SequenceOption,
    options: Sequence[SequenceOption],
    scores: Sequence[float],
    candidates: Sequence[ContactEvent],
    physical: Mapping[int, np.ndarray],
    fps: float,
) -> tuple[float, ...]:
    """Describe chooser margins and discarded contacts before boundary widening."""
    selected_score = next(float(score) for option, score in zip(options, scores, strict=True) if option == selected)
    alternatives = [float(score) for option, score in zip(options, scores, strict=True)
                    if option.sequence != selected.sequence]
    insertions = [float(score) for option, score in zip(options, scores, strict=True)
                  if option.added_contact is not None]
    other_starts = [float(score) for option, score in zip(options, scores, strict=True)
                    if option.edit.serve_frame is not None and option.edit.serve_frame != selected.edit.serve_frame]
    sequence = selected.sequence
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    discarded = [candidate for candidate in candidates
                 if all(abs(candidate.frame - event.frame) > distance for event in sequence.events)]
    strongest = max(discarded, key=lambda candidate: candidate.probability, default=None)
    raw_sides = side_agreement(sequence)
    original_agreement = max(raw_sides[1:3])
    improvements = []
    for candidate in discarded:
        revised = replace(sequence, events=tuple(sorted((*sequence.events, candidate), key=lambda event: event.frame)))
        improvements.append(max(side_agreement(revised)[1:3]) - original_agreement)
    left_gap = right_gap = np.nan
    block = np.full(len(PHYSICAL_FEATURE_NAMES), np.nan)
    if strongest is not None:
        left = [event.frame for event in sequence.events if event.frame < strongest.frame]
        right = [event.frame for event in sequence.events if event.frame > strongest.frame]
        left_gap = (strongest.frame - left[-1]) / fps if left else np.nan
        right_gap = (right[0] - strongest.frame) / fps if right else np.nan
        block = physical[strongest.frame]
    return (
        selected_score, selected_score - max(alternatives) if alternatives else np.nan,
        max(insertions, default=np.nan), max(other_starts, default=np.nan), float(len(discarded)),
        np.nan if strongest is None else strongest.probability, left_gap, right_gap,
        max(improvements, default=np.nan), *raw_sides, *sequence_summary(sequence, fps), *block,
    )


def summarise_gap_maxima(values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Summarise the strongest candidate in each gap with available evidence."""
    if not values:
        return (np.nan, np.nan, np.nan, np.nan)
    maxima = np.asarray([max(group) for group in values], dtype=np.float64)
    ordered = np.sort(maxima)[::-1]
    return (float(ordered[0]), float(ordered[1]) if len(ordered) > 1 else np.nan,
            float(np.mean(maxima)), float(np.sum(maxima)))


def gap_contact_features(
    selected: SequenceOption,
    candidates: Sequence[ContactEvent],
    physical: Mapping[int, np.ndarray],
    fps: float,
    insertion_model: Any,
) -> tuple[float, ...]:
    """Score remaining contacts against the finished bounds and raw side guesses."""
    sequence = selected.sequence
    frames = np.asarray([event.frame for event in sequence.events], dtype=np.int64)
    durations = np.diff(np.asarray((sequence.start_frame, *frames.tolist(), sequence.end_frame), dtype=np.int64))
    positive_gaps = set(np.flatnonzero(durations > 0).tolist())
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    contexts = []
    owners = []
    timing_by_gap: dict[int, list[float]] = {}
    excluded = 0
    for candidate in candidates:
        inside = sequence.start_frame <= candidate.frame < sequence.end_frame
        duplicate = bool(np.any(np.abs(frames - candidate.frame) <= distance))
        if not inside or duplicate:
            excluded += 1
            continue
        gap = int(np.searchsorted(frames, candidate.frame, side="right"))
        timing_by_gap.setdefault(gap, []).append(float(candidate.probability))
        owners.append(gap)
        revised = replace(sequence, events=tuple(sorted((*sequence.events, candidate), key=lambda event: event.frame)))
        contexts.append(SequenceOption(replace(selected.edit, sequence=sequence), candidate, revised))
    scores = positive_probabilities(insertion_model, insertion_features(contexts, physical, fps))
    local_by_gap: dict[int, list[float]] = {}
    for gap, score in zip(owners, scores, strict=True):
        local_by_gap.setdefault(gap, []).append(float(score))
    present_gaps = positive_gaps & set(timing_by_gap)
    return (
        *summarise_gap_maxima(list(local_by_gap.values())),
        *summarise_gap_maxima(list(timing_by_gap.values())),
        float(len(contexts)), float(len(present_gaps)), float(len(positive_gaps)),
        float(len(positive_gaps) - len(present_gaps)), float(excluded),
        float(np.max(durations[list(positive_gaps)]) / fps) if positive_gaps else 0.0,
        float(len(durations) - 1 in present_gaps),
    )


def build_confidence_features(refined: RefinedSequences, insertion_model: Any, fps: float) -> ConfidenceFeatures:
    """Reuse chooser work while keeping its raw side evidence intact.

    Final side alternation is an annotation decision. Confidence needs the
    earlier individual guesses to measure agreement rather than assume it.
    """
    span_ids = tuple(refined.scored_insertion.chosen)
    final_by_span = {sequence.span_id: sequence for sequence in refined.sequences}
    grouped: dict[int, list[int]] = {}
    for index, option in enumerate(refined.pool.options):
        grouped.setdefault(option.span_id, []).append(index)
    base_rows = []
    gap_rows = []
    for span_id in span_ids:
        selected = refined.pool.options[refined.scored_insertion.chosen[span_id]]
        indices = grouped[span_id]
        candidates = refined.pool.later_candidates.get(span_id, ())
        row = list(discarded_contact_features(
            selected, [refined.pool.options[index] for index in indices],
            refined.scored_insertion.scores[indices], candidates, refined.pool.physical, fps,
        ))
        final = final_by_span[span_id]
        guarded = replace(selected.sequence, start_frame=final.start_frame, end_frame=final.end_frame)
        # Only boundary summaries change; chooser margins and raw votes stay fixed.
        row[13:13 + len(SUMMARY_FEATURE_NAMES)] = sequence_summary(guarded, fps)
        base_rows.append(row)
        gap_rows.append(gap_contact_features(
            replace(selected, sequence=guarded), candidates, refined.pool.physical, fps, insertion_model,
        ))
    base = np.asarray(base_rows, dtype=np.float64).reshape(len(span_ids), len(BASE_FEATURE_NAMES))
    gaps = np.asarray(gap_rows, dtype=np.float64).reshape(len(span_ids), len(GAP_EVIDENCE_NAMES))
    return ConfidenceFeatures(span_ids, base, np.column_stack((base, gaps)))


def score_rally_confidence(
    refined: RefinedSequences, model: Any, insertion_model: Any, fps: float,
) -> RallyConfidence:
    """Rank rallies using the sequence bundle's existing insertion model."""
    features = build_confidence_features(refined, insertion_model, fps)
    return RallyConfidence(features, positive_probabilities(model, features.gap))
