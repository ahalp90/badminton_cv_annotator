"""Propose rally contacts and measure each alternative for the fitted selector.

Feature names, column order and NaN placement are part of saved model bundles.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import combinations, pairwise

import numpy as np

from annotator.config import frames_at_fps
from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from annotator.contacts.player_attribution import contact_player_evidence
from annotator.contacts.score_candidates import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
    strongest_spaced_rows,
)
from annotator.inference.prepare_contact_inference import AnnotationEvidence
from annotator.types import ContactEvent, ContactSequence, CourtHalf

EARLIER_SERVE_CANDIDATES = 2
LATER_CONTACT_CANDIDATES = 6

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

FEATURE_NAMES = (
    *(f'before__{name}' for name in SUMMARY_FEATURE_NAMES),
    *(f'after__{name}' for name in SUMMARY_FEATURE_NAMES),
    *(f'before__{name}' for name in SIDE_FEATURE_NAMES),
    *(f'after__{name}' for name in SIDE_FEATURE_NAMES),
    'added_count', 'deleted_count',
    *(f'{kind}_probability_{stat}' for kind in ('added', 'deleted') for stat in ('min', 'max', 'mean')),
    *(f'{kind}__{name}' for kind in ('original_first', 'new_first', 'strongest_addition', 'weakest_deletion')
      for name in CONTACT_FEATURE_NAMES),
    'unused_candidate_count', 'unused_probability_min', 'unused_probability_max', 'unused_probability_mean',
    'usable_candidate_count', 'excluded_candidate_count',
    'gaps_with_candidates', 'positive_gap_count', 'gaps_without_candidates', 'longest_gap_seconds',
    'last_gap_has_candidate',
    'gap_probability_max', 'gap_probability_second', 'gap_probability_mean', 'gap_probability_sum',
    'strongest_unused_left_gap_seconds', 'strongest_unused_right_gap_seconds',
    *(f'strongest_unused__{name}' for name in CONTACT_FEATURE_NAMES),
)


@dataclass(frozen=True)
class RallyContext:
    """Detector evidence shared by all options for one video."""

    fps: float
    initial: tuple[ContactSequence, ...]
    events: tuple[ContactEvent, ...]
    shortlisted_by_span: dict[int, tuple[ContactEvent, ...]]
    physical_by_frame: dict[int, np.ndarray]
    frame_count: int | None = None
    bounds_supplied: bool = False


@dataclass(frozen=True)
class RallyOptions:
    """Alternative contact lists and their feature rows, grouped by initial rally."""

    sequences: tuple[ContactSequence, ...]
    matrix: np.ndarray
    span_rows: dict[int, np.ndarray]
    context: RallyContext


def build_rally_context(
    evidence: AnnotationEvidence, contacts: ScoredContacts,
) -> RallyContext:
    """Prepare initial rallies and shortlisted contacts for fitting and inference."""
    def event_for_row(row: int) -> ContactEvent:
        frame = int(contacts.candidates['frame'][row])
        side, distances = contact_player_evidence(
            frame, evidence.track, evidence.sticky, evidence.bboxes, evidence.scene_courts,
        )
        return ContactEvent(frame, float(contacts.probabilities[row]), side, distances)

    events = tuple(event_for_row(int(row)) for row in contacts.kept_indices)
    initial = initial_sequences(evidence.heuristic_spans, events)
    contacts_by_frame = {event.frame: event for event in events}
    used_rows = set(contacts.kept_indices)
    shortlisted_by_span = {}
    previous_end = -1
    for before in initial:
        serve_rows = shortlist_serve_rows(before, contacts, evidence.fps, previous_end=previous_end)
        rows = [*serve_rows, *shortlist_later_rows(before, contacts, evidence.fps)]
        used_rows.update(rows)
        frames = set()
        for row in rows:
            frame = int(contacts.candidates['frame'][row])
            frames.add(frame)
            # Retained contacts already have player evidence; attribute only additions.
            if frame not in contacts_by_frame:
                contacts_by_frame[frame] = event_for_row(row)
        shortlisted_by_span[before.span_id] = tuple(contacts_by_frame[frame] for frame in sorted(frames))
        previous_end = before.end_frame
    selected_rows = sorted(used_rows)
    selected_frames = contacts.candidates['frame'][selected_rows]
    # Each frame has one feature row. Copy the selection so training contexts
    # do not retain the full video matrix.
    physical_rows = contacts.feature_matrix[selected_rows]
    physical_by_frame = {int(frame): physical for frame, physical in zip(selected_frames, physical_rows, strict=True)}
    return RallyContext(
        evidence.fps, initial, events, shortlisted_by_span, physical_by_frame,
        frame_count=len(evidence.track), bounds_supplied=evidence.bounds_supplied,
    )


def build_rally_options(context: RallyContext, max_edits: int = 3, deletion_candidates: int = 6) -> RallyOptions:
    """Build ordered contact-toggle proposals from detector evidence alone."""
    if max_edits < 0 or deletion_candidates < 0:
        raise ValueError('edit and deletion budgets must be non-negative')
    sequences = []
    matrices = []
    span_rows = {}
    stream_frames = np.asarray([event.frame for event in context.events], dtype=np.int64)
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, context.fps)
    previous_end = -1
    for before in context.initial:
        shortlisted = context.shortlisted_by_span.get(before.span_id, ())
        earliest = min((before.start_frame, *(event.frame for event in shortlisted)))
        first, last = np.searchsorted(stream_frames, (earliest, before.end_frame))
        earliest_start = before.start_frame if context.bounds_supplied else previous_end
        proposals = toggle_proposals(
            before, shortlisted, context.events[first:last], earliest_start, distance, max_edits, deletion_candidates,
        )
        # An earlier serve can absorb stream contacts; include them in unused-contact features.
        considered_by_frame = {event.frame: event for sequence in proposals for event in sequence.events}
        considered_by_frame.update((event.frame, event) for event in shortlisted)
        considered_contacts = tuple(considered_by_frame[frame] for frame in sorted(considered_by_frame))
        before_summary, before_sides = sequence_summary(before, context.fps), side_agreement(before)
        start_row = len(sequences)
        for after in proposals:
            sequences.append(after)
            matrices.append(option_features(before, after, considered_contacts, context, before_summary, before_sides))
        span_rows[before.span_id] = np.arange(start_row, len(sequences), dtype=np.int64)
        previous_end = before.end_frame
    matrix = np.asarray(matrices, dtype=np.float32).reshape(len(sequences), len(FEATURE_NAMES))
    return RallyOptions(tuple(sequences), matrix, span_rows, context)


def shortlist_serve_rows(
    sequence: ContactSequence, scores: ScoredContacts, fps: float, *, previous_end: int,
) -> list[int]:
    """Shortlist earlier serves from the first contact's search interval.

    Stay after the previous rally. Stronger rows win, then earlier frames.
    Keep up to two, spaced clear of the first contact and each earlier pick.
    """
    if not sequence.events:
        return []
    frames = scores.candidates["frame"]
    interval_ids = scores.candidates["interval_id"]
    first_frame = sequence.events[0].frame
    first_row = int(np.searchsorted(frames, first_frame))
    same_interval = interval_ids == interval_ids[first_row]
    in_prefix = (frames >= previous_end) & (frames < first_frame)
    possible = np.flatnonzero(same_interval & in_prefix)
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    return strongest_spaced_rows(
        possible, frames, scores.probabilities, distance,
        limit=EARLIER_SERVE_CANDIDATES, taken_frames=(first_frame,),
    )


def shortlist_later_rows(sequence: ContactSequence, scores: ScoredContacts, fps: float) -> list[int]:
    """Shortlist candidate rows that could be a missed contact after the first one.

    A frame must lie inside the sequence, after its first contact, and more than
    the nearby-contact distance from every contact and from stronger picks.
    Stronger candidates win, then earlier frames.
    """
    if not sequence.events:
        return []
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    frames = scores.candidates["frame"].astype(np.int64)
    eligible = (frames > sequence.events[0].frame) & (frames < sequence.end_frame)
    rows = np.flatnonzero(eligible)
    return strongest_spaced_rows(
        rows, frames, scores.probabilities, distance,
        limit=LATER_CONTACT_CANDIDATES, taken_frames=[event.frame for event in sequence.events],
    )


def toggle_proposals(
    before: ContactSequence, shortlisted: tuple[ContactEvent, ...], stream: tuple[ContactEvent, ...],
    earliest_start: int, distance: int, max_edits: int, deletion_candidates: int,
) -> list[ContactSequence]:
    """Return unique alternatives, unchanged first and then fewest contact edits."""
    original_frames = tuple(event.frame for event in before.events)
    original_set = set(original_frames)
    additions = tuple(event for event in shortlisted
                      if event.frame not in original_set and event.frame < before.end_frame)
    deletions = []
    if before.events and deletion_candidates:
        deletions = [before.events[0]]
        weakest = sorted(before.events[1:], key=lambda event: (event.probability, event.frame))
        deletions.extend(weakest[:deletion_candidates - 1])
    proposals = {original_frames: before}
    for added_count in range(min(max_edits, len(additions)) + 1):
        for added in combinations(additions, added_count):
            start = min((before.start_frame, *(event.frame for event in added)))
            if start < earliest_start:
                continue
            # An earlier serve absorbs the full stream before removing explicit deletions.
            base = {event.frame: event for event in stream if start <= event.frame < before.end_frame}
            base.update((event.frame, event) for event in added)
            for deleted_count in range(min(max_edits - added_count, len(deletions)) + 1):
                for deleted in combinations(deletions, deleted_count):
                    deleted_frames = {event.frame for event in deleted}
                    events = tuple(base[frame] for frame in sorted(base) if frame not in deleted_frames)
                    if any(abs(candidate.frame - event.frame) <= distance
                           for candidate in added for event in events if event.frame != candidate.frame):
                        continue
                    sequence = replace(before, start_frame=start, events=events)
                    proposals.setdefault(tuple(event.frame for event in events), sequence)
    # Only additions move the start earlier, and they cannot be deleted, so
    # contact frames identify each proposal. Zero edits puts the original first.
    ordered_frames = sorted(proposals, key=lambda frames: (len(original_set ^ set(frames)), frames))
    return [proposals[frames] for frames in ordered_frames]


def option_features(
    before: ContactSequence, after: ContactSequence, considered_contacts: tuple[ContactEvent, ...],
    context: RallyContext, before_summary: tuple[float, ...], before_sides: tuple[float, ...],
) -> np.ndarray:
    """Measure a proposed contact sequence against the initial rally evidence."""
    before_frames = {event.frame for event in before.events}
    after_frames = {event.frame for event in after.events}
    added = tuple(event for event in after.events if event.frame not in before_frames)
    deleted = tuple(event for event in before.events if event.frame not in after_frames)
    strongest = min(added, key=lambda event: (-event.probability, event.frame)) if added else None
    weakest = min(deleted, key=lambda event: (event.probability, event.frame)) if deleted else None
    physical_frames = (
        before.events[0].frame if before.events else None,
        after.events[0].frame if after.events else None,
        None if strongest is None else strongest.frame, None if weakest is None else weakest.frame,
    )
    unused_features, unused_frame = unused_evidence(after, considered_contacts, context.fps)
    return np.concatenate((
        before_summary,
        sequence_summary(after, context.fps),
        before_sides,
        side_agreement(after),
        (len(added), len(deleted)),
        probability_summary(added),
        probability_summary(deleted),
        *(physical_or_nan(context.physical_by_frame, frame) for frame in physical_frames),
        unused_features,
        physical_or_nan(context.physical_by_frame, unused_frame),
    ))


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


def physical_or_nan(physical: Mapping[int, np.ndarray], frame: int | None) -> np.ndarray:
    if frame is None:
        return np.full(len(CONTACT_FEATURE_NAMES), np.nan)
    return physical[frame]


def unused_evidence(
    after: ContactSequence, considered_contacts: tuple[ContactEvent, ...], fps: float,
) -> tuple[tuple[float, ...], int | None]:
    """Measure omitted contacts and the gaps where another contact could fit.

    A usable candidate lies inside the rally and far enough from selected contacts.
    Gaps include the intervals before the first and after the last contact.
    """
    frames = np.asarray([event.frame for event in after.events], dtype=np.int64)
    chosen = set(frames.tolist())
    unused = tuple(event for event in considered_contacts if event.frame not in chosen)
    durations = np.diff([after.start_frame, *frames.tolist(), after.end_frame])
    positive_gaps = set(np.flatnonzero(durations > 0).tolist())
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    probabilities_by_gap: dict[int, list[float]] = {}
    usable_count = 0
    for event in unused:
        inside = after.start_frame <= event.frame < after.end_frame
        spaced = not np.any(np.abs(frames - event.frame) <= distance)
        if inside and spaced:
            gap = int(np.searchsorted(frames, event.frame))
            probabilities_by_gap.setdefault(gap, []).append(event.probability)
            usable_count += 1
    gaps_with_candidates = positive_gaps & set(probabilities_by_gap)
    strongest = min(unused, key=lambda event: (-event.probability, event.frame)) if unused else None
    left_gap = right_gap = np.nan
    if strongest is not None and after.start_frame <= strongest.frame < after.end_frame:
        position = int(np.searchsorted(frames, strongest.frame))
        left = int(frames[position - 1]) if position else after.start_frame
        right = int(frames[position]) if position < len(frames) else after.end_frame
        left_gap, right_gap = (strongest.frame - left) / fps, (right - strongest.frame) / fps
    values = (
        float(len(unused)),
        *probability_summary(unused),
        float(usable_count),
        float(len(unused) - usable_count),
        float(len(gaps_with_candidates)),
        float(len(positive_gaps)),
        float(len(positive_gaps - gaps_with_candidates)),
        float(max(durations) / fps),
        float(len(durations) - 1 in gaps_with_candidates),
        *summarise_gap_maxima(list(probabilities_by_gap.values())),
        left_gap,
        right_gap,
    )
    return values, None if strongest is None else strongest.frame


def summarise_gap_maxima(values: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Summarise the strongest candidate in each gap with available evidence."""
    if not values:
        return (np.nan, np.nan, np.nan, np.nan)
    maxima = np.asarray([max(group) for group in values], dtype=np.float64)
    ordered = np.sort(maxima)[::-1]
    return (float(ordered[0]), float(ordered[1]) if len(ordered) > 1 else np.nan,
            float(np.mean(maxima)), float(np.sum(maxima)))


def probability_summary(events: tuple[ContactEvent, ...]) -> tuple[float, float, float]:
    if not events:
        return np.nan, np.nan, np.nan
    values = [event.probability for event in events]
    return min(values), max(values), float(np.mean(values))


def events_between(events: Sequence[ContactEvent], start_frame: int, end_frame: int) -> tuple[ContactEvent, ...]:
    """Return the events inside one half-open frame interval, keeping their order."""
    return tuple(event for event in events if start_frame <= event.frame < end_frame)


def initial_sequences(
    spans: Sequence[tuple[int, int]], events: Sequence[ContactEvent],
) -> tuple[ContactSequence, ...]:
    """Group the initial contacts into ordered, disjoint rally spans.

    Later serve edits can move a start back only as far as the previous span's end.
    """
    sequences = []
    previous_end = -1
    for span_id, (start_frame, end_frame) in enumerate(spans):
        if not previous_end <= start_frame < end_frame:
            raise ValueError(f"span {span_id}: rally spans must be ordered, disjoint and non-empty")
        owned = events_between(events, start_frame, end_frame)
        sequences.append(ContactSequence(span_id, start_frame, end_frame, owned))
        previous_end = end_frame
    return tuple(sequences)
