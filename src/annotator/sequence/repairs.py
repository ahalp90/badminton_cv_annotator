"""Repair serves, delete contacts and propose missed contacts for each rally.

The selector considers these rule-based repairs alongside combinations of several
added or deleted contacts. The repair set also fixes the training weight scale.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import shortlist_later_frames, shortlist_serve_frames
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    events_between,
    scale_frames,
)
from annotator.sequence.features import physical_rows


@dataclass(frozen=True)
class RallyRepairs:
    """Repaired sequences, shortlisted contacts and shared physical evidence."""

    sequences: tuple[ContactSequence, ...]
    candidates: dict[int, tuple[ContactEvent, ...]]
    physical: dict[int, np.ndarray]


def check_events_match_kept_candidates(events: Sequence[ContactEvent], scores: CandidateScores) -> None:
    if np.any(np.diff(scores.frames) <= 0):
        raise ValueError('candidate scores must be in strictly increasing frame order')
    kept_frames = scores.frames[scores.kept].tolist()
    kept_probabilities = scores.probabilities[scores.kept].tolist()
    event_frames = [event.frame for event in events]
    event_probabilities = [event.probability for event in events]
    if event_frames != kept_frames or event_probabilities != kept_probabilities:
        raise ValueError('initial events must be exactly the kept candidates')


def without_frame(sequence: ContactSequence, frame: int) -> ContactSequence:
    return replace(sequence, events=tuple(event for event in sequence.events if event.frame != frame))


def serve_and_deletion_repairs(
    sequence: ContactSequence, earlier: Sequence[ContactEvent], events: Sequence[ContactEvent],
    previous_end: int,
) -> tuple[ContactSequence, ...]:
    """Keep the contacts, repair the serve, delete one contact, or combine both.

    An earlier serve absorbs contacts before the original start. A combined edit
    must preserve its new serve and differ from every simpler repair.
    """
    serves = []
    for candidate in earlier:
        start = min(sequence.start_frame, candidate.frame)
        if start < previous_end:
            continue
        expanded = events_between(events, start, sequence.end_frame)
        if all(event.frame != candidate.frame for event in expanded):
            expanded = tuple(sorted((*expanded, candidate), key=lambda event: event.frame))
        added = replace(sequence, start_frame=start, events=expanded)
        replaced = without_frame(added, sequence.events[0].frame)
        serves.extend(((False, candidate.frame, added), (True, candidate.frame, replaced)))
    seeds = [sequence, *(revised for _replaced, _frame, revised in serves)]
    seeds.extend(without_frame(sequence, event.frame) for event in sequence.events)
    simple_frames = {tuple(event.frame for event in seed.events) for seed in seeds}
    for is_replacement, serve_frame, revised in serves:
        for event in revised.events:
            undoes_serve = event.frame == serve_frame
            replaces_first = not is_replacement and event.frame == sequence.events[0].frame
            if undoes_serve or replaces_first:
                continue
            deleted = without_frame(revised, event.frame)
            if tuple(contact.frame for contact in deleted.events) not in simple_frames:
                seeds.append(deleted)
    return tuple(seeds)


def add_later_contacts(
    repairs: Sequence[ContactSequence], candidates: Sequence[ContactEvent], distance: int,
) -> tuple[ContactSequence, ...]:
    """Keep each repair, then add each later contact that fits its current bounds.

    Recheck spacing after the serve or deletion repair has changed membership.
    Each resulting sequence adds at most one of these later contacts.
    """
    sequences = []
    for repaired in repairs:
        sequences.append(repaired)
        for candidate in candidates:
            inside = repaired.start_frame <= candidate.frame < repaired.end_frame
            too_close = any(abs(candidate.frame - event.frame) <= distance for event in repaired.events)
            if inside and not too_close:
                added = tuple(sorted((*repaired.events, candidate), key=lambda event: event.frame))
                sequences.append(replace(repaired, events=added))
    return tuple(sequences)


def build_rally_repairs(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent], scores: CandidateScores,
    contact_features: np.ndarray, search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None],
) -> RallyRepairs:
    """Shortlist candidates and build rally repairs without scoring fitted models.

    :param initial: Initial rallies in chronological order.
    :param events: Detector-selected contacts, exactly the kept candidates.
    :param scores: All scored contact candidates, in frame order.
    :param contact_features: Structured physical evidence for searched frames.
    :param search_intervals: Half-open intervals indexed by candidate interval IDs.
    :param fps: Video frame rate.
    :param side_for_frame: Raw court-half guess for a shortlisted new contact.
    :return: Direct contact sequences and the evidence shared by their features.
    """
    events_by_frame = {event.frame: event for event in events}
    serve_rows = shortlist_serve_frames(initial, scores, search_intervals, fps)
    later_frames = {sequence.span_id: shortlist_later_frames(sequence, scores, fps) for sequence in initial}
    serve_frames = {int(scores.frames[row]) for rows in serve_rows.values() for row in rows}
    later_frame_set = {frame for frames in later_frames.values() for frame in frames}
    sides = {}
    for frame in sorted((serve_frames | later_frame_set) - set(events_by_frame)):
        side = side_for_frame(frame)
        sides[frame] = None if side is None else Half(side)
    probability_by_frame = dict(zip(scores.frames.tolist(), scores.probabilities.tolist()))
    physical = physical_rows(contact_features, set(events_by_frame) | serve_frames | later_frame_set)
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    sequences = []
    candidates_by_span = {}
    previous_end = -1
    for before in initial:
        later = tuple(ContactEvent(frame, probability_by_frame[frame], sides[frame])
                      for frame in later_frames[before.span_id])
        earlier = []
        for row in serve_rows.get(before.span_id, ()):
            frame = int(scores.frames[row])
            if scores.kept[row]:
                event = events_by_frame[frame]
            else:
                event = ContactEvent(frame, probability_by_frame[frame], sides[frame])
            earlier.append(event)
        candidates = {event.frame: event for event in later}
        for event in earlier:
            if min(before.start_frame, event.frame) >= previous_end:
                candidates[event.frame] = event
        candidates_by_span[before.span_id] = tuple(candidates[frame] for frame in sorted(candidates))
        repairs = serve_and_deletion_repairs(before, earlier, events, previous_end)
        sequences.extend(add_later_contacts(repairs, later, distance))
        previous_end = before.end_frame
    return RallyRepairs(tuple(sequences), candidates_by_span, physical)
