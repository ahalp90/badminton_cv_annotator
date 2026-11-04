"""Shortlist possible serves and missed contacts from detector scores."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    scale_frames,
)
from annotator.sequence.features import physical_rows


@dataclass(frozen=True)
class RallyCandidates:
    """Shortlisted contacts by rally and their shared physical evidence."""

    by_span: dict[int, tuple[ContactEvent, ...]]
    physical: dict[int, np.ndarray]


EARLIER_SERVE_CANDIDATES = 2


LATER_CONTACT_CANDIDATES = 6


def shortlist_serve_frames(
    sequences: Sequence[ContactSequence],
    scores: CandidateScores,
    search_intervals: Sequence[tuple[int, int]],
    fps: float,
) -> dict[int, list[int]]:
    """Shortlist the strongest earlier candidate rows before each first contact.

    Candidates come from the first contact's search interval, starting after
    the previous span when that span ends inside the interval. Stronger rows
    win, then earlier frames. Keep up to two candidates; a short prefix may
    have fewer. Each pick must lie more than the nearby-contact
    distance from the first contact and from earlier picks.

    :param sequences: Initial sequences in chronological order.
    :param scores: Every scored candidate in frame order.
    :param search_intervals: Half-open contact search intervals, indexed by
        ``scores.interval_ids``.
    :param fps: Source frame rate.
    :return: Candidate row indices by span ID, strongest first, for every span
        with a contact.
    """
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    shortlists: dict[int, list[int]] = {}
    previous_end = -1
    for sequence in sequences:
        if sequence.events:
            first_frame = sequence.events[0].frame
            first_row = int(np.searchsorted(scores.frames, first_frame))
            interval_id = int(scores.interval_ids[first_row])
            interval_start, interval_end = search_intervals[interval_id]
            prefix_start = interval_start
            if interval_start <= previous_end < interval_end:
                prefix_start = max(prefix_start, previous_end)
            in_prefix = (scores.frames >= prefix_start) & (scores.frames < first_frame)
            possible = np.flatnonzero((scores.interval_ids == interval_id) & in_prefix)
            strongest_first = sorted(
                possible, key=lambda row: (-float(scores.probabilities[row]), int(scores.frames[row])),
            )
            chosen_rows: list[int] = []
            for row in strongest_first:
                frame = int(scores.frames[row])
                chosen_frames = [first_frame, *(int(scores.frames[chosen]) for chosen in chosen_rows)]
                if all(abs(frame - chosen_frame) > distance for chosen_frame in chosen_frames):
                    chosen_rows.append(int(row))
                if len(chosen_rows) == EARLIER_SERVE_CANDIDATES:
                    break
            shortlists[sequence.span_id] = chosen_rows
        previous_end = sequence.end_frame
    return shortlists


def shortlist_later_frames(sequence: ContactSequence, scores: CandidateScores, fps: float) -> list[int]:
    """Shortlist candidate frames that could be a missed contact after the first one.

    A frame must lie inside the sequence, after its first contact, and more than
    the nearby-contact distance from every contact and from stronger picks.
    Stronger candidates win, then earlier frames.
    """
    if not sequence.events:
        return []
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    frames = scores.frames.astype(np.int64)
    eligible = (frames > sequence.events[0].frame + distance) & (frames < sequence.end_frame)
    for event in sequence.events:
        clear_of_contact = np.abs(frames - event.frame) > distance
        eligible = eligible & clear_of_contact
    rows = np.flatnonzero(eligible & np.isfinite(scores.probabilities))
    strongest_first = sorted(rows, key=lambda row: (-float(scores.probabilities[row]), int(frames[row])))
    selected: list[int] = []
    for row in strongest_first:
        frame = int(frames[row])
        if all(abs(frame - other) > distance for other in selected):
            selected.append(frame)
            if len(selected) == LATER_CONTACT_CANDIDATES:
                break
    return selected


def check_events_match_kept_candidates(events: Sequence[ContactEvent], scores: CandidateScores) -> None:
    if np.any(np.diff(scores.frames) <= 0):
        raise ValueError('candidate scores must be in strictly increasing frame order')
    kept_frames = scores.frames[scores.kept].tolist()
    kept_probabilities = scores.probabilities[scores.kept].tolist()
    event_frames = [event.frame for event in events]
    event_probabilities = [event.probability for event in events]
    if event_frames != kept_frames or event_probabilities != kept_probabilities:
        raise ValueError('initial events must be exactly the kept candidates')


def build_rally_candidates(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent], scores: CandidateScores,
    contact_features: np.ndarray, search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None],
) -> RallyCandidates:
    """Gather shortlisted contacts and their physical evidence for rally proposals.

    :param initial: Initial rallies in chronological order.
    :param events: Detector-selected contacts, exactly the kept candidates.
    :param scores: All scored contact candidates, in frame order.
    :param contact_features: Structured physical evidence for candidate frames.
    :param search_intervals: Half-open intervals indexed by candidate interval IDs.
    :param fps: Video frame rate.
    :param side_for_frame: Raw court-half guess for a shortlisted new contact.
    :return: Shortlisted contacts by rally, with physical evidence for contact features.
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
    candidates_by_span = {}
    for before in initial:
        candidates = {}
        for frame in later_frames[before.span_id]:
            candidates[frame] = ContactEvent(frame, probability_by_frame[frame], sides[frame])
        for row in serve_rows.get(before.span_id, ()):
            frame = int(scores.frames[row])
            if scores.kept[row]:
                event = events_by_frame[frame]
            else:
                event = ContactEvent(frame, probability_by_frame[frame], sides[frame])
            candidates[frame] = event
        candidates_by_span[before.span_id] = tuple(candidates[frame] for frame in sorted(candidates))
    return RallyCandidates(candidates_by_span, physical)
