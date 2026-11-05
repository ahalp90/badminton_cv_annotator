"""Shortlist possible serves and missed contacts from detector scores."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from annotator.config import frames_at_fps
from annotator.contacts.model import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
    strongest_spaced_rows,
)
from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.features import physical_rows


@dataclass(frozen=True)
class RallyCandidates:
    """Shortlisted contacts by rally and their shared physical evidence."""

    by_span: dict[int, tuple[ContactEvent, ...]]
    physical: dict[int, np.ndarray]


EARLIER_SERVE_CANDIDATES = 2


LATER_CONTACT_CANDIDATES = 6


def shortlist_serve_rows(
    sequences: Sequence[ContactSequence],
    scores: ScoredContacts,
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
        the candidate rows' interval IDs.
    :param fps: Source frame rate.
    :return: Candidate row indices by span ID, strongest first, for every span
        with a contact.
    """
    frames = scores.candidates["frame"]
    interval_ids = scores.candidates["interval_id"]
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    shortlists: dict[int, list[int]] = {}
    previous_end = -1
    for sequence in sequences:
        if sequence.events:
            first_frame = sequence.events[0].frame
            first_row = int(np.searchsorted(frames, first_frame))
            interval_id = int(interval_ids[first_row])
            interval_start, interval_end = search_intervals[interval_id]
            prefix_start = interval_start
            if interval_start <= previous_end < interval_end:
                prefix_start = max(prefix_start, previous_end)
            in_prefix = (frames >= prefix_start) & (frames < first_frame)
            possible = np.flatnonzero((interval_ids == interval_id) & in_prefix)
            shortlists[sequence.span_id] = strongest_spaced_rows(
                possible, frames, scores.probabilities, distance,
                limit=EARLIER_SERVE_CANDIDATES, taken_frames=(first_frame,),
            )
        previous_end = sequence.end_frame
    return shortlists


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
    rows = np.flatnonzero(eligible & np.isfinite(scores.probabilities))
    return strongest_spaced_rows(
        rows, frames, scores.probabilities, distance,
        limit=LATER_CONTACT_CANDIDATES, taken_frames=[event.frame for event in sequence.events],
    )


def build_rally_candidates(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent], scores: ScoredContacts,
    search_intervals: Sequence[tuple[int, int]], *, fps: float,
    side_for_frame: Callable[[int], Half | None],
) -> RallyCandidates:
    """Shortlist earlier and missed contacts, retaining their physical evidence.

    Events come from the selected rows of these same scored candidates. Reuse
    their side guesses; only shortlisted additions need a new guess.
    """
    contacts_by_frame = {event.frame: event for event in events}
    serve_rows = shortlist_serve_rows(initial, scores, search_intervals, fps)
    candidates_by_span = {}
    for before in initial:
        rows = [*serve_rows.get(before.span_id, ()), *shortlist_later_rows(before, scores, fps)]
        frames = set()
        for row in rows:
            frame = int(scores.candidates['frame'][row])
            frames.add(frame)
            if frame not in contacts_by_frame:
                contacts_by_frame[frame] = ContactEvent(frame, float(scores.probabilities[row]), side_for_frame(frame))
        candidates_by_span[before.span_id] = tuple(contacts_by_frame[frame] for frame in sorted(frames))
    physical = physical_rows(scores.candidates, contacts_by_frame)
    return RallyCandidates(candidates_by_span, physical)
