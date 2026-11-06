"""Shortlist possible serves and missed contacts from detector scores."""

import numpy as np

from annotator.config import frames_at_fps
from annotator.contacts.model import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
    strongest_spaced_rows,
)
from annotator.sequence.contacts import ContactSequence

EARLIER_SERVE_CANDIDATES = 2


LATER_CONTACT_CANDIDATES = 6


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
    rows = np.flatnonzero(eligible & np.isfinite(scores.probabilities))
    return strongest_spaced_rows(
        rows, frames, scores.probabilities, distance,
        limit=LATER_CONTACT_CANDIDATES, taken_frames=[event.frame for event in sequence.events],
    )
