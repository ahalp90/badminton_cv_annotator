"""Shortlist possible serves and missed contacts from detector scores."""

from collections.abc import Sequence

import numpy as np

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.sequence.contacts import CandidateScores, ContactSequence, scale_frames

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
