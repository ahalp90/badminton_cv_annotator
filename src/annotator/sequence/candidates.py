"""Choose the extra contact frames each rally sequence may use.

Serve candidates sit before a sequence's first contact and may add or replace
it. Later-contact candidates sit after the first contact and may fill a missed
contact. Both lists come from the detector's scored candidates without labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

EARLIER_SERVE_CANDIDATES = 2
LATER_CONTACT_CANDIDATES = 6
SERVE_CANDIDATE_FEATURE_NAMES = (
    "candidate_contact_score",
    "fixed_contact_score",
    "frames_before_fixed_at_30_fps",
    "candidate_from_section_start_at_30_fps",
    "section_length_at_30_fps",
    "candidate_already_kept",
    "candidate_side_known",
    "fixed_side_known",
    "candidate_and_fixed_side_match",
)


@dataclass(frozen=True)
class ServeCandidate:
    """One earlier frame that may be a sequence's missed or mistimed serve."""

    span_id: int
    event: ContactEvent
    first_contact_frame: int
    features: tuple[float, ...]  # one per SERVE_CANDIDATE_FEATURE_NAMES entry


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


def at_30_fps(frame_count: int, fps: float) -> float:
    return float(frame_count) * 30.0 / fps


def serve_candidates(
    sequences: Sequence[ContactSequence],
    shortlists: Mapping[int, Sequence[int]],
    scores: CandidateScores,
    events_by_frame: Mapping[int, ContactEvent],
    sides: Mapping[int, Half | None],
    fps: float,
) -> dict[int, tuple[ServeCandidate, ...]]:
    """Describe each shortlisted serve frame with the serve models' nine inputs.

    A candidate the detector already kept reuses its full-stream event, so a
    serve edit that absorbs it keeps the same contact record.
    """
    candidates: dict[int, tuple[ServeCandidate, ...]] = {}
    for sequence in sequences:
        rows = shortlists.get(sequence.span_id)
        if rows is None:
            continue
        first = sequence.events[0]
        section_length = sequence.end_frame - sequence.start_frame
        described = []
        for row in rows:
            frame = int(scores.frames[row])
            kept = bool(scores.kept[row])
            if kept:
                event = events_by_frame[frame]
            else:
                event = ContactEvent(frame, float(scores.probabilities[row]), sides[frame])
            both_known = event.side is not None and first.side is not None
            features = (
                event.probability,
                first.probability,
                at_30_fps(first.frame - frame, fps),
                at_30_fps(frame - sequence.start_frame, fps),
                at_30_fps(section_length, fps),
                float(kept),
                float(event.side is not None),
                float(first.side is not None),
                float(both_known and event.side == first.side),
            )
            described.append(ServeCandidate(sequence.span_id, event, first.frame, features))
        candidates[sequence.span_id] = tuple(described)
    return candidates
