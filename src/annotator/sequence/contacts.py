"""Contact events, rally contact sequences and the candidate scores behind them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from annotator.config import ScalingKind
from annotator.outcomes.point_winner import Half


@dataclass(frozen=True)
class ContactEvent:
    """One predicted contact and the court half of its raw striker guess."""

    frame: int
    probability: float
    side: Half | None


@dataclass(frozen=True)
class ContactSequence:
    """One half-open rally interval and the contacts it currently owns."""

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


class CandidateScores(NamedTuple):
    """Every scored contact candidate in one video, in frame order.

    ``kept`` marks the candidates the contact detector selected. Those rows are
    the full-stream contact events, including any outside a rally interval.
    """

    frames: np.ndarray  # (candidate,) int
    interval_ids: np.ndarray  # (candidate,) index into the search intervals
    probabilities: np.ndarray  # (candidate,) float
    kept: np.ndarray  # (candidate,) bool


def scale_frames(frames_at_30_fps: int, fps: float) -> int:
    """Scale a frame count defined at 30 fps with the repository's half-up rule."""
    return int(ScalingKind.FRAME_COUNT.scale(float(frames_at_30_fps), float(fps)))


def events_between(events: Sequence[ContactEvent], start_frame: int, end_frame: int) -> tuple[ContactEvent, ...]:
    """Return the events inside one half-open frame interval, keeping their order."""
    return tuple(event for event in events if start_frame <= event.frame < end_frame)


def initial_sequences(
    spans: Sequence[tuple[int, int]], events: Sequence[ContactEvent],
) -> tuple[ContactSequence, ...]:
    """Number chronological half-open rally spans and give each its contacts.

    Later stages rely on the spans being ordered and disjoint: serve edits may
    move a start back only as far as the previous span's end.

    :param spans: Heuristic ``(start_frame, end_frame)`` rally intervals.
    :param events: Full-stream contact events in frame order.
    :return: One contact sequence per span, numbered by position.
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


def apply_choices(
    initial: Sequence[ContactSequence],
    events: Sequence[ContactEvent],
    chosen: Mapping[int, ContactSequence],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Swap in each chosen sequence and rebuild the full contact stream.

    Contacts an initial sequence owned survive only if its chosen sequence keeps
    them. Contacts outside every initial sequence always survive, so an edit
    that deletes an absorbed out-of-span contact leaves it in the full stream.
    """
    owned = {event.frame for sequence in initial for event in sequence.events}
    by_frame: dict[int, ContactEvent] = {}
    sequences = []
    for sequence in initial:
        after = chosen[sequence.span_id]
        sequences.append(after)
        for event in after.events:
            by_frame[event.frame] = event
    for event in events:
        if event.frame not in owned:
            by_frame[event.frame] = event
    return tuple(sequences), tuple(by_frame[frame] for frame in sorted(by_frame))


def stream_payload(sequences: Sequence[ContactSequence], events: Sequence[ContactEvent]) -> dict[str, Any]:
    """Store final sections and the full stream with each contact's side."""
    def event_row(event: ContactEvent) -> list:
        return [event.frame, event.probability, None if event.side is None else event.side.value]

    return {
        'sequences': [
            {'span_id': sequence.span_id, 'start_frame': sequence.start_frame, 'end_frame': sequence.end_frame,
             'events': [event_row(event) for event in sequence.events]}
            for sequence in sequences
        ],
        'events': [event_row(event) for event in events],
    }


def stream_from_payload(payload: dict[str, Any]) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    def event_from_row(row: list) -> ContactEvent:
        frame, probability, side = row
        return ContactEvent(frame, probability, None if side is None else Half(side))

    sequences = tuple(
        ContactSequence(sequence['span_id'], sequence['start_frame'], sequence['end_frame'],
                        tuple(event_from_row(row) for row in sequence['events']))
        for sequence in payload['sequences']
    )
    return sequences, tuple(event_from_row(row) for row in payload['events'])
