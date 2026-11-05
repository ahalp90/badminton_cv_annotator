"""Contact events, rally contact sequences and the candidate scores behind them."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from annotator.outcomes.point_winner import OTHER_HALF, CourtHalf


@dataclass(frozen=True)
class ContactEvent:
    """A contact frame, model probability (NaN for heuristics) and player side."""

    frame: int
    probability: float
    side: CourtHalf | None
    wrist_distances: tuple[float, float] | None = None  # TOP/BOT gaps in body heights.


@dataclass(frozen=True)
class ContactSequence:
    """One half-open rally interval and its contacts in chronological order."""

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


@dataclass(frozen=True)
class FinishedRally:
    """A selected sequence and its final striker, unknown if no player was identified."""

    sequence: ContactSequence
    final_half: CourtHalf | None

    @property
    def first_half(self) -> CourtHalf | None:
        """Derive the first striker from the fitted final striker."""
        if self.final_half is None:
            return None
        return self.final_half if len(self.sequence.events) % 2 else OTHER_HALF[self.final_half]


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


def stream_payload(sequences: Sequence[ContactSequence], events: Sequence[ContactEvent]) -> dict[str, Any]:
    """Store final sections and the full stream with each contact's side."""
    def event_row(event: ContactEvent) -> list:
        probability = None if math.isnan(event.probability) else event.probability
        if probability is not None and not math.isfinite(probability):
            raise ValueError("contact probability must be finite or unknown")
        return [event.frame, probability, None if event.side is None else event.side.value]

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
        if probability is not None and not math.isfinite(probability):
            raise ValueError("contact probability must be finite or unknown")
        return ContactEvent(
            frame, math.nan if probability is None else probability, None if side is None else CourtHalf(side),
        )

    sequences = tuple(
        ContactSequence(sequence['span_id'], sequence['start_frame'], sequence['end_frame'],
                        tuple(event_from_row(row) for row in sequence['events']))
        for sequence in payload['sequences']
    )
    return sequences, tuple(event_from_row(row) for row in payload['events'])
