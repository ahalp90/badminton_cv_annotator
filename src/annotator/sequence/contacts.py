"""Contact events, rally contact sequences and the candidate scores behind them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from annotator.config import ScalingKind
from annotator.outcomes.point_winner import OTHER_HALF, Half


@dataclass(frozen=True)
class ContactEvent:
    """A contact frame, model probability (NaN for heuristics) and player side."""

    frame: int
    probability: float
    side: Half | None


@dataclass(frozen=True)
class ContactSequence:
    """One half-open rally interval and its contacts in chronological order."""

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


@dataclass(frozen=True)
class FinishedRally:
    """A selected sequence and its fitted final striker, including ties."""

    sequence: ContactSequence
    final_half: Half | None

    @property
    def first_half(self) -> Half | None:
        """Derive the first striker from the fitted final striker."""
        if self.final_half is None:
            return None
        return self.final_half if len(self.sequence.events) % 2 else OTHER_HALF[self.final_half]


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

    Each chosen rally owns the contacts inside its bounds. Detector contacts
    outside every chosen rally survive in the full stream.
    """
    by_frame: dict[int, ContactEvent] = {}
    sequences = []
    for sequence in initial:
        after = chosen[sequence.span_id]
        sequences.append(after)
        for event in after.events:
            by_frame[event.frame] = event
    for event in events:
        if not any(sequence.start_frame <= event.frame < sequence.end_frame for sequence in sequences):
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
