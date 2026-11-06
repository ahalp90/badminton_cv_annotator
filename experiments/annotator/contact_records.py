"""Read and write contact sequences saved by annotator experiments."""
import math
from collections.abc import Sequence
from typing import Any

from annotator.types import ContactEvent, ContactSequence, CourtHalf


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
