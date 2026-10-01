"""Widen rally bounds around their contacts without changing which contacts they own."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    events_between,
    scale_frames,
)

BOUNDARY_PADDING_AT_30_FPS = 10


def proposed_bounds(sequence: ContactSequence, padding: int) -> list[int]:
    if not sequence.events:
        return [sequence.start_frame, sequence.end_frame]
    first_frame = min(event.frame for event in sequence.events)
    last_frame = max(event.frame for event in sequence.events)
    start = max(0, min(sequence.start_frame, first_frame - padding))
    end = max(sequence.end_frame, last_frame + padding + 1)
    return [start, end]


def extend_boundaries(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent], fps: float,
    *, frame_count: int | None = None,
) -> tuple[ContactSequence, ...]:
    """Pad each sequence around its first and last contact, keeping its contacts.

    Padding never moves a bound inward and stops at the neighbouring
    sequences' current bounds. When both neighbours reach into one gap, the
    gap's midpoint divides it. A sequence whose padded bounds would take in or
    lose any full-stream contact keeps its current bounds instead.

    :param sequences: Chosen sequences in chronological order, disjoint.
    :param events: Full-stream contact events in frame order.
    :param fps: Source frame rate.
    :param frame_count: Video length; bound extensions stop at this exclusive end.
    :return: Sequences in the same order, with the same contacts.
    """
    padding = scale_frames(BOUNDARY_PADDING_AT_30_FPS, fps)
    bounds = [proposed_bounds(sequence, padding) for sequence in sequences]
    for position in range(len(sequences)):
        if frame_count is not None:
            bounds[position][1] = min(bounds[position][1], frame_count)
        if position:
            bounds[position][0] = max(bounds[position][0], sequences[position - 1].end_frame)
        if position + 1 < len(sequences):
            bounds[position][1] = min(bounds[position][1], sequences[position + 1].start_frame)
    for left_position in range(len(sequences) - 1):
        right_position = left_position + 1
        if bounds[left_position][1] <= bounds[right_position][0]:
            continue
        midpoint = (sequences[left_position].end_frame + sequences[right_position].start_frame) // 2
        bounds[left_position][1] = min(bounds[left_position][1], midpoint)
        bounds[right_position][0] = max(bounds[right_position][0], midpoint)

    extended = []
    for sequence, (start_frame, end_frame) in zip(sequences, bounds, strict=True):
        if events_between(events, start_frame, end_frame) == sequence.events:
            extended.append(replace(sequence, start_frame=start_frame, end_frame=end_frame))
        else:
            extended.append(sequence)
    return tuple(extended)
