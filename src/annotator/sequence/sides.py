"""Assign strikers by alternating court halves across each finished rally."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from annotator.outcomes.point_winner import OTHER_HALF, Half, fit_alternation
from annotator.sequence.contacts import ContactEvent, ContactSequence


def alternating_sides(final_half: Half, event_count: int) -> list[Half]:
    last = event_count - 1
    return [final_half if (last - index) % 2 == 0 else OTHER_HALF[final_half] for index in range(event_count)]


def alternate_sides(
    sequences: Sequence[ContactSequence], events: Sequence[ContactEvent],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Replace each sequence's raw side guesses with its better alternating phase.

    A tied phase vote leaves that sequence's raw per-contact guesses unchanged.
    Contacts outside every sequence keep their raw guesses.

    :param sequences: Final sequences; each contact belongs to at most one.
    :param events: Full-stream contact events in frame order.
    :return: Sequences and full-stream events with the fitted sides.
    """
    side_by_frame: dict[int, Half | None] = {}
    for sequence in sequences:
        final_half = fit_alternation([event.side for event in sequence.events])
        if final_half is None:
            continue
        fitted = alternating_sides(final_half, len(sequence.events))
        for event, side in zip(sequence.events, fitted, strict=True):
            side_by_frame[event.frame] = side
    revised = tuple(replace(event, side=side_by_frame.get(event.frame, event.side)) for event in events)
    by_frame = {event.frame: event for event in revised}
    revised_sequences = tuple(
        replace(sequence, events=tuple(by_frame[event.frame] for event in sequence.events)) for sequence in sequences
    )
    return revised_sequences, revised
