"""Boundary finishing preserves chosen contacts and appropriate video context."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

from annotator.contacts.player_attribution import fit_rally_sides
from annotator.rallies.build_contact_proposals import (
    RallyContext,
    RallyOptions,
    events_between,
    initial_sequences,
)
from annotator.rallies.refine_rallies import extend_boundaries, finish_rallies
from annotator.types import ContactEvent, ContactSequence, CourtHalf, FinishedRally

TOP = CourtHalf.TOP
BOT = CourtHalf.BOT


def test_rally_finishing_uses_wrist_costs_and_preserves_raw_guesses():
    events = (ContactEvent(10, 0.9, TOP, (0.1, 0.2)), ContactEvent(20, 0.8, TOP, (0.1, 2.0)))
    sequence = ContactSequence(0, 0, 30, events)
    finished = fit_rally_sides([sequence])[0]
    assert [event.side for event in finished.sequence.events] == [BOT, TOP]
    assert [event.side for event in sequence.events] == [TOP, TOP]
    assert finished.final_half == TOP


def _sequence(span_id: int, start_frame: int, end_frame: int, frames: list[int]) -> ContactSequence:
    return ContactSequence(span_id, start_frame, end_frame, tuple(ContactEvent(frame, .8, None) for frame in frames))


def _bounds(sequences) -> list[tuple[int, int]]:
    return [(sequence.start_frame, sequence.end_frame) for sequence in sequences]


def _all_events(*sequences: ContactSequence) -> list[ContactEvent]:
    return [event for sequence in sequences for event in sequence.events]


def _finish_selected(
    initial: Sequence[ContactSequence], events: Sequence[ContactEvent],
    choices: dict[int, ContactSequence], fps: float, *, bounds_supplied: bool = False,
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...]]:
    context = RallyContext(
        fps, tuple(initial), tuple(events), {}, {}, frame_count=3000, bounds_supplied=bounds_supplied,
    )
    rows = {span_id: index for index, span_id in enumerate(choices)}
    pool = RallyOptions(
        tuple(choices.values()), np.empty((len(choices), 0)),
        {span_id: np.array([index]) for span_id, index in rows.items()}, context,
    )
    return finish_rallies(pool, rows)


def test_boundaries_pad_around_first_and_last_contact_without_moving_inward():
    padded = _sequence(0, 100, 200, [105, 190])
    already_wide = _sequence(1, 300, 400, [350])
    near_video_start = _sequence(2, 3, 50, [5])
    for sequence, expected in ((padded, (95, 201)), (already_wide, (300, 400)), (near_video_start, (0, 50))):
        assert _bounds(extend_boundaries([sequence], sequence.events, fps=30.0, frame_count=500)) == [expected]
    assert _bounds(extend_boundaries([padded], padded.events, fps=60.0, frame_count=500)) == [(85, 211)]


def test_boundaries_stop_at_neighbours_and_split_a_shared_gap_at_its_midpoint():
    left = _sequence(0, 0, 50, [10, 45])
    right = _sequence(1, 52, 100, [55, 90])
    extended = extend_boundaries([left, right], _all_events(left, right), fps=30.0, frame_count=500)
    assert _bounds(extended) == [(0, 51), (51, 101)]
    assert [sequence.events for sequence in extended] == [left.events, right.events]


def test_boundaries_stay_put_when_padding_would_take_in_another_contact():
    sequence = _sequence(0, 100, 200, [105, 150])
    stray = ContactEvent(97, .6, None)
    assert _bounds(extend_boundaries([sequence], [stray, *sequence.events], fps=30.0, frame_count=500)) == [(100, 200)]


def test_boundaries_leave_a_contact_less_sequence_unchanged():
    left = _sequence(0, 0, 50, [])
    right = _sequence(1, 55, 100, [58])
    assert _bounds(extend_boundaries([left, right], _all_events(left, right), fps=30.0, frame_count=500)) == [(0, 50), (50, 100)]


def _sided(span_id: int, frames: list[int], sides: list[CourtHalf | None]) -> ContactSequence:
    events = tuple(ContactEvent(frame, .8, side) for frame, side in zip(frames, sides, strict=True))
    return ContactSequence(span_id, frames[0] - 5, frames[-1] + 5, events)


def test_alternation_replaces_raw_sides_with_the_better_phase():
    sequence = _sided(0, [10, 20, 30], [TOP, TOP, BOT])
    sequences = fit_rally_sides([sequence])
    assert [event.side for event in sequences[0].sequence.events] == [BOT, TOP, BOT]


def test_tied_alternation_uses_first_known_player_and_preserves_unknown_rallies():
    tied = _sided(0, [10, 20], [TOP, TOP])
    unknown = _sided(1, [100], [None])
    fitted = _sided(2, [200, 210, 220], [TOP, BOT, BOT])
    stray = ContactEvent(150, .7, TOP)
    stream = [*tied.events, *unknown.events, stray, *fitted.events]
    sequences, events = _finish_selected(
        [tied, unknown, fitted], stream, {0: tied, 1: unknown, 2: fitted}, 30, bounds_supplied=True,
    )
    assert [[event.side for event in rally.sequence.events] for rally in sequences] == [
        [TOP, BOT], [None], [TOP, BOT, TOP],
    ]
    assert [(event.frame, event.side) for event in events] == [
        (10, TOP), (20, BOT), (100, None), (150, TOP), (200, TOP), (210, BOT), (220, TOP),
    ]


def test_selected_contacts_carry_final_fit_and_preserve_outside_events():
    original = _sided(0, [10, 20, 30], [TOP, TOP, BOT])
    chosen = replace(original, events=original.events[1:])
    outside = ContactEvent(50, .7, TOP)
    rallies, events = _finish_selected((original,), (*original.events, outside), {0: chosen}, 30, bounds_supplied=True)
    assert rallies[0].final_half is BOT
    assert [(event.frame, event.side) for event in rallies[0].sequence.events] == [(20, TOP), (30, BOT)]
    assert [(event.frame, event.side) for event in events] == [(20, TOP), (30, BOT), (50, TOP)]


def test_padding_stops_at_video_end_without_losing_last_contact():
    sequence = _sequence(0, 30, 100, [50, 99])
    result = extend_boundaries([sequence], sequence.events, fps=30.0, frame_count=100)
    assert _bounds(result) == [(30, 100)]
    assert result[0].events == sequence.events


def test_absorbed_rejected_contact_is_removed_before_padding():
    original = _sequence(0, 110, 200, [120, 140])
    candidate = ContactEvent(100, .9, None)
    rejected = ContactEvent(105, .6, None)
    chosen = replace(original, start_frame=100, events=(candidate, *original.events))
    rallies, stream = _finish_selected((original,), (rejected, *original.events), {0: chosen}, 30)
    sequences = tuple(rally.sequence for rally in rallies)
    assert [event.frame for event in stream] == [100, 120, 140]
    assert sequences[0].start_frame == 90
    assert events_between(stream, sequences[0].start_frame, sequences[0].end_frame) == sequences[0].events


@pytest.mark.parametrize('fps', [30, 60])
def test_discovery_keeps_context_and_removes_rejected_hits_owned_by_chosen_span(fps):
    original = _sequence(0, 0, 30 * fps, [fps, 10 * fps, 12 * fps])
    chosen = replace(original, events=original.events[1:])
    rallies, stream = _finish_selected((original,), original.events, {0: chosen}, fps)
    sequences = tuple(rally.sequence for rally in rallies)
    assert _bounds(sequences) == [(0, 30 * fps)]
    assert stream == chosen.events
    assert events_between(stream, *(_bounds(sequences)[0])) == chosen.events


def test_padding_collision_preserves_chosen_bounds():
    original = _sequence(0, 0, 600, [20, 300, 400])
    chosen = replace(original, start_frame=300, events=original.events[1:])
    outside = ContactEvent(295, .8, None)
    rallies, stream = _finish_selected((original,), (*original.events[:1], outside, *chosen.events), {0: chosen}, 30)
    sequences = tuple(rally.sequence for rally in rallies)
    assert _bounds(sequences) == [(300, 600)]
    assert outside in stream
    assert events_between(stream, 300, 600) == chosen.events


def test_supplied_bounds_stay_exact_and_close_rallies_remain_disjoint():
    left = _sequence(0, 0, 500, [200, 450])
    right = _sequence(1, 505, 700, [510, 650])
    events = _all_events(left, right)
    supplied, _ = _finish_selected((left, right), events, {0: left, 1: right}, 30, bounds_supplied=True)
    assert tuple(rally.sequence for rally in supplied) == (left, right)
    rallies, stream = _finish_selected((left, right), events, {0: left, 1: right}, 30)
    discovered = tuple(rally.sequence for rally in rallies)
    assert discovered[0].end_frame <= discovered[1].start_frame
    for sequence in discovered:
        assert events_between(stream, sequence.start_frame, sequence.end_frame) == sequence.events


def test_chosen_rally_owns_internal_hits_even_when_initial_span_did_not():
    original = _sequence(0, 110, 200, [120, 140])
    chosen = _sequence(0, 100, 200, [100, 120, 140])
    stream = [ContactEvent(80, .5, None), ContactEvent(105, .5, None), *original.events,
              ContactEvent(210, .5, None)]
    rallies, rebuilt = _finish_selected((original,), stream, {0: chosen}, 30, bounds_supplied=True)
    assert [event.frame for event in rebuilt] == [80, 100, 120, 140, 210]
    assert tuple(rally.sequence for rally in rallies) == (chosen,)


def test_applied_choices_keep_out_of_span_contacts():
    events = [ContactEvent(frame, .8, None) for frame in (20, 60, 120, 170, 200, 300)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    delete_second = ContactSequence(0, 0, 100, (events[0],))
    absorb_gap = ContactSequence(1, 120, 250, (events[2], events[3], events[4]))
    rallies, stream = _finish_selected(
        initial, events, {0: delete_second, 1: absorb_gap}, 30, bounds_supplied=True,
    )
    assert [[event.frame for event in rally.sequence.events] for rally in rallies] == [[20], [120, 170, 200]]
    assert [event.frame for event in stream] == [20, 120, 170, 200, 300]


def test_applied_choices_remove_rejected_contact_inside_the_expanded_rally():
    events = [ContactEvent(frame, .8, None) for frame in (20, 120, 170, 200)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    new_serve = ContactEvent(110, .6, None)
    replaced = ContactSequence(1, 110, 250, (new_serve, events[2], events[3]))
    _rallies, stream = _finish_selected(
        initial, events, {0: initial[0], 1: replaced}, 30, bounds_supplied=True,
    )
    assert [event.frame for event in stream] == [20, 110, 170, 200]
