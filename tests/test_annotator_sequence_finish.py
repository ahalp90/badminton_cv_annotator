"""Boundary widening keeps membership; tied side alternation keeps raw guesses."""

from annotator.outcomes.point_winner import Half
from annotator.sequence.boundaries import extend_boundaries
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.sides import alternate_sides

TOP = Half.TOP
BOT = Half.BOT


def _sequence(span_id: int, start_frame: int, end_frame: int, frames: list[int]) -> ContactSequence:
    return ContactSequence(span_id, start_frame, end_frame, tuple(ContactEvent(frame, .8, None) for frame in frames))


def _bounds(sequences) -> list[tuple[int, int]]:
    return [(sequence.start_frame, sequence.end_frame) for sequence in sequences]


def _all_events(*sequences: ContactSequence) -> list[ContactEvent]:
    return [event for sequence in sequences for event in sequence.events]


def test_boundaries_pad_around_first_and_last_contact_without_moving_inward():
    padded = _sequence(0, 100, 200, [105, 190])
    already_wide = _sequence(1, 300, 400, [350])
    near_video_start = _sequence(2, 3, 50, [5])
    for sequence, expected in ((padded, (95, 201)), (already_wide, (300, 400)), (near_video_start, (0, 50))):
        assert _bounds(extend_boundaries([sequence], sequence.events, fps=30.0)) == [expected]
    assert _bounds(extend_boundaries([padded], padded.events, fps=60.0)) == [(85, 211)]


def test_boundaries_stop_at_neighbours_and_split_a_shared_gap_at_its_midpoint():
    left = _sequence(0, 0, 50, [10, 45])
    right = _sequence(1, 52, 100, [55, 90])
    extended = extend_boundaries([left, right], _all_events(left, right), fps=30.0)
    assert _bounds(extended) == [(0, 51), (51, 101)]
    assert [sequence.events for sequence in extended] == [left.events, right.events]


def test_boundaries_stay_put_when_padding_would_take_in_another_contact():
    sequence = _sequence(0, 100, 200, [105, 150])
    stray = ContactEvent(97, .6, None)
    assert _bounds(extend_boundaries([sequence], [stray, *sequence.events], fps=30.0)) == [(100, 200)]


def test_boundaries_leave_a_contact_less_sequence_unchanged():
    left = _sequence(0, 0, 50, [])
    right = _sequence(1, 55, 100, [58])
    assert _bounds(extend_boundaries([left, right], _all_events(left, right), fps=30.0)) == [(0, 50), (50, 100)]


def _sided(span_id: int, frames: list[int], sides: list[Half | None]) -> ContactSequence:
    events = tuple(ContactEvent(frame, .8, side) for frame, side in zip(frames, sides, strict=True))
    return ContactSequence(span_id, frames[0] - 5, frames[-1] + 5, events)


def test_alternation_replaces_raw_sides_with_the_better_phase():
    sequence = _sided(0, [10, 20, 30], [TOP, TOP, BOT])
    sequences, events = alternate_sides([sequence], sequence.events)
    assert [event.side for event in sequences[0].events] == [BOT, TOP, BOT]
    assert [event.side for event in events] == [BOT, TOP, BOT]


def test_tied_alternation_keeps_raw_sides_and_out_of_span_contacts_keep_theirs():
    tied = _sided(0, [10, 20], [TOP, TOP])
    unknown = _sided(1, [100], [None])
    fitted = _sided(2, [200, 210, 220], [TOP, BOT, BOT])
    stray = ContactEvent(150, .7, TOP)
    stream = [*tied.events, *unknown.events, stray, *fitted.events]
    sequences, events = alternate_sides([tied, unknown, fitted], stream)
    assert [[event.side for event in sequence.events] for sequence in sequences] == [
        [TOP, TOP], [None], [TOP, BOT, TOP],
    ]
    assert [(event.frame, event.side) for event in events] == [
        (10, TOP), (20, TOP), (100, None), (150, TOP), (200, TOP), (210, BOT), (220, TOP),
    ]


def test_padding_stops_at_video_end_without_losing_last_contact():
    sequence = _sequence(0, 30, 100, [50, 99])
    result = extend_boundaries([sequence], sequence.events, fps=30.0, frame_count=100)
    assert _bounds(result) == [(30, 100)]
    assert result[0].events == sequence.events
