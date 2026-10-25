"""Contact shortlists, seed proposals and selected full-stream membership."""

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import shortlist_later_frames, shortlist_serve_frames
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    apply_choices,
    initial_sequences,
)
from annotator.sequence.repairs import (
    add_later_contacts,
    check_events_match_kept_candidates,
    serve_and_deletion_repairs,
)


def _scores(rows: list[tuple[int, int, float, bool]]) -> CandidateScores:
    """Build candidate scores from (frame, interval ID, probability, kept) rows."""
    frames, interval_ids, probabilities, kept = zip(*rows, strict=True)
    return CandidateScores(np.asarray(frames), np.asarray(interval_ids), np.asarray(probabilities), np.asarray(kept))


def _frames(sequence: ContactSequence) -> list[int]:
    return [event.frame for event in sequence.events]


def test_initial_sequences_leave_out_of_span_contacts_unowned():
    events = [ContactEvent(10, .9, None), ContactEvent(60, .8, None), ContactEvent(130, .7, None)]
    sequences = initial_sequences([(0, 50), (100, 200)], events)
    assert [_frames(sequence) for sequence in sequences] == [[10], [130]]
    assert [sequence.span_id for sequence in sequences] == [0, 1]


@pytest.mark.parametrize("spans", [[(0, 50), (40, 90)], [(50, 90), (0, 40)], [(10, 10)]])
def test_initial_sequences_reject_overlapping_unordered_or_empty_spans(spans):
    with pytest.raises(ValueError, match="ordered, disjoint and non-empty"):
        initial_sequences(spans, [])


def test_serve_shortlist_clips_to_previous_span_and_breaks_ties_by_earlier_frame():
    scores = _scores([
        (5, 0, .5, False),
        (12, 0, .5, False),
        (30, 0, .7, False),
        (36, 0, .95, False),  # within six frames of the first contact
        (40, 0, .9, True),
        (110, 1, .99, False),  # inside the interval but before the previous span ends
        (130, 1, .6, False),
        (170, 1, .5, False),
        (176, 1, .9, False),
        (180, 1, .8, True),
    ])
    events = [ContactEvent(40, .9, None), ContactEvent(180, .8, None)]
    sequences = initial_sequences([(20, 120), (150, 250), (260, 290)], events)
    shortlists = shortlist_serve_frames(sequences, scores, [(0, 100), (100, 300)], fps=30.0)
    frames_by_span = {span_id: scores.frames[rows].tolist() for span_id, rows in shortlists.items()}
    assert frames_by_span == {0: [30, 5], 1: [130, 170]}


@pytest.mark.parametrize(("rows", "expected"), [
    ([(40, 0, .9, True)], []),
    ([(10, 0, .5, False), (14, 0, .4, False), (40, 0, .9, True)], [0]),
])
def test_serve_shortlist_uses_available_candidates(rows, expected):
    scores = _scores(rows)
    sequences = initial_sequences([(0, 100)], [ContactEvent(40, .9, None)])
    assert shortlist_serve_frames(sequences, scores, [(0, 100)], fps=30.0) == {0: expected}


def test_later_shortlist_skips_near_contacts_and_keeps_six_strongest():
    scores = _scores([
        (20, 0, .9, True),
        (24, 0, .99, False),
        (50, 0, .6, False),
        (54, 0, .6, False),
        (100, 0, .7, True),
        (104, 0, .9, False),
        (120, 0, .1, False),
        (130, 0, .2, False),
        (140, 0, .3, False),
        (150, 0, np.nan, False),
        (160, 0, .4, False),
        (170, 0, .5, False),
        (180, 0, .05, False),
        (200, 0, .8, False),
    ])
    sequence = ContactSequence(0, 0, 200, (ContactEvent(20, .9, None), ContactEvent(100, .7, None)))
    assert shortlist_later_frames(sequence, scores, fps=30.0) == [50, 170, 160, 140, 130, 120]
    assert shortlist_later_frames(ContactSequence(1, 0, 200, ()), scores, fps=30.0) == []


def test_applied_choices_keep_out_of_span_contacts():
    events = [ContactEvent(frame, .8, None) for frame in (20, 60, 120, 170, 200, 300)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    delete_second = ContactSequence(0, 0, 100, (events[0],))
    absorb_gap = ContactSequence(1, 120, 250, (events[2], events[3], events[4]))
    sequences, stream = apply_choices(initial, events, {0: delete_second, 1: absorb_gap})
    assert [_frames(sequence) for sequence in sequences] == [[20], [120, 170, 200]]
    assert [event.frame for event in stream] == [20, 120, 170, 200, 300]


def test_applied_choices_remove_rejected_contact_inside_the_expanded_rally():
    events = [ContactEvent(frame, .8, None) for frame in (20, 120, 170, 200)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    new_serve = ContactEvent(110, .6, None)
    replaced = ContactSequence(1, 110, 250, (new_serve, events[2], events[3]))
    _sequences, stream = apply_choices(initial, events, {0: initial[0], 1: replaced})
    assert [event.frame for event in stream] == [20, 110, 170, 200]


def test_serve_and_deletion_repairs_keep_order_and_skip_redundant_combinations():
    before = ContactSequence(0, 100, 200, (ContactEvent(120, .9, Half.TOP), ContactEvent(160, .8, Half.BOT)))
    earlier = [ContactEvent(frame, .5, None) for frame in (105, 110)]
    seeds = serve_and_deletion_repairs(before, earlier, before.events, previous_end=-1)
    assert [_frames(seed) for seed in seeds] == [
        [120, 160], [105, 120, 160], [105, 160], [110, 120, 160], [110, 160],
        [160], [120], [105, 120], [105], [110, 120], [110],
    ]
    assert seeds[1].start_frame == 100


def test_earlier_serve_absorbs_gap_contacts_and_stops_at_previous_rally():
    gap = ContactEvent(98, .7, Half.BOT)
    before = ContactSequence(1, 100, 200, (ContactEvent(120, .9, Half.TOP), ContactEvent(160, .8, Half.BOT)))
    earlier = [ContactEvent(frame, .5, None) for frame in (90, 97)]
    seeds = serve_and_deletion_repairs(before, earlier, [gap, *before.events], previous_end=95)
    assert not any(90 in _frames(seed) for seed in seeds)
    added = next(seed for seed in seeds if _frames(seed) == [97, 98, 120, 160])
    assert added.start_frame == 97 and added.events[1] is gap


def test_initial_events_must_be_exactly_the_kept_candidates():
    scores = _scores([(20, 0, .9, True), (40, 0, .8, False)])
    check_events_match_kept_candidates((ContactEvent(20, .9, None),), scores)
    with pytest.raises(ValueError, match='exactly the kept candidates'):
        check_events_match_kept_candidates((ContactEvent(40, .8, None),), scores)


def test_later_contacts_are_checked_against_each_repaired_sequence():
    kept = ContactSequence(0, 100, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    deleted_first = ContactSequence(0, 100, 200, kept.events[1:])
    candidates = [ContactEvent(125, .6, None), ContactEvent(140, .5, None), ContactEvent(250, .9, None)]
    repaired = add_later_contacts((kept, deleted_first), candidates, distance=6)
    assert [_frames(sequence) for sequence in repaired] == [
        [120, 160], [120, 140, 160], [160], [125, 160], [140, 160],
    ]
