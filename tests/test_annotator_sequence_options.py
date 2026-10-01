"""Candidate shortlists, sequence edits and the option pool keep their fixed order."""

import numpy as np
import pytest

from annotator.outcomes.point_winner import Half
from annotator.sequence.candidates import (
    ServeCandidate,
    shortlist_later_frames,
    shortlist_serve_frames,
)
from annotator.sequence.contacts import (
    CandidateScores,
    ContactEvent,
    ContactSequence,
    initial_sequences,
)
from annotator.sequence.edits import (
    EditKind,
    SequenceEdit,
    apply_choices,
    option_pool,
    sequence_edits,
)


def _scores(rows: list[tuple[int, int, float, bool]]) -> CandidateScores:
    """Build candidate scores from (frame, interval ID, probability, kept) rows."""
    frames, interval_ids, probabilities, kept = zip(*rows, strict=True)
    return CandidateScores(np.asarray(frames), np.asarray(interval_ids), np.asarray(probabilities), np.asarray(kept))


def _frames(sequence: ContactSequence) -> list[int]:
    return [event.frame for event in sequence.events]


def _serve(span_id: int, frame: int, first_contact_frame: int, probability: float = .5) -> ServeCandidate:
    return ServeCandidate(span_id, ContactEvent(frame, probability, None), first_contact_frame, ())


def _edit_summary(edits) -> list[tuple[str, int | None, int | None]]:
    return [(str(edit.kind), edit.serve_frame, edit.deleted_frame) for edit in edits]


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


def test_sequence_edits_keep_fixed_order_and_skip_redundant_combinations():
    sequence = ContactSequence(0, 100, 200, (ContactEvent(120, .9, Half.TOP), ContactEvent(160, .8, Half.BOT)))
    candidates = [_serve(0, 105, 120), _serve(0, 110, 120)]
    edits = sequence_edits(sequence, candidates, sequence.events, previous_end=-1)
    assert _edit_summary(edits) == [
        ("keep", None, None),
        ("add", 105, None),
        ("replace", 105, None),
        ("add", 110, None),
        ("replace", 110, None),
        ("delete", None, 120),
        ("delete", None, 160),
        ("add_delete", 105, 160),
        ("replace_delete", 105, 160),
        ("add_delete", 110, 160),
        ("replace_delete", 110, 160),
    ]
    assert [_frames(edit.sequence) for edit in edits[1:3]] == [[105, 120, 160], [105, 160]]
    assert edits[1].sequence.start_frame == 100


def test_serve_edit_absorbs_gap_contacts_and_stops_at_previous_span():
    gap_contact = ContactEvent(98, .7, Half.BOT)
    sequence = ContactSequence(1, 100, 200, (ContactEvent(120, .9, Half.TOP), ContactEvent(160, .8, Half.BOT)))
    events = [gap_contact, *sequence.events]
    candidates = [_serve(1, 90, 120), _serve(1, 97, 120)]
    edits = sequence_edits(sequence, candidates, events, previous_end=95)
    serve_edits = [edit for edit in edits if edit.serve_frame is not None]
    assert {edit.serve_frame for edit in serve_edits} == {97}
    added = next(edit for edit in serve_edits if edit.kind == EditKind.ADD)
    assert (added.sequence.start_frame, _frames(added.sequence)) == (97, [97, 98, 120, 160])
    assert added.sequence.events[1] is gap_contact


def test_option_pool_checks_later_contacts_against_each_edited_sequence():
    sequence = ContactSequence(0, 100, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    keep = SequenceEdit(EditKind.KEEP, None, None, sequence)
    delete_first = SequenceEdit(EditKind.DELETE, None, 120, ContactSequence(0, 100, 200, sequence.events[1:]))
    later = [ContactEvent(125, .6, None), ContactEvent(140, .5, None), ContactEvent(250, .9, None)]
    options = option_pool({0: (keep, delete_first)}, {0: later}, fps=30.0)
    summary = [(str(option.edit.kind), None if option.added_contact is None else option.added_contact.frame)
               for option in options]
    assert summary == [("keep", None), ("keep", 140), ("delete", None), ("delete", 125), ("delete", 140)]
    assert _frames(options[1].sequence) == [120, 140, 160]


def test_applied_choices_keep_out_of_span_contacts():
    events = [ContactEvent(frame, .8, None) for frame in (20, 60, 120, 170, 200, 300)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    delete_second = ContactSequence(0, 0, 100, (events[0],))
    absorb_gap = ContactSequence(1, 120, 250, (events[2], events[3], events[4]))
    sequences, stream = apply_choices(initial, events, {0: delete_second, 1: absorb_gap})
    assert [_frames(sequence) for sequence in sequences] == [[20], [120, 170, 200]]
    assert [event.frame for event in stream] == [20, 120, 170, 200, 300]


def test_applied_choices_keep_an_absorbed_gap_contact_the_edit_dropped():
    events = [ContactEvent(frame, .8, None) for frame in (20, 120, 170, 200)]
    initial = initial_sequences([(0, 100), (150, 250)], events)
    new_serve = ContactEvent(110, .6, None)
    replaced = ContactSequence(1, 110, 250, (new_serve, events[2], events[3]))
    _sequences, stream = apply_choices(initial, events, {0: initial[0], 1: replaced})
    assert [event.frame for event in stream] == [20, 110, 120, 170, 200]
