"""Contact shortlists, seed proposals and selected full-stream membership."""

import numpy as np
import pytest

from annotator.contacts.features import contact_feature_dtype
from annotator.contacts.model import ScoredContacts
from annotator.sequence.candidates import (
    shortlist_later_rows,
    shortlist_serve_rows,
)
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    apply_choices,
    initial_sequences,
)
from annotator.sequence.rally_options import toggle_proposals


def _scores(rows: list[tuple[int, int, float, bool]]) -> ScoredContacts:
    """Build candidate scores from (frame, interval ID, probability, kept) rows."""
    frames, interval_ids, probabilities, kept = zip(*rows, strict=True)
    candidates = np.zeros(len(rows), dtype=contact_feature_dtype())
    candidates['frame'] = frames
    candidates['interval_id'] = interval_ids
    return ScoredContacts(candidates, np.asarray(probabilities), np.flatnonzero(kept))


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
    shortlists = shortlist_serve_rows(sequences, scores, [(0, 100), (100, 300)], fps=30.0)
    frames_by_span = {span_id: scores.candidates['frame'][rows].tolist() for span_id, rows in shortlists.items()}
    assert frames_by_span == {0: [30, 5], 1: [130, 170]}


@pytest.mark.parametrize(("rows", "expected"), [
    ([(40, 0, .9, True)], []),
    ([(10, 0, .5, False), (14, 0, .4, False), (40, 0, .9, True)], [0]),
])
def test_serve_shortlist_uses_available_candidates(rows, expected):
    scores = _scores(rows)
    sequences = initial_sequences([(0, 100)], [ContactEvent(40, .9, None)])
    assert shortlist_serve_rows(sequences, scores, [(0, 100)], fps=30.0) == {0: expected}


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
    rows = shortlist_later_rows(sequence, scores, fps=30.0)
    assert scores.candidates['frame'][rows].tolist() == [50, 170, 160, 140, 130, 120]
    assert shortlist_later_rows(ContactSequence(1, 0, 200, ()), scores, fps=30.0) == []


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


def test_toggle_additions_recheck_spacing_after_deletion_and_keep_bounds() -> None:
    before = ContactSequence(0, 100, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    candidates = tuple(ContactEvent(frame, .7, None) for frame in (125, 140, 250))
    options = toggle_proposals(before, candidates, before.events, -1, 6, 2, 2)
    frames = [_frames(sequence) for sequence in options]
    assert [125, 160] in frames
    assert not any(120 in selected and 125 in selected for selected in frames)
    assert not any(250 in selected for selected in frames)


def test_toggle_earlier_start_absorbs_stream_and_respects_previous_rally() -> None:
    before = ContactSequence(1, 110, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    gap = ContactEvent(100, .7, None)
    candidates = tuple(ContactEvent(frame, .6, None) for frame in (80, 90))
    options = toggle_proposals(before, candidates, (gap, *before.events), 85, 6, 1, 0)
    assert not any(80 in _frames(sequence) for sequence in options)
    expanded = next(sequence for sequence in options if _frames(sequence) == [90, 100, 120, 160])
    assert expanded.start_frame == 90 and expanded.events[1] is gap
