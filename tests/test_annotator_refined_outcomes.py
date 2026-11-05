"""Carry the selected rally sides through to outcome preparation."""

from annotator.outcomes.point_winner import CourtHalf
from annotator.sequence.contacts import ContactEvent, ContactSequence, FinishedRally
from annotator.sequence.rally_model import fit_rally_sides


def test_refined_contacts_keep_sequence_membership_and_player_order():
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, CourtHalf.TOP), ContactEvent(60, .8, CourtHalf.BOT))),
        ContactSequence(1, 120, 200, (ContactEvent(150, .7, CourtHalf.TOP),)),
    )
    finished, _stream = fit_rally_sides(sequences, tuple(event for sequence in sequences for event in sequence.events))
    assert [(rally.sequence.start_frame, rally.sequence.end_frame) for rally in finished] == [(0, 100), (120, 200)]
    assert [[event.frame for event in rally.sequence.events] for rally in finished] == [[20, 60], [150]]
    assert [rally.final_half for rally in finished] == [CourtHalf.BOT, CourtHalf.TOP]
    assert [rally.first_half for rally in finished] == [CourtHalf.TOP, CourtHalf.TOP]


def test_tied_or_empty_sequences_have_no_resolved_winner_side(monkeypatch):
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, CourtHalf.TOP), ContactEvent(60, .8, CourtHalf.TOP))),
        ContactSequence(1, 120, 200, ()),
    )
    finished, _stream = fit_rally_sides(sequences, sequences[0].events)
    assert [rally.final_half for rally in finished] == [None, None]

    def unexpected_fit(_guesses):
        raise AssertionError('unexpected second fit')

    monkeypatch.setattr('annotator.outcomes.point_winner.fit_alternation', unexpected_fit)
    assert [rally.final_half for rally in finished] == [None, None]
    assert [rally.first_half for rally in finished] == [None, None]
    assert [len(rally.sequence.events) for rally in finished] == [2, 0]
    assert finished[0].sequence.events[-1].side is CourtHalf.TOP


def test_empty_sequence_can_carry_an_unresolved_fit():
    sequence = ContactSequence(0, 0, 100, ())
    assert FinishedRally(sequence, None).first_half is None
