"""Carry the selected rally sides through to outcome preparation."""

from annotator.contacts.selection import contacts_from_sequences
from annotator.outcomes.point_winner import Half
from annotator.sequence.contacts import ContactEvent, ContactSequence, FinishedRally
from annotator.sequence.rally_model import fit_rally_sides


def test_refined_contacts_keep_sequence_membership_and_player_order():
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, Half.TOP), ContactEvent(60, .8, Half.BOT))),
        ContactSequence(1, 120, 200, (ContactEvent(150, .7, Half.TOP),)),
    )
    finished, _stream = fit_rally_sides(sequences, tuple(event for sequence in sequences for event in sequence.events))
    result = contacts_from_sequences(finished)
    assert [rally.span for rally in result.rallies] == [(0, 100), (120, 200)]
    assert [rally.frames for rally in result.rallies] == [[20, 60], [150]]
    assert [rally.final_half for rally in result.rallies] == [Half.BOT, Half.TOP]
    assert [rally.first_half for rally in result.rallies] == [Half.TOP, Half.TOP]
    assert [(contact.rally_id, contact.contact_frame) for contact in result.filtered_contacts] == [
        (0, 20), (0, 60), (1, 150),
    ]


def test_tied_or_empty_sequences_have_no_resolved_winner_side(monkeypatch):
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, Half.TOP), ContactEvent(60, .8, Half.TOP))),
        ContactSequence(1, 120, 200, ()),
    )
    finished, _stream = fit_rally_sides(sequences, sequences[0].events)
    assert [rally.final_half for rally in finished] == [None, None]

    def unexpected_fit(_guesses):
        raise AssertionError('unexpected second fit')

    monkeypatch.setattr('annotator.outcomes.point_winner.fit_alternation', unexpected_fit)
    result = contacts_from_sequences(finished)
    assert [rally.final_half for rally in result.rallies] == [None, None]
    assert [rally.first_half for rally in result.rallies] == [None, None]
    assert [len(rally.frames) for rally in result.rallies] == [2, 0]
    assert finished[0].sequence.events[-1].side is Half.TOP


def test_empty_sequence_can_carry_an_unresolved_fit():
    sequence = ContactSequence(0, 0, 100, ())
    assert contacts_from_sequences([FinishedRally(sequence, None)]).rallies[0].first_half is None
