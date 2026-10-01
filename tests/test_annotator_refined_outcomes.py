"""Downstream outcomes use the selected sequence's contacts and player sides."""

from annotator.outcomes.point_winner import Half
from annotator.outcomes.video import build_refined_contact_data
from annotator.sequence import ContactEvent, ContactSequence


def test_refined_contacts_keep_sequence_membership_and_player_order():
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, Half.TOP), ContactEvent(60, .8, Half.BOT))),
        ContactSequence(1, 120, 200, (ContactEvent(150, .7, Half.TOP),)),
    )
    result = build_refined_contact_data(sequences)
    assert result.filtered_by_rally == {0: [20, 60], 1: [150]}
    assert result.striker_halves == [Half.BOT, Half.TOP]
    assert result.fitted_first_all == [Half.TOP, Half.TOP]
    assert result.next_servers == [Half.TOP, None]
    assert [(contact.rally_id, contact.contact_frame) for contact in result.filtered_contacts] == [
        (0, 20), (0, 60), (1, 150),
    ]


def test_tied_or_empty_sequences_have_no_resolved_winner_side():
    sequences = (
        ContactSequence(0, 0, 100, (ContactEvent(20, .9, Half.TOP), ContactEvent(60, .8, Half.TOP))),
        ContactSequence(1, 120, 200, ()),
    )
    result = build_refined_contact_data(sequences)
    assert result.striker_halves == [None, None]
    assert result.fitted_first_all == [None, None]
    assert result.n_strokes_list == [2, 0]
