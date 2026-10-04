"""Chooser tie-breaks, the keep fallback and the previous-choice margin guard."""

import numpy as np

from annotator.sequence.candidates import ServeCandidate
from annotator.sequence.choices import (
    choose_best_options,
    choose_edits,
    choose_with_margin,
)
from annotator.sequence.contacts import ContactEvent, ContactSequence
from annotator.sequence.edits import (
    EditKind,
    SequenceEdit,
    SequenceOption,
    option_pool,
    sequence_edits,
)

SEQUENCE = ContactSequence(0, 100, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
LATER_CONTACT = ContactEvent(140, .6, None)


def _edit_options() -> tuple[SequenceOption, ...]:
    """Keep, add/replace 105 and 110, delete 120 or 160, then combinations."""
    candidates = []
    for frame in (105, 110):
        candidates.append(ServeCandidate(0, ContactEvent(frame, .5, None), 120, ()))
    edits = sequence_edits(SEQUENCE, candidates, SEQUENCE.events, previous_end=-1)
    return option_pool({0: edits}, {}, fps=30.0)


def _chosen_edit(options, chosen) -> tuple[str, int | None, int | None]:
    edit = options[chosen[0]].edit
    return str(edit.kind), edit.serve_frame, edit.deleted_frame


def _with_margin_options() -> tuple[SequenceOption, ...]:
    """Keep, keep plus a later contact, delete 120, delete 120 plus a later contact."""
    keep = SequenceEdit(EditKind.KEEP, None, None, SEQUENCE)
    delete_first = SequenceEdit(EditKind.DELETE, None, 120, ContactSequence(0, 100, 200, SEQUENCE.events[1:]))
    return option_pool({0: (keep, delete_first)}, {0: (LATER_CONTACT,)}, fps=30.0)


def test_first_chooser_keeps_the_initial_sequence_unless_an_edit_scores_higher():
    options = _edit_options()
    rows = list(range(len(options)))
    scores = np.full(len(options), .4)
    scores[0] = .5
    assert _chosen_edit(options, choose_edits(options, scores, rows, 0.0)) == ("keep", None, None)
    scores[5] = .5  # equal to keep is not enough
    assert _chosen_edit(options, choose_edits(options, scores, rows, 0.0)) == ("keep", None, None)


def test_first_chooser_breaks_equal_scores_by_kind_then_frame():
    options = _edit_options()
    rows = list(range(len(options)))
    scores = np.full(len(options), .1)
    scores[[2, 5]] = .7  # replace 105 and delete 120
    assert _chosen_edit(options, choose_edits(options, scores, rows, 0.0)) == ("replace", 105, None)
    scores[:] = .1
    scores[[3, 1]] = .7  # add 110 and add 105
    assert _chosen_edit(options, choose_edits(options, scores, rows, 0.0)) == ("add", 105, None)


def test_first_chooser_ignores_edits_below_the_minimum_or_without_a_score():
    options = _edit_options()
    rows = list(range(len(options)))
    scores = np.full(len(options), .1)
    scores[0] = .3
    scores[1] = .4
    scores[2] = np.nan
    assert _chosen_edit(options, choose_edits(options, scores, rows, .5)) == ("keep", None, None)
    assert _chosen_edit(options, choose_edits(options, scores, rows, 0.0)) == ("add", 105, None)


def test_best_option_prefers_no_added_contact_on_equal_scores():
    options = _with_margin_options()
    # Delete without a later contact beats keep with one, although keep is simpler.
    assert choose_best_options(options, np.array([.2, .6, .6, .2])) == {0: 2}
    reordered = (options[1], options[0], options[2], options[3])
    assert choose_best_options(reordered, np.array([.6, .6, .2, .2])) == {0: 1}


def test_later_chooser_needs_a_clear_margin_over_the_previous_choice():
    options = _with_margin_options()
    assert choose_with_margin(options, np.array([.5, .54, .1, .1]), {0: 0}) == {0: 0}
    assert choose_with_margin(options, np.array([.5, .56, .1, .1]), {0: 0}) == {0: 1}
    assert choose_with_margin(options, np.array([.5, .56, .1, .1]), {0: 2}) == {0: 1}
    assert choose_with_margin(options, np.array([.5, .5, .1, .1]), {0: 2}) == {0: 0}


def test_previous_choice_scores_as_its_best_equal_sequence():
    base = _with_margin_options()
    delete_first = base[2]
    same_sequence = SequenceEdit(EditKind.REPLACE_DELETE, 90, 120, delete_first.sequence)
    options = (*base, SequenceOption(same_sequence, None, delete_first.sequence))
    scores = np.array([.1, .58, .3, .1, .55])
    # Scored alone, the previous choice trails by 0.28. Its equal sequence
    # trails by only 0.03.
    assert choose_with_margin(options, scores, {0: 2}) == {0: 2}
