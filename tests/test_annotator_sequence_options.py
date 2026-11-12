"""Contact shortlists, seed proposals and selected full-stream membership."""

import numpy as np
import pytest

from annotator.contacts.build_features import (
    CONTACT_FEATURE_NAMES,
    contact_feature_dtype,
)
from annotator.contacts.score_candidates import ScoredContacts, contact_feature_matrix
from annotator.rallies.build_contact_proposals import (
    FEATURE_NAMES,
    RallyContext,
    build_rally_options,
    initial_sequences,
    propose_contact_additions_and_removals,
    shortlist_later_rows,
    shortlist_serve_rows,
)
from annotator.types import ContactEvent, ContactSequence


def _scores(rows: list[tuple[int, int, float, bool]]) -> ScoredContacts:
    """Build candidate scores from (frame, interval ID, probability, kept) rows."""
    frames, interval_ids, probabilities, kept = zip(*rows, strict=True)
    candidates = np.zeros(len(rows), dtype=contact_feature_dtype())
    candidates['frame'] = frames
    candidates['interval_id'] = interval_ids
    return ScoredContacts(candidates, contact_feature_matrix(candidates), np.asarray(probabilities), np.flatnonzero(kept))


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
    first = shortlist_serve_rows(sequences[0], scores, fps=30.0, previous_end=-1)
    second = shortlist_serve_rows(sequences[1], scores, fps=30.0, previous_end=120)
    assert scores.candidates['frame'][first].tolist() == [30, 5]
    assert scores.candidates['frame'][second].tolist() == [130, 170]
    assert shortlist_serve_rows(sequences[2], scores, fps=30.0, previous_end=250) == []
    # Without a preceding rally, the search interval still excludes the early clip.
    unclipped = shortlist_serve_rows(sequences[1], scores, fps=30.0, previous_end=0)
    assert scores.candidates['frame'][unclipped].tolist() == [110, 130]


@pytest.mark.parametrize(("rows", "expected"), [
    ([(40, 0, .9, True)], []),
    ([(10, 0, .5, False), (14, 0, .4, False), (40, 0, .9, True)], [0]),
])
def test_serve_shortlist_uses_available_candidates(rows, expected):
    scores = _scores(rows)
    sequences = initial_sequences([(0, 100)], [ContactEvent(40, .9, None)])
    assert shortlist_serve_rows(sequences[0], scores, fps=30.0, previous_end=-1) == expected


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
        (150, 0, .01, False),
        (160, 0, .4, False),
        (170, 0, .5, False),
        (180, 0, .05, False),
        (200, 0, .8, False),
    ])
    sequence = ContactSequence(0, 0, 200, (ContactEvent(20, .9, None), ContactEvent(100, .7, None)))
    rows = shortlist_later_rows(sequence, scores, fps=30.0)
    assert scores.candidates['frame'][rows].tolist() == [50, 170, 160, 140, 130, 120]
    assert shortlist_later_rows(ContactSequence(1, 0, 200, ()), scores, fps=30.0) == []


def test_contact_additions_recheck_spacing_after_deletion_and_keep_bounds() -> None:
    before = ContactSequence(0, 100, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    candidates = tuple(ContactEvent(frame, .7, None) for frame in (125, 140, 250))
    options = propose_contact_additions_and_removals(
        before, candidates, before.events, -1, 6, 2, 2, latest_end=before.end_frame,
    )
    frames = [_frames(sequence) for sequence in options]
    assert [125, 160] in frames
    assert not any(120 in selected and 125 in selected for selected in frames)
    assert not any(250 in selected for selected in frames)


def test_earlier_serve_includes_selected_contacts_and_respects_previous_rally() -> None:
    before = ContactSequence(1, 110, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    gap = ContactEvent(100, .7, None)
    candidates = tuple(ContactEvent(frame, .6, None) for frame in (80, 90))
    options = propose_contact_additions_and_removals(
        before, candidates, (gap, *before.events), 85, 6, 1, 0, latest_end=before.end_frame,
    )
    assert not any(80 in _frames(sequence) for sequence in options)
    expanded = next(sequence for sequence in options if _frames(sequence) == [90, 100, 120, 160])
    assert expanded.start_frame == 90 and expanded.events[1] is gap


def test_propose_contact_additions_and_removals_deduplicate_absorbed_contacts_and_prefer_fewer_edits() -> None:
    before = ContactSequence(1, 110, 200, (ContactEvent(120, .9, None), ContactEvent(160, .8, None)))
    earlier = ContactEvent(80, .6, None)
    gap = ContactEvent(90, .7, None)
    # Adding 80 absorbs 90, so explicitly adding both gives the same sequence.
    options = propose_contact_additions_and_removals(
        before, (earlier, gap), (gap, *before.events), 0, 6, 2, 0, latest_end=before.end_frame,
    )
    assert options[0] is before
    assert [_frames(sequence) for sequence in options] == [
        [120, 160], [90, 120, 160], [80, 90, 120, 160],
    ]


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_contact_gap_features_exclude_rally_boundary_gaps(fps: float) -> None:
    events = (ContactEvent(20, .9, None), ContactEvent(60, .7, None))
    sequence = ContactSequence(0, 10, 200, events)
    contact_features = {event.frame: np.zeros(len(CONTACT_FEATURE_NAMES)) for event in events}
    context = RallyContext(fps, (sequence,), events, {}, contact_features, frame_count=200)

    pool = build_rally_options(context, max_edits=0)
    features = dict(zip(FEATURE_NAMES, pool.matrix[0], strict=True))

    assert features['before__longest_gap_seconds'] == pytest.approx(40 / fps)
    assert features['after__longest_gap_seconds'] == pytest.approx(40 / fps)
    assert features['longest_gap_seconds'] == pytest.approx(140 / fps)
    assert features['positive_gap_count'] == 3
    assert pool.matrix.dtype == np.float32
