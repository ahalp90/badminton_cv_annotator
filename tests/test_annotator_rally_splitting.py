"""Late gap splits preserve selected hits, original bounds and raw hitter evidence."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from annotator.contacts.build_features import (
    CONTACT_FEATURE_NAMES,
    contact_feature_dtype,
)
from annotator.contacts.score_candidates import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
)
from annotator.inference.learned import predict_from_rally_options
from annotator.rallies.build_contact_proposals import (
    FEATURE_NAMES,
    RallyContext,
    build_rally_options,
)
from annotator.rallies.refine_rallies import (
    REVIEW_FEATURE_NAMES,
    RallySelector,
    predict_rally_options,
    score_rally_review,
)
from annotator.rallies.split_and_correct_rallies import (
    build_split_context,
    split_and_correct_rallies,
)
from annotator.rallies.split_rallies import split_rallies
from annotator.types import ContactEvent, ContactSequence, CourtHalf
from shared.frame_timing import frames_at_fps


def options(fps=30.0, *, supplied=False):
    frames = [frames_at_fps(frame, fps) for frame in (120, 150, 400, 470)]
    # The second pair starts on Bot; a whole-rally fit overwrites its side guesses.
    sides = (CourtHalf.TOP, CourtHalf.BOT, CourtHalf.BOT, CourtHalf.TOP)
    events = tuple(ContactEvent(frame, .95, side) for frame, side in zip(frames, sides, strict=True))
    sequence = ContactSequence(7, frames_at_fps(100, fps), frames_at_fps(500, fps), events)
    outside = ContactEvent(frames_at_fps(550, fps), .95, CourtHalf.TOP)
    context = RallyContext(
        fps, (sequence,), (*events, outside), {},
        {event.frame: np.zeros(len(CONTACT_FEATURE_NAMES)) for event in (*events, outside)},
        frame_count=frames_at_fps(600, fps), bounds_supplied=supplied,
    )
    return build_rally_options(context, max_edits=0)


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_split_keeps_outer_bounds_contacts_and_refits_raw_sides(fps, annotator_models):
    pool = options(fps)
    refined = predict_rally_options(annotator_models.rally_selector, pool)
    rallies, events, sources = split_rallies(refined)
    first, second = [rally.sequence for rally in rallies]
    assert sources == (7, 7)
    assert (first.span_id, second.span_id) == (0, 1)
    assert first.start_frame == refined.sequences[0].start_frame
    assert second.end_frame == refined.sequences[0].end_frame
    assert first.end_frame == first.events[-1].frame + 1
    assert second.start_frame == second.events[0].frame
    assert [event.side for event in first.events] == [CourtHalf.TOP, CourtHalf.BOT]
    assert [event.side for event in second.events] == [CourtHalf.BOT, CourtHalf.TOP]
    assert [(event.frame, event.probability) for event in events] == [
        (event.frame, event.probability) for event in refined.events
    ]
    assert events[-1] is refined.events[-1]


def test_supplied_bounds_bypass_splitting(annotator_models):
    refined = predict_rally_options(annotator_models.rally_selector, options(supplied=True))
    rallies, events, sources = split_rallies(refined)
    assert rallies is refined.rallies
    assert events is refined.events
    assert sources == (7,)


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
@pytest.mark.parametrize('extra_frame', [0, 1])
def test_split_threshold_is_strict_and_scaled(fps, extra_frame, annotator_models):
    pool = options(fps)
    sequence = pool.context.initial[0]
    start = sequence.events[0].frame
    events = (sequence.events[0], replace(sequence.events[1], frame=start + frames_at_fps(90, fps) + extra_frame))
    sequence = replace(sequence, events=events)
    context = replace(pool.context, initial=(sequence,), events=events,
                      contact_features_by_frame={event.frame: np.zeros(len(CONTACT_FEATURE_NAMES)) for event in events})
    refined = predict_rally_options(annotator_models.rally_selector, build_rally_options(context, max_edits=0))
    rallies, _, _ = split_rallies(refined)
    assert len(rallies) == 1 + extra_frame
    if not extra_frame:
        assert rallies[0].sequence.start_frame == refined.sequences[0].start_frame
        assert rallies[0].sequence.end_frame == refined.sequences[0].end_frame


def evidence_and_contacts(pool, additions=()):
    frames = sorted({event.frame for event in pool.context.events} | set(additions))
    rows = np.zeros(len(frames), dtype=contact_feature_dtype())
    rows['frame'] = frames
    contacts = ScoredContacts(rows, np.zeros((len(rows), len(CONTACT_FEATURE_NAMES)), dtype=np.float32),
                              np.full(len(rows), .9), np.arange(len(rows)))
    inputs = SimpleNamespace(fps=pool.context.fps, track=np.zeros((pool.context.frame_count, 3)),
                             players=None, wrist_gaps=None, bboxes=None, scene_courts=(),
                             bounds_supplied=pool.context.bounds_supplied)
    return inputs, contacts


def test_final_inference_scores_each_piece_and_handles_empty_video(annotator_models):
    pool = options()
    inputs, contacts = evidence_and_contacts(pool)
    prediction = predict_from_rally_options(inputs, None, contacts, pool, annotator_models)
    count_column = REVIEW_FEATURE_NAMES.index('final__predicted_event_count')
    assert prediction.review.features.span_ids == (0, 1)
    np.testing.assert_array_equal(prediction.review.features.matrix[:, count_column], [2, 2])
    assert prediction.rally_confidence == tuple(prediction.review.scores)
    assert [event.side for event in prediction.sequences[1].events] == [CourtHalf.BOT, CourtHalf.TOP]
    assert prediction.events[-1] is pool.context.events[-1]
    empty_pool = build_rally_options(replace(pool.context, initial=(), events=()), max_edits=0)
    empty = predict_from_rally_options(inputs, None, contacts, empty_pool, annotator_models)
    assert empty.rallies == empty.events == empty.source_span_ids == empty.rally_confidence == ()


@pytest.mark.parametrize('supplied', [False, True])
def test_second_pass_leaves_ordinary_and_supplied_predictions_unchanged(supplied, annotator_models):
    pool = options(supplied=supplied)
    if not supplied:
        before = pool.context.initial[0]
        before = replace(before, events=before.events[:2])
        pool = build_rally_options(replace(pool.context, initial=(before,), events=before.events), max_edits=0)
    inputs, contacts = evidence_and_contacts(pool)
    first = predict_rally_options(annotator_models.rally_selector, pool)
    review = score_rally_review(first, annotator_models.review_model)
    prediction = predict_from_rally_options(inputs, None, contacts, pool, annotator_models)
    assert prediction.events == first.events
    assert replace(prediction.rallies[0].sequence, span_id=7) == first.sequences[0]
    np.testing.assert_array_equal(prediction.review.features.matrix, review.features.matrix)
    np.testing.assert_array_equal(prediction.review.scores, review.scores)


def test_second_pass_recovers_edge_hits_without_crossing_split_midpoint(annotator_models):
    pool = options()
    inputs, contacts = evidence_and_contacts(pool, additions=(90, 180, 260, 290, 370, 510))
    first = predict_rally_options(annotator_models.rally_selector, pool)
    context, _, _, _ = build_split_context(inputs, contacts, first)
    assert context.piece_bounds == {0: (100, 272), 1: (278, 500)}
    assert [event.frame for event in context.shortlisted_by_span[0]] == [180, 260]
    assert [event.frame for event in context.shortlisted_by_span[1]] == [290, 370]
    proposals = build_rally_options(context)
    for sequence in proposals.sequences:
        earliest, latest = context.piece_bounds[sequence.span_id]
        assert all(earliest <= event.frame < latest for event in sequence.events)
    for piece in context.initial:
        assert piece in proposals.sequences

    def prefer_more_contacts(matrix):
        scores = matrix[:, FEATURE_NAMES.index('after__predicted_event_count')] / 10
        return np.column_stack((1 - scores, scores))

    selector = RallySelector(SimpleNamespace(predict_proba=prefer_more_contacts))
    corrected = split_and_correct_rallies(inputs, contacts, first, selector)
    rallies, events = corrected.rallies, corrected.events
    sources, review = corrected.source_span_ids, corrected.review_features
    assert [[event.frame for event in rally.sequence.events] for rally in rallies] == [
        [120, 150, 180, 260], [290, 370, 400, 470],
    ]
    assert rallies[0].sequence.end_frame <= 275 <= rallies[1].sequence.start_frame
    assert rallies[0].sequence.start_frame == 100 and rallies[1].sequence.end_frame == 500
    assert sources == (7, 7) and review.span_ids == (0, 1)
    assert events[-1] is pool.context.events[-1]


def test_correction_preserves_ordinary_rally_beside_split_parent(annotator_models):
    pool = options()
    ordinary_events = (ContactEvent(600, .95, CourtHalf.BOT), ContactEvent(630, .95, CourtHalf.TOP))
    ordinary = ContactSequence(9, 580, 650, ordinary_events)
    all_events = (*pool.context.events, *ordinary_events)
    context = replace(
        pool.context, initial=(*pool.context.initial, ordinary), events=all_events, frame_count=700,
        contact_features_by_frame={event.frame: np.zeros(len(CONTACT_FEATURE_NAMES)) for event in all_events},
    )
    pool = build_rally_options(context, max_edits=0)
    inputs, contacts = evidence_and_contacts(pool)
    first = predict_rally_options(annotator_models.rally_selector, pool)
    before_review = score_rally_review(first, annotator_models.review_model)
    prediction = predict_from_rally_options(inputs, None, contacts, pool, annotator_models)
    assert replace(prediction.rallies[-1].sequence, span_id=9) == first.sequences[-1]
    assert prediction.events[-2:] == first.events[-2:]
    np.testing.assert_array_equal(prediction.review.features.matrix[-1], before_review.features.matrix[-1])
    assert prediction.rally_confidence[-1] == before_review.scores[-1]


@pytest.mark.parametrize('fps', [25.0, 30.0, 60.0])
def test_midpoint_margin_prevents_duplicate_hits_across_pieces(fps, annotator_models):
    pool = options(fps)
    parent_hits = pool.context.initial[0].events
    midpoint = (parent_hits[1].frame + parent_hits[2].frame + 1) // 2
    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    margin = (distance + 1) // 2
    left_edge, right_edge = midpoint - margin - 1, midpoint + margin
    inputs, contacts = evidence_and_contacts(pool, (left_edge, midpoint - 1, midpoint + 1, right_edge))
    first = predict_rally_options(annotator_models.rally_selector, pool)

    def prefer_more_contacts(matrix):
        scores = matrix[:, FEATURE_NAMES.index('after__predicted_event_count')] / 10
        return np.column_stack((1 - scores, scores))

    selector = RallySelector(SimpleNamespace(predict_proba=prefer_more_contacts))
    rallies = split_and_correct_rallies(inputs, contacts, first, selector).rallies
    left_last = rallies[0].sequence.events[-1].frame
    right_first = rallies[1].sequence.events[0].frame
    assert (left_last, right_first) == (left_edge, right_edge)
    assert right_first - left_last > distance
