"""Late gap splits preserve selected hits, original bounds and raw hitter evidence."""

from dataclasses import replace

import numpy as np
import pytest

from annotator.contacts.build_features import CONTACT_FEATURE_NAMES
from annotator.inference.learned import predict_from_rally_options
from annotator.rallies.build_contact_proposals import RallyContext, build_rally_options
from annotator.rallies.refine_rallies import predict_rally_options, score_rally_review
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


def test_final_inference_inherits_presplit_review_and_handles_empty_video(annotator_models):
    pool = options()
    refined = predict_rally_options(annotator_models.rally_selector, pool)
    review = score_rally_review(refined, annotator_models.review_model)
    prediction = predict_from_rally_options(None, None, pool, annotator_models)
    np.testing.assert_array_equal(prediction.review.features.matrix, review.features.matrix)
    assert prediction.rally_confidence == (review.scores[0], review.scores[0])
    assert len(prediction.sequences) == 2
    empty_pool = build_rally_options(replace(pool.context, initial=(), events=()), max_edits=0)
    empty = predict_from_rally_options(None, None, empty_pool, annotator_models)
    assert empty.rallies == empty.events == empty.source_span_ids == empty.rally_confidence == ()
