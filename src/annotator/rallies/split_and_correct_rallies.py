"""Reconsider each split piece within its own part of the parent rally."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from annotator.contacts.score_candidates import (
    NEARBY_CONTACT_DISTANCE_AT_30_FPS,
    ScoredContacts,
)
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rallies.build_contact_proposals import (
    RallyContext,
    RallyOptions,
    build_rally_options,
    build_sequence_context,
)
from annotator.rallies.refine_rallies import (
    RallyPrediction,
    RallyReviewFeatures,
    RallySelector,
    build_review_features,
    predict_rally_options,
)
from annotator.rallies.split_rallies import split_rallies
from annotator.types import ContactEvent, FinishedRally
from shared.frame_timing import frames_at_fps


@dataclass(frozen=True)
class CorrectedRallies:
    """Final rallies and the measured inputs reused by confidence and training."""

    rallies: tuple[FinishedRally, ...]
    events: tuple[ContactEvent, ...]
    source_span_ids: tuple[int, ...]
    review_features: RallyReviewFeatures
    piece_options: RallyOptions | None


def build_split_context(
    inputs: PreparedAnnotationInputs, contacts: ScoredContacts, prediction: RallyPrediction,
) -> tuple[RallyContext | None, tuple[FinishedRally, ...], tuple[ContactEvent, ...], tuple[int, ...]]:
    """Split finished parents and rebuild candidate shortlists for their pieces.

    Return the split-piece context, all rallies, the full contact stream and
    parent IDs. Ordinary rallies have no second-pass context. A small exclusion
    margin around each midpoint keeps hits in neighbouring pieces more than the
    nearby-contact distance apart. Contact features continue to use surrounding
    frames from the full video.
    """
    rallies, events, source_span_ids = split_rallies(prediction)
    counts = Counter(source_span_ids)
    if all(count == 1 for count in counts.values()):
        return None, rallies, events, source_span_ids

    distance = frames_at_fps(NEARBY_CONTACT_DISTANCE_AT_30_FPS, inputs.fps)
    margin = (distance + 1) // 2
    initial = []
    piece_bounds = {}
    raw_events = {}
    originals = {}
    for source, count in counts.items():
        if count > 1:
            selected = prediction.pool.sequences[prediction.chosen[source]]
            originals[source] = {event.frame: event for event in selected.events}
    for position, (rally, source) in enumerate(zip(rallies, source_span_ids, strict=True)):
        if counts[source] == 1:
            continue
        piece = rally.sequence
        piece = replace(piece, events=tuple(originals[source][event.frame] for event in piece.events))
        raw_events.update((event.frame, event) for event in piece.events)
        earliest, latest = piece.start_frame, piece.end_frame
        if position and source_span_ids[position - 1] == source:
            left_hit = rallies[position - 1].sequence.events[-1].frame
            earliest = (left_hit + piece.events[0].frame + 1) // 2 + margin
        if position + 1 < len(rallies) and source_span_ids[position + 1] == source:
            right_hit = rallies[position + 1].sequence.events[0].frame
            latest = (piece.events[-1].frame + right_hit + 1) // 2 - margin
        initial.append(piece)
        piece_bounds[piece.span_id] = earliest, latest
    context_events = tuple(raw_events.get(event.frame, event) for event in events)
    context = build_sequence_context(inputs, contacts, tuple(initial), context_events, piece_bounds=piece_bounds)
    return context, rallies, events, source_span_ids


def split_and_correct_rallies(
    inputs: PreparedAnnotationInputs, contacts: ScoredContacts, prediction: RallyPrediction, selector: RallySelector,
) -> CorrectedRallies:
    """Correct split pieces once and measure review features for the final rallies.

    Ordinary rallies keep their first-pass choices and feature rows. Split pieces
    use their own alternatives, original hitter evidence and final bounds.
    """
    context, rallies, events, source_span_ids = build_split_context(inputs, contacts, prediction)
    original_features = build_review_features(prediction)
    original_rows = dict(zip(original_features.span_ids, original_features.matrix, strict=True))
    final_rows = [original_rows[source] for source in source_span_ids]
    options = None
    if context is not None:
        options = build_rally_options(context, selector.max_edits, selector.deletion_candidates)
        corrected = predict_rally_options(selector, options)
        corrected_rallies = {rally.sequence.span_id: rally for rally in corrected.rallies}
        corrected_features = build_review_features(corrected)
        corrected_rows = dict(zip(corrected_features.span_ids, corrected_features.matrix, strict=True))
        final = []
        for position, rally in enumerate(rallies):
            span_id = rally.sequence.span_id
            final.append(corrected_rallies.get(span_id, rally))
            if span_id in corrected_rows:
                final_rows[position] = corrected_rows[span_id]
        rallies = tuple(final)
        events = corrected.events
    span_ids = tuple(rally.sequence.span_id for rally in rallies)
    matrix = np.asarray(final_rows, dtype=np.float64).reshape(len(span_ids), original_features.matrix.shape[1])
    return CorrectedRallies(rallies, events, source_span_ids, RallyReviewFeatures(span_ids, matrix), options)
