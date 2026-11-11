"""Reconsider each split piece within its own part of the parent rally."""

from collections import Counter
from dataclasses import replace

import numpy as np

from annotator.contacts.score_candidates import ScoredContacts
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.rallies.build_contact_proposals import (
    RallyContext,
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


def build_split_context(
    inputs: PreparedAnnotationInputs, contacts: ScoredContacts, prediction: RallyPrediction,
) -> tuple[RallyContext | None, tuple[FinishedRally, ...], tuple[ContactEvent, ...], tuple[int, ...]]:
    """Split finished parents and rebuild candidate shortlists for their pieces.

    Return the split-piece context, all rallies, the full contact stream and
    parent IDs. Ordinary rallies have no second-pass context. Midpoints between
    sibling hits bound candidate membership; temporal features remain unchanged.
    """
    rallies, events, source_span_ids = split_rallies(prediction)
    counts = Counter(source_span_ids)
    if all(count == 1 for count in counts.values()):
        return None, rallies, events, source_span_ids

    initial = []
    limits = {}
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
            earliest = (left_hit + piece.events[0].frame + 1) // 2
        if position + 1 < len(rallies) and source_span_ids[position + 1] == source:
            right_hit = rallies[position + 1].sequence.events[0].frame
            latest = (piece.events[-1].frame + right_hit + 1) // 2
        initial.append(piece)
        limits[piece.span_id] = earliest, latest
    context_events = tuple(raw_events.get(event.frame, event) for event in events)
    context = build_sequence_context(inputs, contacts, tuple(initial), context_events, contact_limits=limits)
    return context, rallies, events, source_span_ids


def correct_split_rallies(
    inputs: PreparedAnnotationInputs, contacts: ScoredContacts, prediction: RallyPrediction, selector: RallySelector,
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...], tuple[int, ...], RallyReviewFeatures]:
    """Correct split pieces once and measure review features for the final rallies.

    Ordinary rallies keep their first-pass choices and feature rows. Split pieces
    use their own alternatives, original hitter evidence and final bounds.
    """
    context, rallies, events, source_span_ids = build_split_context(inputs, contacts, prediction)
    original_features = build_review_features(prediction)
    original_rows = dict(zip(original_features.span_ids, original_features.matrix, strict=True))
    final_rows = [original_rows[source] for source in source_span_ids]
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
    return rallies, events, source_span_ids, RallyReviewFeatures(span_ids, matrix)
