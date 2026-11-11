"""Split refined rallies at long hit gaps, then fit each piece's player order."""

from dataclasses import replace
from itertools import pairwise

from annotator.config import MAX_RALLY_GAP_AT_30_FPS
from annotator.contacts.player_attribution import fit_rally_sides
from annotator.rallies.refine_rallies import RallyPrediction
from annotator.types import ContactEvent, FinishedRally
from shared.frame_timing import frames_at_fps


def split_rallies(
    prediction: RallyPrediction,
) -> tuple[tuple[FinishedRally, ...], tuple[ContactEvent, ...], tuple[int, ...]]:
    """Split gaps longer than three seconds while preserving the outer bounds.

    Supplied bounds stay fixed. Split pieces use the selected proposal's original
    hitter evidence, since fitting the whole rally has already changed its sides.
    Return final rallies, the full contact stream and each rally's source span ID.
    """
    context = prediction.pool.context
    if context.bounds_supplied:
        return prediction.rallies, prediction.events, tuple(sequence.span_id for sequence in prediction.sequences)

    max_gap = frames_at_fps(MAX_RALLY_GAP_AT_30_FPS, context.fps)
    rallies = []
    source_span_ids = []
    for rally in prediction.rallies:
        sequence = rally.sequence
        split_positions = [
            index for index in range(1, len(sequence.events))
            if sequence.events[index].frame - sequence.events[index - 1].frame > max_gap
        ]
        if not split_positions:
            rallies.append(replace(rally, sequence=replace(sequence, span_id=len(rallies))))
            source_span_ids.append(sequence.span_id)
            continue

        proposal = prediction.pool.sequences[prediction.chosen[sequence.span_id]]
        positions = [0, *split_positions, len(proposal.events)]
        pieces = []
        for first, end in pairwise(positions):
            events = proposal.events[first:end]
            pieces.append(replace(
                sequence, span_id=len(rallies) + len(pieces), events=events,
                start_frame=sequence.start_frame if first == 0 else events[0].frame,
                end_frame=sequence.end_frame if end == len(proposal.events) else events[-1].frame + 1,
            ))
            source_span_ids.append(sequence.span_id)
        rallies.extend(fit_rally_sides(pieces))

    fitted = {event.frame: event for rally in rallies for event in rally.sequence.events}
    events = tuple(fitted.get(event.frame, event) for event in prediction.events)
    return tuple(rallies), events, tuple(source_span_ids)
