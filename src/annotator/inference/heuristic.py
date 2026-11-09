"""Turn accepted heuristic hits into rallies with alternating top/bottom hitters."""

import math

from annotator.contacts.player_attribution import (
    contact_player_evidence,
    fit_rally_sides,
)
from annotator.inference.prepare_annotation_inputs import PreparedAnnotationInputs
from annotator.types import (
    ContactCandidate,
    ContactEvent,
    ContactSequence,
    FinishedRally,
)


def select_heuristic_contacts(
    inputs: PreparedAnnotationInputs,
) -> tuple[FinishedRally, ...]:
    """Select heuristic contacts and fit alternating players within each rally.

    Keep contacts that pass the wrist and spacing checks, then remove those
    on excluded footage.
    """
    frames_by_rally: dict[int, list[int]] = {}
    for contact in accepted_heuristic_contacts(inputs.heuristic_contacts):
        frames_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    sequences = []
    for rally_id, (start, end) in enumerate(inputs.heuristic_spans):
        frames = frames_by_rally.get(rally_id, [])
        events = []
        for frame in frames:
            if inputs.exclusion_mask[frame]:
                continue
            side, distances = contact_player_evidence(
                frame, inputs.track, inputs.players, inputs.wrist_gaps, inputs.bboxes, inputs.scene_courts,
            )
            # Heuristic contacts have no model probability.
            events.append(ContactEvent(frame, math.nan, side, distances))
        sequences.append(ContactSequence(rally_id, start, end, tuple(events)))
    return fit_rally_sides(sequences)


def accepted_heuristic_contacts(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Keep candidates unless a wrist or spacing check rejected them."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.temporal_proximity_suppressed is not True
    ]
