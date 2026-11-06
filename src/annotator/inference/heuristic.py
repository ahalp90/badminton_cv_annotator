"""Turn accepted heuristic hits into rallies with alternating top/bottom hitters."""

import math

from annotator.contacts.player_attribution import (
    contact_player_evidence,
    fit_rally_sides,
)
from annotator.inference.prepare_contact_inference import AnnotationEvidence
from annotator.types import (
    ContactCandidate,
    ContactEvent,
    ContactSequence,
    FinishedRally,
)


def select_heuristic_contacts(
    evidence: AnnotationEvidence, rejection_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[FinishedRally, ...]:
    """Select heuristic contacts and fit alternating players within each rally.

    Keep contacts that pass the wrist and spacing checks, then remove those
    on excluded footage. If that removes every accepted contact in a rally,
    append a diagnostic row when ``rejection_diagnostics`` is supplied.
    """
    frames_by_rally: dict[int, list[int]] = {}
    for contact in accepted_heuristic_contacts(evidence.heuristic_contacts):
        frames_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    sequences = []
    for rally_id, (start, end) in enumerate(evidence.heuristic_spans):
        frames = frames_by_rally.get(rally_id, [])
        events = []
        for frame in frames:
            if evidence.exclusion_mask[frame]:
                continue
            side, distances = contact_player_evidence(
                frame, evidence.track, evidence.sticky, evidence.bboxes, evidence.scene_courts,
            )
            # Heuristic contacts have no model probability.
            events.append(ContactEvent(frame, math.nan, side, distances))
        sequences.append(ContactSequence(rally_id, start, end, tuple(events)))
        if frames and not events and rejection_diagnostics is not None:
            rejection_diagnostics.append({
                'rule': 'all_contacts_on_believed_mask',
                'rally_id': rally_id,
                'start_frame': start,
                'end_frame': end,
                'first_rejected_frame': frames[0],
                'rejection_grade': '',
            })
    return fit_rally_sides(sequences)


def accepted_heuristic_contacts(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Keep candidates unless a wrist or spacing check rejected them."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
