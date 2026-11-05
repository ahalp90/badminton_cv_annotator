"""Filter heuristic contacts and fit the player order used by outcomes."""

import math

from annotator.contacts.attribution import attribute_half
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import ContactEvent, ContactSequence, FinishedRally
from annotator.sequence.rally_model import fit_rally_sides
from annotator.types import ContactCandidate


def scoring_filter(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Return contacts accepted by the scorer's wrist and suppression gates."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]


def select_heuristic_contacts(
    evidence: AnnotationEvidence, rejection_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[FinishedRally, ...]:
    """Finish heuristic rallies from their gate-passing, unmasked contacts.

    Append a diagnostic when every gate-passing contact in a rally is masked.
    """
    scored_contacts = scoring_filter(evidence.heuristic_contacts)
    scored_by_rally: dict[int, list[int]] = {}
    events_by_rally: dict[int, list[ContactEvent]] = {}
    for contact in scored_contacts:
        frame = contact.contact_frame
        scored_by_rally.setdefault(contact.rally_id, []).append(frame)
        if evidence.exclusion_mask[frame]:
            continue
        side = attribute_half(frame, evidence.track, evidence.sticky, evidence.bboxes, evidence.scene_courts)
        # Heuristic contacts have no model probability; only model inference reads it.
        event = ContactEvent(frame, math.nan, side)
        events_by_rally.setdefault(contact.rally_id, []).append(event)

    sequences = []
    for rally_id, (start, end) in enumerate(evidence.heuristic_spans):
        selected = tuple(events_by_rally.get(rally_id, ()))
        sequences.append(ContactSequence(rally_id, start, end, selected))
        before_mask = scored_by_rally.get(rally_id, [])
        if before_mask and not selected and rejection_diagnostics is not None:
            rejection_diagnostics.append({
                'rule': 'all_contacts_on_believed_mask',
                'rally_id': rally_id,
                'start_frame': start,
                'end_frame': end,
                'first_rejected_frame': before_mask[0],
                'rejection_grade': '',
            })
    return fit_rally_sides(sequences)
