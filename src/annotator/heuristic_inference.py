"""Filter heuristic contacts and fit the player order used by outcomes."""

import math

from annotator.contacts.attribution import contact_player_evidence
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    FinishedRally,
    fit_rally_sides,
)
from annotator.types import ContactCandidate


def select_heuristic_contacts(
    evidence: AnnotationEvidence, rejection_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[FinishedRally, ...]:
    """Finish heuristic rallies from their gate-passing, unmasked contacts.

    Append a diagnostic when every gate-passing contact in a rally is masked.
    """
    frames_by_rally: dict[int, list[int]] = {}
    for contact in scoring_filter(evidence.heuristic_contacts):
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
            # Heuristic contacts have no fitted probability; learned inference reads it.
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


def scoring_filter(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Return contacts accepted by the scorer's wrist and suppression gates."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]
