"""Select contacts and fit the alternating player order before rally outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass

from annotator.courts.scenes import court_at_frame
from annotator.outcomes import point_winner
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import FinishedRally
from annotator.types import ContactCandidate

OTHER_HALF = point_winner.OTHER_HALF


@dataclass(frozen=True)
class OutcomeRally:
    """Selected frames and fitted player order for one rally."""

    span: tuple[int, int]
    frames: list[int]
    frames_before_mask: list[int]
    final_half: point_winner.Half | None

    @property
    def first_half(self) -> point_winner.Half | None:
        """Derive the first striker from the fitted final striker."""
        if self.final_half is None:
            return None
        return self.final_half if len(self.frames) % 2 else OTHER_HALF[self.final_half]


@dataclass(frozen=True)
class ContactData:
    """Rallies and accepted contacts used by downstream stages."""

    rallies: list[OutcomeRally]
    filtered_contacts: list[ContactCandidate]


def scoring_filter(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Return contacts accepted by the scorer's wrist and suppression gates."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]


def select_heuristic_contacts(evidence: AnnotationEvidence) -> ContactData:
    """Filter heuristic contacts and fit each rally's alternating striker phase."""
    scored_contacts = scoring_filter(evidence.heuristic_contacts)
    filtered_contacts = [
        contact for contact in scored_contacts if not evidence.exclusion_mask[contact.contact_frame]
    ]
    scored_by_rally: dict[int, list[int]] = {}
    for contact in scored_contacts:
        scored_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    rallies = []
    for rally_id, span in enumerate(evidence.heuristic_spans):
        frames = filtered_by_rally.get(rally_id, [])
        guesses = []
        for frame in frames:
            frame_net_band = court_at_frame(evidence.scene_courts, frame).net_band
            guesses.append(point_winner.attribute_half(
                frame, evidence.track, evidence.sticky, evidence.bboxes, frame_net_band,
            ))
        rallies.append(OutcomeRally(span, frames, scored_by_rally.get(rally_id, []),
                                    point_winner.fit_alternation(guesses)))
    return ContactData(rallies, filtered_contacts)


def contacts_from_sequences(finished: Sequence[FinishedRally]) -> ContactData:
    """Carry selected contacts and their already-fitted sides into outcomes."""
    rallies = []
    contacts = []
    for rally in finished:
        sequence = rally.sequence
        frames = [event.frame for event in sequence.events]
        contacts.extend(ContactCandidate(sequence.span_id, frame, None, None, None) for frame in frames)
        rallies.append(OutcomeRally((sequence.start_frame, sequence.end_frame), frames, frames, rally.final_half))
    return ContactData(rallies, contacts)
