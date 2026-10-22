"""Select contacts and fit the alternating player order before rally outcomes."""

from collections.abc import Sequence
from dataclasses import dataclass

from annotator.courts.scenes import court_at_frame
from annotator.outcomes import point_winner
from annotator.preparation import AnnotationEvidence
from annotator.sequence.contacts import ContactSequence
from annotator.types import ContactCandidate

OTHER_HALF = point_winner.OTHER_HALF


@dataclass(frozen=True)
class ContactData:
    """Filtered contacts and fitted hitting-order fields for downstream stages."""

    spans: list[tuple[int, int]]
    filtered_contacts: list[ContactCandidate]
    scored_by_rally: dict[int, list[int]]
    filtered_by_rally: dict[int, list[int]]
    striker_halves: list[point_winner.Half | None]
    n_strokes_list: list[int]
    next_servers: list[point_winner.Half | None]
    fitted_first_all: list[point_winner.Half | None]


def scoring_filter(contacts: list[ContactCandidate]) -> list[ContactCandidate]:
    """Return contacts accepted by the scorer's wrist and suppression gates."""
    return [
        contact
        for contact in contacts
        if contact.wrist_near is not False and contact.suppressed is not True
    ]


def first_stroke_half(
    final_half: point_winner.Half, n_strokes: int,
) -> point_winner.Half:
    """Return a rally's fitted first-stroke half from its fitted final half."""
    final_stroke_is_first_half = (n_strokes - 1) % 2 == 0
    return final_half if final_stroke_is_first_half else OTHER_HALF[final_half]


def select_heuristic_contacts(evidence: AnnotationEvidence) -> ContactData:
    """Filter heuristic contacts and fit each rally's alternating striker phase."""
    scored_contacts = scoring_filter(evidence.heuristic_contacts)
    filtered_contacts = [
        contact
        for contact in scored_contacts
        if not evidence.exclusion_mask[contact.contact_frame]
    ]
    scored_by_rally: dict[int, list[int]] = {}
    for contact in scored_contacts:
        scored_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)
    filtered_by_rally: dict[int, list[int]] = {}
    for contact in filtered_contacts:
        filtered_by_rally.setdefault(contact.rally_id, []).append(contact.contact_frame)

    striker_halves: list[point_winner.Half | None] = []
    for rally_id in range(len(evidence.heuristic_spans)):
        guesses = []
        for frame in filtered_by_rally.get(rally_id, []):
            frame_net_band = court_at_frame(evidence.scene_courts, frame).net_band
            guesses.append(point_winner.attribute_half(
                frame, evidence.track, evidence.sticky, evidence.bboxes, frame_net_band,
            ))
        striker_halves.append(point_winner.fit_alternation(guesses))
    n_strokes_list = [len(filtered_by_rally.get(rally_id, [])) for rally_id in range(len(evidence.heuristic_spans))]
    next_servers = point_winner.next_server_half(striker_halves, n_strokes_list)
    fitted_first_all = [
        first_stroke_half(half, n) if half is not None else None
        for half, n in zip(striker_halves, n_strokes_list)
    ]
    return ContactData(
        spans=evidence.heuristic_spans,
        filtered_contacts=filtered_contacts,
        scored_by_rally=scored_by_rally,
        filtered_by_rally=filtered_by_rally,
        striker_halves=striker_halves,
        n_strokes_list=n_strokes_list,
        next_servers=next_servers,
        fitted_first_all=fitted_first_all,
    )


def contacts_from_sequences(sequences: Sequence[ContactSequence]) -> ContactData:
    """Carry final model contacts and sides into landing and winner prediction.

    The sequence models already selected contacts and fitted player order.
    Use those sides directly; re-attributing players here could use different
    court geometry from the model that selected the sequence.
    """
    spans = []
    contacts = []
    frames_by_rally = {}
    striker_halves = []
    for sequence in sequences:
        spans.append((sequence.start_frame, sequence.end_frame))
        frames = [event.frame for event in sequence.events]
        if frames:
            frames_by_rally[sequence.span_id] = frames
        contacts.extend(ContactCandidate(sequence.span_id, frame, None, None, None) for frame in frames)
        striker_halves.append(point_winner.fit_alternation([event.side for event in sequence.events]))
    counts = [len(sequence.events) for sequence in sequences]
    first_halves = [
        first_stroke_half(half, count) if half is not None else None
        for half, count in zip(striker_halves, counts, strict=True)
    ]
    return ContactData(
        spans=spans,
        filtered_contacts=contacts,
        scored_by_rally=frames_by_rally,
        filtered_by_rally=frames_by_rally,
        striker_halves=striker_halves,
        n_strokes_list=counts,
        next_servers=point_winner.next_server_half(striker_halves, counts),
        fitted_first_all=first_halves,
    )
