"""Build the finished contact sequences each rally may choose between.

A sequence edit keeps the detector's contacts, repairs the serve, deletes one
contact, or combines a serve repair with one deletion. A sequence option is an
edit plus at most one added later contact. Every chooser stage scores the same
option pool, so its order is part of the selected behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from annotator.contacts.model import NEARBY_CONTACT_DISTANCE_AT_30_FPS
from annotator.sequence.candidates import ServeCandidate
from annotator.sequence.contacts import (
    ContactEvent,
    ContactSequence,
    events_between,
    scale_frames,
)


class EditKind(StrEnum):
    """Edit kinds in their fixed tie-break and one-hot column order."""

    KEEP = "keep"
    ADD = "add"
    REPLACE = "replace"
    DELETE = "delete"
    ADD_DELETE = "add_delete"
    REPLACE_DELETE = "replace_delete"


EDIT_KINDS = tuple(EditKind)
SERVE_EDIT_KINDS = (EditKind.ADD, EditKind.REPLACE)
SERVE_KIND_BY_EDIT = {
    EditKind.ADD: EditKind.ADD,
    EditKind.REPLACE: EditKind.REPLACE,
    EditKind.ADD_DELETE: EditKind.ADD,
    EditKind.REPLACE_DELETE: EditKind.REPLACE,
}
COMBINED_KIND_BY_SERVE_KIND = {EditKind.ADD: EditKind.ADD_DELETE, EditKind.REPLACE: EditKind.REPLACE_DELETE}


@dataclass(frozen=True)
class SequenceEdit:
    """One serve repair, deletion or both, and the sequence it produces."""

    kind: EditKind
    serve_frame: int | None
    deleted_frame: int | None
    sequence: ContactSequence

    @property
    def serve_kind(self) -> EditKind | None:
        """Return ADD or REPLACE when this edit uses a serve candidate."""
        return SERVE_KIND_BY_EDIT.get(self.kind)


@dataclass(frozen=True)
class SequenceOption:
    """A sequence edit, optionally followed by one added later contact."""

    edit: SequenceEdit
    added_contact: ContactEvent | None
    sequence: ContactSequence

    @property
    def span_id(self) -> int:
        return self.sequence.span_id


def edit_priority(edit: SequenceEdit) -> tuple[int, int, int]:
    """Order equal-scored edits: simpler kinds first, then earlier frames."""
    first_frame = edit.serve_frame if edit.serve_frame is not None else edit.deleted_frame
    return (
        EDIT_KINDS.index(edit.kind),
        -1 if first_frame is None else first_frame,
        -1 if edit.deleted_frame is None else edit.deleted_frame,
    )


def option_priority(option: SequenceOption) -> tuple[int, tuple[int, int, int]]:
    """Order equal-scored options: fewer added contacts first, then the edit order."""
    return (int(option.added_contact is not None), edit_priority(option.edit))


def without_frame(sequence: ContactSequence, frame: int) -> ContactSequence:
    return replace(sequence, events=tuple(event for event in sequence.events if event.frame != frame))


def build_serve_edits(
    sequence: ContactSequence,
    candidates: Sequence[ServeCandidate],
    events: Sequence[ContactEvent],
    previous_end: int,
) -> list[SequenceEdit]:
    """Add each serve candidate, or let it replace the first contact.

    Moving the start earlier also absorbs any full-stream contact between the
    candidate and the original start. A candidate that would reach into the
    previous span is skipped.
    """
    edits = []
    for candidate in candidates:
        frame = candidate.event.frame
        new_start = min(sequence.start_frame, frame)
        if new_start < previous_end:
            continue
        expanded = events_between(events, new_start, sequence.end_frame)
        added = expanded
        if all(event.frame != frame for event in expanded):
            added = tuple(sorted((*expanded, candidate.event), key=lambda event: event.frame))
        replaced = tuple(event for event in added if event.frame != candidate.first_contact_frame)
        added_sequence = ContactSequence(sequence.span_id, new_start, sequence.end_frame, added)
        edits.append(SequenceEdit(EditKind.ADD, frame, None, added_sequence))
        edits.append(SequenceEdit(EditKind.REPLACE, frame, None, replace(added_sequence, events=replaced)))
    return edits


def sequence_edits(
    sequence: ContactSequence,
    candidates: Sequence[ServeCandidate],
    events: Sequence[ContactEvent],
    previous_end: int,
) -> tuple[SequenceEdit, ...]:
    """Return keep, serve, single-deletion and serve-plus-deletion edits in fixed order.

    A combined edit is skipped when it undoes its own serve candidate, when an
    added serve then deletes the first contact (that is a replace), or when its
    contacts match a simpler edit.

    :param sequence: One initial rally sequence.
    :param candidates: Its serve candidates, strongest first; empty without contacts.
    :param events: Full-stream contact events in frame order.
    :param previous_end: End frame of the previous initial span, or -1.
    :return: Edits in the order the choosers score and tie-break them.
    """
    edits = [SequenceEdit(EditKind.KEEP, None, None, sequence)]
    serve_edits = build_serve_edits(sequence, candidates, events, previous_end)
    edits.extend(serve_edits)
    for event in sequence.events:
        edits.append(SequenceEdit(EditKind.DELETE, None, event.frame, without_frame(sequence, event.frame)))
    simple_frame_lists = {tuple(event.frame for event in edit.sequence.events) for edit in edits}
    for serve_edit in serve_edits:
        for event in serve_edit.sequence.events:
            undoes_serve = event.frame == serve_edit.serve_frame
            deletes_first_after_add = serve_edit.kind == EditKind.ADD and event.frame == sequence.events[0].frame
            if undoes_serve or deletes_first_after_add:
                continue
            revised = without_frame(serve_edit.sequence, event.frame)
            if tuple(later.frame for later in revised.events) in simple_frame_lists:
                continue
            combined_kind = COMBINED_KIND_BY_SERVE_KIND[serve_edit.kind]
            edits.append(SequenceEdit(combined_kind, serve_edit.serve_frame, event.frame, revised))
    return tuple(edits)


def option_pool(
    edits_by_span: Mapping[int, Sequence[SequenceEdit]],
    later_candidates: Mapping[int, Sequence[ContactEvent]],
    fps: float,
) -> tuple[SequenceOption, ...]:
    """Follow each edit by itself, then by each compatible later contact.

    A later contact must fall inside the edited sequence and more than the
    nearby-contact distance from each of its contacts.
    """
    distance = scale_frames(NEARBY_CONTACT_DISTANCE_AT_30_FPS, fps)
    options = []
    for span_id, edits in edits_by_span.items():
        for edit in edits:
            options.append(SequenceOption(edit, None, edit.sequence))
            edited = edit.sequence
            for candidate in later_candidates.get(span_id, ()):
                if not edited.start_frame <= candidate.frame < edited.end_frame:
                    continue
                if any(abs(candidate.frame - event.frame) <= distance for event in edited.events):
                    continue
                events = tuple(sorted((*edited.events, candidate), key=lambda event: event.frame))
                options.append(SequenceOption(edit, candidate, replace(edited, events=events)))
    return tuple(options)


def apply_choices(
    initial: Sequence[ContactSequence],
    events: Sequence[ContactEvent],
    chosen: Mapping[int, ContactSequence],
) -> tuple[tuple[ContactSequence, ...], tuple[ContactEvent, ...]]:
    """Swap in each chosen sequence and rebuild the full contact stream.

    Contacts an initial sequence owned survive only if its chosen sequence keeps
    them. Contacts outside every initial sequence always survive, so an edit
    that deletes an absorbed out-of-span contact leaves it in the full stream.
    """
    owned = {event.frame for sequence in initial for event in sequence.events}
    by_frame: dict[int, ContactEvent] = {}
    sequences = []
    for sequence in initial:
        after = chosen[sequence.span_id]
        sequences.append(after)
        for event in after.events:
            by_frame[event.frame] = event
    for event in events:
        if event.frame not in owned:
            by_frame[event.frame] = event
    return tuple(sequences), tuple(by_frame[frame] for frame in sorted(by_frame))
