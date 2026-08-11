"""Pure helpers for the H3/R8 accepted-contact opener experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Protocol

import numpy as np

from annotator.inpaint_guard import DEGRADED, FABRICATED, NO_FLAG, SUSPECT_FLAT

MIN_PATH_FRAMES = 5
MAX_LARGEST_STEP_RATIO = 8.0
ROBUST_TREND_MIN_DECREASE_BH = 0.05


class MotionMeasurements(Protocol):
    """Path fields used by the fixed local eligibility rule."""

    n_frames: int
    largest_step_ratio: float


class DistanceTrend(Protocol):
    """Robust player-distance trend used by both motion verdicts."""

    fitted_decrease_bh: float


class PreContactVerdict(StrEnum):
    """Three-way pre-contact trajectory verdict."""

    INCOMING = "incoming"
    NOT_INCOMING = "not_incoming"
    UNAVAILABLE = "unavailable"


class SequentialCategory(StrEnum):
    """Terminal category from the outgoing-first search."""

    VISIBLE_SERVE = "visible_serve"
    FIRST_VISIBLE_POST_SERVE = "first_visible_post_serve_contact"
    NOT_ENOUGH_TRAJECTORY = "not_enough_shuttle_trajectory_to_tell"
    NO_CREDIBLE_CONTACT = "no_credible_accepted_contact"


class IncomingSearchCategory(StrEnum):
    """Terminal category from the earliest-incoming predecessor search."""

    VISIBLE_SERVE = "visible_serve"
    FIRST_VISIBLE_POST_SERVE = "first_visible_post_serve_contact"
    PREDECESSOR_EVIDENCE_UNAVAILABLE = "predecessor_evidence_unavailable"
    NO_MEASURED_INCOMING = "no_measured_incoming"
    NO_INCOMING_WITH_UNAVAILABLE = "no_incoming_anchor_with_unavailable_evidence"
    NO_ACCEPTED_CONTACT = "no_accepted_contact"


class PredecessorAdmission(StrEnum):
    """Rule which admitted the nearest earlier accepted contact."""

    NONE = "none"
    ORDINARY = "ordinary_window"
    HIGH_SHOT = "high_shot_oob"


class IncomingStopReason(StrEnum):
    """Exact reason the incoming-only search stopped."""

    NO_ACCEPTED_CONTACT = "no_accepted_contact"
    NO_MEASURED_INCOMING = "no_measured_incoming"
    NO_INCOMING_WITH_UNAVAILABLE = "no_incoming_anchor_with_unavailable_evidence"
    NO_PREDECESSOR = "no_predecessor"
    PREDECESSOR_BEYOND_ADMISSION = "predecessor_beyond_admission_rules"
    PREDECESSOR_ADMITTED_ORDINARY = "predecessor_admitted_by_ordinary_window"
    PREDECESSOR_ADMITTED_HIGH_SHOT = "predecessor_admitted_by_high_shot_oob"


@dataclass(frozen=True, slots=True)
class FrozenContactEvidence:
    """Frozen GT-free evidence for one chronological accepted contact."""

    frame: int
    pre_verdict: PreContactVerdict
    credible_outgoing: bool


@dataclass(frozen=True, slots=True)
class SequentialSearchResult:
    """Selected contact and classification from the outgoing-first search."""

    category: SequentialCategory
    selected_frame: int | None
    selected_rank: int | None
    skipped_contacts: int


@dataclass(frozen=True, slots=True)
class HighShotState:
    """Measured half-open production ``high_shot_oob`` interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("high-shot state must be a non-empty half-open interval")


@dataclass(frozen=True, slots=True)
class IncomingSearchResult:
    """Anchor, predecessor admission, and category from the incoming search."""

    category: IncomingSearchCategory
    stop_reason: IncomingStopReason
    anchor_frame: int | None
    anchor_rank: int | None
    predecessor_frame: int | None
    predecessor_rank: int | None
    contact_gap: int | None
    admission: PredecessorAdmission


def rebuild_guard_codes(track: np.ndarray, existing_codes: np.ndarray, halo_frames: int) -> np.ndarray:
    """Rebuild recurrence grades with a different literal source-frame halo."""
    track_array = np.asarray(track)
    codes_array = np.asarray(existing_codes)
    if track_array.ndim != 2 or track_array.shape[1] < 2:
        raise ValueError("track must have shape (n_frames, at least 2)")
    if codes_array.ndim != 1 or len(codes_array) != len(track_array):
        raise ValueError("existing_codes must have one value per track frame")
    if np.any(~np.isin(codes_array, (NO_FLAG, FABRICATED, SUSPECT_FLAT, DEGRADED))):
        raise ValueError("existing_codes contains an unknown recurrence grade")
    if isinstance(halo_frames, bool) or not isinstance(halo_frames, (int, np.integer)) or halo_frames < 0:
        raise ValueError("halo_frames must be a non-negative integer")

    core = (codes_array == FABRICATED) | (codes_array == SUSPECT_FLAT)
    halo = np.zeros(len(track_array), dtype=bool)
    edges = np.diff(np.concatenate(([False], core, [False])).astype(np.int8))
    for start in np.flatnonzero(edges == 1):
        start = int(start)
        halo[max(0, start - int(halo_frames)):start] = True
    for stop in np.flatnonzero(edges == -1):
        stop = int(stop)
        halo[stop:min(len(track_array), stop + int(halo_frames))] = True

    on_attractor = np.zeros(len(track_array), dtype=bool)
    for pos_x, pos_y in np.unique(track_array[core, :2], axis=0):
        on_attractor |= (track_array[:, 0] == pos_x) & (track_array[:, 1] == pos_y)

    rebuilt = np.zeros(len(track_array), dtype=np.uint8)
    rebuilt[(halo | on_attractor) & ~core] = DEGRADED
    rebuilt[codes_array == SUSPECT_FLAT] = SUSPECT_FLAT
    rebuilt[codes_array == FABRICATED] = FABRICATED
    stored_blank = (track_array[:, 0] == 0) & (track_array[:, 1] == 0)
    rebuilt[stored_blank] = NO_FLAG
    return rebuilt


def path_is_eligible(
    motion: MotionMeasurements,
    contact_gap: int,
    maximum_contact_gap: int,
) -> bool:
    """Apply the fixed H3/R8 path-length, contact-gap, and jump checks."""
    _validate_contact_gaps(contact_gap, maximum_contact_gap)
    return (
        motion.n_frames >= MIN_PATH_FRAMES
        and contact_gap <= maximum_contact_gap
        and motion.largest_step_ratio <= MAX_LARGEST_STEP_RATIO
    )


def classify_pre_motion(
    motion: MotionMeasurements | None,
    trend: DistanceTrend | None,
    contact_gap: int | None,
    maximum_contact_gap: int,
) -> PreContactVerdict:
    """Classify incoming evidence while preserving unavailable paths."""
    _validate_maximum_contact_gap(maximum_contact_gap)
    if motion is None or trend is None or contact_gap is None:
        return PreContactVerdict.UNAVAILABLE
    if not path_is_eligible(motion, contact_gap, maximum_contact_gap):
        return PreContactVerdict.UNAVAILABLE
    if trend.fitted_decrease_bh >= ROBUST_TREND_MIN_DECREASE_BH:
        return PreContactVerdict.INCOMING
    return PreContactVerdict.NOT_INCOMING


def has_outgoing_motion(
    motion: MotionMeasurements | None,
    trend: DistanceTrend | None,
    contact_gap: int | None,
    maximum_contact_gap: int,
) -> bool:
    """Collapse absent or ineligible post-contact evidence into false."""
    _validate_maximum_contact_gap(maximum_contact_gap)
    if motion is None or trend is None or contact_gap is None:
        return False
    return (
        path_is_eligible(motion, contact_gap, maximum_contact_gap)
        and trend.fitted_decrease_bh <= -ROBUST_TREND_MIN_DECREASE_BH
    )


def sequential_outgoing_search(contacts: Sequence[FrozenContactEvidence]) -> SequentialSearchResult:
    """Select the first contact with frozen positive outgoing evidence."""
    _validate_contacts(contacts)
    for accepted_rank, contact in enumerate(contacts, start=1):
        if not contact.credible_outgoing:
            continue
        if contact.pre_verdict is PreContactVerdict.INCOMING:
            category = SequentialCategory.FIRST_VISIBLE_POST_SERVE
        elif contact.pre_verdict is PreContactVerdict.NOT_INCOMING:
            category = SequentialCategory.VISIBLE_SERVE
        else:
            category = SequentialCategory.NOT_ENOUGH_TRAJECTORY
        return SequentialSearchResult(category, contact.frame, accepted_rank, accepted_rank - 1)

    return SequentialSearchResult(
        SequentialCategory.NO_CREDIBLE_CONTACT,
        selected_frame=None,
        selected_rank=None,
        skipped_contacts=len(contacts),
    )


def incoming_predecessor_search(
    contacts: Sequence[FrozenContactEvidence],
    *,
    ordinary_max_gap_frames: int,
    high_shot_state: HighShotState | None,
    high_shot_endpoint_buffer_frames: int,
) -> IncomingSearchResult:
    """Inspect the nearest predecessor of the earliest incoming contact."""
    _validate_contacts(contacts)
    if ordinary_max_gap_frames < 1:
        raise ValueError("ordinary_max_gap_frames must be positive")
    if high_shot_endpoint_buffer_frames < 0:
        raise ValueError("high_shot_endpoint_buffer_frames must be non-negative")

    if not contacts:
        return _terminal_incoming_result(
            IncomingSearchCategory.NO_ACCEPTED_CONTACT,
            IncomingStopReason.NO_ACCEPTED_CONTACT,
        )

    anchor_index = next(
        (index for index, contact in enumerate(contacts) if contact.pre_verdict is PreContactVerdict.INCOMING),
        None,
    )
    if anchor_index is None:
        if any(contact.pre_verdict is PreContactVerdict.UNAVAILABLE for contact in contacts):
            return _terminal_incoming_result(
                IncomingSearchCategory.NO_INCOMING_WITH_UNAVAILABLE,
                IncomingStopReason.NO_INCOMING_WITH_UNAVAILABLE,
            )
        return _terminal_incoming_result(
            IncomingSearchCategory.NO_MEASURED_INCOMING,
            IncomingStopReason.NO_MEASURED_INCOMING,
        )

    anchor = contacts[anchor_index]
    anchor_rank = anchor_index + 1
    if anchor_index == 0:
        return IncomingSearchResult(
            IncomingSearchCategory.FIRST_VISIBLE_POST_SERVE,
            IncomingStopReason.NO_PREDECESSOR,
            anchor.frame,
            anchor_rank,
            predecessor_frame=None,
            predecessor_rank=None,
            contact_gap=None,
            admission=PredecessorAdmission.NONE,
        )

    predecessor = contacts[anchor_index - 1]
    predecessor_rank = anchor_index
    contact_gap = anchor.frame - predecessor.frame
    admission = _predecessor_admission(
        predecessor.frame,
        anchor.frame,
        ordinary_max_gap_frames,
        high_shot_state,
        high_shot_endpoint_buffer_frames,
    )
    if admission is PredecessorAdmission.NONE:
        return IncomingSearchResult(
            IncomingSearchCategory.FIRST_VISIBLE_POST_SERVE,
            IncomingStopReason.PREDECESSOR_BEYOND_ADMISSION,
            anchor.frame,
            anchor_rank,
            predecessor.frame,
            predecessor_rank,
            contact_gap,
            admission,
        )

    stop_reason = (
        IncomingStopReason.PREDECESSOR_ADMITTED_ORDINARY
        if admission is PredecessorAdmission.ORDINARY
        else IncomingStopReason.PREDECESSOR_ADMITTED_HIGH_SHOT
    )
    category = (
        IncomingSearchCategory.PREDECESSOR_EVIDENCE_UNAVAILABLE
        if predecessor.pre_verdict is PreContactVerdict.UNAVAILABLE
        else IncomingSearchCategory.VISIBLE_SERVE
    )
    return IncomingSearchResult(
        category,
        stop_reason,
        anchor.frame,
        anchor_rank,
        predecessor.frame,
        predecessor_rank,
        contact_gap,
        admission,
    )


def _predecessor_admission(
    predecessor_frame: int,
    anchor_frame: int,
    ordinary_max_gap_frames: int,
    high_shot_state: HighShotState | None,
    endpoint_buffer_frames: int,
) -> PredecessorAdmission:
    """Admit a candidate by timing or by a measured high-shot state."""
    if anchor_frame - predecessor_frame <= ordinary_max_gap_frames:
        return PredecessorAdmission.ORDINARY
    if high_shot_state is None:
        return PredecessorAdmission.NONE

    brackets_state = predecessor_frame <= high_shot_state.start and high_shot_state.end <= anchor_frame
    predecessor_near_start = high_shot_state.start - predecessor_frame <= endpoint_buffer_frames
    anchor_near_end = anchor_frame - high_shot_state.end <= endpoint_buffer_frames
    if brackets_state and predecessor_near_start and anchor_near_end:
        return PredecessorAdmission.HIGH_SHOT
    return PredecessorAdmission.NONE


def _terminal_incoming_result(
    category: IncomingSearchCategory,
    stop_reason: IncomingStopReason,
) -> IncomingSearchResult:
    """Build a no-anchor result without an invented contact frame."""
    return IncomingSearchResult(
        category,
        stop_reason,
        anchor_frame=None,
        anchor_rank=None,
        predecessor_frame=None,
        predecessor_rank=None,
        contact_gap=None,
        admission=PredecessorAdmission.NONE,
    )


def _validate_contacts(contacts: Sequence[FrozenContactEvidence]) -> None:
    """Require the accepted-contact order supplied by the frozen evidence."""
    if any(current.frame <= previous.frame for previous, current in pairwise(contacts)):
        raise ValueError("accepted contacts must be strictly chronological")


def _validate_contact_gaps(contact_gap: int, maximum_contact_gap: int) -> None:
    """Validate the positive local contact-gap inputs."""
    _validate_maximum_contact_gap(maximum_contact_gap)
    if contact_gap < 1:
        raise ValueError("contact_gap must be positive")


def _validate_maximum_contact_gap(maximum_contact_gap: int) -> None:
    """Validate the fixed local path gap limit."""
    if maximum_contact_gap < 1:
        raise ValueError("maximum_contact_gap must be positive")
