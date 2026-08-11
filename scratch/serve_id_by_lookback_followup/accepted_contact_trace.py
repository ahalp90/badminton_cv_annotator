"""GT-free helpers for the sequential accepted-contact opener experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import NamedTuple, Protocol

import numpy as np

from annotator.types import true_runs

MIN_PATH_FRAMES = 5
MAX_LARGEST_STEP_RATIO = 4.0
ROBUST_TREND_MIN_DECREASE_BH = 0.05


class MotionMeasurements(Protocol):
    """Fields used from the PR #82 motion measurement."""

    n_frames: int
    largest_step_ratio: float


class DistanceTrend(Protocol):
    """Field used from the PR #82 robust distance trend."""

    fitted_decrease_bh: float


class PostContactRun(NamedTuple):
    """A maximal usable shuttle run after an accepted contact."""

    start: int
    end: int
    frames_from_contact: int


class IncomingVerdict(StrEnum):
    """Three-way PR #82 pre-contact motion result."""

    INCOMING = "incoming"
    NOT_INCOMING = "not_incoming"
    UNAVAILABLE = "unavailable"


class OpenerCategory(StrEnum):
    """Terminal result of the sequential opener search."""

    VISIBLE_SERVE = "visible_serve"
    FIRST_VISIBLE_POST_SERVE = "first_visible_post_serve_contact"
    NOT_ENOUGH_TRAJECTORY = "not_enough_shuttle_trajectory_to_tell"
    NO_CREDIBLE_CONTACT = "no_credible_accepted_contact"


@dataclass(frozen=True, slots=True)
class AcceptedContactEvidence:
    """GT-free outgoing evidence for one chronological accepted contact."""

    frame: int
    credible_outgoing: bool


@dataclass(frozen=True, slots=True)
class OpenerSearchResult:
    """Selected accepted contact and its pre-contact classification."""

    category: OpenerCategory
    selected_frame: int | None
    selected_rank: int | None
    skipped_contacts: int


def closest_post_contact_run(
    usable: np.ndarray,
    contact_frame: int,
    lookahead_frames: int,
    same_scene_mask: np.ndarray | None = None,
) -> PostContactRun | None:
    """Select the earliest maximal run in ``(contact, contact + lookahead]``."""
    usable_array = np.asarray(usable)
    if usable_array.ndim != 1:
        raise ValueError("usable must be a one-dimensional mask")
    if not 0 <= contact_frame < len(usable_array):
        raise ValueError("contact_frame must index the usable mask")
    if lookahead_frames < 0:
        raise ValueError("lookahead_frames must be non-negative")

    if same_scene_mask is not None:
        scene_array = np.asarray(same_scene_mask)
        if scene_array.shape != usable_array.shape:
            raise ValueError("same_scene_mask must have the same shape as usable")
        usable_array = usable_array.astype(bool) & scene_array.astype(bool)
    else:
        usable_array = usable_array.astype(bool)

    window_start = contact_frame + 1
    window_end = min(len(usable_array), contact_frame + lookahead_frames + 1)
    runs = true_runs(usable_array[window_start:window_end])
    if not runs:
        return None

    relative_start, relative_end = runs[0]
    start = window_start + relative_start
    end = window_start + relative_end
    return PostContactRun(start, end, start - contact_frame)


def has_credible_outgoing_motion(
    motion: MotionMeasurements | None,
    trend: DistanceTrend | None,
    frames_from_contact: int | None,
    maximum_frames_from_contact: int,
) -> bool:
    """Collapse unavailable evidence into the binary outgoing predicate."""
    if maximum_frames_from_contact < 1:
        raise ValueError("contact gaps must be positive")
    if motion is None or trend is None or frames_from_contact is None:
        return False
    if frames_from_contact < 1:
        raise ValueError("contact gaps must be positive")
    return _common_path_eligible(
        motion,
        frames_from_contact,
        maximum_frames_from_contact,
    ) and trend.fitted_decrease_bh <= -ROBUST_TREND_MIN_DECREASE_BH


def _common_path_eligible(
    motion: MotionMeasurements,
    frames_from_contact: int,
    maximum_frames_from_contact: int,
) -> bool:
    """Apply the shared fixed path-length, gap, and jump checks."""
    return (
        motion.n_frames >= MIN_PATH_FRAMES
        and frames_from_contact <= maximum_frames_from_contact
        and motion.largest_step_ratio <= MAX_LARGEST_STEP_RATIO
    )


def search_accepted_contacts(
    contacts: Sequence[AcceptedContactEvidence],
    incoming_check: Callable[[int], IncomingVerdict],
) -> OpenerSearchResult:
    """Select the first credible contact, then request its PR #82 pre verdict."""
    if any(current.frame <= previous.frame for previous, current in pairwise(contacts)):
        raise ValueError("accepted contacts must be strictly chronological")

    for accepted_rank, contact in enumerate(contacts, start=1):
        if not contact.credible_outgoing:
            continue
        incoming = incoming_check(contact.frame)
        if incoming is IncomingVerdict.INCOMING:
            category = OpenerCategory.FIRST_VISIBLE_POST_SERVE
        elif incoming is IncomingVerdict.NOT_INCOMING:
            category = OpenerCategory.VISIBLE_SERVE
        else:
            category = OpenerCategory.NOT_ENOUGH_TRAJECTORY
        return OpenerSearchResult(category, contact.frame, accepted_rank, accepted_rank - 1)

    return OpenerSearchResult(
        OpenerCategory.NO_CREDIBLE_CONTACT,
        selected_frame=None,
        selected_rank=None,
        skipped_contacts=len(contacts),
    )
