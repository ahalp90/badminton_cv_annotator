"""Shared court halves, annotator settings and player/contact records."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import NamedTuple

import numpy as np


class CourtHalf(StrEnum):
    """Camera-relative court half of a player or landing: far (Top) or near (Bot)."""

    TOP = 'Top'
    BOT = 'Bot'


OTHER_HALF = {CourtHalf.TOP: CourtHalf.BOT, CourtHalf.BOT: CourtHalf.TOP}


class SmoothingMode(StrEnum):
    """Select how invisible frames contribute to span smoothing."""

    ZERO_FILL = 'zero_fill'
    IGNORE_INVISIBLE = 'ignore_invisible'


class SpanOpen(StrEnum):
    """Where to start a rally within the region between long rests.

    REGION_START opens a rally at every region's start. BACK_FILL also opens at
    the region's start, but requires a sustained fast burst within that region.
    Setting ``ResolvedAnnotatorConfig.span_open`` to None starts at the first
    qualifying fast burst instead.
    """

    REGION_START = 'region_start'
    BACK_FILL = 'back_fill'


class ReentryGuardVariant(StrEnum):
    """Checks for a shuttle returning after leaving the top of the frame.

    REENTRY_ONLY requires it to return near the top and descend. TWO_SIDED also
    requires its last position before disappearing to be near the top.
    """

    TWO_SIDED = 'two-sided'
    REENTRY_ONLY = 'reentry-only'


class ContactCandidate(NamedTuple):
    """A contact candidate with wrist-distance and suppression results."""

    rally_id: int
    contact_frame: int
    wrist_near: bool | None
    suppressed: bool | None


class Slot(IntEnum):
    """Storage-row indices pinned to sticky_anchor's public constants.

    ``SLOT_TOP = 0`` and ``SLOT_BOTTOM = 1`` let annotator code index sticky's
    per-slot arrays directly.
    Sticky's pick order is bottom-first and is deliberately not modelled here;
    enum definition and iteration order must never be read as pick order.
    """

    TOP = 0
    BOTTOM = 1


# COCO wrist/ankle indices shared by player tracking and contact measurements.
WRIST_L, WRIST_R = 9, 10
ANKLE_L, ANKLE_R = 15, 16


class StickyResult(NamedTuple):
    """Frame-aligned evidence from the sticky player picker.

    The second axis of every per-slot field is ``[TOP, BOTTOM]``.

    :param distances: ``(t,)`` nearest finite wrist gap in body-height units;
        positive infinity outside analysed segments and NaN when an analysed
        frame has no finite picked-player gap.
    :param picks: ``(t, 2)`` raw pose-slot indices; ``-1`` means no accepted
        player for that court half.
    :param standing_count: ``(t,)`` number of standing in-court detections.
    :param ankle_pos: ``(t, 2, 2)`` mean ankle ``[x, y]`` normalised by video
        resolution; NaN where no player was picked.
    :param bbox_height: ``(t, 2)`` picked-player box heights in pixels; NaN
        where no player was picked.
    :param distances_per_slot: ``(t, 2)`` wrist gaps in body-height units before
        the nearest-slot collapse, with the same infinity and NaN sentinels as
        ``distances``.
    :param wrist_dist_px: ``(t, 2)`` wrist gaps in pixels on visible shuttle
        frames; positive infinity outside analysed segments and NaN when
        unavailable inside them.
    :param analysed: ``(t,)`` boolean mask of frames visited by the picker.
    """

    distances: np.ndarray
    picks: np.ndarray
    standing_count: np.ndarray
    ankle_pos: np.ndarray
    bbox_height: np.ndarray
    distances_per_slot: np.ndarray
    wrist_dist_px: np.ndarray
    analysed: np.ndarray


@dataclass(frozen=True)
class ContactEvent:
    """A contact frame, model probability (NaN for heuristics) and player side."""

    frame: int
    probability: float
    side: CourtHalf | None
    wrist_distances: tuple[float, float] | None = None  # TOP/BOT gaps in body heights.


@dataclass(frozen=True)
class ContactSequence:
    """One half-open rally interval and its contacts in chronological order."""

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


@dataclass(frozen=True)
class FinishedRally:
    """A selected sequence and its final striker, unknown if no player was identified."""

    sequence: ContactSequence
    final_half: CourtHalf | None

    @property
    def first_half(self) -> CourtHalf | None:
        """Derive the first striker from the fitted final striker."""
        if self.final_half is None:
            return None
        return self.final_half if len(self.sequence.events) % 2 else OTHER_HALF[self.final_half]
