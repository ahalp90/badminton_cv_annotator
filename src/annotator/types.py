"""Court halves, player measurements, contact records and rally settings."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import NamedTuple

import numpy as np

# Court halves and tracked players


class CourtHalf(StrEnum):
    """Player or landing half as seen by the camera: Top is far, Bot is near."""

    TOP = 'Top'
    BOT = 'Bot'


OTHER_HALF = {CourtHalf.TOP: CourtHalf.BOT, CourtHalf.BOT: CourtHalf.TOP}


class Slot(IntEnum):
    """Top/bottom row indices in the tracked-player arrays.

    Matches sticky_anchor: top row 0, bottom row 1. The tracker chooses the
    bottom player first; these values specify storage order, not pick order.
    """

    TOP = 0
    BOTTOM = 1


# COCO wrist and ankle joint indices.
WRIST_L, WRIST_R = 9, 10
ANKLE_L, ANKLE_R = 15, 16


class StickyResult(NamedTuple):
    """Tracked players and their measurements, one row per video frame.

    Player tracking fills picks and body positions first. Wrist measurement
    then fills the distance arrays in place. Player axis: [TOP, BOTTOM].

    :param distances: (frames,) nearest wrist gap in body heights. Infinity
        outside tracked intervals; NaN if no usable gap within an interval.
    :param picks: (frames, 2) indices into the original pose detections.
        -1 means no player selected on that half.
    :param standing_count: (frames,) standing in-court detections.
    :param ankle_pos: (frames, 2, 2) mean ankle xy, divided by video width and
        height. NaN where no player was selected.
    :param bbox_height: (frames, 2) selected players' box heights in pixels.
        NaN where no player was selected.
    :param distances_per_slot: (frames, 2) wrist gaps in body heights, before
        choosing the nearest player. Same infinity/NaN meanings as distances.
    :param wrist_dist_px: (frames, 2) wrist gaps in pixels, measured only for
        visible shuttle frames. Infinity outside tracked intervals; NaN when
        unavailable within them.
    :param analysed: (frames,) True where player tracking ran.
    """

    distances: np.ndarray
    picks: np.ndarray
    standing_count: np.ndarray
    ankle_pos: np.ndarray
    bbox_height: np.ndarray
    distances_per_slot: np.ndarray
    wrist_dist_px: np.ndarray
    analysed: np.ndarray


# Contacts and rallies


class ContactCandidate(NamedTuple):
    """Heuristic contact and the results of its wrist and spacing checks.

    Includes rejected candidates so callers can inspect why they were dropped.
    ``rally_id`` indexes the initial rally spans. ``wrist_near`` is None when
    unchecked; ``suppressed`` is None when spacing was not checked. Supplied
    contact frames start with both checks unset.
    """

    rally_id: int
    contact_frame: int
    wrist_near: bool | None
    suppressed: bool | None


@dataclass(frozen=True)
class ContactEvent:
    """Contact frame, score and hitter, before or after fitting player order.

    ``side`` starts as the nearest-player guess. ``fit_rally_sides`` replaces
    it with the chosen alternating side. None means unknown at either stage.
    ``probability`` is the contact-model score, or NaN for a heuristic contact.
    ``wrist_distances`` is (Top, Bot) in body heights, or None when the two
    players cannot both be compared.
    """

    frame: int
    probability: float
    side: CourtHalf | None
    wrist_distances: tuple[float, float] | None = None  # TOP/BOT gaps in body heights.


@dataclass(frozen=True)
class ContactSequence:
    """Rally bounds and contacts in time order, from initial detection to selection.

    Frame range includes start, excludes end. ``span_id`` identifies the
    initial rally across alternative contact lists and later boundary changes.
    Contact sides are individual guesses until ``fit_rally_sides`` assigns
    one alternating order.
    """

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


@dataclass(frozen=True)
class FinishedRally:
    """Selected contacts after fitting one alternating player order.

    ``final_half`` is None if no contact identified a player.
    """

    sequence: ContactSequence
    final_half: CourtHalf | None

    @property
    def first_half(self) -> CourtHalf | None:
        """First hitter, derived from the final hitter and contact count."""
        if self.final_half is None:
            return None
        return self.final_half if len(self.sequence.events) % 2 else OTHER_HALF[self.final_half]


# Rally and contact detection choices


class SmoothingMode(StrEnum):
    """Whether to include invisible shuttle coordinates when smoothing the track."""

    ZERO_FILL = 'zero_fill'
    IGNORE_INVISIBLE = 'ignore_invisible'


class SpanOpen(StrEnum):
    """Where to start a rally between long rests.

    REGION_START uses the start of every such interval. BACK_FILL uses the
    same start, but only if the interval includes a sustained fast burst.
    Use None for the setting to start at the first fast burst instead.
    """

    REGION_START = 'region_start'
    BACK_FILL = 'back_fill'


class ReentryGuardVariant(StrEnum):
    """Checks for a shuttle returning after leaving the top of the image.

    REENTRY_ONLY requires a return near the top, moving downwards. TWO_SIDED
    also requires the last position before disappearing to be near the top.
    """

    TWO_SIDED = 'two-sided'
    REENTRY_ONLY = 'reentry-only'
