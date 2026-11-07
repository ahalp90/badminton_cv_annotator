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


class PlayerRow(IntEnum):
    """Index the player axis of tracked arrays: top = 0, bottom = 1."""

    TOP = 0
    BOTTOM = 1


# COCO wrist and ankle joint indices.
WRIST_L, WRIST_R = 9, 10
ANKLE_L, ANKLE_R = 15, 16


class TrackedPlayers(NamedTuple):
    """Selected top/bottom players, one row per video frame.

    Detection indices address the original pose arrays; -1 means no selection.
    Ankle xy is divided by video width/height. Missing positions/heights are NaN.
    """

    detection_indices: np.ndarray  # (frames, players)
    ankle_xy: np.ndarray  # (frames, players, xy)
    box_heights_px: np.ndarray  # (frames, players)


class WristGaps(NamedTuple):
    """Shuttle-to-wrist distances in body heights, for contact detection/attribution.

    Infinity outside tracked intervals; NaN for unavailable gaps inside them.
    Each player's distance uses its nearer wrist and mean nearby box height.
    """

    nearest: np.ndarray  # (frames,)
    per_player: np.ndarray  # (frames, players), top then bottom


# Contacts and rallies


class ContactCandidate(NamedTuple):
    """Raw heuristic contact, kept for inspection even if rejected.

    ``rally_id`` indexes the initial rally ranges. Supplied contacts and
    candidates found without wrist measurements have both checks unset.
    """

    rally_id: int
    contact_frame: int
    wrist_near: bool | None
    temporal_proximity_suppressed: bool | None


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

    INCLUDE_INVISIBLE = 'include_invisible'
    IGNORE_INVISIBLE = 'ignore_invisible'
