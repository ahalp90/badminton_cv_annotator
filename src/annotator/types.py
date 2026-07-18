"""Shared annotator declarations for fps scaling and storage slots."""
from __future__ import annotations

import math
from enum import IntEnum, StrEnum
from typing import NamedTuple

from .fps_constants import BASE_FPS


class ScalingKind(StrEnum):
    """Describe how a base-30 value scales with the video's frame rate.

    Per-frame speeds shrink as fps rises, frame counts grow, dimensionless
    values never scale; arithmetic must stay identical to ``scale_for_fps``
    until the threading stage rewires the table onto these declarations.
    """

    PER_FRAME_SPEED = 'per_frame_speed'
    FRAME_COUNT = 'frame_count'
    DIMENSIONLESS = 'dimensionless'

    def scale(self, value: float, fps: float) -> float | int:
        """Scale one base-30 value, requiring a positive finite frame rate."""
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f'fps must be positive and finite, got {fps!r}')
        if self is ScalingKind.PER_FRAME_SPEED:
            return value * BASE_FPS / fps
        if self is ScalingKind.FRAME_COUNT:
            return max(1, math.floor(value * fps / BASE_FPS + 0.5))
        return value


class DeadMaskMode(StrEnum):
    """Select the producer policy for the per-frame dead-time mask."""

    REPLAY = 'replay'
    COMPOSITION = 'composition'
    UNION = 'union'


class ContactCandidate(NamedTuple):
    """One raw contact candidate and its independent gate/suppression verdicts."""

    rally_id: int
    contact_frame: int
    proximity_ok: bool | None
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
