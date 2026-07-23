"""Shared annotator declarations for fps scaling and storage slots."""
from __future__ import annotations

import math
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, NamedTuple

from .fps_constants import BASE_FPS

if TYPE_CHECKING:
    from .rally_segmentation import ServeStartClose, ServeStartMode


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


class SmoothingMode(StrEnum):
    """Select how invisible frames contribute to span smoothing."""

    ZERO_FILL = 'zero_fill'
    IGNORE_INVISIBLE = 'ignore_invisible'


class SpanOpen(StrEnum):
    """Where a rally span opens; segment_video(span_open=...), default None.

    None keeps today's burst-open rule bit-for-bit: a span opens at the first
    qualifying fast burst in its active region. The two named rules trade that:
      REGION_START drops the qualifying-burst gate entirely and opens a span at
      every active region's start (each maximal run of non-long-rest frames).
      BACK_FILL keeps the qualifying-burst gate unchanged (a region with no
      qualifying fast run yields no rally) but moves the emitted span's start
      back from the burst to the region start.
    """

    REGION_START = 'region_start'
    BACK_FILL = 'back_fill'


class ReentryGuardVariant(StrEnum):
    """Which sides of a high-shot gap the re-entry buffer tests."""

    TWO_SIDED = 'two-sided'
    REENTRY_ONLY = 'reentry-only'


class ContactCandidate(NamedTuple):
    """One raw contact candidate and its independent gate/suppression verdicts."""

    rally_id: int
    contact_frame: int
    proximity_ok: bool | None
    wrist_near: bool | None
    suppressed: bool | None


class ServeStartConfig(NamedTuple):
    """Policy-only serve-start request; ``threshold_bh`` is a body-height multiple, the sticky lane's only unit."""

    threshold_bh: float
    mode: 'ServeStartMode'
    close: 'ServeStartClose | None' = None
    stillness_threshold_bh: float | None = None


class Slot(IntEnum):
    """Storage-row indices pinned to sticky_anchor's public constants.

    ``SLOT_TOP = 0`` and ``SLOT_BOTTOM = 1`` let annotator code index sticky's
    per-slot arrays directly.
    Sticky's pick order is bottom-first and is deliberately not modelled here;
    enum definition and iteration order must never be read as pick order.
    """

    TOP = 0
    BOTTOM = 1
