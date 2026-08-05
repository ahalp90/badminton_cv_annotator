"""Shared annotator declarations for FPS scaling and storage slots.

This module also owns shuttle-track primitives and pose-array conventions used
by rally segmentation and point-winner attribution.
"""
from __future__ import annotations

import warnings
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .fps_constants import ScalingKind as ScalingKind

if TYPE_CHECKING:
    from .rally_segmentation import ServeStartClose, ServeStartMode


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


# ---------------------------------------------------------------------------
# Shared shuttle-track primitives (rally segmentation and replay masking both import these)
# ---------------------------------------------------------------------------
def compute_speed(track: np.ndarray) -> np.ndarray:
    """Per-frame shuttle speed, NaN where the step is not fully visible.

    Speed at frame i is the L2 displacement of `(x, y)` from frame i-1 to i.
    Frame 0 has no predecessor and both endpoint frames must have visibility 1,
    else the step is unmeasured and reads NaN (so nan-aware stats skip it).

    :param track: `(t, 3)` `[x_norm, y_norm, visibility]` whole-video track.
    :return: `(t,)` speed in norm-units/frame; NaN on frame 0 and on any step
        touching a non-visible frame.
    """
    xy = track[:, :2]  # (t, 2) normalised position
    visibility = track[:, 2]  # (t,)
    step = np.diff(xy, axis=0)  # (t-1, 2) frame i-1 -> i
    step_speed = np.linalg.norm(step, axis=1)  # (t-1,)
    both_visible = (visibility[:-1] == 1) & (visibility[1:] == 1)  # (t-1,) both ends of the step

    speed = np.full(len(track), np.nan)  # (t,) frame-indexed; frame 0 stays NaN
    speed[1:] = np.where(both_visible, step_speed, np.nan)
    return speed


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True in a boolean mask, as half-open `[start, end)` ranges.

    Vectorised via edge detection on the zero-padded int mask: +1 marks a run
    start, -1 marks one-past a run end. Shared with replay masking's court-absence
    signal, which masks whole absent runs.

    :param mask: `(t,)` boolean.
    :return: list of `(start, end)` with `mask[start:end]` all True.
    """
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])  # sentinels force edges at the ends
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median that ignores NaN, one value per input frame.

    Pads both ends with NaN so every frame gets a full-width window and the
    output keeps length t; nanmedian drops the pad and any NaN steps. Shared
    with replay masking's slow-motion signal.

    :param values: `(t,)` values, may contain NaN.
    :param window: window width in frames.
    :return: `(t,)` centred rolling median; NaN only where a whole window is NaN.
    """
    left = window // 2
    right = window - 1 - left
    padded = np.concatenate([np.full(left, np.nan), values, np.full(right, np.nan)])
    windows = sliding_window_view(padded, window)  # (t, window)
    with warnings.catch_warnings():
        # An all-NaN window (e.g. a fully untracked span) is expected and yields
        # NaN by design; silence the RuntimeWarning rather than let it spam logs.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


# COCO wrist/ankle keypoint indices in the (t, n_max, 17, 2) pose keypoint arrays. Rally segmentation
# (rally_segmentation.py) and point_winner both read these for the sticky picker, attribution
# and landing kinematics.
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
