"""Stage 8: rally segmentation and contact detection (scraper_spec.md section 6).

Trajectory rules over a whole-video TrackNetV3 shuttle track, the `(t, 3)`
`[x_norm, y_norm, visibility]` npy that `shuttle_extractor.py:244-249` writes
(x, y already normalised to [0, 1] by video resolution, visibility passed
through). Speed everywhere below is per-frame L2 displacement of `(x, y)` on
frames where visibility is 1.

Three primitives (`compute_speed`, `true_runs`, `rolling_nanmedian`) are
re-exported from `annotator.types` because stage 9 reuses them. Its slow-motion
signal is defined against the same per-frame speed, so re-deriving it there
would be a second source of truth. All per-frame arrays here share one
frame-index space `[0, t)`; that invariant is what lets rally spans, contacts
and masks line up downstream.

`segment_video` takes four off-by-default keyword options that each preserve
today's behaviour exactly when left at their default:
  - `thresholds`: a `Stage8Thresholds` preset used instead of the module globals;
    None reads the globals through the low-level opt-out path.
  - `serve_start`: `ServeStartOptions` gating rally openings on a serve-setup
    lookback (the shuttle sitting near a court-scale player through the last
    second before the burst).
  - `span_open`: a `SpanOpen` rule that changes where a span opens (region start
    vs the qualifying burst).
  - `replay_mask`: a `(t,)` bool dead-time mask applied at entry (via
    `apply_replay_mask`), freezing replay/off-rally frames to rest before speed.

The contact chain keeps raw impulse candidates, applies a body-unit wrist gate,
then applies greedy suppression. The three operations are separate helpers and
`segment_video` composes them in that order.

Run as `python -m annotator.rally_segmentation --shuttle-dir ...` with
PYTHONPATH=src.
"""
import argparse
import csv
import logging
import sys
import warnings
from enum import StrEnum
from numbers import Real
from pathlib import Path
from typing import Mapping, NamedTuple

import numpy as np

from .config import (
    BaseAnnotatorConfig,
    CONTACT_FRAMES_CSV,
    END_REST_FRAMES,
    PROXIMITY_MAX,
    RALLY_SPANS_CSV,
    REST_SPEED,
    REST_WINDOW,
    SMOOTH_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
    Stage8Thresholds,
)
from .doubles_flag import read_whole_video_flags
from .fps_constants import FpsConstants, scale_for_fps
from .types import (
    ANKLE_L,
    ANKLE_R,
    ContactCandidate,
    ReentryGuardVariant,
    Slot,
    SmoothingMode,
    SpanOpen,
    StickyResult,
    WRIST_L,
    WRIST_R,
    compute_speed,
    rolling_nanmedian,
    true_runs,
)

# sticky_anchor is part of BST-X, not the scraper package. Keep the import seam
# at the package boundary so the picker remains the single implementation.
_BST_X_ROOT = Path(__file__).resolve().parents[1] / 'bst_x'
if str(_BST_X_ROOT) not in sys.path:
    sys.path.insert(0, str(_BST_X_ROOT))

from preparing_data.heuristics import sticky_anchor  # noqa: E402
from preparing_data.heuristics.base import ClipContext, RawClip  # noqa: E402

log = logging.getLogger(__name__)

# Fraction of a window that must be tracked (visibility 1) for the window to
# read as "seeing the shuttle". Below this the window is mostly untracked and
# counts as rest (spec s6 "visibility mostly 0 across the window"). Not in
# config: it is the numeric reading of "mostly", not a swept rule threshold.
VISIBILITY_REST_FRAC = 0.5
QUIET_START_REST_FRACTION = 0.8

# Contact-chain constants: the base-30 table in fps_constants.py scaled once
# to the 25 fps surface these module defaults serve.
IMPULSE_FLOOR_HALF_WINDOW_FRAMES = scale_for_fps(25.0).impulse_floor_half_window_frames
CONTACT_DEDUP_RADIUS_FRAMES = scale_for_fps(25.0).contact_dedup_radius_frames
CONTACT_IMPULSE_MULTIPLE = 4.0
FLOOR_EPS = 1e-4
BODY_UNIT_WRIST_THRESHOLD = 1.4
CONTACT_SUPPRESSION_RADIUS_FRAMES = scale_for_fps(25.0).contact_suppression_radius_frames


def scale_thresholds(
    thresholds: Stage8Thresholds, fps: float, *,
    constants: FpsConstants | None = None, overrides_base30: Mapping[str, float] | None = None,
) -> Stage8Thresholds:
    """Replace a preset's fps-dependent fields from the base-30 table; the preset
    contributes only its non-fps fields. Returned fields are final.
    """
    values = scale_for_fps(fps) if constants is None else constants
    overrides = {} if overrides_base30 is None else overrides_base30
    return thresholds._replace(
        rest_speed=values.rest_speed, rest_window=values.rest_window,
        start_speed=values.start_speed, start_min_frames=values.start_min_frames,
        smooth_window=values.smooth_window, end_rest_frames=values.end_rest_frames,
        impulse_floor_half_window_frames=values.impulse_floor_half_window_frames,
        contact_dedup_radius_frames=values.contact_dedup_radius_frames,
        contact_suppression_radius_frames=values.contact_suppression_radius_frames,
        contact_impulse_multiple=overrides.get('contact_impulse_multiple', thresholds.contact_impulse_multiple),
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred zero-inclusive rolling mean with a shrinking edge window.

    This is a plain mean: zeros go in like any other value, nothing is
    excluded, and any masking is the caller's job.

    :param values: `(t,)` values, no NaN.
    :param window: window width in frames.
    :return: `(t,)` centred mean; edge frames average their partial window.
    """
    kernel = np.ones(window)
    counts = np.convolve(np.ones_like(values), kernel, mode='same')  # samples per position
    sums = np.convolve(values, kernel, mode='same')
    return sums / counts


def _nan_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that ignores NaN with a shrinking edge window.

    :param values: `(t,)` values, may contain NaN.
    :param window: window width in frames.
    :return: `(t,)` centred mean; NaN where a whole window is NaN.
    """
    kernel = np.ones(window)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    counts = np.convolve(valid.astype(float), kernel, mode='same')
    sums = np.convolve(filled, kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums / counts


# ---------------------------------------------------------------------------
# Selectable-option types (off by default; see segment_video)
# ---------------------------------------------------------------------------
class CourtGeo(NamedTuple):
    """A court geometry used to filter person detections.

    The caller constructs it from tracked calibration geometry. Feet inside `net_band` claim
    NEITHER court half because the net line carries 3D model error.

    :param x_range: foot-point x bounds, pixels.
    :param y_range: foot-point y bounds, pixels.
    :param net_band: top/bottom half-split band (low, high), pixels.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    net_band: tuple[float, float]


class ServeSetupInputs(NamedTuple):
    """Per-frame evidence for the body-height-unit serve-setup gate.

    Ankles are image-fraction centroids and heights are image-HEIGHT fractions.
    ``wrist_dist`` is the raw wrist-to-shuttle distance as an image-height
    fraction, with ``+inf`` outside analysed coverage and NaN when a frame or
    slot has no measured distance. Dividing the raw pixel distance by the same
    image height used for ``top_height`` and ``bot_height`` makes the gate's
    ratio cancel exactly to body heights, with no hidden normalisation step.
    """

    count: np.ndarray
    wrist_dist: np.ndarray
    analysed: np.ndarray
    top_ankles: np.ndarray
    bot_ankles: np.ndarray
    top_height: np.ndarray
    bot_height: np.ndarray

    def validate(self) -> None:
        """Validate shapes, dtypes, and the count contract."""
        arrays = {name: np.asarray(value) for name, value in self._asdict().items()}
        trailing_shapes = {
            'count': (), 'wrist_dist': (2,), 'analysed': (),
            'top_ankles': (2,), 'bot_ankles': (2,),
            'top_height': (), 'bot_height': (),
        }
        for name, trailing in trailing_shapes.items():
            value = arrays[name]
            if value.ndim != 1 + len(trailing) or value.shape[1:] != trailing:
                raise ValueError(f'{name} has wrong shape/rank')
        if len({value.shape[0] for value in arrays.values()}) != 1:
            raise ValueError('ServeSetupInputs fields must have equal first-axis length')

        count = arrays['count']
        if (np.issubdtype(count.dtype, np.bool_) or
                not np.issubdtype(count.dtype, np.number) or
                np.issubdtype(count.dtype, np.complexfloating)):
            raise ValueError('count must be numeric real values')
        if not np.all(np.isfinite(count)) or np.any(count < 0) or np.any(count != np.floor(count)):
            raise ValueError('count must be finite, nonnegative, integer-valued reals')
        if not np.issubdtype(arrays['analysed'].dtype, np.bool_):
            raise ValueError('analysed must have boolean dtype')
        for name in ('wrist_dist', 'top_ankles', 'bot_ankles', 'top_height', 'bot_height'):
            if not np.issubdtype(arrays[name].dtype, np.floating):
                raise ValueError(f'{name} must have floating-point dtype')


def series_drift(points: np.ndarray) -> tuple[float, int]:
    """Return median-half drift for a sentinel-coded point series.

    ``points`` may contain NaN rows and the paired-zero ``(0, 0)`` sentinel;
    it is not suitable for arbitrary geometry where the origin is meaningful.
    Both coordinates must be finite and the pair must not be zero, while
    ``(0, y)`` and ``(x, 0)`` remain detected.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError('points must have shape (n, 2)')
    if (np.issubdtype(points.dtype, np.bool_) or
            not np.issubdtype(points.dtype, np.number) or
            np.issubdtype(points.dtype, np.complexfloating)):
        raise ValueError('points must have real numeric dtype')
    detected = np.all(np.isfinite(points), axis=1) & np.any(points != 0, axis=1)
    points = points[detected]
    detected_count = len(points)
    if detected_count < 2:
        return float('nan'), detected_count
    split = (detected_count + 1) // 2
    first = np.median(points[:split], axis=0)
    second = np.median(points[split:], axis=0)
    return float(np.linalg.norm(second - first)), detected_count


# Presence floor per required player in the serve-setup gate.
PLAYER_PRESENT_MIN_FRAC = 0.5


def serve_setup_still(
    inputs: ServeSetupInputs,
    claimed_serve_frame: int,
    window_frames: int,
    threshold_bh: float,
    slots: tuple[Slot, ...],
) -> bool:
    """Return whether every requested player is still through the serve frame."""
    inputs.validate()
    if isinstance(window_frames, bool) or not isinstance(window_frames, (int, np.integer)) or window_frames <= 0:
        raise ValueError('window_frames must be a positive integer')
    t = len(inputs.count)
    if isinstance(claimed_serve_frame, bool) or not isinstance(claimed_serve_frame, (int, np.integer)):
        raise ValueError('claimed_serve_frame must be an integer in range [0, t)')
    if not 0 <= claimed_serve_frame < t:
        raise ValueError('claimed_serve_frame must be in range [0, t)')
    if not np.isfinite(threshold_bh) or threshold_bh < 0:
        raise ValueError('threshold_bh must be finite and nonnegative')
    # Element check before set(): an unhashable member must ValueError, not TypeError.
    if (not isinstance(slots, tuple) or not slots or
            any(not isinstance(slot, Slot) for slot in slots) or
            len(set(slots)) != len(slots)):
        raise ValueError('slots must be a nonempty duplicate-free tuple of Slot values')

    end = int(claimed_serve_frame) + 1
    window = slice(max(0, end - int(window_frames)), end)
    for slot in slots:
        ankles = inputs.top_ankles[window] if slot is Slot.TOP else inputs.bot_ankles[window]
        heights = inputs.top_height[window] if slot is Slot.TOP else inputs.bot_height[window]
        drift, _ = series_drift(ankles)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            body_unit = float(np.nanmean(heights))
        if not np.isfinite(body_unit) or body_unit <= 0:
            return False
        ratio = drift / body_unit
        if not np.isfinite(ratio) or ratio > threshold_bh:
            return False
    return True


def build_serve_setup_inputs(
    sticky: 'StickyResult', resolution: tuple[float, float],
) -> ServeSetupInputs:
    """Build validated sticky-sourced evidence for the serve-setup gate."""
    if (not isinstance(resolution, tuple) or len(resolution) != 2 or
            any(isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or value <= 0 for value in resolution)):
        raise ValueError('resolution must be two finite positive real components')

    height = float(resolution[1])
    count = np.asarray(sticky.standing_count).copy()
    wrist_dist = np.asarray(sticky.wrist_dist_px, dtype=np.float64).copy() / height
    analysed = np.asarray(sticky.analysed, dtype=bool).copy()
    top_ankles = np.full_like(sticky.ankle_pos[:, Slot.TOP], np.nan, dtype=float)
    bot_ankles = np.full_like(sticky.ankle_pos[:, Slot.BOTTOM], np.nan, dtype=float)
    top_height = np.full(len(count), np.nan, dtype=float)
    bot_height = np.full(len(count), np.nan, dtype=float)

    for slot, ankles_out, heights_out in (
        (Slot.TOP, top_ankles, top_height), (Slot.BOTTOM, bot_ankles, bot_height),
    ):
        ankles = sticky.ankle_pos[:, slot]
        box_height = sticky.bbox_height[:, slot]
        ankle_valid = np.all(np.isfinite(ankles), axis=1) & np.any(ankles != 0, axis=1)
        height_valid = np.isfinite(box_height) & (box_height > 0)
        ankles_out[ankle_valid] = ankles[ankle_valid]
        heights_out[height_valid] = box_height[height_valid] / height

    inputs = ServeSetupInputs(
        count=count, wrist_dist=wrist_dist, analysed=analysed,
        top_ankles=top_ankles, bot_ankles=bot_ankles,
        top_height=top_height, bot_height=bot_height,
    )
    inputs.validate()
    return inputs


class ServeStartMode(StrEnum):
    """What a region whose bursts are none of them serve-setup-preceded does.

    TRIM keeps the span at the region's first burst (the stock pick), so coverage is only
    ever traded by a later start on the QUALIFYING regions. REJECT drops the region outright,
    the stronger anti-weld / anti-spurious lever, at the risk of dropping a GT rally that
    happens to sit in a no-qualify region.
    """

    TRIM = 'trim'
    REJECT = 'reject'


class ServeStartClose(StrEnum):
    """Where a split span closes; the serve-start split axis (None = single span, off).

    In split mode every serve-setup-qualifying burst opens a span. BURST closes the previous
    span exactly at the next qualifying burst, so the split spans union back to the single
    span and coverage is unchanged. LAST_REST closes it at the start of the last rest run
    before that burst, leaving the between-rally dead tail (where the junk contacts live)
    outside the span, at the risk of uncovering a rally whose own serve failed the gate.
    """

    BURST = 'burst'
    LAST_REST = 'last_rest'


class ServeStartOptions(NamedTuple):
    """Serve-start gating for segment_video(serve_start=...).

    The sticky setup evidence is built by the caller from the unmasked track. The committed
    measurement convention builds serve-start evidence before any replay mask is applied.

    :param dist: retired raw-distance carrier. It must remain None.
    :param threshold: serve-setup gate distance as a multiple of body height.
    :param mode: fallback for a region with no qualifying burst (TRIM / REJECT).
    :param close: optional split placement; None opens one span per region (the default).
    :param diagnostics: optional caller-supplied dict; when given, the span rule fills it in
        place with the per-call region counts / spacings (single writer, valid to read straight
        after the call IN THE SAME PROCESS: the in-place fill does not cross a multiprocessing
        worker boundary, so the pooled sweep runner leaves it None). None (the default) collects nothing.
    :param setup: sticky-sourced evidence (build_serve_setup_inputs).
    :param stillness_threshold_bh: optional stillness bound in body heights for the sticky
        gate; None (the default) leaves the stillness check off. Sticky path only.
    :param lookback_frames: resolved setup-window length in frames (the fps table's
        serve_start_lookback_frames row); required with setup. Sticky path only.
    :param stillness_window_frames: resolved stillness-window length in frames (the
        serve_stillness_window_frames row); required once stillness_threshold_bh is set.
        Sticky path only.
    """

    dist: np.ndarray | None
    threshold: float
    mode: ServeStartMode
    close: ServeStartClose | None = None
    diagnostics: dict | None = None
    setup: ServeSetupInputs | None = None
    stillness_threshold_bh: float | None = None
    lookback_frames: int | None = None
    stillness_window_frames: int | None = None


# Stage 6 bridge fixed body-height window. The resolved path supplies this explicitly.
BODY_UNIT_HALF_WINDOW = 12


# ---------------------------------------------------------------------------
# Rally spans
# ---------------------------------------------------------------------------
def _gap_is_high_shot_oob(track: np.ndarray, gap_start: int, constants: FpsConstants) -> bool:
    run_start = gap_start
    while (run_start > 0 and track[run_start - 1, 2] == 1
           and gap_start - run_start < constants.high_shot_oob_lookback_frames):
        run_start -= 1
    n_visible = gap_start - run_start
    if n_visible < constants.high_shot_oob_min_visible_frames:
        return False
    first_xy = track[run_start, :2]
    last_xy = track[gap_start - 1, :2]
    mean_velocity = (last_xy - first_xy) / (n_visible - 1)
    return bool(last_xy[1] + constants.high_shot_oob_extrap_frames * mean_velocity[1] < 0.0)


def _gap_passes_reentry_guard(
    track: np.ndarray, gap_start: int, gap_end: int, variant: ReentryGuardVariant, buffer: float,
    constants: FpsConstants,
) -> bool:
    if gap_end >= len(track):
        return True
    stop = gap_end
    limit = min(gap_end + constants.reentry_lookahead_frames, len(track))
    while stop < limit and track[stop, 2] == 1:
        stop += 1
    n_visible = stop - gap_end
    if n_visible < constants.reentry_min_visible_frames:
        return False
    descending = (track[stop - 1, 1] - track[gap_end, 1]) / (n_visible - 1) > 0.0
    near_top = track[gap_end, 1] <= buffer
    if variant is ReentryGuardVariant.TWO_SIDED:
        return bool(track[gap_start - 1, 1] <= buffer and near_top and descending)
    return bool(near_top and descending)


def _gap_state_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Stage8Thresholds, constants: FpsConstants, demotion_bound: int,
    reentry_guard_variant: ReentryGuardVariant | None, reentry_guard_buffer: float | None,
) -> np.ndarray:
    speed_median = rolling_nanmedian(speed, thresholds.rest_window)
    slow = speed_median < thresholds.rest_speed
    high_shot_oob = np.zeros(len(track), dtype=bool)
    dead = np.zeros(len(track), dtype=bool)
    for gap_start, gap_end in true_runs(track[:, 2] != 1):
        holds_open = _gap_is_high_shot_oob(track, gap_start, constants)
        if holds_open and reentry_guard_variant is not None:
            assert reentry_guard_buffer is not None
            holds_open = _gap_passes_reentry_guard(
                track, gap_start, gap_end, reentry_guard_variant, reentry_guard_buffer, constants,
            )
        if holds_open:
            demotion_frame = min(gap_start + demotion_bound, gap_end)
            high_shot_oob[gap_start:demotion_frame] = True
            dead[demotion_frame:gap_end] = True
        elif gap_end - gap_start > constants.blip_max_frames:
            dead[gap_start:gap_end] = True
    return dead | (slow & ~high_shot_oob)


def _rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Stage8Thresholds | None = None, *,
    constants: FpsConstants | None = None, gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None, reentry_guard_buffer: float | None = None,
) -> np.ndarray:
    """Per-frame rest flag: slow OR mostly untracked across the window (spec s6).

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param track: `(t, 3)` track, for the visibility column.
    :param thresholds: a preset to read rest_window / rest_speed from; None reads the
        module globals through the low-level opt-out path.
    :return: `(t,)` bool, True where the frame reads as rest.
    """
    if gap_state_demotion_bound is not None:
        assert thresholds is not None and constants is not None
        return _gap_state_rest_mask(
            speed, track, thresholds, constants, gap_state_demotion_bound,
            reentry_guard_variant, reentry_guard_buffer,
        )
    rest_window = REST_WINDOW if thresholds is None else thresholds.rest_window
    rest_speed = REST_SPEED if thresholds is None else thresholds.rest_speed
    speed_median = rolling_nanmedian(speed, rest_window)  # (t,)
    slow = speed_median < rest_speed  # NaN windows read not-slow here...
    visible = (track[:, 2] == 1).astype(float)  # (t,) 1.0 where tracked
    frac_visible = _rolling_mean(visible, rest_window)  # (t,) fraction tracked in window
    mostly_untracked = frac_visible < VISIBILITY_REST_FRAC  # ...and the OR below catches them
    return slow | mostly_untracked


def _rally_regions(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Stage8Thresholds | None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Shared region scaffold for the span-opening rules.

    Reads start_speed / end_rest_frames / start_min_frames from `thresholds` (or the
    module globals when None), exactly as `_find_rally_spans` does, so every opening
    rule sees the same fast runs and active regions.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: a preset, or None to read the module globals.
    :return: `(fast_runs, rest_runs, regions)`. fast_runs are the sustained-fast runs (>=
        start_min_frames above start_speed); rest_runs are every at_rest run; regions are the
        maximal non-long-rest runs (a long rest is a rest run >= end_rest_frames).
    """
    start_speed = START_SPEED if thresholds is None else thresholds.start_speed
    end_rest_frames = END_REST_FRAMES if thresholds is None else thresholds.end_rest_frames
    start_min_frames = START_MIN_FRAMES if thresholds is None else thresholds.start_min_frames

    fast = np.nan_to_num(speed, nan=0.0) > start_speed  # (t,) NaN steps are not fast
    rest_runs = true_runs(at_rest)
    long_rest = np.zeros(len(speed), dtype=bool)  # (t,) frames inside an extended rest
    for start, end in rest_runs:
        if end - start >= end_rest_frames:
            long_rest[start:end] = True
    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= start_min_frames]
    regions = true_runs(~long_rest)
    return fast_runs, rest_runs, regions


def _find_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Stage8Thresholds | None = None,
) -> list[tuple[int, int]]:
    """Segment the video into rally spans between extended rest.

    A long rest (a rest run >= END_REST_FRAMES) separates rallies. Inside each
    stretch of non-long-rest frames, the rally starts at the first sustained
    burst of fast frames (START_MIN_FRAMES consecutive above START_SPEED, the
    acceleration-from-rest signature) and ends where the following long rest
    begins. A stretch with no such burst (e.g. a brief tracker twitch) yields
    no rally.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: a preset to read the boundary thresholds from; None reads the
        module globals through the low-level opt-out path.
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    fast_runs, _rest_runs, regions = _rally_regions(speed, at_rest, thresholds)

    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        # The first qualifying fast run that opens inside this active region is
        # the acceleration out of the preceding rest; the region's end is the
        # onset of the next extended rest (or the video end).
        burst_start = next(
            (start for start, _ in fast_runs if region_start <= start < region_end),
            None,
        )
        if burst_start is None:
            continue
        spans.append((int(burst_start), int(region_end)))
    return spans


def _find_rally_spans_span_open(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Stage8Thresholds | None, span_open: SpanOpen,
) -> list[tuple[int, int]]:
    """Span finder under a SpanOpen rule (no serve gating).

    REGION_START drops the qualifying-burst gate: every active region yields one span
    `(region_start, region_end)`. BACK_FILL keeps the gate (a region needs a qualifying
    fast burst) but opens the span at region_start instead of at the burst.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: a preset, or None to read the module globals.
    :param span_open: REGION_START or BACK_FILL.
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    fast_runs, _rest_runs, regions = _rally_regions(speed, at_rest, thresholds)
    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        if span_open is SpanOpen.REGION_START:
            spans.append((int(region_start), int(region_end)))
            continue
        # BACK_FILL: the region still has to carry a qualifying burst to count as a rally.
        has_burst = any(region_start <= start < region_end for start, _ in fast_runs)
        if has_burst:
            spans.append((int(region_start), int(region_end)))
    return spans


def _find_rally_spans_quiet_start(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Stage8Thresholds, window: int,
) -> list[tuple[int, int]]:
    fast_runs, _rest_runs, regions = _rally_regions(speed, at_rest, thresholds)
    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        bursts = [start for start, _end in fast_runs if region_start <= start < region_end]
        if not bursts:
            continue
        quiet_burst = next(
            (start for start in bursts if len(at_rest[max(0, start - window):start])
             and at_rest[max(0, start - window):start].mean() >= QUIET_START_REST_FRACTION),
            None,
        )
        spans.append((int(bursts[0] if quiet_burst is None else quiet_burst), int(region_end)))
    return spans


def court_scale_slots(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray, court_geo: CourtGeo,
) -> np.ndarray:
    """Original pose-slot indices of the court-scale detections, ascending.

    The filter returns original pose-slot identities rather than recovering
    them through score equality, which can alias detections with tied scores.

    :param frame_bboxes: (16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param frame_scores: (16,) detection scores, NaN on padding slots.
    :param court_geo: the court geometry to filter against.
    :return: (k,) int slot indices into the frame's pose arrays.
    """
    valid = np.isfinite(frame_scores)
    x1, y1, x2, y2 = frame_bboxes[valid].T  # each (m,) pixels
    foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
    x_lo, x_hi = court_geo.x_range
    y_lo, y_hi = court_geo.y_range
    in_court = (x_lo <= foot_x) & (foot_x <= x_hi) & (y_lo <= y2) & (y2 <= y_hi)
    return np.flatnonzero(valid)[in_court]


def _serve_distance_ratio_passes(
    window_dist: np.ndarray, window_height: np.ndarray, threshold_bh: float,
) -> bool:
    """Return whether paired distance evidence passes the body-height threshold.

    The finite-distance mask selects the matching heights. Sticky setup construction writes both
    values for a picked slot on the same frame.
    """
    mask = np.isfinite(window_dist)
    if not mask.any():
        return False
    ratio = np.median(window_dist[mask]) / np.mean(window_height[mask])
    return bool(ratio <= threshold_bh)


def _last_rest_close(rest_runs: list[tuple[int, int]], open_frame: int, next_burst: int) -> int:
    """Where a split span closes under close='last_rest'.

    The START of the last at_rest run (any length) that ends at or before the next qualifying
    burst AND opens after this span's own open, so the dead tail between the previous rally's
    last action and the next serve falls outside the span (that tail is where the between-rally
    junk contacts sit). Falls back to next_burst when no rest run sits between the two opens.

    :param rest_runs: every `(start, end)` at_rest run, ascending (true_runs order).
    :param open_frame: this span's opening (qualifying) burst frame.
    :param next_burst: the next qualifying burst frame (where close='burst' would cut).
    :return: the close frame (half-open span end).
    """
    rest_starts = [rest_start for rest_start, rest_end in rest_runs
                   if rest_end <= next_burst and rest_start > open_frame]
    return rest_starts[-1] if rest_starts else next_burst  # ascending, so [-1] is the last run


def _valid_serve_window(value: object, name: str) -> int:
    """Return one positive integer window length."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _valid_serve_threshold(value: object, name: str) -> float:
    """Return one finite nonnegative serve threshold."""
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _sticky_serve_setup_before(
    setup: ServeSetupInputs, burst: int, threshold: float, lookback_frames: int,
    stillness_threshold_bh: float | None, stillness_window_frames: int | None,
) -> bool:
    """Apply the sticky coverage gate and its three serve-setup lanes."""
    if not 0 <= burst < len(setup.count):
        raise ValueError('claimed_serve_frame must be in range [0, t)')
    setup_window = slice(max(0, burst - lookback_frames), burst)
    if setup_window.start == setup_window.stop:
        return False
    if stillness_threshold_bh is None:
        stillness_window = setup_window
    else:
        assert stillness_window_frames is not None
        stillness_window = slice(max(0, burst + 1 - stillness_window_frames), burst + 1)
    coverage_start = min(setup_window.start, stillness_window.start)
    coverage_stop = max(setup_window.stop, stillness_window.stop)
    if not np.all(setup.analysed[coverage_start:coverage_stop]):
        return False

    count = setup.count[setup_window]
    median_count = float(np.median(count))
    distances = setup.wrist_dist[setup_window]
    heights = (setup.top_height[setup_window], setup.bot_height[setup_window])
    valid = np.empty(distances.shape, dtype=bool)
    for slot in Slot:
        valid[:, slot] = np.isfinite(distances[:, slot]) & np.isfinite(heights[slot])

    if median_count >= 2:
        # Presence floor AND the primitive's minimum detections: below two valid
        # rows a drift cannot be split into halves, so the window fails closed
        # even with the stillness gate off.
        if any(np.mean(valid[:, slot]) < PLAYER_PRESENT_MIN_FRAC or
               np.count_nonzero(valid[:, slot]) < 2 for slot in Slot):
            return False
        if not any(
            _serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold)
            for slot in Slot
        ):
            return False
        return stillness_threshold_bh is None or serve_setup_still(
            setup, burst, stillness_window_frames, stillness_threshold_bh, (Slot.TOP, Slot.BOTTOM),
        )

    if median_count >= 1:
        for slot in Slot:
            slot_valid = valid[:, slot]
            if np.mean(slot_valid) < PLAYER_PRESENT_MIN_FRAC or np.count_nonzero(slot_valid) < 2:
                continue
            if not _serve_distance_ratio_passes(distances[:, slot], heights[slot], threshold):
                continue
            if stillness_threshold_bh is None:
                return True
            masked = setup._replace(
                top_ankles=setup.top_ankles.copy(), bot_ankles=setup.bot_ankles.copy(),
                top_height=setup.top_height.copy(), bot_height=setup.bot_height.copy(),
            )
            ankles_out = masked.top_ankles if slot is Slot.TOP else masked.bot_ankles
            heights_out = masked.top_height if slot is Slot.TOP else masked.bot_height
            stillness_distances = setup.wrist_dist[stillness_window, slot]
            stillness_ankles = ankles_out[stillness_window]
            stillness_heights = heights_out[stillness_window]
            stillness_valid = (
                np.isfinite(stillness_distances) & np.all(np.isfinite(stillness_ankles), axis=1) &
                np.isfinite(stillness_heights)
            )
            stillness_indexes = np.arange(stillness_window.start, stillness_window.stop)
            ankles_out[stillness_indexes[~stillness_valid]] = np.nan
            heights_out[stillness_indexes[~stillness_valid]] = np.nan
            if serve_setup_still(masked, burst, stillness_window_frames, stillness_threshold_bh, (slot,)):
                return True
        return False

    # TODO: position-based count means a tight close-up CAN read 1 and route partial; the old
    # "close-ups read zero" claim was size-banded and does not describe the permanent
    # court-scale box and slot paths.
    return False


def _serve_start_find_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Stage8Thresholds | None,
    options: ServeStartOptions, span_open: SpanOpen | None,
) -> list[tuple[int, int]]:
    """Span finder that opens only at a serve-setup-preceded burst.

    Same region / long-rest / fast-run structure as the stock finder. The change: a rally
    opens at a fast burst whose sticky setup lookback passes the serve-setup gate. A region with
    no qualifying burst is handled by the mode: TRIM falls back to the first burst (span survives
    at the stock start), REJECT drops it.

    `options.close` controls what a QUALIFYING region does when span_open is None (the default):
    None opens one span at the FIRST qualifying burst running to region end; BURST / LAST_REST
    open a span at EVERY qualifying burst. span_open=BACK_FILL back-fills a qualifying region to
    a single span opening at region_start (serve-start with span_open=REGION_START is rejected
    by segment_video before this is reached).

    :param speed: (t,) per-frame speed (NaN on non-visible steps).
    :param at_rest: (t,) per-frame rest flag.
    :param thresholds: a preset, or None to read the module globals.
    :param options: the serve-start gate inputs and mode.
    :param span_open: None (burst-open) or BACK_FILL (open qualifying regions at region_start).
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    if options.dist is not None:
        raise ValueError('legacy serve-start dist is no longer supported; supply setup')
    if options.setup is None:
        raise ValueError('serve-start setup must be supplied')
    threshold = _valid_serve_threshold(options.threshold, 'threshold')
    lookback_frames = _valid_serve_window(options.lookback_frames, 'lookback_frames')
    if options.stillness_threshold_bh is not None:
        stillness_threshold_bh = _valid_serve_threshold(
            options.stillness_threshold_bh, 'stillness_threshold_bh',
        )
        stillness_window_frames = _valid_serve_window(
            options.stillness_window_frames, 'stillness_window_frames',
        )
    else:
        stillness_threshold_bh = None
        # Still validated when supplied: a bad window would silently wrap the
        # coverage-gate slice even with the stillness gate off.
        stillness_window_frames = (
            None if options.stillness_window_frames is None
            else _valid_serve_window(options.stillness_window_frames, 'stillness_window_frames')
        )
    options.setup.validate()

    def qualifies(burst: int) -> bool:
        return _sticky_serve_setup_before(
            options.setup, burst, threshold, lookback_frames,
            stillness_threshold_bh, stillness_window_frames,
        )

    mode = options.mode
    close = options.close

    fast_runs, rest_runs, regions = _rally_regions(speed, at_rest, thresholds)

    spans: list[tuple[int, int]] = []
    no_qualify_regions: list[tuple[int, int]] = []
    n_regions_with_burst = 0
    # Per-region qualifying-burst counts (spans per region under split) and the frames between
    # consecutive qualifying bursts (the double-fire read: a small spacing means a second serve
    # signature fired just after a span opened, cutting one rally in two). Diagnosed, not suppressed.
    qualifying_counts: list[int] = []
    qualifying_spacings: list[int] = []
    for region_start, region_end in regions:
        bursts = [start for start, _ in fast_runs if region_start <= start < region_end]
        if not bursts:
            continue  # no burst at all: stock forms no span here either, not a serve-start drop
        n_regions_with_burst += 1
        # Gate every burst, not just up to the first: the split modes open a span at each, and
        # the double-fire diagnostics need the whole per-region qualifying picture regardless.
        qualifying = [start for start in bursts if qualifies(start)]
        qualifying_counts.append(len(qualifying))
        qualifying_spacings.extend(int(later - earlier)
                                   for earlier, later in zip(qualifying, qualifying[1:]))

        if not qualifying:
            # No qualifying burst: the mode owns the region. TRIM keeps the span at the stock
            # first burst; REJECT drops it. Both record the region as no-qualify.
            if mode is ServeStartMode.TRIM:
                spans.append((int(bursts[0]), int(region_end)))
            no_qualify_regions.append((int(region_start), int(region_end)))
        elif span_open is SpanOpen.BACK_FILL:
            # Serve gate decides qualification; the span back-fills to the region start.
            spans.append((int(region_start), int(region_end)))
        elif close is None:
            # Split off: one span at the first qualifying burst, running to region end.
            spans.append((int(qualifying[0]), int(region_end)))
        else:
            # Split on: every qualifying burst opens a span, closing where the next one opens
            # (close='burst') or at the last rest run before it (close='last_rest'); the last
            # span runs to region end. Under close='burst' the spans union to the single span.
            for idx, open_frame in enumerate(qualifying):
                if idx + 1 < len(qualifying):
                    next_burst = qualifying[idx + 1]
                    close_frame = (next_burst if close is ServeStartClose.BURST
                                   else _last_rest_close(rest_runs, open_frame, next_burst))
                else:
                    close_frame = region_end
                spans.append((int(open_frame), int(close_frame)))

    if options.diagnostics is not None:
        options.diagnostics.clear()
        options.diagnostics.update({
            'n_regions_with_burst': n_regions_with_burst,
            'n_qualified': n_regions_with_burst - len(no_qualify_regions),
            'n_no_qualify': len(no_qualify_regions),
            'no_qualify_regions': no_qualify_regions,
            'qualifying_counts': qualifying_counts,
            'qualifying_spacings': qualifying_spacings,
        })
    return spans


# ---------------------------------------------------------------------------
# Replay / dead-time mask
# ---------------------------------------------------------------------------
def apply_replay_mask(track: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Freeze replay/off-rally frames to the last live position so they read as rest.

    Returns a new `(t, 3)` array; `track` is not mutated. For each contiguous True run in
    `mask`, the run's xy (columns 0-1) is set to the xy of the last frame BEFORE the run, and
    its visibility (column 2) forced to 1. A run that starts at frame 0 has no earlier frame,
    so it takes the xy of the first frame AFTER it instead.

    Why: stage 8 reads invisible or NaN-speed frames as not-rest, so replay closeups hold rally
    regions open. Freezing the position makes masked footage read as sustained sub-REST_SPEED
    rest (masked frames count as rest). Forcing visibility avoids the NaN-speed path reopening
    the region.

    Fail loud on a length mismatch, and on an all-True mask (nothing live to anchor a frozen
    position to, and a fully-masked video is senseless). An all-False mask has no True runs, so
    the untouched copy returns bit-identical by construction.

    :param track: `(t, 3)` `[x_norm, y_norm, visibility]` whole-video track.
    :param mask: `(t,)` bool, True on replay/off-rally frames (stage-9 `1_replay.npy` convention).
    :return: a new `(t, 3)` track with masked frames frozen to rest.
    """
    if len(mask) != len(track):
        raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
    if mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')

    frozen = track.copy()
    for start, end in true_runs(mask):
        # start-1 is the last live frame before the run; a run at frame 0 has none, so anchor to
        # end (the first live frame after it). The not-all-True guard above guarantees it exists.
        anchor = start - 1 if start > 0 else end
        frozen[start:end, :2] = track[anchor, :2]
        frozen[start:end, 2] = 1
    return frozen


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------
# The rule this chain replaced (measured s25-s26; kept here as context and as the fallback):
# a junction was a contact when the direction changed by over 30 degrees AND both smoothed
# segment speeds exceeded 0.005 (25 fps per-frame units), dedup sharpest-angle-first; the
# wrist gate then kept contacts within 0.125 image-fractions of a wrist. End to end that read
# 60.9% recall at 54.0% precision (+/-10 frames); the impulse chain below reads 0.8296
# recall at 0.7120 precision (re-earned sticky table, m4/r9, +/-10). If the impulse chain
# ever needs retiring, the measured simple fallback is
# OR(angle > 60 with speeds > 0.0035) plus the body-unit gate: 73.8% recall / 55.3% precision.
def span_impulses(
    track: np.ndarray, start: int, end: int, thresholds: Stage8Thresholds | None = None, *,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> np.ndarray | None:
    """Return ``|v_out - v_in|`` for each junction in one rally span.

    The track is already replay-masked when this is called from ``segment_video``.
    Junction ``k`` sits at local frame ``k + 1`` and touches three straddling frames.
    Track xy inputs are finite. IGNORE_INVISIBLE makes a float copy whose invisible
    xy values are NaN. A NaN output means its smoothing window was unmeasurable;
    downstream comparisons drop the corresponding junction.
    """
    smooth_window = SMOOTH_WINDOW if thresholds is None else thresholds.smooth_window
    span = track[start:end]
    if len(span) < smooth_window + 2:
        return None

    if smoothing_mode is SmoothingMode.ZERO_FILL:
        smooth_x = _rolling_mean(span[:, 0], smooth_window)
        smooth_y = _rolling_mean(span[:, 1], smooth_window)
    else:
        smooth_xy = span[:, :2].astype(float, copy=True)
        smooth_xy[span[:, 2] != 1] = np.nan
        smooth_x = _nan_rolling_mean(smooth_xy[:, 0], smooth_window)
        smooth_y = _nan_rolling_mean(smooth_xy[:, 1], smooth_window)
    velocity = np.diff(np.column_stack([smooth_x, smooth_y]), axis=0)
    return np.linalg.norm(velocity[1:] - velocity[:-1], axis=1)


def rolling_floor(
    values: np.ndarray, around_visible: np.ndarray,
    half_window: int = IMPULSE_FLOOR_HALF_WINDOW_FRAMES,
) -> np.ndarray:
    """Median floor over visible junctions within a frame-count window.

    ``half_window`` counts junctions on each side at 25 fps. A junction with no
    visible neighbour in its window receives NaN and cannot pass the rule.
    """
    visible_values = np.where(around_visible, values, np.nan)
    floor = np.full(len(values), np.nan)
    for junction_index in range(len(values)):
        window_start = max(junction_index - half_window, 0)
        window_end = min(junction_index + half_window + 1, len(values))
        window = visible_values[window_start:window_end]
        if np.isfinite(window).any():
            floor[junction_index] = np.nanmedian(window)
    return floor


def impulse_cell_candidates(
    track: np.ndarray, start: int, end: int, thresholds: Stage8Thresholds | None = None, *,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> list[tuple[int, float]]:
    """Find raw impulse candidates and retain their impulse for suppression.

    The measured rule is pure impulse. It uses the three-frame visibility mask,
    a rolling impulse floor, and largest-impulse-first de-duplication at the
    three-frame boundary used at 25 fps.
    """
    span = track[start:end]
    impulses = span_impulses(track, start, end, thresholds, smoothing_mode=smoothing_mode)
    if impulses is None:
        return []

    around_visible = (
        (span[:-2, 2] == 1) & (span[1:-1, 2] == 1) & (span[2:, 2] == 1)
    )
    half_window = IMPULSE_FLOOR_HALF_WINDOW_FRAMES if thresholds is None else thresholds.impulse_floor_half_window_frames
    dedup_radius = CONTACT_DEDUP_RADIUS_FRAMES if thresholds is None else thresholds.contact_dedup_radius_frames
    floors = rolling_floor(impulses, around_visible, half_window)
    impulse_multiple = CONTACT_IMPULSE_MULTIPLE if thresholds is None else thresholds.contact_impulse_multiple
    impulse_pass = impulses / np.maximum(floors, FLOOR_EPS) > impulse_multiple
    is_contact = impulse_pass & around_visible

    candidate_local = np.flatnonzero(is_contact) + 1
    candidate_impulses = impulses[is_contact]
    kept: list[tuple[int, float]] = []
    # Stable sort: equal impulses keep the earlier frame, matching suppression's
    # (-impulse, frame) ordering. Exact ties occur in real data (sset_01 has one),
    # so an unstable sort here is a platform-dependent output.
    for candidate_index in np.argsort(-candidate_impulses, kind='stable'):
        local_frame = int(candidate_local[candidate_index])
        if all(
            abs(local_frame - other_frame) >= dedup_radius
            for other_frame, _other_impulse in kept
        ):
            kept.append((local_frame, float(candidate_impulses[candidate_index])))
    kept.sort()
    return [(start + local_frame, impulse) for local_frame, impulse in kept]


def detect_contact_flags(
    track: np.ndarray, start: int, end: int, thresholds: Stage8Thresholds | None = None, *,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> list[tuple[int, float]]:
    """Independently invoke the raw contact finder and retain ``(frame, impulse)`` flags."""
    return impulse_cell_candidates(track, start, end, thresholds, smoothing_mode=smoothing_mode)


def contact_proximity_ok(
    track: np.ndarray, positions: np.ndarray | None, contact_frame: int
) -> bool | None:
    """Guardrail: does a tracked player sit near the shuttle at the contact frame?

    Never filters a contact; it annotates one. When no positions were supplied
    the check is unmeasured, which returns None (serialised blank downstream):
    a guardrail with no evidence must not read as a pass.

    :param track: `(t, 3)` whole-video track.
    :param positions: `(t, 2, 2)` `[slot, xy]` court positions, or None.
    :param contact_frame: whole-video frame index of the contact.
    :return: True/False when measured, None when no positions were supplied.
    """
    if positions is None:
        return None
    shuttle_xy = track[contact_frame, :2]  # (2,)
    player_xy = positions[contact_frame]  # (2, 2) [slot, xy]
    distances = np.linalg.norm(player_xy - shuttle_xy, axis=1)  # (2,) per slot
    if np.all(np.isnan(distances)):
        # Positions exist but both slots failed this frame: measured, unconfirmed.
        return False
    return bool(np.nanmin(distances) <= PROXIMITY_MAX)


def wrist_contact_near(sticky_distances: np.ndarray | None, contact_frame: int) -> bool | None:
    """The single-frame body-unit wrist gate on one contact, in player-box-height units.

    Mirrors `contact_proximity_ok`'s three-way verdict. None distances mean the gate never
    ran (no body-unit or pose/court inputs), which returns None (serialised blank
    downstream): raw candidates stand, per the recall-first convention. A NaN frame is
    measured-but-unconfirmed and fails closed to False, the measured arm's behaviour.
    """
    if sticky_distances is None:
        return None
    distance = sticky_distances[contact_frame]
    return bool(np.isfinite(distance) and distance <= BODY_UNIT_WRIST_THRESHOLD)


def suppress_contact_flags(
    flags: list[tuple[int, float]],
    radius: int = CONTACT_SUPPRESSION_RADIUS_FRAMES,
) -> list[int]:
    """Greedy argmax suppression over ``(frame, impulse)`` flags.

    Flags are ranked by descending impulse and then ascending frame. A flag is
    accepted only when it is at least ``radius`` frames from every accepted flag.
    The default radius is the base-30 nine, eight frames at the 25 fps surface.
    """
    ordered = sorted(flags, key=lambda flag: (-flag[1], flag[0]))
    accepted: list[int] = []
    for frame, _impulse in ordered:
        if all(abs(frame - other) >= radius for other in accepted):
            accepted.append(frame)
    return sorted(accepted)


def tracker_segments(homography_rows, court_present, n_frames):
    """Return scene-row intervals intersected with court-present runs.

    :param homography_rows: Scene rows with named ``start_frame`` and ``end_frame`` bounds.
    :param court_present: `(n_frames,)` boolean court-detection mask.
    :param n_frames: Number of frames in the aligned video arrays.
    :return: Maximal court-present half-open intervals within each scene row.
    """
    if (
        not isinstance(court_present, np.ndarray)
        or court_present.shape != (n_frames,)
        or court_present.dtype != np.bool_
    ):
        raise ValueError('court_present must have shape (n_frames,) and bool dtype')

    parsed_rows = []
    for row in homography_rows:
        try:
            start = int(row['start_frame'])
            end = int(row['end_frame'])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError('homography row bounds must be integers') from exc
        if end < start:
            raise ValueError('homography rows must not be reversed')
        parsed_rows.append((start, end))

    parsed_rows.sort()
    for previous, current in zip(parsed_rows, parsed_rows[1:]):
        if current[0] < previous[1]:
            raise ValueError('homography rows must not overlap')

    segments = []
    for row_start, row_end in parsed_rows:
        start = max(0, row_start)
        end = min(n_frames, row_end)
        if start >= end:
            continue
        run_start = None
        for frame in range(start, end):
            if court_present[frame] and run_start is None:
                run_start = frame
            elif not court_present[frame] and run_start is not None:
                segments.append((run_start, frame))
                run_start = None
        if run_start is not None:
            segments.append((run_start, end))
    return segments


def build_sticky_result(
    track: np.ndarray, segments: list[tuple[int, int]],
    pose_bboxes: np.ndarray, pose_scores: np.ndarray, pose_kps: np.ndarray,
    pose_ndet: np.ndarray, gate_video_id: str,
    gate_court_info: dict[str, dict], gate_resolution_table: object,
    resolution: tuple[float, float], half_window: int = BODY_UNIT_HALF_WINDOW,
) -> StickyResult:
    """Run one sticky analysis loop over the supplied tracker segments.

    A tracker segment is one EMA lifetime. The sequential loop deliberately runs
    on every frame, including non-contact frames, because skipped frames change
    the picker's EMA state and therefore later candidate choices.
    """
    params = sticky_anchor.StickyAnchorParams()
    raw = RawClip(
        kps=pose_kps,
        bboxes=pose_bboxes,
        scores=pose_scores,
        # The picker does not read keypoint scores, but RawClip requires the field.
        kp_scores=np.zeros((*pose_scores.shape, 17), dtype=np.float32),
        ndet=pose_ndet,
    )
    ctx = ClipContext(gate_video_id, gate_court_info, gate_resolution_table)
    court_info = ctx.all_court_info[ctx.vid]
    halfcourt_centre = sticky_anchor.compute_halfcourt_centres(court_info)
    n_frames = len(track)
    picks_out = np.full((n_frames, 2), -1, dtype=int)
    standing_count = np.zeros(n_frames, dtype=int)
    ankle_pos = np.full((n_frames, 2, 2), np.nan)
    bbox_height = np.full((n_frames, 2), np.nan)
    # Per-slot fail-closed sentinel: +inf outside tracker segments, NaN once a tracker
    # segment is analysed but a slot carries no finite gap.
    distances_per_slot = np.full((n_frames, 2), np.inf, dtype=np.float64)
    wrist_dist_px = np.full((n_frames, 2), np.inf, dtype=np.float64)
    analysed = np.zeros(n_frames, dtype=bool)

    for start, end in segments:
        ema = halfcourt_centre.copy()
        for frame in range(start, end):
            analysed[frame] = True
            distances_per_slot[frame] = np.nan
            wrist_dist_px[frame] = np.nan
            analysis = sticky_anchor.analyse_frame(raw, frame, ema, halfcourt_centre, ctx, params)
            standing_count[frame] = analysis.standing_in_court_count
            if analysis.picks is None:
                ema[:] = halfcourt_centre
                continue
            assert analysis.court_base_pos is not None
            assert analysis.bboxes is not None
            assert analysis.filtered_to_raw is not None
            for slot in Slot:
                pick = analysis.picks[slot]
                if pick < 0:
                    ema[slot] = halfcourt_centre[slot]
                    continue
                candidate_position = analysis.court_base_pos[pick]
                if sticky_anchor.in_generous_court(
                    candidate_position, params.update_gate_eps,
                ):
                    ema[slot] = (
                        params.ema_alpha * candidate_position
                        + (1 - params.ema_alpha) * ema[slot]
                    )
                raw_slot = int(analysis.filtered_to_raw[pick])
                picks_out[frame, slot] = raw_slot
                box = analysis.bboxes[pick]
                bbox_height[frame, slot] = box[3] - box[1]
                ankles = pose_kps[frame, raw_slot, (ANKLE_L, ANKLE_R), :]
                ankle_pos[frame, slot] = ankles.mean(axis=0) / np.asarray(resolution)

    width, height = resolution
    for start, end in segments:
        for frame in range(start, end):
            shuttle_x = track[frame, 0] * width
            shuttle_y = track[frame, 1] * height
            for half in Slot:
                pick = picks_out[frame, half]
                if pick < 0:
                    continue
                wrists = pose_kps[frame, pick, (WRIST_L, WRIST_R), :]
                numerator = float(np.hypot(wrists[:, 0] - shuttle_x, wrists[:, 1] - shuttle_y).min())
                window = bbox_height[max(start, frame - half_window):min(end, frame + half_window + 1), half]
                if not np.isfinite(window).any():
                    raise ValueError(
                        f'sticky body-unit distance: no accepted finite height for slot {half} '
                        f'at frame {frame}'
                    )
                divisor = float(np.nanmean(window))
                if not np.isfinite(divisor) or divisor <= 0.0:
                    raise ValueError(
                        f'sticky body-unit distance: non-finite or non-positive body-scale '
                        f'denominator for slot {half} at frame {frame}'
                    )
                distances_per_slot[frame, half] = numerator / divisor
                if track[frame, 2] == 1:
                    wrist_dist_px[frame, half] = numerator

    gaps = np.full(n_frames, np.inf)
    for frame in np.flatnonzero(analysed):
        finite_distances = distances_per_slot[frame][np.isfinite(distances_per_slot[frame])]
        gaps[frame] = float(finite_distances.min()) if len(finite_distances) else float('nan')

    return StickyResult(
        gaps, picks_out, standing_count, ankle_pos, bbox_height, distances_per_slot, wrist_dist_px,
        analysed,
    )


def find_rally_spans(
    track: np.ndarray, thresholds: Stage8Thresholds | None = None,
    serve_start: ServeStartOptions | None = None, span_open: SpanOpen | None = None,
    *, constants: FpsConstants | None = None, gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None, reentry_guard_buffer: float | None = None,
    quiet_start_window: int | None = None,
) -> list[tuple[int, int]]:
    """Span-only segmentation; deliberately performs no contact extraction."""
    if (reentry_guard_variant is None) != (reentry_guard_buffer is None):
        raise ValueError('reentry guard needs both a variant and a buffer, or neither')
    if reentry_guard_variant is not None and gap_state_demotion_bound is None:
        raise ValueError('reentry guard requires gap_state_demotion_bound')
    speed = compute_speed(track)
    if gap_state_demotion_bound is not None:
        at_rest = _rest_mask(
            speed, track, thresholds, constants=constants, gap_state_demotion_bound=gap_state_demotion_bound,
            reentry_guard_variant=reentry_guard_variant, reentry_guard_buffer=reentry_guard_buffer,
        )
    else:
        # The low-level opt-out retains the original module-global call path.
        at_rest = _rest_mask(speed, track) if thresholds is None else _rest_mask(speed, track, thresholds)
    if serve_start is not None:
        return _serve_start_find_rally_spans(speed, at_rest, thresholds, serve_start, span_open)
    if quiet_start_window is not None:
        assert thresholds is not None
        return _find_rally_spans_quiet_start(speed, at_rest, thresholds, quiet_start_window)
    if span_open is not None:
        return _find_rally_spans_span_open(speed, at_rest, thresholds, span_open)
    return _find_rally_spans(speed, at_rest) if thresholds is None else _find_rally_spans(speed, at_rest, thresholds)


def assemble_contacts(
    track: np.ndarray, positions: np.ndarray | None, spans: list[tuple[int, int]],
    thresholds: Stage8Thresholds | None, sticky_distances: np.ndarray | None,
    suppression_radius: int | None,
    *, smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> list[ContactCandidate]:
    """Detect, gate, and suppress contacts for already-selected spans."""
    raw_flags = [(rally_id, frame, impulse) for rally_id, (start, end) in enumerate(spans)
                 for frame, impulse in detect_contact_flags(
                     track, start, end, thresholds, smoothing_mode=smoothing_mode,
                 )]
    if sticky_distances is None:
        return [ContactCandidate(r, f, contact_proximity_ok(track, positions, f), None, None)
                for r, f, _ in raw_flags]
    gated = [(f, impulse) for _r, f, impulse in raw_flags if wrist_contact_near(sticky_distances, f)]
    radius = ((CONTACT_SUPPRESSION_RADIUS_FRAMES if thresholds is None else thresholds.contact_suppression_radius_frames)
              if suppression_radius is None else suppression_radius)
    accepted = set(suppress_contact_flags(gated, radius=radius))
    gate_frames = {frame for frame, _ in gated}
    return [ContactCandidate(r, f, contact_proximity_ok(track, positions, f), f in gate_frames,
                             f in gate_frames and f not in accepted)
            for r, f, _ in raw_flags]


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None, *,
    thresholds: Stage8Thresholds | None = None,
    serve_start: ServeStartOptions | None = None,
    span_open: SpanOpen | None = None,
    replay_mask: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    suppression_radius: int | None = None,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
    constants: FpsConstants | None = None,
    gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
    quiet_start_window: int | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Full stage-8 pass over one video's shuttle track.

    Every keyword option is off by default and each default preserves today's behaviour
    exactly. `thresholds=None` reads the module globals through the low-level opt-out path.

    :param track: `(t, 3)` whole-video track.
    :param positions: optional `(t, 2, 2)` court positions for the proximity guardrail.
    :param thresholds: a `Stage8Thresholds` preset used instead of the globals, or None.
    :param serve_start: `ServeStartOptions` gating rally openings on sticky serve-setup evidence,
        or None. Its setup inputs are built from the UNMASKED track by the caller (the committed
        measurement convention); serve-start was only ever measured with masking off, so combining
        it with `replay_mask` is unmeasured territory.
    :param span_open: a `SpanOpen` rule (REGION_START / BACK_FILL) changing where a span opens,
        or None (today's burst-open rule). `serve_start` with REGION_START raises ValueError, and
        `serve_start.close` (a split) with BACK_FILL raises too (BACK_FILL is one span per region).
    :param replay_mask: `(t,)` bool dead-time mask (True = dead), applied at entry via
        `apply_replay_mask` before speed is computed, or None.
    :param sticky_distances: optional `(t,)` body-unit shuttle-to-nearest-wrist gaps. NaN fails
        closed. Production callers supply the cached sticky distances.
    :param suppression_radius: optional contact suppression radius; None keeps the shipped 9-frame default.
    :param smoothing_mode: span coordinate smoothing policy; ZERO_FILL preserves
        the shipped rule and IGNORE_INVISIBLE drops invisible xy from each mean.
    :return: `(spans, contacts)` where spans is `[(start_frame, end_frame), ...]`
        (rally_id is the list index) and contacts is
        `ContactCandidate(rally_id, contact_frame, proximity_ok, wrist_near, suppressed)`.
        Every detected candidate is a row (the RAW set, kept for recall-first uses).
        `wrist_near` is the pure wrist-gate verdict and `suppressed` records a gate-passing
        candidate that lost the suppression-radius contest. Both are blank when no gate inputs
        are supplied, so every raw candidate stands.
    """
    if serve_start is not None and span_open is SpanOpen.REGION_START:
        raise ValueError(
            'serve_start with span_open=REGION_START is contradictory: REGION_START drops the '
            'qualifying gate serve_start refines. Use span_open=BACK_FILL under serve gating '
            '(the two forms coincide there).'
        )
    if serve_start is not None and serve_start.close is not None and span_open is SpanOpen.BACK_FILL:
        raise ValueError(
            'serve_start.close (split placement) with span_open=BACK_FILL is contradictory: '
            'BACK_FILL emits one span per qualifying region at region_start, so there is nothing '
            'to split. Drop the split close, or use it without BACK_FILL.'
        )
    # Argument validation precedes any span work so bad combinations fail loudly
    # here, never as a numpy error from deep inside span finding.
    if sticky_distances is not None:
        if sticky_distances.shape != (len(track),):
            raise ValueError('sticky_distances must have shape (len(track),)')
    if replay_mask is not None:
        track = apply_replay_mask(track, replay_mask)

    if spans is None:
        spans = find_rally_spans(
            track, thresholds, serve_start, span_open, constants=constants,
            gap_state_demotion_bound=gap_state_demotion_bound,
            reentry_guard_variant=reentry_guard_variant, reentry_guard_buffer=reentry_guard_buffer,
            quiet_start_window=quiet_start_window,
        )

    gate_ran = sticky_distances is not None
    if not gate_ran:
        return spans, assemble_contacts(
            track, positions, spans, thresholds, None, suppression_radius,
            smoothing_mode=smoothing_mode,
        )

    return spans, assemble_contacts(
        track, positions, spans, thresholds, sticky_distances, suppression_radius,
        smoothing_mode=smoothing_mode,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_SPAN_OPEN_CHOICES = {'region-start': SpanOpen.REGION_START, 'back-fill': SpanOpen.BACK_FILL}


def _format_bool(value: bool | None) -> str:
    """Serialise a guardrail bool for the CSV: 'True'/'False', blank when unmeasured.

    Matches the config's bool encoding note (consumers parse `== 'True'`).
    """
    if value is None:
        return ''
    return 'True' if value else 'False'


def _load_positions(pos_dir: Path | None, video_id: str) -> np.ndarray | None:
    """Load `<video_id>_pos.npy` from pos_dir if both are present, else None."""
    if pos_dir is None:
        return None
    pos_path = pos_dir / f'{video_id}_pos.npy'
    if not pos_path.exists():
        log.info('no positions for %s, proximity_ok left blank', video_id)
        return None
    return np.load(pos_path)


def _load_replay_mask(mask_dir: Path | None, video_id: str) -> np.ndarray | None:
    """Load `<video_id>_dead_mask.npy` from mask_dir if present, else None.

    A missing file means the video runs unmasked. A present-but-invalid mask hits
    `run_video`'s fail-loud checks, which the per-video log-and-skip in `main` catches.
    """
    if mask_dir is None:
        return None
    mask_path = mask_dir / f'{video_id}_dead_mask.npy'
    if not mask_path.exists():
        log.info('no dead mask for %s, running unmasked', video_id)
        return None
    return np.load(mask_path)


def _read_string_id_table(path: Path, label: str):
    """Read a table with unique string IDs and retain the indexed DataFrame."""
    import pandas as pd

    table = pd.read_csv(path, dtype={'id': str, 'video_id': str})
    id_column = 'id' if 'id' in table.columns else 'video_id' if 'video_id' in table.columns else None
    if id_column is None:
        raise ValueError(f'{label}: expected an id or video_id column')
    table[id_column] = table[id_column].astype(str)
    if table[id_column].duplicated().any():
        duplicate_ids = sorted(table.loc[table[id_column].duplicated(), id_column].unique())
        raise ValueError(f'{label}: duplicate IDs {duplicate_ids}')
    return table.set_index(id_column)


def _read_fps_table(path: Path):
    """Read a unique string-id FPS table written by stage 11."""
    table = _read_string_id_table(path, 'fps CSV')
    if 'fps' not in table.columns:
        raise ValueError('fps CSV: expected an fps column')
    return {str(video_id): float(row.fps) for video_id, row in table.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 8: rally spans and contacts from shuttle tracks.')
    parser.add_argument('--shuttle-dir', type=Path, required=True,
                        help='Directory of <video_id>.npy (t, 3) shuttle tracks')
    parser.add_argument('--pos-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_pos.npy court positions')
    parser.add_argument('--mask-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_dead_mask.npy dead-time masks '
                             '(True = dead); a missing file runs that video unmasked')
    parser.add_argument('--doubles-csv', type=Path, default=None,
                        help='Optional doubles flags CSV; only whole-video False rows are processed')
    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument('--fps-csv', type=Path, default=None,
                           help='per-video id,fps table written by stage 11')
    fps_group.add_argument('--fps', type=float, default=None,
                           help='CFR override for every video in this run')
    parser.add_argument('--span-open', choices=tuple(_SPAN_OPEN_CHOICES), default=None,
                        help='optional span-opening rule: region-start (every active region '
                             'yields a span) or back-fill (a qualifying region opens at its '
                             'start). Default: open at the first qualifying burst')
    parser.add_argument('--rally-spans-csv', type=Path, default=RALLY_SPANS_CSV)
    parser.add_argument('--contact-frames-csv', type=Path, default=CONTACT_FRAMES_CSV)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if not args.shuttle_dir.is_dir():
        raise FileNotFoundError(f'shuttle dir not found: {args.shuttle_dir}')
    if args.fps is None and args.fps_csv is None:
        parser.error('one of --fps or --fps-csv is required before processing videos')

    fps_by_id = _read_fps_table(args.fps_csv) if args.fps_csv is not None else {}
    span_open = _SPAN_OPEN_CHOICES[args.span_open] if args.span_open is not None else None
    args.rally_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    args.contact_frames_csv.parent.mkdir(parents=True, exist_ok=True)

    span_rows: list[tuple[str, int, int, int]] = []
    contact_rows: list[tuple[str, int, int, str, str, str]] = []
    track_paths = sorted(args.shuttle_dir.glob('*.npy'))
    input_track_paths = track_paths
    from .batch_report import VideoOutcome, publish_batch_report
    from .run_video import run_video

    outcomes_by_path: dict[Path, VideoOutcome] = {}
    all_excluded_error = None
    if args.doubles_csv is not None:
        whole_video_flags = read_whole_video_flags(args.doubles_csv)
        filtered_track_paths = []
        for track_path in track_paths:
            video_id = track_path.stem
            if video_id not in whole_video_flags:
                outcomes_by_path[track_path] = VideoOutcome(
                    video_id, 'excluded', reason='no doubles row; not assuming singles',
                )
                log.warning('excluding %s: no doubles row; not assuming singles', video_id)
            elif whole_video_flags[video_id]:
                outcomes_by_path[track_path] = VideoOutcome(
                    video_id, 'excluded', reason='flagged doubles',
                )
                log.warning('excluding %s: flagged doubles', video_id)
            else:
                filtered_track_paths.append(track_path)
        # One excluded clip is a log line; the whole batch excluded must block. A flags
        # CSV with no whole-video rows (e.g. this module's own per-rally CLI output)
        # would otherwise empty the batch and exit 0.
        if track_paths and not filtered_track_paths:
            all_excluded_error = ValueError(
                f'{args.doubles_csv}: the doubles filter excluded every video in the batch; '
                'refusing to write empty outputs'
            )
        track_paths = filtered_track_paths

    for track_path in track_paths:
        video_id = track_path.stem
        if args.fps is None and video_id not in fps_by_id:
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'skipped', reason='absent from fps CSV',
            )
            log.warning('skipping %s: absent from fps CSV', video_id)
            continue
        try:
            if args.fps is not None:
                fps = args.fps
            else:
                fps = fps_by_id[video_id]
            track = np.load(track_path)
            positions = _load_positions(args.pos_dir, video_id)
            replay_mask = _load_replay_mask(args.mask_dir, video_id)
            result = run_video(
                track,
                fps=fps,
                base=BaseAnnotatorConfig(span_open=span_open),
                positions=positions,
                raw_exclusion_mask=(
                    replay_mask if replay_mask is not None
                    else np.zeros(len(track), dtype=bool)
                ),
                court_optional=True,
                stop_after_segmentation=True,
            )
            spans, contacts = result.spans, result.contacts
        except Exception as exc:  # log-and-skip per video: one bad track must not sink the batch
            exception_text = ' '.join(str(exc).split()) or type(exc).__name__
            outcomes_by_path[track_path] = VideoOutcome(
                video_id, 'skipped', reason=exception_text,
            )
            log.warning('skipping %s: %s', video_id, exc)
            continue
        outcomes_by_path[track_path] = VideoOutcome(
            video_id, 'processed', rallies=len(spans), contacts=len(contacts),
        )
        for rally_id, (start, end) in enumerate(spans):
            span_rows.append((video_id, rally_id, start, end))
        for contact in contacts:
            contact_rows.append((
                video_id, contact.rally_id, contact.contact_frame,
                _format_bool(contact.proximity_ok), _format_bool(contact.wrist_near),
                _format_bool(contact.suppressed),
            ))
        log.info('%s: %d rallies, %d contacts', video_id, len(spans), len(contacts))

    outcomes = [outcomes_by_path[track_path] for track_path in input_track_paths]
    if all_excluded_error is None:
        with args.rally_spans_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['video_id', 'rally_id', 'start_frame', 'end_frame'])
            writer.writerows(span_rows)
        # wrist_near is the pure body-unit gate verdict; suppressed is true only for a gate-passing
        # candidate that lost the suppression contest. Both are blank when a video ran without gate
        # inputs (the gate never ran), so its raw candidates stand. Every detected candidate is
        # written (the RAW set), so nothing recall-first loses its input.
        with args.contact_frames_csv.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow([
                'video_id', 'rally_id', 'contact_frame', 'proximity_ok', 'wrist_near', 'suppressed',
            ])
            writer.writerows(contact_rows)
        log.info('wrote %d rally spans, %d contacts', len(span_rows), len(contact_rows))

    try:
        publish_batch_report(
            outcomes, args.rally_spans_csv, all_excluded=all_excluded_error is not None,
        )
    except Exception as publication_error:
        if all_excluded_error is not None:
            raise all_excluded_error from publication_error
        raise
    if all_excluded_error is not None:
        raise all_excluded_error


if __name__ == '__main__':
    main()
