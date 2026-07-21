"""Stage 8: rally segmentation and contact detection (scraper_spec.md section 6).

Trajectory rules over a whole-video TrackNetV3 shuttle track, the `(t, 3)`
`[x_norm, y_norm, visibility]` npy that `shuttle_extractor.py:244-249` writes
(x, y already normalised to [0, 1] by video resolution, visibility passed
through). Speed everywhere below is per-frame L2 displacement of `(x, y)` on
frames where visibility is 1.

Three primitives (`compute_speed`, `true_runs`, `rolling_nanmedian`) are public
because stage 9 reuses them: its slow-motion signal is defined against this
stage's per-frame speed, so re-deriving it there would be a second source of
truth. All per-frame arrays here share one frame-index space `[0, t)`; that
invariant is what lets rally spans, contacts and masks line up downstream.

`segment_video` takes four off-by-default keyword options that each preserve
today's behaviour exactly when left at their default:
  - `thresholds`: a `Stage8Thresholds` preset used instead of the module globals;
    None reads the globals through the same code path as today (so the sweep
    runner's `_patch_stage8` global-patching keeps working unchanged).
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
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    BEST_CONFIG_THRESHOLDS,
    END_REST_FRAMES,
    PROXIMITY_MAX,
    REST_SPEED,
    REST_WINDOW,
    SHIPPED_THRESHOLDS,
    SMOOTH_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
    Stage8Thresholds,
)
from .doubles_flag import read_whole_video_flags
from scraper.config import CONTACT_FRAMES_CSV, RALLY_SPANS_CSV
from .fps_constants import FpsConstants, scale_for_fps
from .types import ContactCandidate, ReentryGuardVariant, Slot, SmoothingMode, SpanOpen

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
        min_dir_change_deg=overrides.get('min_dir_change_deg', thresholds.min_dir_change_deg),
    )


# ---------------------------------------------------------------------------
# Shared primitives (stage 9 imports these)
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
    start, -1 marks one-past a run end. Shared with stage 9's court-absence
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
    with stage 9's slow-motion signal.

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
class CourtBox(NamedTuple):
    """The pilot court geometry the serve-start builders filter against.

    Pilot-scoped: the caller constructs it (the stand-in occupancy box or the
    homography quad bounding box) and passes it to the builders, so no pilot
    geometry lives in this module. Feet inside `mid_band` claim NEITHER court
    half (the net line carries 3D model error); the stand-in band is zero-width.

    :param x_range: foot-point x bounds, pixels.
    :param y_range: foot-point y bounds, pixels.
    :param height_band: court-player bbox pixel-height band.
    :param mid_band: top/bottom half-split band (low, high), pixels.
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    height_band: tuple[float, float]
    mid_band: tuple[float, float]


class WideshotInputs(NamedTuple):
    """Per-frame wide-shot gate inputs, precomputed once by the caller.

    Built by `build_serve_start_wideshot_inputs` from the same raw pose boxes the
    serve-start distance array uses, so the gate never recomputes per burst.
    """

    count: np.ndarray  # (t,) court-scale detection count per frame
    top_foot: np.ndarray  # (t, 2) best top-half court-scale foot, image-fraction; NaN when absent
    bot_foot: np.ndarray  # (t, 2) same for the bottom half


class ServeSetupInputs(NamedTuple):
    """Per-frame evidence for the body-height-unit serve-setup gate.

    Ankles are image-fraction centroids and heights are image-HEIGHT fractions.
    Distances are body-height-unit gaps, with ``+inf`` outside analysed
    coverage. The next batch supplies the builder that fills these fields from
    the sticky cache.
    """

    count: np.ndarray
    distances: np.ndarray
    analysed: np.ndarray
    top_ankles: np.ndarray
    bot_ankles: np.ndarray
    top_height: np.ndarray
    bot_height: np.ndarray

    def validate(self) -> None:
        """Validate shapes, dtypes, and the count contract."""
        arrays = {name: np.asarray(value) for name, value in self._asdict().items()}
        trailing_shapes = {
            'count': (), 'distances': (2,), 'analysed': (),
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
        for name in ('distances', 'top_ankles', 'bot_ankles', 'top_height', 'bot_height'):
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


# Presence floor per required player; consumed by the next batch.
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
    distances = np.asarray(sticky.distances_per_slot, dtype=np.float64).copy()
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
        valid = np.all(np.isfinite(ankles), axis=1)
        valid &= np.any(ankles != 0, axis=1)
        valid &= np.isfinite(box_height) & (box_height > 0)
        ankles_out[valid] = ankles[valid]
        heights_out[valid] = box_height[valid] / height

    inputs = ServeSetupInputs(
        count=count, distances=distances, analysed=analysed,
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

    Carries everything the serve-start span rule reads. The distance and wide-shot
    inputs are precomputed by the builders below from the UNMASKED track (the
    committed measurement convention: the arrays are built before any replay mask
    is applied, and serve-start was only ever measured with masking off).

    The serve-setup gate has two forms, chosen by whether `height` is supplied:
      - `height=None` (the default): `threshold` is a raw image-fraction and the gate passes a
        burst when the median lookback distance is `<= threshold`. Bit-for-bit today's gate.
      - `height` supplied: `threshold` is a MULTIPLE of the nearest player's body height, and the
        gate passes when the median lookback distance divided by the mean lookback bbox height is
        `<= threshold`. Expressing the same rule in body-height units makes its DEFINITION robust
        to camera framing, which a raw image-fraction is not: the same serve setup reads a median
        0.031 image-fraction on the pilot but 0.117 on vid-15 (a different zoom), so a fixed
        image-fraction does not travel between videos.

    Off by default: leave `height` None and nothing changes. The body-height form still needs a
    per-video number (the portable part is the definition, not one shared constant): reading the
    pilot box-height sweep, a threshold of ~0.75 body heights keeps 105/113 pilot serves, the
    closest swept row to the raw 0.10 gate's 103/113 keep (the 0.70 row's 101/113 ties it from
    below). Vid-15 needs its own value: there its serve-setup and dead-time populations overlap
    in body-height units, so no single constant separates them on both videos.

    :param dist: `(t,)` nearest court-scale player bbox-centre distance (build_serve_start_dist).
    :param threshold: serve-setup gate distance; a raw image-fraction when `height` is None, else a
        multiple of body height (see the class docstring).
    :param mode: fallback for a region with no qualifying burst (TRIM / REJECT).
    :param wideshot: optional wide-shot refinement inputs; None leaves it off (the default).
    :param close: optional split placement; None opens one span per region (the default).
    :param diagnostics: optional caller-supplied dict; when given, the span rule fills it in
        place with the per-call region counts / spacings (single writer, valid to read straight
        after the call IN THE SAME PROCESS: the in-place fill does not cross a multiprocessing
        worker boundary, so the pooled sweep runner leaves it None). None (the default) collects nothing.
    :param height: optional `(t,)` nearest-player bbox height, image-fraction
        (build_serve_start_box_height), sharing dist's finite frames. None (the default) keeps the
        raw-fraction gate; supplying it switches the gate to the body-height form above.
    :param setup: optional sticky-sourced evidence (build_serve_setup_inputs). Supplying it
        dispatches the three-lane sticky gate; exactly one of dist and setup is required, and
        the legacy wideshot/height fields belong to the dist path only.
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
    wideshot: WideshotInputs | None = None
    close: ServeStartClose | None = None
    diagnostics: dict | None = None
    height: np.ndarray | None = None
    setup: ServeSetupInputs | None = None
    stillness_threshold_bh: float | None = None
    lookback_frames: int | None = None
    stillness_window_frames: int | None = None


# The last second before a burst (25 fps): the serve-setup lookback window.
SERVE_START_LOOKBACK_FRAMES = 25

# Wide-shot refinement gate constants (off unless ServeStartOptions.wideshot is set). Over the
# same lookback the burst must also look like the serve WIDE SHOT (both players standing in
# position): median court-scale detection count >= 2, BOTH court halves occupied (a court-scale
# box present in >= half the lookback frames per half), and per-half total foot drift <= 0.05
# image-fraction.
WIDESHOT_COUNT_MED_MIN = 2.0      # median court-scale detections over the lookback
WIDESHOT_SLOT_PRESENT_FRAC = 0.5  # a half counts as occupied when present >= this fraction
WIDESHOT_DRIFT_MAX = 0.05         # max per-half total foot drift, image-fraction
WIDESHOT_DRIFT_END_FRAMES = 10    # drift = gap between head/tail means over up-to-10 present feet
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
        module globals (the default path, so the sweep's global patching still binds).
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
        module globals (the default path; the sweep monkey-patches this whole function
        under quiet-start, so its two-arg call stays intact).
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    start_speed = START_SPEED if thresholds is None else thresholds.start_speed
    end_rest_frames = END_REST_FRAMES if thresholds is None else thresholds.end_rest_frames
    start_min_frames = START_MIN_FRAMES if thresholds is None else thresholds.start_min_frames

    fast = np.nan_to_num(speed, nan=0.0) > start_speed  # (t,) NaN steps are not fast

    long_rest = np.zeros(len(speed), dtype=bool)  # (t,) frames inside an extended rest
    for start, end in true_runs(at_rest):
        if end - start >= end_rest_frames:
            long_rest[start:end] = True

    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= start_min_frames]

    spans: list[tuple[int, int]] = []
    for region_start, region_end in true_runs(~long_rest):
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


# ---------------------------------------------------------------------------
# Serve-start gating (opens a span only at a serve-setup-preceded burst)
# ---------------------------------------------------------------------------
# The dead time between rallies is saturated with fast bursts (tracker jitter, carry, replays),
# so the stock first-burst rule opens spans a long way before the real serve. Serve-start opens a
# span only at a burst whose last second reads like SERVE SETUP: the shuttle sits close to a
# court-scale player (held for the toss). The court geometry is a caller-supplied CourtBox, so no
# pilot-scoped geometry lives here.
def court_scale_boxes(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray, court_box: CourtBox,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame's court-scale person boxes; the serve-start builders' shared filter.

    Public: stage 10's point-winner attribution and landing filter (the same court-scale
    candidate set, re-detected per window frame) import this too.

    Keeps the detections whose foot point (bottom-centre) sits inside the court region AND
    whose pixel height is court-player scale; both builders filter through here so the rule
    lives once. Detection validity is `np.isfinite(scores)`, NOT a >= 0.5 cutoff: the pose
    extraction already floored scores at 0.30, so `isfinite` equals the scoping's ndet.
    Padding slots carry NaN scores and drop out.

    :param frame_bboxes: (16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param frame_scores: (16,) detection scores, NaN on padding slots.
    :param court_box: the court geometry to filter against.
    :return: (x1, y1, x2, y2, scores) filtered to the court-scale detections, each (k,).
    """
    slots = court_scale_slots(frame_bboxes, frame_scores, court_box)
    x1, y1, x2, y2 = frame_bboxes[slots].T  # each (k,) pixels
    return x1, y1, x2, y2, frame_scores[slots]


def court_scale_slots(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray, court_box: CourtBox,
) -> np.ndarray:
    """Original pose-slot indices of the court-scale detections, ascending.

    The filter's mask logic lives here once. Callers that need the detections'
    identities (the contact gate) take slots directly; a score-equality lookup
    can alias two detections that share a score.

    :param frame_bboxes: (16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param frame_scores: (16,) detection scores, NaN on padding slots.
    :param court_box: the court geometry to filter against.
    :return: (k,) int slot indices into the frame's pose arrays.
    """
    valid = np.isfinite(frame_scores)
    x1, y1, x2, y2 = frame_bboxes[valid].T  # each (m,) pixels
    foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
    box_height = y2 - y1
    x_lo, x_hi = court_box.x_range
    y_lo, y_hi = court_box.y_range
    height_lo, height_hi = court_box.height_band
    in_court = (x_lo <= foot_x) & (foot_x <= x_hi) & (y_lo <= y2) & (y2 <= y_hi)
    in_scale = (height_lo <= box_height) & (box_height <= height_hi)
    return np.flatnonzero(valid)[in_court & in_scale]


# Dies at Stage 7 with the frozen sweep.
def _build_serve_start_metrics(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
    court_box: CourtBox, resolution: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame nearest court-scale detection: bbox-centre distance AND that box's height.

    The single source of truth for "the nearest court-scale player" behind both serve-start
    builders: one loop picks the nearest detection per visible-shuttle frame and reads off its
    distance and its bbox height, so the two arrays share their finite frame set and their
    nearest detection by construction (the body-height gate divides one by the other and
    relies on exactly that pairing).

    :return: `(dist, box_height)`, each `(t,)` image-fraction; NaN where the shuttle is
        invisible or no court-scale detection is present.
    """
    n_frames = len(track)
    width, height = resolution

    dist = np.full(n_frames, np.nan)
    box_height = np.full(n_frames, np.nan)
    for frame in np.flatnonzero(track[:, 2] == 1):
        x1, y1, x2, y2, _ = court_scale_boxes(bboxes[frame], scores[frame], court_box)
        if len(x1) == 0:
            continue
        centre_x = (x1 + x2) / 2.0 / width
        centre_y = (y1 + y2) / 2.0 / height
        gaps = np.hypot(centre_x - track[frame, 0], centre_y - track[frame, 1])  # image-fraction
        nearest = int(np.argmin(gaps))
        dist[frame] = gaps[nearest]
        box_height[frame] = (y2[nearest] - y1[nearest]) / height  # image-fraction
    return dist, box_height


# Dies at Stage 7 with the frozen sweep.
def build_serve_start_dist(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
    court_box: CourtBox, resolution: tuple[float, float],
) -> np.ndarray:
    """Per-frame nearest court-scale player bbox-centre distance; the serve-start gate input.

    For every frame where the shuttle is visible, take the frame's court-scale detections
    (the rule lives in `court_scale_boxes`) and return the smallest normalised
    (image-fraction) distance from the shuttle to any of their bbox centres. NaN where the
    shuttle is invisible or no court-scale detection is present.

    :param track: (t, 3) [x_norm, y_norm, visibility] whole-video track (built UNMASKED).
    :param bboxes: (t, 16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param scores: (t, 16) detection scores, NaN on padding slots.
    :param court_box: the court geometry the foot-point filter uses.
    :param resolution: (width, height) the shuttle xy and bbox centres normalise by.
    :return: (t,) float; NaN where the shuttle is invisible or no court-scale player is present.
    """
    return _build_serve_start_metrics(track, bboxes, scores, court_box, resolution)[0]


# Dies at Stage 7 with the frozen sweep.
def build_serve_start_box_height(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
    court_box: CourtBox, resolution: tuple[float, float],
) -> np.ndarray:
    """Per-frame bbox height of the SAME nearest court-scale player build_serve_start_dist measures.

    The body-height yardstick the body-height serve-setup gate divides by: the nearest court-scale
    box's pixel height as an image-fraction, NaN where build_serve_start_dist is NaN (shuttle
    invisible or no court-scale detection present). Both builders read
    `_build_serve_start_metrics`'s single loop, so the two arrays share their finite frame set and
    their nearest detection frame-for-frame, which the body-height gate relies on to median
    distance and mean height over the same lookback.

    :param track: (t, 3) [x_norm, y_norm, visibility] whole-video track (built UNMASKED).
    :param bboxes: (t, 16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param scores: (t, 16) detection scores, NaN on padding slots.
    :param court_box: the court geometry the foot-point filter uses.
    :param resolution: (width, height) the bbox centres and heights normalise by.
    :return: (t,) float; the nearest court-scale box height as an image-fraction, NaN where
        build_serve_start_dist is NaN.
    """
    return _build_serve_start_metrics(track, bboxes, scores, court_box, resolution)[1]


def _serve_setup_before(dist: np.ndarray, burst_start: int, threshold: float) -> bool:
    """Does the lookback before a burst look like serve setup?

    True when the median of the finite nearest-court-scale-centre distances over the
    SERVE_START_LOOKBACK_FRAMES frames immediately before burst_start is <= threshold. A
    lookback with no finite frame (the shuttle never visible near a court-scale player in that
    second) can't evidence setup, so it reads False (NaN fails). Near the video start the
    lookback truncates to the frames that exist.

    :param dist: (t,) per-frame nearest-court-scale-centre distance, NaN where undefined.
    :param burst_start: first frame of the candidate burst.
    :param threshold: the gate distance (image-fraction).
    :return: True when the pre-burst second reads as serve setup.
    """
    lookback = dist[max(0, burst_start - SERVE_START_LOOKBACK_FRAMES):burst_start]
    finite = lookback[np.isfinite(lookback)]
    if len(finite) == 0:
        return False
    return bool(np.median(finite) <= threshold)


def _serve_setup_before_boxheight(
    dist: np.ndarray, height: np.ndarray, burst_start: int, threshold_bh: float,
) -> bool:
    """Body-height-normalised form of the serve-setup gate.

    The portable version of `_serve_setup_before`: rather than comparing the raw lookback median
    distance against a fixed image-fraction, it divides that median by the mean nearest-player bbox
    height over the same lookback and tests the ratio against `threshold_bh`, a multiple of a body
    height. The measured definition from the pilot box-height sweep (raw = median finite lookback
    distances, denom = mean of the finite lookback nearest-box heights), so a burst passes when
    `median(distances) / mean(heights) <= threshold_bh`.

    `dist` and `height` come off `_build_serve_start_metrics`'s single loop, so the median and the
    mean run over the same lookback frames: `finite` is dist's mask, and the heights it selects
    are the matching ones. Fails closed like `_serve_setup_before`: an all-NaN lookback carries
    no evidence of setup.

    :param dist: (t,) per-frame nearest-court-scale-centre distance, NaN where undefined.
    :param height: (t,) that same detection's bbox height, image-fraction, NaN on the same frames.
    :param burst_start: first frame of the candidate burst.
    :param threshold_bh: the gate distance as a multiple of body height.
    :return: True when the pre-burst second reads as serve setup in body-height units.
    """
    lookback = slice(max(0, burst_start - SERVE_START_LOOKBACK_FRAMES), burst_start)
    window_dist = dist[lookback]
    window_height = height[lookback]
    finite = np.isfinite(window_dist)
    if not finite.any():
        return False
    raw = np.median(window_dist[finite])
    denom = np.mean(window_height[finite])
    return bool(raw / denom <= threshold_bh)


# Dies at Stage 7 with the frozen sweep.
def build_serve_start_wideshot_inputs(
    bboxes: np.ndarray, scores: np.ndarray, court_box: CourtBox, resolution: tuple[float, float],
) -> WideshotInputs:
    """Per-frame court-scale count and per-half best feet; the wide-shot gate input.

    For EVERY frame (shuttle visibility is irrelevant to the wide shot), count the frame's
    court-scale detections (the rule lives in `court_scale_boxes`, shared with
    build_serve_start_dist) and keep the highest-score court-scale foot per court half
    (top: foot y < the court mid-band low edge; bottom: foot y >= the high edge; feet inside
    the band claim neither half), normalised to image-fraction.

    :param bboxes: (t, 16, 4) xyxy person boxes in pixels, NaN-padded past the detections.
    :param scores: (t, 16) detection scores, NaN on padding slots.
    :param court_box: the court geometry (foot-point filter plus the half-split mid band).
    :param resolution: (width, height) the feet normalise by.
    :return: WideshotInputs of (t,) counts and (t, 2) per-half feet (NaN when a half is empty).
    """
    n_frames = len(bboxes)
    width, height = resolution

    count = np.zeros(n_frames, dtype=int)
    top_foot = np.full((n_frames, 2), np.nan)
    bot_foot = np.full((n_frames, 2), np.nan)
    for frame in range(n_frames):
        x1, _, x2, y2, cs_scores = court_scale_boxes(bboxes[frame], scores[frame], court_box)
        count[frame] = len(x1)
        if len(x1) == 0:
            continue
        foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
        # Feet inside the mid band claim NEITHER half (net-line 3D model error); a zero-width
        # band makes y2 >= band_hi exactly the (y2 < mid) split.
        band_lo, band_hi = court_box.mid_band
        in_top_half = y2 < band_lo
        in_bot_half = y2 >= band_hi
        for half_idx, foot_out in ((np.flatnonzero(in_top_half), top_foot),
                                   (np.flatnonzero(in_bot_half), bot_foot)):
            if len(half_idx):
                best = half_idx[np.argmax(cs_scores[half_idx])]
                foot_out[frame] = (foot_x[best] / width, y2[best] / height)
    return WideshotInputs(count=count, top_foot=top_foot, bot_foot=bot_foot)


def _slot_total_drift(foot_series: np.ndarray) -> tuple[float, int]:
    """Total foot drift and present-count for one court half over a lookback.

    Drift is the gap between the means of the first and last WIDESHOT_DRIFT_END_FRAMES
    present feet (net relocation, robust to end jitter), NaN when the series has no more
    feet than one window. Between 11 and 19 present feet the two windows partially overlap
    and damp the reading; that is the measured arithmetic and stays.

    :param foot_series: (n, 2) image-fraction feet, NaN rows where the half is empty.
    :return: (total_drift, present_count).
    """
    present = np.flatnonzero(np.isfinite(foot_series[:, 0]))
    if len(present) <= WIDESHOT_DRIFT_END_FRAMES:
        # One window's worth of feet or fewer: head and tail would fully overlap and read
        # exactly 0.0, silently passing the drift bound even for a sprinting player. Abstain
        # to NaN, which the gate fails closed. Only a truncated lookback near the video start
        # can get here past the presence gate (a full 25-frame lookback guarantees >= 13
        # present per half), so the audited keep rates, all full-lookback windows, cannot move.
        return float('nan'), len(present)
    head = foot_series[present[:WIDESHOT_DRIFT_END_FRAMES]].mean(axis=0)
    tail = foot_series[present[-WIDESHOT_DRIFT_END_FRAMES:]].mean(axis=0)
    return float(np.linalg.norm(tail - head)), len(present)


def _wide_shot_before(inputs: WideshotInputs, burst_start: int) -> bool:
    """Does the lookback before a burst look like the serve wide shot?

    Three conditions over the SERVE_START_LOOKBACK_FRAMES frames immediately before
    burst_start: median court-scale detection count >= WIDESHOT_COUNT_MED_MIN, both court
    halves occupied (each half's best box present in >= WIDESHOT_SLOT_PRESENT_FRAC of the
    lookback), and every occupied half's total foot drift <= WIDESHOT_DRIFT_MAX. A NaN drift
    (under two present feet) fails. Near the video start the lookback truncates.

    :param inputs: the precomputed per-frame gate inputs.
    :param burst_start: first frame of the candidate burst.
    :return: True when the pre-burst second reads as the serve wide shot.
    """
    lookback = slice(max(0, burst_start - SERVE_START_LOOKBACK_FRAMES), burst_start)
    count = inputs.count[lookback]
    if len(count) == 0 or np.median(count) < WIDESHOT_COUNT_MED_MIN:
        return False

    top_drift, top_present = _slot_total_drift(inputs.top_foot[lookback])
    bot_drift, bot_present = _slot_total_drift(inputs.bot_foot[lookback])
    present_min = WIDESHOT_SLOT_PRESENT_FRAC * len(count)
    if top_present < present_min or bot_present < present_min:
        return False

    # Max over the two halves: both players must be still, so the noisier half binds. NaN
    # halves drop out and no finite drift at all fails (only a truncated lookback near the
    # video start can reach the no-finite branch).
    finite_drifts = [drift for drift in (top_drift, bot_drift) if np.isfinite(drift)]
    if not finite_drifts:
        return False
    return bool(max(finite_drifts) <= WIDESHOT_DRIFT_MAX)


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
    end = burst + 1
    setup_window = slice(max(0, end - lookback_frames), end)
    stillness_window = slice(max(0, end - stillness_window_frames), end) if stillness_window_frames is not None else setup_window
    if not np.all(setup.analysed[min(setup_window.start, stillness_window.start):end]):
        return False

    count = setup.count[setup_window]
    median_count = float(np.median(count))
    distances = setup.distances[setup_window]
    ankles = (setup.top_ankles[setup_window], setup.bot_ankles[setup_window])
    heights = (setup.top_height[setup_window], setup.bot_height[setup_window])
    valid = np.empty(distances.shape, dtype=bool)
    for slot in Slot:
        valid[:, slot] = (
            np.isfinite(distances[:, slot]) & np.all(np.isfinite(ankles[slot]), axis=1) &
            np.isfinite(heights[slot])
        )

    if median_count >= 2:
        # Presence floor AND the primitive's minimum detections: below two valid
        # rows a drift cannot be split into halves, so the window fails closed
        # even with the stillness gate off.
        if any(np.mean(valid[:, slot]) < PLAYER_PRESENT_MIN_FRAC or
               np.count_nonzero(valid[:, slot]) < 2 for slot in Slot):
            return False
        nearest = np.full(len(distances), np.nan)
        for frame in range(len(distances)):
            finite = distances[frame, np.isfinite(distances[frame])]
            if len(finite):
                nearest[frame] = np.min(finite)
        finite_nearest = nearest[np.isfinite(nearest)]
        if not len(finite_nearest) or np.median(finite_nearest) > threshold:
            return False
        return stillness_threshold_bh is None or serve_setup_still(
            setup, burst, stillness_window_frames, stillness_threshold_bh, (Slot.TOP, Slot.BOTTOM),
        )

    if median_count >= 1:
        for slot in Slot:
            slot_valid = valid[:, slot]
            if np.mean(slot_valid) < PLAYER_PRESENT_MIN_FRAC or np.count_nonzero(slot_valid) < 2:
                continue
            finite_distances = distances[:, slot][slot_valid]
            if not len(finite_distances) or np.median(finite_distances) > threshold:
                continue
            if stillness_threshold_bh is None:
                return True
            masked = setup._replace(
                top_ankles=setup.top_ankles.copy(), bot_ankles=setup.bot_ankles.copy(),
                top_height=setup.top_height.copy(), bot_height=setup.bot_height.copy(),
            )
            ankles_out = masked.top_ankles if slot is Slot.TOP else masked.bot_ankles
            heights_out = masked.top_height if slot is Slot.TOP else masked.bot_height
            stillness_distances = setup.distances[stillness_window, slot]
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
    opens at a fast burst whose lookback passes the serve-setup gate (the raw-fraction
    `_serve_setup_before`, or the body-height `_serve_setup_before_boxheight` when
    `options.height` is supplied), and, with the optional wide-shot refinement installed, the
    wide-shot gate too (`_wide_shot_before`). A region with no qualifying burst is handled by the
    mode: TRIM falls back to the first burst (span survives at the stock start), REJECT drops it.

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
    if (options.dist is None) == (options.setup is None):
        raise ValueError('exactly one of dist and setup must be supplied')
    threshold = _valid_serve_threshold(options.threshold, 'threshold')
    if options.setup is not None:
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
    else:
        dist = options.dist
        wideshot = options.wideshot
        height = options.height

        def qualifies(burst: int) -> bool:
            """Serve-setup gate (raw-fraction, or body-height-normalised when options.height is set),
            AND the wide-shot gate when the refinement is on."""
            if height is None:
                setup = _serve_setup_before(dist, burst, threshold)
            else:
                setup = _serve_setup_before_boxheight(dist, height, burst, threshold)
            if not setup:
                return False
            return wideshot is None or _wide_shot_before(wideshot, burst)

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
    # (-impulse, frame) ordering. Exact ties occur in real data (the pilot has one),
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


def detect_contacts(
    track: np.ndarray, start: int, end: int, thresholds: Stage8Thresholds | None = None, *,
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> list[int]:
    """Raw contact frames from ``impulse_cell_candidates``, in ascending order.

    The candidate rule uses a three-frame visibility mask, a rolling impulse
    floor, and largest-impulse-first de-duplication. Use
    ``detect_contact_flags`` when suppression needs the impulse value.
    """
    return [
        frame for frame, _impulse in detect_contact_flags(
            track, start, end, thresholds, smoothing_mode=smoothing_mode,
        )
    ]


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


# COCO wrist/ankle keypoint indices in the (t, n_max, 17, 2) pose keypoint arrays. Defined
# here because stage 8 owns the pose-array conventions; point_winner reads them for
# attribution and landing kinematics.
WRIST_L, WRIST_R = 9, 10
ANKLE_L, ANKLE_R = 15, 16


def body_unit_dist_at_frame(
    frame: int, track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray, kps: np.ndarray,
    court_box: CourtBox, width: float, height: float, half_window: int = BODY_UNIT_HALF_WINDOW,
) -> float:
    """Body-unit shuttle-to-nearest-wrist gap at one frame, or NaN.

    This is the gate wrapper around point-winner's own body-unit machinery.
    Keeping the wrapper here lets the contact chain invoke the measured stage
    without copying its box-window association or slot recovery.
    """
    if track[frame, 2] != 1:
        return float('nan')
    candidate_slots = court_scale_slots(bboxes[frame], scores[frame], court_box)
    if len(candidate_slots) == 0:
        return float('nan')

    from . import point_winner

    x1, y1, x2, y2 = bboxes[frame, candidate_slots].T
    gaps = point_winner.body_unit_gaps(
        frame, x1, y1, x2, y2, [int(slot) for slot in candidate_slots], bboxes, scores, kps,
        court_box, track, width, height, half_window,
    )
    finite_gaps = gaps[np.isfinite(gaps)]
    return float(finite_gaps.min()) if len(finite_gaps) else float('nan')


def wrist_contact_near(body_unit_dist: np.ndarray | None, contact_frame: int) -> bool | None:
    """The single-frame body-unit wrist gate on one contact, in player-box-height units.

    Mirrors `contact_proximity_ok`'s three-way verdict. None distances mean the gate never
    ran (no body-unit or pose/court inputs), which returns None (serialised blank
    downstream): raw candidates stand, per the recall-first convention. A NaN frame is
    measured-but-unconfirmed and fails closed to False, the measured arm's behaviour.
    """
    if body_unit_dist is None:
        return None
    distance = body_unit_dist[contact_frame]
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


class StickyResult(NamedTuple):
    """Cached sticky evidence. ``bbox_height`` is in pixels."""

    distances: np.ndarray
    picks: np.ndarray
    standing_count: np.ndarray
    ankle_pos: np.ndarray
    bbox_height: np.ndarray
    distances_per_slot: np.ndarray
    analysed: np.ndarray


def build_sticky_result(
    track: np.ndarray, spans: list[tuple[int, int]],
    pose_bboxes: np.ndarray, pose_scores: np.ndarray, pose_kps: np.ndarray,
    pose_ndet: np.ndarray, gate_video_id: str,
    gate_court_info: dict[str, dict], gate_resolution_table: object,
    court_box: CourtBox, resolution: tuple[float, float], half_window: int = BODY_UNIT_HALF_WINDOW,
) -> StickyResult:
    """Run one sticky analysis loop over the supplied spans.

    A span is one clip for EMA purposes. The sequential loop deliberately runs
    on every frame, including non-contact frames, because skipped frames change
    the picker's EMA state and therefore later candidate choices.
    """
    from . import point_winner

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
    gaps = np.full(n_frames, np.inf)
    picks_out = np.full((n_frames, 2), -1, dtype=int)
    standing_count = np.zeros(n_frames, dtype=int)
    ankle_pos = np.full((n_frames, 2, 2), np.nan)
    bbox_height = np.full((n_frames, 2), np.nan)
    # Same fail-closed sentinel as the collapsed series: +inf outside the spans,
    # NaN once a frame is visited but a slot carries no finite gap.
    distances_per_slot = np.full((n_frames, 2), np.inf, dtype=np.float64)
    analysed = np.zeros(n_frames, dtype=bool)

    for start, end in spans:
        ema = halfcourt_centre.copy()
        for frame in range(start, end):
            analysed[frame] = True
            distances_per_slot[frame] = np.nan
            analysis = sticky_anchor.analyse_frame(raw, frame, ema, halfcourt_centre, ctx, params)
            standing_count[frame] = analysis.standing_in_court_count
            if analysis.picks is None:
                gaps[frame] = np.nan
                ema[:] = halfcourt_centre
                continue
            assert analysis.court_base_pos is not None
            assert analysis.bboxes is not None
            assert analysis.filtered_to_raw is not None
            frame_has_zero = False
            picked_raw_slots: list[int] = []
            picked_slots: list[Slot] = []
            for slot in Slot:
                pick = analysis.picks[slot]
                if pick < 0:
                    frame_has_zero = True
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
                picked_raw_slots.append(raw_slot)
                picked_slots.append(slot)
                box = analysis.bboxes[pick]
                bbox_height[frame, slot] = box[3] - box[1]
                ankles = pose_kps[frame, raw_slot, (ANKLE_L, ANKLE_R), :]
                ankle_pos[frame, slot] = ankles.mean(axis=0) / np.asarray(resolution)

            # A partial pick is valid. Only the retained raw slots enter the
            # body-unit denominator association; no doubles/count machinery is
            # part of this contact gate.
            if frame_has_zero and not picked_raw_slots:
                gaps[frame] = np.nan
                continue
            picked_boxes = pose_bboxes[frame, picked_raw_slots].astype(np.float64)
            x1, y1, x2, y2 = picked_boxes.T
            try:
                candidate_gaps = point_winner.body_unit_gaps(
                    frame, x1, y1, x2, y2, picked_raw_slots,
                    pose_bboxes, pose_scores, pose_kps, court_box,
                    track, resolution[0], resolution[1], half_window,
                )
            except ValueError as exc:
                if not str(exc).startswith('body-unit gap:'):
                    raise
                gaps[frame] = np.nan
                continue
            for slot, gap in zip(picked_slots, candidate_gaps):
                if np.isfinite(gap):
                    distances_per_slot[frame, slot] = gap
            finite_gaps = candidate_gaps[np.isfinite(candidate_gaps)]
            gaps[frame] = float(np.min(finite_gaps)) if len(finite_gaps) else float('nan')

    return StickyResult(
        gaps, picks_out, standing_count, ankle_pos, bbox_height, distances_per_slot, analysed,
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
        # The frozen sweep rebinds _rest_mask to a (speed, track) replacement, so
        # the OFF path must keep the legacy call shapes until Stage 7 retires it.
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
    thresholds: Stage8Thresholds | None, body_unit_dist: np.ndarray | None,
    suppression_radius: int | None,
    *, smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
) -> list[ContactCandidate]:
    """Detect, gate, and suppress contacts for already-selected spans."""
    raw_flags = [(rally_id, frame, impulse) for rally_id, (start, end) in enumerate(spans)
                 for frame, impulse in detect_contact_flags(
                     track, start, end, thresholds, smoothing_mode=smoothing_mode,
                 )]
    if body_unit_dist is None:
        return [ContactCandidate(r, f, contact_proximity_ok(track, positions, f), None, None)
                for r, f, _ in raw_flags]
    gated = [(f, impulse) for _r, f, impulse in raw_flags if wrist_contact_near(body_unit_dist, f)]
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
    body_unit_dist: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
    pose_bboxes: np.ndarray | None = None,
    pose_scores: np.ndarray | None = None,
    pose_kps: np.ndarray | None = None,
    pose_ndet: np.ndarray | None = None,
    court_box: CourtBox | None = None,
    gate_video_id: str | None = None,
    gate_court_info: dict[str, dict] | None = None,
    gate_resolution_table: object | None = None,
    suppression_radius: int | None = None,
    body_unit_half_window: int | None = None,
    resolution: tuple[float, float] = (1920.0, 1080.0),
    smoothing_mode: SmoothingMode = SmoothingMode.ZERO_FILL,
    constants: FpsConstants | None = None,
    gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None,
    reentry_guard_buffer: float | None = None,
    quiet_start_window: int | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Full stage-8 pass over one video's shuttle track.

    Every keyword option is off by default and each default preserves today's behaviour
    exactly. `thresholds=None` reads the module globals through the same code path as today,
    so the sweep runner's `_patch_stage8` global-patching (and its gap-state / quiet-start /
    nan-smoothing monkey-patches) keeps working unchanged.

    :param track: `(t, 3)` whole-video track.
    :param positions: optional `(t, 2, 2)` court positions for the proximity guardrail.
    :param thresholds: a `Stage8Thresholds` preset used instead of the globals, or None.
    :param serve_start: `ServeStartOptions` gating rally openings on a serve-setup lookback, or
        None. The gate reads distance as a raw image-fraction unless `serve_start.height` is
        supplied, which switches it to the framing-robust body-height form (see ServeStartOptions).
        Its distance / height / wide-shot inputs are built from the UNMASKED track by the caller
        (the committed measurement convention); serve-start was only ever measured with masking
        off, so combining it with `replay_mask` is unmeasured territory.
    :param span_open: a `SpanOpen` rule (REGION_START / BACK_FILL) changing where a span opens,
        or None (today's burst-open rule). `serve_start` with REGION_START raises ValueError, and
        `serve_start.close` (a split) with BACK_FILL raises too (BACK_FILL is one span per region).
    :param replay_mask: `(t,)` bool dead-time mask (True = dead), applied at entry via
        `apply_replay_mask` before speed is computed, or None.
    :param body_unit_dist: optional `(t,)` body-unit shuttle-to-nearest-wrist gaps. NaN fails
        closed. If absent, all four pose inputs must be supplied for the gate to run.
    :param pose_bboxes: `(t, n, 4)` pose boxes for the body-unit gate.
    :param pose_scores: `(t, n)` pose scores aligned to `pose_bboxes`.
    :param pose_kps: `(t, n, 17, 2)` pose keypoints aligned to `pose_bboxes`.
    :param pose_ndet: `(t,)` raw detection counts consumed by sticky_anchor.
    :param court_box: per-video CourtBox used by point-winner's body-unit denominator association.
    :param gate_video_id: string video ID used by sticky_anchor's homography context.
    :param gate_court_info: `{video_id: court_info}` from the homography table.
    :param gate_resolution_table: resolution DataFrame indexed by string video ID.
    :param suppression_radius: optional contact suppression radius; None keeps the shipped 9-frame default.
    :param body_unit_half_window: resolved body-height window. Required with explicit
        thresholds; bare callers retain the Stage 6 bridge's legacy 12-frame default.
    :param resolution: `(width, height)` of the track and pose pixels.
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
    # Stage 6 bridge: bare calls preserve the frozen sweep/pilot's module-dispatched globals.
    if body_unit_half_window is None:
        if thresholds is not None:
            raise ValueError('explicit thresholds require body_unit_half_window')
        body_unit_half_window = BODY_UNIT_HALF_WINDOW
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
    gate_inputs = (
        pose_bboxes, pose_scores, pose_kps, pose_ndet, court_box,
        gate_video_id, gate_court_info, gate_resolution_table,
    )
    if sticky_distances is not None:
        if sticky_distances.shape != (len(track),):
            raise ValueError('sticky_distances must have shape (len(track),)')
        if body_unit_dist is not None or any(value is not None for value in gate_inputs):
            raise ValueError('sticky_distances cannot be combined with other gate inputs')
    if body_unit_dist is not None and any(value is not None for value in gate_inputs):
        raise ValueError('body_unit_dist cannot be combined with pose gate inputs')
    if body_unit_dist is None and any(value is not None for value in gate_inputs):
        if not all(value is not None for value in gate_inputs):
            raise ValueError(
                'sticky gate requires pose_bboxes, pose_scores, pose_kps, pose_ndet, '
                'court_box, gate_video_id, gate_court_info, and gate_resolution_table'
            )

    gate_track = track
    if replay_mask is not None:
        track = apply_replay_mask(track, replay_mask)

    if spans is None:
        spans = find_rally_spans(
            track, thresholds, serve_start, span_open, constants=constants,
            gap_state_demotion_bound=gap_state_demotion_bound,
            reentry_guard_variant=reentry_guard_variant, reentry_guard_buffer=reentry_guard_buffer,
            quiet_start_window=quiet_start_window,
        )

    gate_ran = sticky_distances is not None or body_unit_dist is not None or all(value is not None for value in gate_inputs)
    if not gate_ran:
        return spans, assemble_contacts(
            track, positions, spans, thresholds, None, suppression_radius,
            smoothing_mode=smoothing_mode,
        )

    distances = sticky_distances
    if body_unit_dist is None:
        if distances is None:
            distances = build_sticky_result(
                gate_track, spans, pose_bboxes, pose_scores, pose_kps, pose_ndet,
                gate_video_id, gate_court_info, gate_resolution_table, court_box, resolution,
                body_unit_half_window,
            ).distances
        return spans, assemble_contacts(
            track, positions, spans, thresholds, distances, suppression_radius,
            smoothing_mode=smoothing_mode,
        )
    return spans, assemble_contacts(
        track, positions, spans, thresholds, body_unit_dist, suppression_radius,
        smoothing_mode=smoothing_mode,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# CLI preset names -> the shipped threshold sets. Library callers pass a Stage8Thresholds
# directly; the CLI only exposes the two committed presets (spec: library-first).
_THRESHOLD_PRESETS = {'shipped': SHIPPED_THRESHOLDS, 'best': BEST_CONFIG_THRESHOLDS}
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
    `apply_replay_mask`'s fail-loud checks inside segment_video, which the per-video
    log-and-skip in `main` catches.
    """
    if mask_dir is None:
        return None
    mask_path = mask_dir / f'{video_id}_dead_mask.npy'
    if not mask_path.exists():
        log.info('no dead mask for %s, running unmasked', video_id)
        return None
    return np.load(mask_path)


def _gate_array_path(gate_dir: Path, video_id: str, kind: str) -> Path:
    """Find one per-video raw pose array under the batch gate directory."""
    candidates = (
        gate_dir / f'{video_id}_{kind}.npy',
        gate_dir / f'{video_id}_raw_{kind}.npy',
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _load_gate_arrays(
    gate_dir: Path, video_id: str, track_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Load and validate all four raw arrays, or None when none are present."""
    paths = {
        kind: _gate_array_path(gate_dir, video_id, kind)
        for kind in ('bboxes', 'scores', 'kps', 'ndet')
    }
    present = [path.exists() for path in paths.values()]
    if not any(present):
        return None
    if not all(present):
        missing = ', '.join(kind for kind, path in paths.items() if not path.exists())
        raise ValueError(f'{video_id}: gate arrays are incomplete; missing {missing}')

    bboxes = np.load(paths['bboxes'])
    scores = np.load(paths['scores'])
    kps = np.load(paths['kps'])
    ndet = np.load(paths['ndet'])
    if len(bboxes) != track_length or len(scores) != track_length or len(kps) != track_length \
            or len(ndet) != track_length:
        raise ValueError(f'{video_id}: gate arrays do not match track length {track_length}')
    if bboxes.ndim != 3 or bboxes.shape[-1] != 4:
        raise ValueError(f'{video_id}: bboxes shape {bboxes.shape} is not (frames, detections, 4)')
    if scores.shape != bboxes.shape[:2] or kps.shape[:2] != bboxes.shape[:2]:
        raise ValueError(f'{video_id}: gate array detection axes disagree')
    if kps.ndim != 4 or kps.shape[2:] != (17, 2):
        raise ValueError(f'{video_id}: kps shape {kps.shape} is not (frames, detections, 17, 2)')
    if ndet.ndim != 1:
        raise ValueError(f'{video_id}: ndet shape {ndet.shape} is not (frames,)')
    return bboxes, scores, kps, ndet


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


def _load_gate_context(
    homography_csv: Path, resolution_csv: Path, court_box_csv: Path,
) -> tuple[dict[str, dict], object, dict[str, CourtBox]]:
    """Load the three batch gate tables with string-keyed joins."""
    from pipeline.court_utils import get_court_info

    homography = _read_string_id_table(homography_csv, 'homography CSV')
    resolution = _read_string_id_table(resolution_csv, 'resolution table')
    court_rows = _read_string_id_table(court_box_csv, 'CourtBox table')
    required = {
        'x_lo', 'x_hi', 'y_lo', 'y_hi', 'height_lo', 'height_hi', 'mid_lo', 'mid_hi',
    }
    missing = sorted(required - set(court_rows.columns))
    if missing:
        raise ValueError(f'CourtBox table: missing columns {missing}')
    all_court_info = {
        str(video_id): get_court_info(homography, str(video_id))
        for video_id in homography.index
    }
    court_boxes = {
        str(video_id): CourtBox(
            x_range=(float(row.x_lo), float(row.x_hi)),
            y_range=(float(row.y_lo), float(row.y_hi)),
            height_band=(float(row.height_lo), float(row.height_hi)),
            mid_band=(float(row.mid_lo), float(row.mid_hi)),
        )
        for video_id, row in court_rows.iterrows()
    }
    return all_court_info, resolution, court_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 8: rally spans and contacts from shuttle tracks.')
    parser.add_argument('--shuttle-dir', type=Path, required=True,
                        help='Directory of <video_id>.npy (t, 3) shuttle tracks')
    parser.add_argument('--pos-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_pos.npy court positions')
    parser.add_argument('--mask-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_dead_mask.npy dead-time masks '
                             '(True = dead); a missing file runs that video unmasked')
    parser.add_argument('--gate-dir', '--pose-dir', dest='gate_dir', type=Path, default=None,
                        help='Optional directory of per-video gate arrays: '
                             '<video_id>_{bboxes,scores,kps,ndet}.npy')
    parser.add_argument('--homography-csv', type=Path, default=None,
                        help='Homography table for sticky-anchor projection')
    parser.add_argument('--resolution-csv', type=Path, default=None,
                        help='Per-video resolution table for sticky-anchor projection')
    parser.add_argument('--court-box-csv', type=Path, default=None,
                        help='Per-video CourtBox table with id/video_id and '
                             'x_lo,x_hi,y_lo,y_hi,height_lo,height_hi,mid_lo,mid_hi')
    parser.add_argument('--doubles-csv', type=Path, default=None,
                        help='Optional doubles flags CSV; only whole-video False rows are processed')
    parser.add_argument('--thresholds', choices=tuple(_THRESHOLD_PRESETS), default='shipped',
                        help='which threshold preset to segment with (default: shipped)')
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

    declared_thresholds = _THRESHOLD_PRESETS[args.thresholds]
    fps_by_id = _read_fps_table(args.fps_csv) if args.fps_csv is not None else {}
    span_open = _SPAN_OPEN_CHOICES[args.span_open] if args.span_open is not None else None
    gate_options = (args.gate_dir, args.homography_csv, args.resolution_csv, args.court_box_csv)
    if any(option is not None for option in gate_options) and not all(
        option is not None for option in gate_options
    ):
        raise ValueError(
            '--gate-dir, --homography-csv, --resolution-csv, and --court-box-csv '
            'must be supplied together'
        )
    gate_context = None
    if all(option is not None for option in gate_options):
        gate_context = _load_gate_context(
            args.homography_csv, args.resolution_csv, args.court_box_csv,
        )
    args.rally_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    args.contact_frames_csv.parent.mkdir(parents=True, exist_ok=True)

    span_rows: list[tuple[str, int, int, int]] = []
    contact_rows: list[tuple[str, int, int, str, str, str]] = []
    track_paths = sorted(args.shuttle_dir.glob('*.npy'))
    if args.doubles_csv is not None:
        whole_video_flags = read_whole_video_flags(args.doubles_csv)
        filtered_track_paths = []
        for track_path in track_paths:
            video_id = track_path.stem
            if video_id not in whole_video_flags:
                log.warning('excluding %s: no doubles row; not assuming singles', video_id)
            elif whole_video_flags[video_id]:
                log.warning('excluding %s: flagged doubles', video_id)
            else:
                filtered_track_paths.append(track_path)
        track_paths = filtered_track_paths

    for track_path in track_paths:
        video_id = track_path.stem
        if args.fps is None and video_id not in fps_by_id:
            log.warning('skipping %s: absent from fps CSV', video_id)
            continue
        try:
            if args.fps is not None:
                fps = args.fps
            else:
                fps = fps_by_id[video_id]
            thresholds = scale_thresholds(declared_thresholds, fps)
            body_unit_half_window = scale_for_fps(fps).body_unit_half_window
            track = np.load(track_path)
            positions = _load_positions(args.pos_dir, video_id)
            replay_mask = _load_replay_mask(args.mask_dir, video_id)
            gate_arrays = None if args.gate_dir is None else _load_gate_arrays(
                args.gate_dir, video_id, len(track),
            )
            if gate_arrays is None:
                log.warning('%s: gate inputs absent; running without contact gate', video_id)
                spans, contacts = segment_video(
                    track, positions, thresholds=thresholds, span_open=span_open,
                    replay_mask=replay_mask, body_unit_half_window=body_unit_half_window,
                )
            else:
                all_court_info, resolution_table, court_boxes = gate_context
                if video_id not in all_court_info or video_id not in resolution_table.index \
                        or video_id not in court_boxes:
                    raise ValueError(f'{video_id}: missing gate-table row')
                row = resolution_table.loc[video_id]
                resolution = (float(row['width']), float(row['height']))
                spans, contacts = segment_video(
                    track, positions, thresholds=thresholds, span_open=span_open,
                    replay_mask=replay_mask, pose_bboxes=gate_arrays[0],
                    pose_scores=gate_arrays[1], pose_kps=gate_arrays[2],
                    pose_ndet=gate_arrays[3], court_box=court_boxes[video_id],
                    gate_video_id=video_id, gate_court_info=all_court_info,
                    gate_resolution_table=resolution_table, resolution=resolution,
                    body_unit_half_window=body_unit_half_window,
                )
        except Exception as exc:  # log-and-skip per video: one bad track must not sink the batch
            log.warning('skipping %s: %s', video_id, exc)
            continue
        for rally_id, (start, end) in enumerate(spans):
            span_rows.append((video_id, rally_id, start, end))
        for contact in contacts:
            contact_rows.append((
                video_id, contact.rally_id, contact.contact_frame,
                _format_bool(contact.proximity_ok), _format_bool(contact.wrist_near),
                _format_bool(contact.suppressed),
            ))
        log.info('%s: %d rallies, %d contacts', video_id, len(spans), len(contacts))

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


if __name__ == '__main__':
    main()
