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

`wrist_dist` is a fifth off-by-default option: a `(t,)` shuttle-to-nearest-wrist gap
(`build_wrist_shuttle_dist`) that fills each contact's `wrist_near` verdict (the contact wrist
check). None leaves every verdict blank, so the raw candidate set is unchanged.

Run as `python -m scraper.stage8_rally_segmentation --shuttle-dir ...` with
PYTHONPATH=src.
"""
import argparse
import csv
import logging
import warnings
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import (
    BEST_CONFIG_THRESHOLDS,
    CONTACT_FRAMES_CSV,
    END_REST_FRAMES,
    MIN_CONTACT_SPEED,
    MIN_DIR_CHANGE_DEG,
    PROXIMITY_MAX,
    RALLY_SPANS_CSV,
    REST_SPEED,
    REST_WINDOW,
    SHIPPED_THRESHOLDS,
    SMOOTH_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
    Stage8Thresholds,
    WRIST_SHUTTLE_MAX,
)

log = logging.getLogger(__name__)

# Fraction of a window that must be tracked (visibility 1) for the window to
# read as "seeing the shuttle". Below this the window is mostly untracked and
# counts as rest (spec s6 "visibility mostly 0 across the window"). Not in
# config: it is the numeric reading of "mostly", not a swept rule threshold.
VISIBILITY_REST_FRAC = 0.5


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
    """Centred rolling mean with a shrinking window at the edges (no NaN handling).

    Used only for the visibility fraction, which is a clean 0/1 array.

    :param values: `(t,)` values, no NaN.
    :param window: window width in frames.
    :return: `(t,)` centred mean; edge frames average their partial window.
    """
    kernel = np.ones(window)
    counts = np.convolve(np.ones_like(values), kernel, mode='same')  # samples per position
    sums = np.convolve(values, kernel, mode='same')
    return sums / counts


# ---------------------------------------------------------------------------
# Selectable-option types (off by default; see segment_video)
# ---------------------------------------------------------------------------
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
    """

    dist: np.ndarray
    threshold: float
    mode: ServeStartMode
    wideshot: WideshotInputs | None = None
    close: ServeStartClose | None = None
    diagnostics: dict | None = None
    height: np.ndarray | None = None


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


# ---------------------------------------------------------------------------
# Rally spans
# ---------------------------------------------------------------------------
def _rest_mask(speed: np.ndarray, track: np.ndarray, thresholds: Stage8Thresholds | None = None) -> np.ndarray:
    """Per-frame rest flag: slow OR mostly untracked across the window (spec s6).

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param track: `(t, 3)` track, for the visibility column.
    :param thresholds: a preset to read rest_window / rest_speed from; None reads the
        module globals (the default path, so the sweep's global patching still binds).
    :return: `(t,)` bool, True where the frame reads as rest.
    """
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


# ---------------------------------------------------------------------------
# Serve-start gating (opens a span only at a serve-setup-preceded burst)
# ---------------------------------------------------------------------------
# The dead time between rallies is saturated with fast bursts (tracker jitter, carry, replays),
# so the stock first-burst rule opens spans a long way before the real serve. Serve-start opens a
# span only at a burst whose last second reads like SERVE SETUP: the shuttle sits close to a
# court-scale player (held for the toss). The court geometry is a caller-supplied CourtBox, so no
# pilot-scoped geometry lives here.
def _court_scale_boxes(
    frame_bboxes: np.ndarray, frame_scores: np.ndarray, court_box: CourtBox,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame's court-scale person boxes; the serve-start builders' shared filter.

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
    valid = np.isfinite(frame_scores)
    x1, y1, x2, y2 = frame_bboxes[valid].T  # each (m,) pixels
    foot_x = (x1 + x2) / 2.0  # bottom-centre; foot y is y2
    box_height = y2 - y1
    x_lo, x_hi = court_box.x_range
    y_lo, y_hi = court_box.y_range
    height_lo, height_hi = court_box.height_band
    in_court = (x_lo <= foot_x) & (foot_x <= x_hi) & (y_lo <= y2) & (y2 <= y_hi)
    in_scale = (height_lo <= box_height) & (box_height <= height_hi)
    court_scale = in_court & in_scale
    return (x1[court_scale], y1[court_scale], x2[court_scale], y2[court_scale],
            frame_scores[valid][court_scale])


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
        x1, y1, x2, y2, _ = _court_scale_boxes(bboxes[frame], scores[frame], court_box)
        if len(x1) == 0:
            continue
        centre_x = (x1 + x2) / 2.0 / width
        centre_y = (y1 + y2) / 2.0 / height
        gaps = np.hypot(centre_x - track[frame, 0], centre_y - track[frame, 1])  # image-fraction
        nearest = int(np.argmin(gaps))
        dist[frame] = gaps[nearest]
        box_height[frame] = (y2[nearest] - y1[nearest]) / height  # image-fraction
    return dist, box_height


def build_serve_start_dist(
    track: np.ndarray, bboxes: np.ndarray, scores: np.ndarray,
    court_box: CourtBox, resolution: tuple[float, float],
) -> np.ndarray:
    """Per-frame nearest court-scale player bbox-centre distance; the serve-start gate input.

    For every frame where the shuttle is visible, take the frame's court-scale detections
    (the rule lives in `_court_scale_boxes`) and return the smallest normalised
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


def build_serve_start_wideshot_inputs(
    bboxes: np.ndarray, scores: np.ndarray, court_box: CourtBox, resolution: tuple[float, float],
) -> WideshotInputs:
    """Per-frame court-scale count and per-half best feet; the wide-shot gate input.

    For EVERY frame (shuttle visibility is irrelevant to the wide shot), count the frame's
    court-scale detections (the rule lives in `_court_scale_boxes`, shared with
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
        x1, _, x2, y2, cs_scores = _court_scale_boxes(bboxes[frame], scores[frame], court_box)
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
    dist = options.dist
    threshold = options.threshold
    mode = options.mode
    wideshot = options.wideshot
    close = options.close
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
def detect_contacts(
    track: np.ndarray, start: int, end: int, thresholds: Stage8Thresholds | None = None,
) -> list[int]:
    """Contact frames inside one rally span, from smoothed-velocity reversals.

    Smooth `(x, y)` over SMOOTH_WINDOW to survive TrackNetV3 jitter, take the
    per-frame velocity vectors, and flag a junction where the incoming and
    outgoing velocity turn by more than MIN_DIR_CHANGE_DEG with both segment
    speeds above MIN_CONTACT_SPEED and the three frames around the reversal all
    visible. A real contact often trips several adjacent junctions once
    smoothed; de-dup keeps the sharpest-angle junction and drops any other
    within SMOOTH_WINDOW frames of it (the spec is silent on de-dup: the true
    contact vertex carries the largest reversal, so sharpest-angle-wins).

    :param track: `(t, 3)` whole-video track.
    :param start: rally span start frame (inclusive).
    :param end: rally span end frame (exclusive).
    :param thresholds: a preset to read smooth_window / min_dir_change_deg / min_contact_speed
        from; None reads the module globals (the default path; the sweep monkey-patches this
        whole function under nan-smoothing, so its three-arg call stays intact).
    :return: contact frames in whole-video frame indices, ascending.
    """
    smooth_window = SMOOTH_WINDOW if thresholds is None else thresholds.smooth_window
    min_dir_change_deg = MIN_DIR_CHANGE_DEG if thresholds is None else thresholds.min_dir_change_deg
    min_contact_speed = MIN_CONTACT_SPEED if thresholds is None else thresholds.min_contact_speed

    span = track[start:end]  # (n, 3) rally-local view
    if len(span) < smooth_window + 2:
        return []  # too short to smooth and difference twice

    # Invisible frames carry zero-filled (0, 0) xy, so windows straddling a
    # visibility gap average the corner in and can seed reversal candidates at
    # gap edges. Measured on the pilot (sweep --nan-smoothing arm, 2026-07-08):
    # NaN-masking removes ~73% of near-gap candidates and +4% relative precision
    # at +/-5, but costs 1.1 recall points there (gaps often start AT a fast
    # contact, so some gap-edge candidates are the only match for it). Left
    # as-is under the recall-first ruling; revisit if the precision stage moves.
    smooth_x = _rolling_mean(span[:, 0], smooth_window)  # (n,)
    smooth_y = _rolling_mean(span[:, 1], smooth_window)  # (n,)
    smoothed = np.column_stack([smooth_x, smooth_y])  # (n, 2)
    visibility = span[:, 2]  # (n,)

    velocity = np.diff(smoothed, axis=0)  # (n-1, 2) segment j spans local frames j -> j+1
    v_in = velocity[:-1]  # (n-2, 2) segment into junction k+1
    v_out = velocity[1:]  # (n-2, 2) segment out of junction k+1
    speed_in = np.linalg.norm(v_in, axis=1)  # (n-2,)
    speed_out = np.linalg.norm(v_out, axis=1)  # (n-2,)

    # Angle between incoming and outgoing velocity at each interior junction.
    # Guard the zero-speed denominator so a stalled segment reads as no turn.
    denom = speed_in * speed_out
    safe = denom > 0
    cos_angle = np.ones(len(denom))  # (n-2,) default 1.0 -> 0 deg where unsafe
    cos_angle[safe] = np.sum(v_in[safe] * v_out[safe], axis=1) / denom[safe]
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))  # (n-2,)

    # All per-junction arrays share index k in [0, n-3]; junction k sits at
    # local frame k+1, touching local frames k, k+1, k+2.
    around_visible = (visibility[:-2] == 1) & (visibility[1:-1] == 1) & (visibility[2:] == 1)
    sharp_turn = angle_deg > min_dir_change_deg
    fast_enough = (speed_in > min_contact_speed) & (speed_out > min_contact_speed)
    is_contact = sharp_turn & fast_enough & around_visible

    candidate_local = np.flatnonzero(is_contact) + 1  # local frame of each candidate
    candidate_angle = angle_deg[is_contact]

    kept: list[int] = []
    for idx in np.argsort(-candidate_angle):  # sharpest angle first
        frame = int(candidate_local[idx])
        if all(abs(frame - other) >= smooth_window for other in kept):
            kept.append(frame)
    kept.sort()
    return [start + frame for frame in kept]


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


# COCO wrist keypoint indices in the (t, n_max, 17, 2) pose keypoint arrays.
WRIST_L, WRIST_R = 9, 10


def build_wrist_shuttle_dist(
    track: np.ndarray, kps: np.ndarray, ndet: np.ndarray, resolution: tuple[float, float],
) -> np.ndarray:
    """Per-frame smallest normalised gap from the shuttle to any player's wrist.

    The contact wrist check's input, precomputed once by the caller (like the serve-start
    builders precompute their `(t,)` arrays). For every frame where the shuttle is visible,
    take the frame's real detections' wrists, divide by the same resolution that normalised
    the shuttle track so both sit in the [0, 1] image space, and record the smallest L2 gap to
    any wrist. NaN where the shuttle is invisible, no player is detected, or every wrist is NaN.

    The arithmetic is the scoping script's `min_wrist_distance`
    (scripts/wrist_contact_separation.py) promoted into the pipeline: identical per-frame gap,
    so the array reproduces the item-7 separation the wrist gate was tuned on.

    :param track: `(t, 3)` `[x_norm, y_norm, visibility]` whole-video track (built UNMASKED;
        contact frames are visible, so masking never moves a measured value).
    :param kps: `(t, n_max, 17, 2)` pixel keypoints, NaN-padded past the per-frame ndet.
    :param ndet: `(t,)` per-frame detection count.
    :param resolution: `(width, height)` the shuttle xy and the wrist pixels normalise by.
    :return: `(t,)` float; NaN where undefined (invisible shuttle / no detection / all wrists NaN).
    """
    n_frames = len(track)
    resolution_xy = np.array(resolution)
    dist = np.full(n_frames, np.nan)
    for frame in np.flatnonzero(track[:, 2] == 1):
        n = int(ndet[frame])
        if n == 0:
            continue  # no player detected: nothing to measure against, stays NaN
        wrists_norm = kps[frame, :n][:, (WRIST_L, WRIST_R), :] / resolution_xy  # (n, 2, 2)
        gaps = np.linalg.norm(wrists_norm - track[frame, :2], axis=-1)  # (n, 2) per detection per wrist
        if np.all(np.isnan(gaps)):
            continue  # every wrist NaN (undetected joints): unmeasured, stays NaN
        dist[frame] = float(np.nanmin(gaps))
    return dist


def wrist_contact_near(wrist_dist: np.ndarray | None, contact_frame: int) -> bool | None:
    """The wrist gate on one contact: was a player's wrist near the shuttle at the turn?

    Mirrors `contact_proximity_ok`'s three-way verdict. When no wrist distances were supplied
    the gate is unmeasured, which returns None (serialised blank downstream): a gate with no
    evidence must not read as a pass. When they were supplied but the frame's value is NaN (no
    player detected, or every wrist undetected) the gate is measured-but-unconfirmed, so it
    fails closed to False. Otherwise it passes when the gap is within WRIST_SHUTTLE_MAX.

    :param wrist_dist: `(t,)` per-frame shuttle-to-nearest-wrist gap (build_wrist_shuttle_dist),
        or None when no pose was supplied.
    :param contact_frame: whole-video frame index of the contact.
    :return: True (wrist near, keep) / False (no wrist near) when measured, None when unmeasured.
    """
    if wrist_dist is None:
        return None
    dist = wrist_dist[contact_frame]
    if np.isnan(dist):
        # Pose supplied but no measurable wrist at this frame: measured, unconfirmed.
        return False
    return bool(dist <= WRIST_SHUTTLE_MAX)


def segment_video(
    track: np.ndarray, positions: np.ndarray | None = None, *,
    thresholds: Stage8Thresholds | None = None,
    serve_start: ServeStartOptions | None = None,
    span_open: SpanOpen | None = None,
    replay_mask: np.ndarray | None = None,
    wrist_dist: np.ndarray | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int, bool | None, bool | None]]]:
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
    :param wrist_dist: optional `(t,)` shuttle-to-nearest-wrist gap (build_wrist_shuttle_dist)
        for the contact wrist check; None (the default) leaves every contact's `wrist_near`
        blank, exactly as `positions=None` leaves `proximity_ok` blank.
    :return: `(spans, contacts)` where spans is `[(start_frame, end_frame), ...]`
        (rally_id is the list index) and contacts is
        `[(rally_id, contact_frame, proximity_ok, wrist_near), ...]`. Every detected candidate
        is a row (the RAW set, kept for recall-first uses); `wrist_near` is the filter verdict
        (True = a wrist was near the shuttle at the turn). The FILTERED set downstream consumers
        default to is the rows with `wrist_near` True; a blank `wrist_near` (no `wrist_dist`
        supplied) means the filter never ran, so every raw candidate stands.
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
    if replay_mask is not None:
        track = apply_replay_mask(track, replay_mask)

    speed = compute_speed(track)

    # Rest mask. thresholds None keeps today's exact two-arg module-dispatched call so the
    # sweep's gap-state monkey-patch (and the patched globals) still bind; a preset threads
    # its rest thresholds through instead.
    at_rest = _rest_mask(speed, track) if thresholds is None else _rest_mask(speed, track, thresholds)

    # Rally spans. Serve-start and span_open are the landed options; absent both, thresholds
    # None keeps today's exact module-dispatched _find_rally_spans (quiet-start-patchable).
    if serve_start is not None:
        spans = _serve_start_find_rally_spans(speed, at_rest, thresholds, serve_start, span_open)
    elif span_open is not None:
        spans = _find_rally_spans_span_open(speed, at_rest, thresholds, span_open)
    elif thresholds is None:
        spans = _find_rally_spans(speed, at_rest)
    else:
        spans = _find_rally_spans(speed, at_rest, thresholds)

    contacts: list[tuple[int, int, bool | None, bool | None]] = []
    for rally_id, (start, end) in enumerate(spans):
        frames = (detect_contacts(track, start, end) if thresholds is None
                  else detect_contacts(track, start, end, thresholds))
        for contact_frame in frames:
            proximity_ok = contact_proximity_ok(track, positions, contact_frame)
            wrist_near = wrist_contact_near(wrist_dist, contact_frame)
            contacts.append((rally_id, contact_frame, proximity_ok, wrist_near))
    return spans, contacts


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


def _load_resolution_map(resolution_csv: Path | None) -> dict[str, tuple[float, float]] | None:
    """Map `<video_id> -> (width, height)` from the resolution CSV, or None when not given.

    Keyed by the id as a string so it matches the track file stem directly. The wrist pixels
    normalise by the same resolution the shuttle track did, so this is the SAME source the
    shuttle extractor read.
    """
    if resolution_csv is None:
        return None
    with resolution_csv.open(newline='', encoding='utf-8') as handle:
        return {
            str(row['id']): (float(row['width']), float(row['height']))
            for row in csv.DictReader(handle)
        }


def _load_wrist_dist(
    pose_dir: Path | None, resolution_map: dict[str, tuple[float, float]] | None,
    video_id: str, track: np.ndarray,
) -> np.ndarray | None:
    """Build the `<video_id>` shuttle-to-nearest-wrist array from its pose npys, or None.

    Needs both the pose (`<video_id>_kps.npy` + `<video_id>_ndet.npy`) and a resolution entry;
    any missing input leaves the wrist gate off for that video (`wrist_near` blank), exactly as a
    missing positions/mask file leaves those columns off. kps is memory-mapped (it is large).
    """
    if pose_dir is None or resolution_map is None:
        return None
    kps_path = pose_dir / f'{video_id}_kps.npy'
    ndet_path = pose_dir / f'{video_id}_ndet.npy'
    if not (kps_path.exists() and ndet_path.exists()):
        log.info('no pose for %s, wrist_near left blank', video_id)
        return None
    if video_id not in resolution_map:
        log.info('no resolution for %s, wrist_near left blank', video_id)
        return None
    kps = np.load(kps_path, mmap_mode='r')
    ndet = np.load(ndet_path)
    return build_wrist_shuttle_dist(track, kps, ndet, resolution_map[video_id])


def main() -> None:
    parser = argparse.ArgumentParser(description='Stage 8: rally spans and contacts from shuttle tracks.')
    parser.add_argument('--shuttle-dir', type=Path, required=True,
                        help='Directory of <video_id>.npy (t, 3) shuttle tracks')
    parser.add_argument('--pos-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_pos.npy court positions')
    parser.add_argument('--mask-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_dead_mask.npy dead-time masks '
                             '(True = dead); a missing file runs that video unmasked')
    parser.add_argument('--pose-dir', type=Path, default=None,
                        help='Optional directory of <video_id>_kps.npy and <video_id>_ndet.npy '
                             'pose arrays for the contact wrist check; needs --resolution-csv too')
    parser.add_argument('--resolution-csv', type=Path, default=None,
                        help='Optional id,width,height CSV (the shuttle-normalisation source); '
                             'with --pose-dir it drives the wrist_near column, else it is blank')
    parser.add_argument('--thresholds', choices=tuple(_THRESHOLD_PRESETS), default='shipped',
                        help='which threshold preset to segment with (default: shipped)')
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

    thresholds = _THRESHOLD_PRESETS[args.thresholds]
    span_open = _SPAN_OPEN_CHOICES[args.span_open] if args.span_open is not None else None
    resolution_map = _load_resolution_map(args.resolution_csv)

    args.rally_spans_csv.parent.mkdir(parents=True, exist_ok=True)
    args.contact_frames_csv.parent.mkdir(parents=True, exist_ok=True)

    span_rows: list[tuple[str, int, int, int]] = []
    contact_rows: list[tuple[str, int, int, str, str]] = []
    for track_path in sorted(args.shuttle_dir.glob('*.npy')):
        video_id = track_path.stem
        try:
            track = np.load(track_path)
            positions = _load_positions(args.pos_dir, video_id)
            replay_mask = _load_replay_mask(args.mask_dir, video_id)
            wrist_dist = _load_wrist_dist(args.pose_dir, resolution_map, video_id, track)
            spans, contacts = segment_video(track, positions, thresholds=thresholds,
                                            span_open=span_open, replay_mask=replay_mask,
                                            wrist_dist=wrist_dist)
        except Exception as exc:  # log-and-skip per video: one bad track must not sink the batch
            log.warning('skipping %s: %s', video_id, exc)
            continue
        for rally_id, (start, end) in enumerate(spans):
            span_rows.append((video_id, rally_id, start, end))
        for rally_id, contact_frame, proximity_ok, wrist_near in contacts:
            contact_rows.append((video_id, rally_id, contact_frame,
                                 _format_bool(proximity_ok), _format_bool(wrist_near)))
        log.info('%s: %d rallies, %d contacts', video_id, len(spans), len(contacts))

    with args.rally_spans_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'rally_id', 'start_frame', 'end_frame'])
        writer.writerows(span_rows)
    # wrist_near is the contact wrist-check verdict: True = keep, blank = filter not run (no pose).
    # Every detected candidate is written (the RAW set); downstream defaults to the wrist_near-True
    # rows (the FILTERED set), so nothing recall-first loses its input.
    with args.contact_frames_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['video_id', 'rally_id', 'contact_frame', 'proximity_ok', 'wrist_near'])
        writer.writerows(contact_rows)
    log.info('wrote %d rally spans, %d contacts', len(span_rows), len(contact_rows))


if __name__ == '__main__':
    main()
