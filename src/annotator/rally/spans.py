"""Rally-span detection from shuttle motion and optional serve evidence."""

import numpy as np

from ..config import (
    END_REST_FRAMES,
    REST_SPEED,
    REST_WINDOW,
    START_MIN_FRAMES,
    START_SPEED,
    RallySegmentationThresholds,
)
from ..fps_constants import FpsConstants
from ..types import (
    ReentryGuardVariant,
    SpanOpen,
    compute_speed,
    rolling_nanmedian,
    true_runs,
)
from .serve import (
    ServeStartClose,
    ServeStartMode,
    ServeStartOptions,
    _resolve_serve_gate,
)
from .trajectory import _rolling_mean


# Fraction of a window that must be tracked for it to read as seeing the shuttle.
VISIBILITY_REST_FRAC = 0.5
QUIET_START_REST_FRACTION = 0.8


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
    speed: np.ndarray, track: np.ndarray, thresholds: RallySegmentationThresholds, constants: FpsConstants, demotion_bound: int,
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
    speed: np.ndarray, track: np.ndarray, thresholds: RallySegmentationThresholds | None = None, *,
    constants: FpsConstants | None = None, gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None, reentry_guard_buffer: float | None = None,
) -> np.ndarray:
    """Per-frame rest flag: slow or mostly untracked across the window.

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
    speed: np.ndarray, at_rest: np.ndarray, thresholds: RallySegmentationThresholds | None,
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
    speed: np.ndarray, at_rest: np.ndarray, thresholds: RallySegmentationThresholds | None = None,
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
    speed: np.ndarray, at_rest: np.ndarray, thresholds: RallySegmentationThresholds | None, span_open: SpanOpen,
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
    speed: np.ndarray, at_rest: np.ndarray, thresholds: RallySegmentationThresholds, window: int,
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
    speed: np.ndarray, at_rest: np.ndarray, thresholds: RallySegmentationThresholds | None,
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
    gate = _resolve_serve_gate(options)

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
        qualifying = [start for start in bursts if gate.qualifies(start)]
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


def find_rally_spans(
    track: np.ndarray, thresholds: RallySegmentationThresholds | None = None,
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
