"""Rally-span detection from shuttle motion and optional serve evidence."""

import itertools

import numpy as np

from ..config import ResolvedAnnotatorConfig, Thresholds
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
    resolve_serve_gate,
)
from .trajectory import _rolling_mean

# Minimum tracked fraction of a window before shuttle speed can rule out rest.
VISIBILITY_REST_FRAC = 0.5
QUIET_START_REST_FRACTION = 0.8


def gap_is_high_shot_outside_frame(track: np.ndarray, gap_start: int, thresholds: Thresholds) -> bool:
    run_start = gap_start
    while (run_start > 0 and track[run_start - 1, 2] == 1
           and gap_start - run_start < thresholds.high_shot_oob_lookback_frames):
        run_start -= 1
    n_visible = gap_start - run_start
    if n_visible < thresholds.high_shot_oob_min_visible_frames:
        return False
    first_xy = track[run_start, :2]
    last_xy = track[gap_start - 1, :2]
    mean_velocity = (last_xy - first_xy) / (n_visible - 1)
    return bool(last_xy[1] + thresholds.high_shot_oob_extrap_frames * mean_velocity[1] < 0.0)


def gap_passes_reentry_guard(
    track: np.ndarray, gap_start: int, gap_end: int, variant: ReentryGuardVariant, buffer: float,
    thresholds: Thresholds,
) -> bool:
    if gap_end >= len(track):
        return True
    stop = gap_end
    limit = min(gap_end + thresholds.reentry_lookahead_frames, len(track))
    while stop < limit and track[stop, 2] == 1:
        stop += 1
    n_visible = stop - gap_end
    if n_visible < thresholds.reentry_min_visible_frames:
        return False
    descending = (track[stop - 1, 1] - track[gap_end, 1]) / (n_visible - 1) > 0.0
    near_top = track[gap_end, 1] <= buffer
    if variant is ReentryGuardVariant.TWO_SIDED:
        return bool(track[gap_start - 1, 1] <= buffer and near_top and descending)
    return bool(near_top and descending)


def gap_state_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Thresholds, demotion_bound: int,
    reentry_guard_variant: ReentryGuardVariant | None, reentry_guard_buffer: float | None,
) -> np.ndarray:
    speed_median = rolling_nanmedian(speed, thresholds.rest_window)
    slow = speed_median < thresholds.rest_speed
    high_shot_oob = np.zeros(len(track), dtype=bool)
    dead = np.zeros(len(track), dtype=bool)
    for gap_start, gap_end in true_runs(track[:, 2] != 1):
        holds_open = gap_is_high_shot_outside_frame(track, gap_start, thresholds)
        if holds_open and reentry_guard_variant is not None:
            assert reentry_guard_buffer is not None
            holds_open = gap_passes_reentry_guard(
                track, gap_start, gap_end, reentry_guard_variant, reentry_guard_buffer, thresholds,
            )
        if holds_open:
            demotion_frame = min(gap_start + demotion_bound, gap_end)
            high_shot_oob[gap_start:demotion_frame] = True
            dead[demotion_frame:gap_end] = True
        elif gap_end - gap_start > thresholds.blip_max_frames:
            dead[gap_start:gap_end] = True
    return dead | (slow & ~high_shot_oob)


def build_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Thresholds, *,
    gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None, reentry_guard_buffer: float | None = None,
) -> np.ndarray:
    """Per-frame rest flag: slow or mostly untracked across the window.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param track: `(t, 3)` track, for the visibility column.
    :param thresholds: final per-video numeric thresholds.
    :return: `(t,)` bool, True where the frame reads as rest.
    """
    if gap_state_demotion_bound is not None:
        return gap_state_rest_mask(
            speed, track, thresholds, gap_state_demotion_bound,
            reentry_guard_variant, reentry_guard_buffer,
        )
    rest_window = thresholds.rest_window
    rest_speed = thresholds.rest_speed
    speed_median = rolling_nanmedian(speed, rest_window)  # (t,)
    slow = speed_median < rest_speed  # NaN windows read not-slow here...
    visible = (track[:, 2] == 1).astype(float)  # (t,) 1.0 where tracked
    frac_visible = _rolling_mean(visible, rest_window)  # (t,) fraction tracked in window
    mostly_untracked = frac_visible < VISIBILITY_REST_FRAC  # ...and the OR below catches them
    return slow | mostly_untracked


def rally_regions(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Find fast bursts, rest periods and the regions between long rests.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: final per-video numeric thresholds.
    :return: `(fast_runs, rest_runs, regions)`. fast_runs are the sustained-fast runs (>=
        start_min_frames above start_speed); rest_runs are every at_rest run; regions are the
        maximal non-long-rest runs (a long rest is a rest run >= end_rest_frames).
    """
    start_speed = thresholds.start_speed
    end_rest_frames = thresholds.end_rest_frames
    start_min_frames = thresholds.start_min_frames

    fast = np.nan_to_num(speed, nan=0.0) > start_speed  # (t,) NaN steps are not fast
    rest_runs = true_runs(at_rest)
    long_rest = np.zeros(len(speed), dtype=bool)  # (t,) frames inside an extended rest
    for start, end in rest_runs:
        if end - start >= end_rest_frames:
            long_rest[start:end] = True
    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= start_min_frames]
    regions = true_runs(~long_rest)
    return fast_runs, rest_runs, regions


def burst_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds,
) -> list[tuple[int, int]]:
    """Start each rally at a fast burst and end it at the next long rest.

    A long rest lasts at least ``thresholds.end_rest_frames``. Between long rests,
    a rally starts at the first burst above ``thresholds.start_speed`` for at
    least ``thresholds.start_min_frames``. A region with no such burst yields no
    rally. The final rally can extend to the end of the video.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: final per-video numeric thresholds.
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    fast_runs, _rest_runs, regions = rally_regions(speed, at_rest, thresholds)

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


def active_region_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds, span_open: SpanOpen,
) -> list[tuple[int, int]]:
    """Span finder under a SpanOpen rule (no serve gating).

    REGION_START drops the qualifying-burst gate: every active region yields one span
    `(region_start, region_end)`. BACK_FILL keeps the gate (a region needs a qualifying
    fast burst) but opens the span at region_start instead of at the burst.

    :param speed: `(t,)` per-frame speed (NaN on non-visible steps).
    :param at_rest: `(t,)` per-frame rest flag.
    :param thresholds: final per-video numeric thresholds.
    :param span_open: REGION_START or BACK_FILL.
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    fast_runs, _rest_runs, regions = rally_regions(speed, at_rest, thresholds)
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


def quiet_start_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds, window: int,
) -> list[tuple[int, int]]:
    fast_runs, _rest_runs, regions = rally_regions(speed, at_rest, thresholds)
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


def last_rest_close(rest_runs: list[tuple[int, int]], open_frame: int, next_burst: int) -> int:
    """End a rally at the start of the last rest period before the next serve.

    The rest must start after this rally opens and end at or before the next
    qualifying burst. If no rest fits, end the rally at that next burst.

    :param rest_runs: every `(start, end)` at_rest run, ascending (true_runs order).
    :param open_frame: this span's opening (qualifying) burst frame.
    :param next_burst: the next qualifying burst frame (where close='burst' would cut).
    :return: the close frame (half-open span end).
    """
    rest_starts = [rest_start for rest_start, rest_end in rest_runs
                   if rest_end <= next_burst and rest_start > open_frame]
    return rest_starts[-1] if rest_starts else next_burst  # ascending, so [-1] is the last run


def serve_qualified_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds,
    options: ServeStartOptions, span_open: SpanOpen | None,
) -> list[tuple[int, int]]:
    """Find rallies by checking for serve preparation before fast shuttle bursts.

    A burst qualifies when its preceding frames show serve preparation. If none
    qualify in a region, TRIM starts a rally at its first burst and REJECT drops
    the region. Regions with no fast bursts yield no rally.

    With ``options.close=None``, start at the first qualifying burst and end at
    the region's end. BURST and LAST_REST instead open a rally at each qualifying
    burst and choose where the previous rally ends. BACK_FILL overrides splitting
    and opens one rally at the region's start. REGION_START is incompatible with
    serve detection and is rejected by ``segment_video`` before this call.

    :param speed: (t,) per-frame speed (NaN on non-visible steps).
    :param at_rest: (t,) per-frame rest flag.
    :param thresholds: final per-video numeric thresholds.
    :param options: the serve-start gate inputs and mode.
    :param span_open: None (burst-open) or BACK_FILL (open qualifying regions at region_start).
    :return: list of `(start_frame, end_frame)` half-open rally spans.
    """
    gate = resolve_serve_gate(options)

    mode = options.mode
    close = options.close

    fast_runs, rest_runs, regions = rally_regions(speed, at_rest, thresholds)

    spans: list[tuple[int, int]] = []
    no_qualify_regions: list[tuple[int, int]] = []
    n_regions_with_burst = 0
    # Closely spaced serve detections can split one rally in two. Record their
    # spacing for diagnostics without changing which bursts qualify.
    qualifying_counts: list[int] = []
    qualifying_spacings: list[int] = []
    for region_start, region_end in regions:
        bursts = [start for start, _ in fast_runs if region_start <= start < region_end]
        if not bursts:
            continue
        n_regions_with_burst += 1
        # Check every burst for splitting and diagnostics, even in single-span mode.
        qualifying = [start for start in bursts if gate.qualifies(start)]
        qualifying_counts.append(len(qualifying))
        qualifying_spacings.extend(int(later - earlier)
                                   for earlier, later in itertools.pairwise(qualifying))

        if not qualifying:
            if mode is ServeStartMode.TRIM:
                spans.append((int(bursts[0]), int(region_end)))
            no_qualify_regions.append((int(region_start), int(region_end)))
        elif span_open is SpanOpen.BACK_FILL:
            spans.append((int(region_start), int(region_end)))
        elif close is None:
            spans.append((int(qualifying[0]), int(region_end)))
        else:
            for idx, open_frame in enumerate(qualifying):
                if idx + 1 < len(qualifying):
                    next_burst = qualifying[idx + 1]
                    close_frame = (next_burst if close is ServeStartClose.BURST
                                   else last_rest_close(rest_runs, open_frame, next_burst))
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
    track: np.ndarray, resolved: ResolvedAnnotatorConfig,
    serve_start: ServeStartOptions | None = None,
) -> list[tuple[int, int]]:
    """Find spans using one video's resolved numeric values and opening policy."""
    thresholds = resolved.thresholds
    speed = compute_speed(track)
    at_rest = build_rest_mask(
        speed, track, thresholds, gap_state_demotion_bound=resolved.gap_state_demotion_bound,
        reentry_guard_variant=resolved.reentry_guard_variant,
        reentry_guard_buffer=resolved.reentry_guard_buffer,
    )
    if serve_start is not None:
        return serve_qualified_rally_spans(speed, at_rest, thresholds, serve_start, resolved.span_open)
    if resolved.quiet_start_window is not None:
        return quiet_start_rally_spans(speed, at_rest, thresholds, resolved.quiet_start_window)
    if resolved.span_open is not None:
        return active_region_rally_spans(speed, at_rest, thresholds, resolved.span_open)
    return burst_rally_spans(speed, at_rest, thresholds)
