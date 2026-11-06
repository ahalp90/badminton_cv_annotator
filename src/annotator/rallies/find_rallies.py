"""Find initial rally ranges from shuttle movement and rests.

``segment_video`` masks excluded footage, finds rally ranges, then detects
heuristic contacts. Missing shuttle frames can still belong to a rally when
the visible trajectory suggests a high shot leaving and returning to view.
"""

import numpy as np

from annotator.array_calc_helpers import (
    compute_speed,
    rolling_mean,
    rolling_nanmedian,
    true_runs,
)
from annotator.config import FpsScaledAnnotatorConfig, Thresholds
from annotator.contacts.heuristics import assemble_contacts
from annotator.excluded_frames.replay import apply_replay_mask
from annotator.types import ContactCandidate, ReentryGuardVariant, SpanOpen

# Minimum tracked fraction of a window before shuttle speed can rule out rest.
VISIBILITY_REST_FRAC = 0.5


def segment_video(
    track: np.ndarray, *,
    video_config: FpsScaledAnnotatorConfig,
    exclusion_mask: np.ndarray | None = None,
    sticky_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Mask excluded footage, find rallies, then check their contact candidates.

    :param track: (frames, 3) normalised shuttle xy and visibility.
    :param video_config: Settings already scaled for this video's frame rate.
    :param exclusion_mask: Optional footage mask; excluded positions are frozen.
    :param sticky_distances: Optional nearest-wrist distances in body heights.
    :param spans: Supplied ranges skip rally detection; incl start, excl end.
    :return: Rally ranges and all candidates, including wrist/spacing failures.
    """
    if sticky_distances is not None and sticky_distances.shape != (len(track),):
        raise ValueError('sticky_distances must have shape (len(track),)')
    if exclusion_mask is not None:
        track = apply_replay_mask(track, exclusion_mask)
    if spans is None:
        spans = find_rally_spans(track, video_config)
    return spans, assemble_contacts(
        track, spans, video_config.thresholds, sticky_distances,
        smoothing_mode=video_config.smoothing_mode,
    )


def find_rally_spans(
    track: np.ndarray, video_config: FpsScaledAnnotatorConfig,
) -> list[tuple[int, int]]:
    """Find initial rally ranges using the configured rest and start rules."""
    thresholds = video_config.thresholds
    speed = compute_speed(track)
    at_rest = build_rest_mask(
        speed, track, thresholds, gap_state_demotion_bound=video_config.gap_state_demotion_bound,
        reentry_guard_variant=video_config.reentry_guard_variant,
        reentry_guard_buffer=video_config.reentry_guard_buffer,
    )
    if video_config.span_open is not None:
        return active_region_rally_spans(speed, at_rest, thresholds, video_config.span_open)
    return burst_rally_spans(speed, at_rest, thresholds)


# Rest and missing-shuttle rules

def build_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Thresholds, *,
    gap_state_demotion_bound: int | None = None,
    reentry_guard_variant: ReentryGuardVariant | None = None, reentry_guard_buffer: float | None = None,
) -> np.ndarray:
    """Mark frames that look like rest from speed and shuttle visibility.

    With ``gap_state_demotion_bound``, allow short gaps and likely high shots.
    Otherwise, mark a frame as rest when its window is slow or mostly untracked.

    :param speed: One speed per frame; NaN for steps with missing positions.
    :param track: (frames, 3) shuttle xy and visibility.
    :param thresholds: Numeric settings already scaled for the video.
    :return: One boolean per frame; True means rest.
    """
    if gap_state_demotion_bound is not None:
        return gap_state_rest_mask(
            speed, track, thresholds, gap_state_demotion_bound,
            reentry_guard_variant, reentry_guard_buffer,
        )
    rest_window = thresholds.rest_window
    rest_speed = thresholds.rest_speed
    speed_median = rolling_nanmedian(speed, rest_window)
    slow = speed_median < rest_speed
    visible = (track[:, 2] == 1).astype(float)
    frac_visible = rolling_mean(visible, rest_window)
    # Missing speed alone is inconclusive; low visibility also counts as rest.
    mostly_untracked = frac_visible < VISIBILITY_REST_FRAC
    return slow | mostly_untracked


def gap_state_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Thresholds, demotion_bound: int,
    reentry_guard_variant: ReentryGuardVariant | None, reentry_guard_buffer: float | None,
) -> np.ndarray:
    """Mark slow movement and long unexplained visibility gaps as rest.

    Allow up to ``demotion_bound`` missing frames for a likely high shot;
    mark the rest of the gap as rest. Other gaps count as rest when longer than
    ``blip_max_frames``. Optional re-entry checks confirm the high-shot guess.
    """
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


def gap_is_high_shot_outside_frame(track: np.ndarray, gap_start: int, thresholds: Thresholds) -> bool:
    """Check whether the shuttle's last visible movement projects above the image.

    Use consecutive visible frames immediately before the gap. This allows a
    high shot to stay in play while the shuttle is out of view.
    """
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
    """Check that the shuttle returns near the top of the image, moving downwards.

    TWO_SIDED also checks the position before the gap. A gap reaching video end
    passes because there is no return to inspect.
    """
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


# Rally starts and ends

def active_region_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds, span_open: SpanOpen,
) -> list[tuple[int, int]]:
    """Start rallies at the beginning of each interval between long rests.

    BACK_FILL requires a sustained fast burst somewhere in that interval.
    REGION_START keeps every interval. Ranges include start, exclude end.
    """
    fast_runs, _rest_runs, regions = rally_regions(speed, at_rest, thresholds)
    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        if span_open is SpanOpen.REGION_START:
            spans.append((int(region_start), int(region_end)))
            continue
        has_burst = any(region_start <= start < region_end for start, _ in fast_runs)
        if has_burst:
            spans.append((int(region_start), int(region_end)))
    return spans


def burst_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds,
) -> list[tuple[int, int]]:
    """Start at the first sustained fast burst after a long rest.

    End at the next long rest or video end. Skip intervals with no qualifying
    burst. Returned frame ranges include start, exclude end.
    """
    fast_runs, _rest_runs, regions = rally_regions(speed, at_rest, thresholds)

    spans: list[tuple[int, int]] = []
    for region_start, region_end in regions:
        # Keep the first sustained burst in this interval between long rests.
        burst_start = next(
            (start for start, _ in fast_runs if region_start <= start < region_end),
            None,
        )
        if burst_start is None:
            continue
        spans.append((int(burst_start), int(region_end)))
    return spans


def rally_regions(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Find fast bursts, rest runs and intervals between long rests.

    Fast bursts exceed start_speed for at least start_min_frames. A long rest
    lasts at least end_rest_frames. Return (fast_runs, all_rest_runs, regions),
    each as frame ranges including start and excluding end.
    """
    start_speed = thresholds.start_speed
    end_rest_frames = thresholds.end_rest_frames
    start_min_frames = thresholds.start_min_frames

    fast = np.nan_to_num(speed, nan=0.0) > start_speed
    rest_runs = true_runs(at_rest)
    long_rest = np.zeros(len(speed), dtype=bool)
    for start, end in rest_runs:
        if end - start >= end_rest_frames:
            long_rest[start:end] = True
    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= start_min_frames]
    regions = true_runs(~long_rest)
    return fast_runs, rest_runs, regions
