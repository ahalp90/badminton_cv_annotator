"""Find initial rally ranges from shuttle movement and rests.

``segment_video`` masks excluded footage, finds rally ranges, then detects
heuristic contacts. Missing shuttle frames can still belong to a rally when
the visible trajectory suggests a high shot leaving and returning to view.
"""

import numpy as np

from annotator.array_calc_helpers import (
    compute_speed,
    rolling_nanmedian,
    true_runs,
)
from annotator.config import FpsScaledAnnotatorConfig, Thresholds
from annotator.contacts.heuristics import assemble_contacts
from annotator.excluded_frames.replay import apply_replay_mask
from annotator.types import ContactCandidate


def segment_video(
    track: np.ndarray, *,
    video_config: FpsScaledAnnotatorConfig,
    exclusion_mask: np.ndarray | None = None,
    nearest_wrist_distances: np.ndarray | None = None,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list[tuple[int, int]], list[ContactCandidate]]:
    """Mask excluded footage, find rallies, then check their contact candidates.

    :param track: (frames, 3) normalised shuttle xy and visibility.
    :param video_config: Settings already scaled for this video's frame rate.
    :param exclusion_mask: Optional footage mask; excluded positions are frozen.
    :param nearest_wrist_distances: Optional nearest-wrist distances in body heights.
    :param spans: Supplied ranges skip rally detection; incl start, excl end.
    :return: Rally ranges and all candidates, including wrist/spacing failures.
    """
    if nearest_wrist_distances is not None and nearest_wrist_distances.shape != (len(track),):
        raise ValueError('nearest_wrist_distances must have shape (len(track),)')
    if exclusion_mask is not None:
        track = apply_replay_mask(track, exclusion_mask)
    if spans is None:
        spans = find_rally_spans(track, video_config)
    return spans, assemble_contacts(
        track, spans, video_config.thresholds, nearest_wrist_distances,
        smoothing_mode=video_config.smoothing_mode,
    )


def find_rally_spans(
    track: np.ndarray, video_config: FpsScaledAnnotatorConfig,
) -> list[tuple[int, int]]:
    """Find initial rally ranges using the configured rest and start rules."""
    thresholds = video_config.thresholds
    speed = compute_speed(track)
    at_rest = build_rest_mask(
        speed, track, thresholds, max_high_shot_gap_frames=video_config.max_high_shot_gap_frames,
        reentry_guard_buffer=video_config.reentry_guard_buffer,
    )
    return active_region_rally_spans(speed, at_rest, thresholds, video_config.require_fast_burst)


# Rest and missing-shuttle rules

def build_rest_mask(
    speed: np.ndarray, track: np.ndarray, thresholds: Thresholds, *,
    max_high_shot_gap_frames: int, reentry_guard_buffer: float,
) -> np.ndarray:
    """Mark slow movement and long unexplained visibility gaps as rest.

    A likely high shot may stay in play for max_high_shot_gap_frames missing
    frames. It must leave and return near the image top, moving down on return.
    Other gaps count entirely as rest when they exceed
    max_unexplained_shuttle_gap_frames.
    """
    speed_median = rolling_nanmedian(speed, thresholds.rest_window)
    slow = speed_median < thresholds.rest_speed
    high_shot_oob = np.zeros(len(track), dtype=bool)
    dead = np.zeros(len(track), dtype=bool)
    for gap_start, gap_end in true_runs(track[:, 2] != 1):
        likely_high_shot = gap_is_high_shot_outside_frame(track, gap_start, thresholds)
        if likely_high_shot:
            likely_high_shot = gap_passes_reentry_guard(
                track, gap_start, gap_end, reentry_guard_buffer, thresholds,
            )
        if likely_high_shot:
            in_play_until = min(gap_start + max_high_shot_gap_frames, gap_end)
            high_shot_oob[gap_start:in_play_until] = True
            dead[in_play_until:gap_end] = True
        elif gap_end - gap_start > thresholds.max_unexplained_shuttle_gap_frames:
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
    track: np.ndarray, gap_start: int, gap_end: int, buffer: float,
    thresholds: Thresholds,
) -> bool:
    """Check that the shuttle returns near the top of the image, moving downwards.

    Check the last position before the gap too. A gap reaching video end
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
    return bool(track[gap_start - 1, 1] <= buffer and near_top and descending)


# Rally starts and ends

def active_region_rally_spans(
    speed: np.ndarray, at_rest: np.ndarray, thresholds: Thresholds, require_fast_burst: bool,
) -> list[tuple[int, int]]:
    """Start rallies at the beginning of each interval between long rests.

    Optionally require sustained fast motion somewhere in the interval.
    Returned ranges include start, exclude end.
    """
    long_rest = np.zeros(len(speed), dtype=bool)
    for start, end in true_runs(at_rest):
        if end - start >= thresholds.end_rest_frames:
            long_rest[start:end] = True
    regions = true_runs(~long_rest)
    fast = np.nan_to_num(speed, nan=0.0) > thresholds.start_speed
    fast_runs = [(start, end) for start, end in true_runs(fast) if end - start >= thresholds.start_min_frames]

    spans = []
    for region_start, region_end in regions:
        has_burst = any(region_start <= start < region_end for start, _ in fast_runs)
        if not require_fast_burst or has_burst:
            spans.append((int(region_start), int(region_end)))
    return spans
