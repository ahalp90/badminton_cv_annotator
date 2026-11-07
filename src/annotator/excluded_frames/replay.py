"""Exclude sustained court absence and slow-motion footage from rally detection.

Camera changes alone do not distinguish live alternate coverage from replays.
"""

import logging

import numpy as np

from annotator.array_calc_helpers import compute_speed, rolling_nanmedian, true_runs
from shared.frame_timing import frames_at_fps, speed_at_fps

log = logging.getLogger(__name__)

# Exclusion settings stay fixed when rally-detection settings are tuned.
COURT_ABSENCE_MIN_FRAMES_BASE30 = 15
SLOWMO_MEDIAN_WINDOW_BASE30 = 5
SLOWMO_MIN_MOVING_SPEED_BASE30 = 0.002
# A lower fraction avoids mistaking rally-end deceleration for slow motion.
SLOWMO_SPEED_FRAC = 0.15


def combine_mask(
    court_present: np.ndarray,
    track: np.ndarray,
    rally_spans: list[tuple[int, int]],
    n_frames: int,
    fps: float,
    *, rejected_shuttle_frames: np.ndarray,
) -> np.ndarray:
    """Combine sustained court absence and slow-motion flags, before short-run filtering."""
    court = court_absence_signal(court_present, n_frames, fps)
    velocity = velocity_drop_signal(
        track, rally_spans, n_frames, fps,
        rejected_shuttle_frames=rejected_shuttle_frames, baseline_exclude=court,
    )
    return court | velocity


def filter_exclusion_mask(raw_mask: np.ndarray, n_frames: int, min_frames: int) -> np.ndarray:
    """Check mask length and retain only excluded runs long enough to trust."""
    if len(raw_mask) != n_frames:
        raise ValueError(f'mask length {len(raw_mask)} != track length {n_frames}')
    if raw_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    return filter_short_exclusion_runs(raw_mask, min_frames)


def filter_short_exclusion_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep complete True runs lasting at least min_frames; remove shorter ones.

    Input and output: one boolean per frame. Filtering an already filtered
    mask leaves it unchanged.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('mask must be a one-dimensional boolean array')
    if isinstance(min_frames, bool) or not isinstance(min_frames, (int, np.integer)) or min_frames < 1:
        raise ValueError(f'min_frames must be a positive integer, got {min_frames!r}')

    filtered = np.zeros_like(mask)
    for start, end in true_runs(mask):
        if end - start >= min_frames:
            filtered[start:end] = True
    return filtered


def apply_replay_mask(track: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Freeze excluded footage so segmentation reads it as rest.

    Return a copy with masked positions fixed to the preceding live frame and
    visibility set to one so NaN speed cannot reopen a rally. A mask starting at
    frame zero uses the first live frame after it. Raise ValueError if every frame
    is masked: no live position to copy.
    """
    if len(mask) != len(track):
        raise ValueError(f'mask length {len(mask)} != track length {len(track)}')
    if mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')

    frozen = track.copy()
    for start, end in true_runs(mask):
        anchor = start - 1 if start > 0 else end
        frozen[start:end, :2] = track[anchor, :2]
        frozen[start:end, 2] = 1
    return frozen


# Court absence and slow motion

def court_absence_signal(court_present: np.ndarray, n_frames: int, fps: float) -> np.ndarray:
    """Exclude whole court-absent runs that reach the minimum duration.

    Scale the duration to video fps. Short gaps in court detection stay unmasked.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if len(court_present) != n_frames:
        raise ValueError(f'court-present length {len(court_present)} != n_frames {n_frames}')
    absent = ~court_present.astype(bool)
    window = frames_at_fps(COURT_ABSENCE_MIN_FRAMES_BASE30, fps)
    for start, end in true_runs(absent):
        if end - start >= window:
            mask[start:end] = True
    return mask


def velocity_drop_signal(
    track: np.ndarray, rally_spans: list[tuple[int, int]], n_frames: int, fps: float,
    *, rejected_shuttle_frames: np.ndarray, baseline_exclude: np.ndarray,
) -> np.ndarray:
    """Flag visible shuttle movement much slower than the video's usual rally speed.

    Compare rolling-median speed with the median over rally frames. Require
    movement at least as fast as the rest threshold: ordinary pauses between
    rallies must remain available for commentary matching.

    Skip speed measurements if either frame is marked ``rejected_shuttle_frames``.
    ``baseline_exclude`` removes frames from the usual-speed estimate only.
    Both masks contain one boolean per video frame.
    A ``rejected_shuttle_frames`` frame can still be marked slow motion based on nearby
    speeds. Leave windows with no speed measurements unflagged.

    :return: One boolean per video frame; True means possible slow motion.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if not rally_spans:
        log.info('no rallies for estimating usual shuttle speed; velocity-drop signal all-False')
        return mask
    if len(track) != n_frames:
        raise ValueError(f'shuttle track length {len(track)} != n_frames {n_frames}')

    speed = compute_speed(track)
    unmeasured_steps = rejected_shuttle_frames[1:] | rejected_shuttle_frames[:-1]
    speed[1:][unmeasured_steps] = np.nan
    in_rally = np.zeros(len(track), dtype=bool)
    for start, end in rally_spans:
        in_rally[start:end] = True
    in_rally = in_rally & ~baseline_exclude

    rally_speed = speed[in_rally]
    if not np.any(~np.isnan(rally_speed)):
        log.info('no visible rally-span frames; velocity-drop signal all-False')
        return mask
    rally_median = float(np.nanmedian(rally_speed))
    if rally_median <= 0:
        log.info('rally median speed is zero; velocity-drop signal all-False')
        return mask

    rolling_median = rolling_nanmedian(speed, frames_at_fps(SLOWMO_MEDIAN_WINDOW_BASE30, fps))
    visible = track[:, 2] == 1
    below_norm = rolling_median < (SLOWMO_SPEED_FRAC * rally_median)
    moving = rolling_median >= speed_at_fps(SLOWMO_MIN_MOVING_SPEED_BASE30, fps)
    mask[below_norm & moving & visible] = True
    return mask
