"""Exclude sustained court absence and slow-motion footage from rally detection.

Missing inputs contribute an all-False component. Camera changes alone do not
distinguish live alternate coverage from replays.
"""

import logging

import numpy as np

from annotator.array_calc_helpers import compute_speed, rolling_nanmedian, true_runs
from annotator.config import (
    SLOWMO_SPEED_FRAC,
    BaseAnnotatorConfig,
    resolve,
)

log = logging.getLogger(__name__)


def combine_mask(
    court_present: np.ndarray | None,
    track: np.ndarray | None,
    rally_spans: list[tuple[int, int]] | None,
    n_frames: int,
    fps: float,
    *, non_evidence: np.ndarray | None = None,
) -> np.ndarray:
    """Combine sustained court absence and slow-motion flags, before short-run filtering."""
    court = court_absence_signal(court_present, n_frames, fps)
    velocity = velocity_drop_signal(
        track, rally_spans, n_frames, fps,
        non_evidence=non_evidence, baseline_exclude=court,
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
    frame zero uses the first live frame after it. An entirely masked track has
    no anchor and is rejected.
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

def court_absence_signal(court_present: np.ndarray | None, n_frames: int, fps: float) -> np.ndarray:
    """Exclude whole court-absent runs that reach the minimum duration.

    Scale the duration to video fps. Short detector blips stay unmasked.
    A missing court-present array gives an all-False mask.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if court_present is None:
        log.info('court-present mask missing; court-absence signal all-False')
        return mask
    if len(court_present) != n_frames:
        raise ValueError(f'court-present length {len(court_present)} != n_frames {n_frames}')
    absent = ~court_present.astype(bool)
    # Replay signals use FPS-scaled defaults independently of heuristic overrides.
    window = resolve(BaseAnnotatorConfig(), fps).thresholds.court_absent_window
    for start, end in true_runs(absent):
        if end - start >= window:
            mask[start:end] = True
    return mask


def velocity_drop_signal(
    track: np.ndarray | None, rally_spans: list[tuple[int, int]] | None, n_frames: int, fps: float,
    *, non_evidence: np.ndarray | None = None, baseline_exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Flag visible shuttle movement much slower than the video's usual rally speed.

    Compare rolling-median speed with the median over rally frames. Require
    movement at least as fast as the rest threshold: ordinary pauses between
    rallies must remain available for commentary matching.

    Steps touching ``non_evidence`` frames are omitted from speed measurement.
    ``baseline_exclude`` removes frames from the usual-speed estimate only.
    A flagged shuttle frame can still be classified from nearby measured steps;
    a window with no measurements is left unflagged.

    :return: One boolean per video frame; True means possible slow motion.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if track is None or not rally_spans:
        log.info('shuttle track or rally spans missing; velocity-drop signal all-False')
        return mask
    if len(track) != n_frames:
        raise ValueError(f'shuttle track length {len(track)} != n_frames {n_frames}')
    _validate_optional_frame_mask('non_evidence', non_evidence, len(track))
    _validate_optional_frame_mask('baseline_exclude', baseline_exclude, len(track))

    speed = compute_speed(track)
    if non_evidence is not None:
        unmeasured_steps = non_evidence[1:] | non_evidence[:-1]
        speed[1:][unmeasured_steps] = np.nan
    in_rally = np.zeros(len(track), dtype=bool)
    for start, end in rally_spans:
        in_rally[start:end] = True
    if baseline_exclude is not None:
        in_rally &= ~baseline_exclude

    rally_speed = speed[in_rally]
    if not np.any(~np.isnan(rally_speed)):
        log.info('no visible rally-span frames; velocity-drop signal all-False')
        return mask
    rally_median = float(np.nanmedian(rally_speed))
    if rally_median <= 0:
        log.info('rally median speed is zero; velocity-drop signal all-False')
        return mask

    # Replay signals use FPS-scaled defaults independently of heuristic overrides.
    values = resolve(BaseAnnotatorConfig(), fps).thresholds
    rolling_median = rolling_nanmedian(speed, values.rest_window)
    visible = track[:, 2] == 1
    below_norm = rolling_median < (SLOWMO_SPEED_FRAC * rally_median)
    moving = rolling_median >= values.rest_speed
    mask[below_norm & moving & visible] = True
    return mask


def _validate_optional_frame_mask(name: str, values: np.ndarray | None, n_frames: int) -> None:
    """Validate one optional frame-aligned boolean mask."""
    if values is not None and (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype != np.bool_
        or len(values) != n_frames
    ):
        raise ValueError(f'{name} must be a one-dimensional np.bool_ array of length {n_frames}')
