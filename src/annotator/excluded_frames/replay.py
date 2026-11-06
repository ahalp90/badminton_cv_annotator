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

# ---------------------------------------------------------------------------
# Signal 1: court absence
# ---------------------------------------------------------------------------
def court_absence_signal(court_present: np.ndarray | None, n_frames: int, fps: float) -> np.ndarray:
    """Fire across court-absent runs that reach the fps-scaled window.

    A sustained absence (>= the window) masks its whole run; a one- or two-frame
    detector blip does not. The run-length gate masks the entire sustained
    absence rather than a single window-sized slice.

    :param court_present: `(frames,)` bool court-present flag, or None.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if court_present is None:
        log.info('court-present mask missing; court-absence signal all-False')
        return mask
    if len(court_present) != n_frames:
        raise ValueError(f'court-present length {len(court_present)} != n_frames {n_frames}')
    absent = ~court_present.astype(bool)  # (frames,)
    # Replay signals use FPS-scaled defaults independently of heuristic overrides.
    window = resolve(BaseAnnotatorConfig(), fps).thresholds.court_absent_window
    for start, end in true_runs(absent):
        if end - start >= window:
            mask[start:end] = True
    return mask


# ---------------------------------------------------------------------------
# Signal 3: velocity drop (slow motion)
# ---------------------------------------------------------------------------
def _validate_optional_frame_mask(name: str, values: np.ndarray | None, n_frames: int) -> None:
    """Validate one optional frame-aligned boolean mask."""
    if values is not None and (
        not isinstance(values, np.ndarray)
        or values.ndim != 1
        or values.dtype != np.bool_
        or len(values) != n_frames
    ):
        raise ValueError(f'{name} must be a one-dimensional np.bool_ array of length {n_frames}')


def velocity_drop_signal(
    track: np.ndarray | None, rally_spans: list[tuple[int, int]] | None, n_frames: int, fps: float,
    *, non_evidence: np.ndarray | None = None, baseline_exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Fire where visible shuttle speed drops well below the rally norm (slow-mo).

    The rally norm is the median per-frame speed across all rally-span frames of
    the video. A frame fires when its rolling median speed sits in the slow-mo
    band (at or above REST_SPEED but below SLOWMO_SPEED_FRAC of the norm) AND
    the shuttle is visible. Invisible frames are the court-absence signal's job.
    Genuine rest (below REST_SPEED) deliberately does NOT fire: a resting
    shuttle is the between-rallies state, and that is exactly where the
    commentary pairing joins with live spans; masking rest would hold every
    post-rally chunk out of pairing. Slow motion means moving, slowly.

    Speed steps touching a non_evidence frame are unmeasured: they feed neither the baseline nor the rolling median; edge frames of a graded run can still fire on neighbouring measured evidence, and a window with no measured step reads not-slow.

    :param track: `(t, 3)` whole-video shuttle track, or None.
    :param rally_spans: `[(start, end), ...]` rally spans for this video, or None/empty.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if track is None or not rally_spans:
        log.info('shuttle track or rally spans missing; velocity-drop signal all-False')
        return mask
    if len(track) != n_frames:
        raise ValueError(f'shuttle track length {len(track)} != n_frames {n_frames}')
    _validate_optional_frame_mask('non_evidence', non_evidence, len(track))
    _validate_optional_frame_mask('baseline_exclude', baseline_exclude, len(track))

    speed = compute_speed(track)  # (t,) per-frame, NaN on non-visible steps
    if non_evidence is not None:
        unmeasured_steps = non_evidence[1:] | non_evidence[:-1]  # (t-1,) either endpoint is graded
        speed[1:][unmeasured_steps] = np.nan
    in_rally = np.zeros(len(track), dtype=bool)  # (t,) frames inside any rally span
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
    rolling_median = rolling_nanmedian(speed, values.rest_window)  # (t,)
    visible = track[:, 2] == 1  # (t,)
    below_norm = rolling_median < (SLOWMO_SPEED_FRAC * rally_median)  # NaN windows read not-slow
    moving = rolling_median >= values.rest_speed  # rest is not slow-mo (see docstring)
    mask[below_norm & moving & visible] = True
    return mask


# ---------------------------------------------------------------------------
# Combination
# ---------------------------------------------------------------------------
def combine_mask(
    court_present: np.ndarray | None,
    track: np.ndarray | None,
    rally_spans: list[tuple[int, int]] | None,
    n_frames: int,
    fps: float,
    *, non_evidence: np.ndarray | None = None,
) -> np.ndarray:
    """Combine court absence and shuttle-speed evidence for off-rally masking.

    :return: `(n_frames,)` bool mask, True where any signal fires.
    """
    court = court_absence_signal(court_present, n_frames, fps)
    velocity = velocity_drop_signal(
        track, rally_spans, n_frames, fps,
        non_evidence=non_evidence, baseline_exclude=court,
    )
    return court | velocity


def filter_short_exclusion_runs(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Keep each raw detector run in full when it is long enough.

    The input contains raw detector flags. For each raw run ``[start, end)``,
    filtering keeps the whole run when ``end - start >= min_frames``. Applying
    this function to an already filtered mask therefore leaves it unchanged.

    :param mask: one-dimensional boolean raw detector flags.
    :param min_frames: positive number of consecutive flags needed to keep a run.
    :return: one-dimensional boolean mask of duration-filtered whole runs.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError('mask must be a one-dimensional boolean array')
    if isinstance(min_frames, bool) or not isinstance(min_frames, (int, np.integer)) or min_frames < 1:
        raise ValueError(f'min_frames must be a positive integer, got {min_frames!r}')

    believed = np.zeros_like(mask)
    for start, end in true_runs(mask):
        if end - start >= min_frames:
            believed[start:end] = True
    return believed


def filter_exclusion_mask(raw_mask: np.ndarray, n_frames: int, min_frames: int) -> np.ndarray:
    """Validate a dead-time mask and discard runs shorter than the trusted minimum."""
    if len(raw_mask) != n_frames:
        raise ValueError(f'mask length {len(raw_mask)} != track length {n_frames}')
    if raw_mask.all():
        raise ValueError('mask is all True: no live frame to anchor a frozen position to')
    return filter_short_exclusion_runs(raw_mask, min_frames)


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
