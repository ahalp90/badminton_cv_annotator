"""Shuttle-trajectory transforms shared by rally spans and contacts."""

import numpy as np

from ..types import true_runs


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


def apply_replay_mask(track: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Freeze replay/off-rally frames to the last live position so they read as rest.

    Returns a new `(t, 3)` array; `track` is not mutated. For each contiguous True run in
    `mask`, the run's xy (columns 0-1) is set to the xy of the last frame BEFORE the run, and
    its visibility (column 2) forced to 1. A run that starts at frame 0 has no earlier frame,
    so it takes the xy of the first frame AFTER it instead.

    Why: rally segmentation reads invisible or NaN-speed frames as not-rest, so replay closeups hold rally
    regions open. Freezing the position makes masked footage read as sustained sub-REST_SPEED
    rest (masked frames count as rest). Forcing visibility avoids the NaN-speed path reopening
    the region.

    Fail loud on a length mismatch, and on an all-True mask (nothing live to anchor a frozen
    position to, and a fully-masked video is senseless). An all-False mask has no True runs, so
    the untouched copy returns bit-identical by construction.

    :param track: `(t, 3)` `[x_norm, y_norm, visibility]` whole-video track.
    :param mask: `(t,)` bool, True on replay/off-rally frames (replay-mask `1_replay.npy` convention).
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
