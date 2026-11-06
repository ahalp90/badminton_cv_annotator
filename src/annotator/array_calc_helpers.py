"""Small array calculations shared by annotator stages."""

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def compute_speed(track: np.ndarray) -> np.ndarray:
    """Shuttle distance per frame in normalised image coordinates.

    Frame i measures movement from i-1 to i. NaN at frame zero or when either
    endpoint is invisible. Input: (frames, 3) xy and visibility.
    """
    xy = track[:, :2]
    visibility = track[:, 2]
    step = np.diff(xy, axis=0)
    step_speed = np.linalg.norm(step, axis=1)
    both_visible = (visibility[:-1] == 1) & (visibility[1:] == 1)

    speed = np.full(len(track), np.nan)
    speed[1:] = np.where(both_visible, step_speed, np.nan)
    return speed


def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Consecutive True runs as (start, end) ranges, incl start, excl end."""
    padded = np.concatenate([[0], mask.astype(np.int8), [0]])  # zeros close runs that reach either array end
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def rolling_nanmedian(values: np.ndarray, window: int) -> np.ndarray:
    """Centred median, one output per input, ignoring missing values.

    Pad with NaN at the ends. Return NaN where the entire window is missing.
    """
    left = window // 2
    right = window - 1 - left
    padded = np.concatenate([np.full(left, np.nan), values, np.full(right, np.nan)])
    windows = sliding_window_view(padded, window)  # (t, window)
    with warnings.catch_warnings():
        # Entirely missing windows are expected; keep their NaN without a warning.
        warnings.simplefilter('ignore', category=RuntimeWarning)
        return np.nanmedian(windows, axis=1)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred mean, incl. zeroes; average by sample count available (smaller at array edges).

    Input values must have no NaNs. Mask unwanted samples before calling.
    """
    kernel = np.ones(window)
    counts = np.convolve(np.ones_like(values), kernel, mode='same')  # samples per position
    sums = np.convolve(values, kernel, mode='same')
    return sums / counts


def rolling_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred mean ignoring NaNs; divide by the non-NaN sample count in each window.

    Return NaN if no samples remain.
    """
    kernel = np.ones(window)
    valid = ~np.isnan(values)
    filled = np.where(valid, values, 0.0)
    counts = np.convolve(valid.astype(float), kernel, mode='same')
    sums = np.convolve(filled, kernel, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        return sums / counts
