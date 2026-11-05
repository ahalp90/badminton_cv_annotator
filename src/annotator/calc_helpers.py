"""Small array calculations shared by annotator stages."""

import warnings

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


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
    start, -1 marks one-past a run end. Shared with replay masking's court-absence
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
    with replay masking's slow-motion signal.

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
