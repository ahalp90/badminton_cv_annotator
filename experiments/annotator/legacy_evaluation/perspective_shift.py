"""Historical camera-change signal; current replay masking uses court absence and speed."""
import logging

import numpy as np

log = logging.getLogger(__name__)
PERSPECTIVE_SHIFT_THRESHOLD = 0.05
HOMOGRAPHY_CORNER_COLS = [
    'upleft_x', 'upleft_y',
    'upright_x', 'upright_y',
    'downleft_x', 'downleft_y',
    'downright_x', 'downright_y',
]


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: the smallest value whose cumulative weight reaches half.

    Ties resolve to the lower value (searchsorted left). Used to pick the
    dominant broadcast view, weighting each segment by its frame duration.

    :param values: `(k,)` values.
    :param weights: `(k,)` non-negative weights.
    :return: the weighted-median value.
    """
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    half = 0.5 * cumulative[-1]
    idx = int(np.searchsorted(cumulative, half))
    return float(sorted_values[min(idx, len(sorted_values) - 1)])



def perspective_shift_signal(homography_rows: list[dict], n_frames: int) -> np.ndarray:
    """Fire on segments whose court corners deviate from the dominant view.

    The dominant broadcast view is the duration-weighted median of each corner
    coordinate across all segments; a segment whose mean corner displacement
    from it (normalised by the reference court's bounding-box diagonal) exceeds
    PERSPECTIVE_SHIFT_THRESHOLD has a different view and its frames fire.

    This is a descriptive camera-change signal, not a replay classification.
    The normalisation uses the reference court's diagonal, so no frame resolution
    is needed.

    :param homography_rows: per-segment dict rows for this video; may be empty.
    :param n_frames: video frame count (the mask length).
    :return: `(n_frames,)` bool signal.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if not homography_rows:
        log.info('homography rows missing; perspective-shift signal all-False')
        return mask

    starts = np.array([int(row['start_frame']) for row in homography_rows])  # (n_seg,)
    ends = np.array([int(row['end_frame']) for row in homography_rows])  # (n_seg,)
    durations = (ends - starts).astype(float)  # (n_seg,) weight
    corners = np.array(
        [[float(row[col]) for col in HOMOGRAPHY_CORNER_COLS] for row in homography_rows]
    )  # (n_seg, 8)

    reference = np.array(
        [_weighted_median(corners[:, col], durations) for col in range(corners.shape[1])]
    )  # (8,) dominant view
    seg_xy = corners.reshape(len(homography_rows), 4, 2)  # (n_seg, 4, 2) [corner, xy]
    ref_xy = reference.reshape(4, 2)  # (4, 2)

    mean_displacement = np.linalg.norm(seg_xy - ref_xy, axis=2).mean(axis=1)  # (n_seg,)
    ref_span = ref_xy.max(axis=0) - ref_xy.min(axis=0)  # (2,) bbox width, height
    diagonal = float(np.hypot(*ref_span))
    if diagonal <= 0:
        log.info('degenerate reference court (zero diagonal); perspective-shift signal all-False')
        return mask

    shifted = (mean_displacement / diagonal) > PERSPECTIVE_SHIFT_THRESHOLD  # (n_seg,)
    for seg in np.flatnonzero(shifted):
        mask[starts[seg]:ends[seg]] = True
    return mask
