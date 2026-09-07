"""Shared shape checks for model and line-recovered court corners."""

import numpy as np

DEFAULT_AREA_BOUNDS = (0.01, 0.95)


def _is_convex(corners_norm: np.ndarray) -> bool:
    """:return: True when the TL->TR->BR->BL quad is convex (consistent turn direction).

    :param corners_norm: (4, 2) normalised xy in TL, TR, BR, BL order
    """
    edges = np.roll(corners_norm, -1, axis=0) - corners_norm  # (4, 2) edge vectors around the quad
    next_edges = np.roll(edges, -1, axis=0)  # (4, 2)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]  # (4,)
    return bool((crosses > 0).all() or (crosses < 0).all())


def _shoelace_area(corners_norm: np.ndarray) -> float:
    """:return: quad area in normalised units via the shoelace formula.

    :param corners_norm: (4, 2) normalised xy in TL, TR, BR, BL order
    """
    x = corners_norm[:, 0]
    y = corners_norm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quadrants_ok(corners: np.ndarray) -> bool:
    """:return: True when each corner sits in its expected quadrant about the centroid.

    y grows downward, so TL is the top-left quadrant (x<cx and y<cy), and so on
    clockwise. A swapped or mislabelled corner lands in the wrong quadrant.

    Deliberately stricter than pure vertex order: real broadcast and handheld
    courts project to roughly axis-aligned trapezoids, so a diamond-like quad
    (which this rejects) is treated as suspect by design; flagging fails closed.

    :param corners: (4, 2) xy in TL, TR, BR, BL order (any consistent pixel space)
    """
    cx, cy = corners.mean(axis=0)
    x = corners[:, 0]
    y = corners[:, 1]
    tl_ok = x[0] < cx and y[0] < cy
    tr_ok = x[1] > cx and y[1] < cy
    br_ok = x[2] > cx and y[2] > cy
    bl_ok = x[3] < cx and y[3] > cy
    return bool(tl_ok and tr_ok and br_ok and bl_ok)


def _geometry_flags(
    corners: np.ndarray,
    frame_wh: tuple[float, float],
    area_bounds: tuple[float, float] = DEFAULT_AREA_BOUNDS,
) -> tuple[str, ...]:
    """Shape-only validity flags for a corner quad.

    The area check runs as a fraction of the ORIGINAL frame, not of the padded
    model square: in pad mode the letterbox shrinks normalised areas by the
    content fraction, which would make a fixed threshold mean different things
    per aspect ratio and per resize mode (and skew a pad-vs-squash comparison).
    Positive scaling of either image axis preserves convexity and the quadrant check.

    :param corners: (4, 2) xy in TL, TR, BR, BL order
    :param frame_wh: (width, height) of the frame the corners live in
    :param area_bounds: (min, max) allowed quad area as a fraction of the frame
    :return: any of ('non_convex', 'bad_area', 'bad_corner_order'); empty when clean
    """
    if not np.isfinite(corners).all():
        return ("non_convex", "bad_area", "bad_corner_order")
    flags: list[str] = []
    if not _is_convex(corners):
        flags.append("non_convex")
    lo, hi = area_bounds
    area_frac = _shoelace_area(corners) / (frame_wh[0] * frame_wh[1])
    if area_frac < lo or area_frac > hi:
        flags.append("bad_area")
    if not _quadrants_ok(corners):
        flags.append("bad_corner_order")
    return tuple(flags)
