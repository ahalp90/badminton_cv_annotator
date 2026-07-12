"""Badminton court model for hand annotation: named line intersections and homography fits.

Court dimensions are fixed by the BWF, so any four or more clicked line
intersections with known court coordinates pin the court-to-image homography.
Projecting a corner's court coordinates through that homography gives its pixel
position even when the corner lies beyond the visible frame. That is the whole
trick behind off-frame ground truth: the visible markings determine where the
unseen corner must be.

Convention matches ``fallback.py`` on the court-fallback branch: court metres,
origin at the TL corner, x across the 6.10 m width, y down-court along the
13.40 m length, corner slots ordered TL TR BR BL. "Far" and "near" describe the
view on screen (far baseline = the one at the top of a normal court view). The
court is symmetric end to end, so any consistent far/near reading of a frame
yields a valid homography. Constants are duplicated here rather than imported
from fallback.py so the annotation tooling stands alone; unify when the
branches merge.
"""
from __future__ import annotations

import cv2
import numpy as np

COURT_WIDTH_M = 6.10
COURT_LENGTH_M = 13.40
NET_Y_M = COURT_LENGTH_M / 2  # 6.70; the net line is not painted, kept for reference

CORNER_NAMES = ("tl", "tr", "br", "bl")
CORNER_DESCRIPTIONS = ("far-left", "far-right", "near-right", "near-left")
# (slot=4, xy=2) court metres, TL TR BR BL
CORNER_COURT_M = np.array(
    [
        [0.0, 0.0],
        [COURT_WIDTH_M, 0.0],
        [COURT_WIDTH_M, COURT_LENGTH_M],
        [0.0, COURT_LENGTH_M],
    ],
    dtype=np.float32,
)

# Painted lines by the court coordinate they hold constant. The x-family runs
# down-court at constant x, the y-family runs across at constant y. Insets per
# the BWF court diagram: singles sideline 0.46 m, doubles long service line
# 0.76 m off the baseline, short service line 1.98 m off the net.
X_LINES: dict[str, float] = {
    "left_doubles": 0.0,
    "left_singles": 0.46,
    "centre": 3.05,
    "right_singles": COURT_WIDTH_M - 0.46,
    "right_doubles": COURT_WIDTH_M,
}
Y_LINES: dict[str, float] = {
    "far_baseline": 0.0,
    "far_long_service": 0.76,
    "far_short_service": NET_Y_M - 1.98,
    "near_short_service": NET_Y_M + 1.98,
    "near_long_service": COURT_LENGTH_M - 0.76,
    "near_baseline": COURT_LENGTH_M,
}


def _build_landmarks() -> dict[str, tuple[float, float]]:
    """Cross every y-family line with every x-family line into named points.

    All 30 crossings are painted intersections: the four long lines run baseline
    to baseline, and the centre line's two halves cover every y-line position.

    :return: name -> (court_x_m, court_y_m), ordered far to near then left to
        right so cycling through the catalogue in the UI is predictable
    """
    points: dict[str, tuple[float, float]] = {}
    for y_name, y_m in Y_LINES.items():
        for x_name, x_m in X_LINES.items():
            points[f"{y_name}_x_{x_name}"] = (x_m, y_m)
    return points


LANDMARKS: dict[str, tuple[float, float]] = _build_landmarks()
LANDMARK_NAMES: tuple[str, ...] = tuple(LANDMARKS)

# The four outer corners are themselves catalogue crossings; these are their
# names in corner slot order TL TR BR BL (pinned by a test against
# CORNER_COURT_M).
CORNER_LANDMARK_NAMES = (
    "far_baseline_x_left_doubles",
    "far_baseline_x_right_doubles",
    "near_baseline_x_right_doubles",
    "near_baseline_x_left_doubles",
)


def project_points(homography: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Project court-metre points to image pixels. Off-frame results stay unclamped.

    :param homography: (3, 3) court -> image
    :param points_m: (n, 2) court coords in metres
    :return: (n, 2) image pixels, float64
    """
    reshaped = np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(reshaped, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)


def project_corners(homography: np.ndarray) -> np.ndarray:
    """:return: (4, 2) the TL TR BR BL court corners in image pixels, unclamped."""
    return project_points(homography, CORNER_COURT_M)


# Floor on the normalised DLT system's sigma_8 / sigma_1. Healthy landmark
# spreads sit orders of magnitude above it; degenerate sets collapse to ~1e-16.
DLT_CONDITION_FLOOR = 1e-8


def _dlt_conditioning(court_pts: np.ndarray, image_pts: np.ndarray) -> float:
    """How well the correspondences pin all 8 homography degrees of freedom.

    Residuals cannot be trusted alone: a set that is mostly collinear (say five
    points on the far baseline plus one elsewhere) fits its own inputs with zero
    residual while the map is unconstrained everywhere else. The tell is the
    direct-linear-transform design matrix losing rank, so this returns its
    sigma_8 / sigma_1 after Hartley normalisation of both point sets.

    :param court_pts: (n, 2) court coords in metres
    :param image_pts: (n, 2) matching image pixels
    :return: the singular-value ratio; ~0 when the fit is underdetermined
    """

    def normalise(points: np.ndarray) -> np.ndarray:
        centred = points - points.mean(axis=0)
        spread = np.sqrt((centred**2).sum(axis=1)).mean()
        return centred * (np.sqrt(2.0) / max(spread, 1e-12))

    src = normalise(court_pts)
    dst = normalise(image_pts)
    x, y = src[:, 0], src[:, 1]
    u, v = dst[:, 0], dst[:, 1]
    zeros = np.zeros(len(src))
    ones = np.ones(len(src))
    rows_u = np.column_stack([-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u])
    rows_v = np.column_stack([zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v])
    singular = np.linalg.svd(np.vstack([rows_u, rows_v]), compute_uv=False)
    return float(singular[7] / singular[0])


def fit_homography(court_pts: np.ndarray, image_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography from court metres to image pixels.

    Plain least squares over every pair (no RANSAC): the pairs are deliberate
    hand clicks, so outlier rejection would only hide a bad click that the
    residuals should expose instead.

    :param court_pts: (n, 2) court coords in metres, n >= 4
    :param image_pts: (n, 2) matching image pixels
    :return: (homography, residuals): the (3, 3) court -> image map and the (n,)
        per-point reprojection error in px
    :raises ValueError: on fewer than 4 pairs, mismatched shapes, or a
        degenerate fit (collinear or repeated landmarks)
    """
    court = np.asarray(court_pts, dtype=np.float64)
    image = np.asarray(image_pts, dtype=np.float64)
    if court.ndim != 2 or court.shape[1] != 2 or court.shape != image.shape:
        raise ValueError(f"expected matching (n, 2) arrays, got {court.shape} and {image.shape}")
    if court.shape[0] < 4:
        raise ValueError(f"need at least 4 landmark pairs to fit a homography, got {court.shape[0]}")
    if _dlt_conditioning(court, image) < DLT_CONDITION_FLOOR:
        raise ValueError("degenerate landmark spread: pick intersections off a single line")
    homography, _ = cv2.findHomography(court, image, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("degenerate homography: landmarks are collinear or repeated")
    residuals = np.linalg.norm(project_points(homography, court) - image, axis=1)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("degenerate homography: landmarks are collinear or repeated")
    return homography, residuals


def extrapolation_errors(clicked_corners_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Distance between where the fit puts each corner and where it was clicked.

    The leave-one-out check: fit the homography from landmarks only, then compare
    its corner projections against corners that were actually clicked. Run on a
    full-court frame this measures how far an extrapolated corner would sit from
    the truth, which is the number that says whether to trust the method.

    :param clicked_corners_px: (4, 2) clicked corners in slot order TL TR BR BL
    :param homography: (3, 3) court -> image, fitted from landmarks
    :return: (4,) per-slot error in px
    """
    clicked = np.asarray(clicked_corners_px, dtype=np.float64)
    return np.linalg.norm(project_corners(homography) - clicked, axis=1)
