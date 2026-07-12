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

from dataclasses import dataclass

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


# Four correspondences fit a homography exactly, zero residual by construction,
# so a mislabelled crossing among 4 points is invisible to every residual gate.
# Five is the smallest count where a wrong-name click shows up. Grafted from the
# tour tool per docs/annotator_unification_brief.md.
FIT_MIN_POINTS = 5


def fit_homography(
    court_pts: np.ndarray, image_pts: np.ndarray, *, min_points: int = FIT_MIN_POINTS
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares homography from court metres to image pixels.

    Plain least squares over every pair (no RANSAC): the pairs are deliberate
    hand clicks, so outlier rejection would only hide a bad click that the
    residuals should expose instead.

    :param court_pts: (n, 2) court coords in metres, n >= min_points
    :param image_pts: (n, 2) matching image pixels
    :param min_points: pair floor; the default refuses exact 4-point fits, whose
        zero residual hides a mislabelled landmark. Only lower it for callers
        that gate correctness some other way
    :return: (homography, residuals): the (3, 3) court -> image map and the (n,)
        per-point reprojection error in px
    :raises ValueError: on fewer than ``min_points`` pairs, mismatched shapes,
        or a degenerate fit (collinear or repeated landmarks)
    """
    court = np.asarray(court_pts, dtype=np.float64)
    image = np.asarray(image_pts, dtype=np.float64)
    if court.ndim != 2 or court.shape[1] != 2 or court.shape != image.shape:
        raise ValueError(f"expected matching (n, 2) arrays, got {court.shape} and {image.shape}")
    if court.shape[0] < min_points:
        raise ValueError(f"need at least {min_points} landmark pairs, got {court.shape[0]}: "
                         "4 fit a homography exactly, so a mislabelled landmark shows no residual")
    if _dlt_conditioning(court, image) < DLT_CONDITION_FLOOR:
        raise ValueError("degenerate landmark spread: pick intersections off a single line")
    homography, _ = cv2.findHomography(court, image, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("degenerate homography: landmarks are collinear or repeated")
    residuals = np.linalg.norm(project_points(homography, court) - image, axis=1)
    if not np.all(np.isfinite(residuals)):
        raise ValueError("degenerate homography: landmarks are collinear or repeated")
    return homography, residuals


# Reprojection gate on the WORST point rather than the mean: least squares
# smears one bad click across the good ones, so a 30 px misidentification among
# six clicks leaves the mean near 1 px while the worst point sticks out.
# Calibration (12 Jul 2026, docs/annotator_unification_brief.md): honest loupe
# clicks jitter 1-2 px, so at few points, where the fit has little slack to
# absorb them, anything over 3 px native is an anomaly. At higher counts honest
# worsts grow (the shipped 18-25 point frames peaked at 8.8 px) and the data
# only has to out-resolve auto reads, whose quality cliff sits at 10 px at the
# 1280-wide reference (about 15 px native on 1920 footage); those thresholds
# scale with frame width.
LOWCOUNT_MAX_POINTS = 6
LOWCOUNT_FAIL_PX = 3.0
REF_FRAME_WIDTH = 1280
HIGHCOUNT_FAIL_REFPX = 10.0
HIGHCOUNT_WARN_REFPX = HIGHCOUNT_FAIL_REFPX * 2 / 3


@dataclass(frozen=True)
class FitCheck:
    """Verdict of the reprojection gate over one fit's residuals.

    :param level: "ok", "warn" (usable, confirm deliberately) or "fail"
    :param reason: what tripped; empty when ok
    :param worst_px: largest per-point reprojection error
    :param rms_px: rms reprojection error, bookkeeping rather than the gate
    """

    level: str
    reason: str
    worst_px: float
    rms_px: float


def check_fit(residuals: np.ndarray, frame_width: int) -> FitCheck:
    """Gate a fit by its worst-point reprojection, calibrated to point count.

    :param residuals: (n,) per-point reprojection errors in native px
    :param frame_width: frame width in px; scales the high-count thresholds
    :return: the verdict; "fail" means discard the fit, "warn" means show the
        numbers and require a deliberate confirm
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    worst = float(residuals.max())
    rms = float(np.sqrt(np.mean(residuals**2)))
    if residuals.size <= LOWCOUNT_MAX_POINTS:
        if worst > LOWCOUNT_FAIL_PX:
            return FitCheck("fail", f"worst point reprojects {worst:.1f} px, over the {LOWCOUNT_FAIL_PX:g} px "
                                    f"floor for {residuals.size} points (a mislabelled crossing?)", worst, rms)
        return FitCheck("ok", "", worst, rms)
    fail_px = HIGHCOUNT_FAIL_REFPX * frame_width / REF_FRAME_WIDTH
    warn_px = HIGHCOUNT_WARN_REFPX * frame_width / REF_FRAME_WIDTH
    if worst > fail_px:
        return FitCheck("fail", f"worst point reprojects {worst:.1f} px, over the {fail_px:.1f} px ceiling "
                                f"(mislabelled crossings reproject 20 px or more)", worst, rms)
    if worst >= warn_px:
        return FitCheck("warn", f"worst point reprojects {worst:.1f} px (warn from {warn_px:.1f} px, "
                                f"hard fail over {fail_px:.1f} px)", worst, rms)
    return FitCheck("ok", "", worst, rms)


def quad_is_camera_valid(quad: np.ndarray) -> bool:
    """Whether a projected quad is a convex behind-baseline TL TR BR BL rectangle.

    Catches a fit that converged to a mirror or a crossed quad, which residuals
    alone pass. Convex means the four edge turns share one sign; behind-baseline
    means the near baseline sits below the far one in image y (TL above BL, TR
    above BR; smaller y is higher up the frame). Maths grafted from the tour
    tool's _quad_is_camera_valid.

    :param quad: (4, 2) projected corners in TL TR BR BL order
    """
    crosses = []
    for corner in range(4):
        here = quad[corner]
        nxt = quad[(corner + 1) % 4]
        after = quad[(corner + 2) % 4]
        crosses.append((nxt[0] - here[0]) * (after[1] - nxt[1]) - (nxt[1] - here[1]) * (after[0] - nxt[0]))
    convex = all(cross > 0 for cross in crosses) or all(cross < 0 for cross in crosses)
    top_left, top_right, bottom_right, bottom_left = quad
    behind_baseline = top_left[1] < bottom_left[1] and top_right[1] < bottom_right[1]
    return bool(convex and behind_baseline)


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
