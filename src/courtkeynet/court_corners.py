"""Classical-CV court recovery for the frames CourtKeyNet cannot fully gate.

The wrapper (see wrapper.py) returns four court corners per frame with a
per-corner peak and a validity gate. It fails closed whenever any corner's peak
sits under the floor. Two failure populations motivated this module: broadcast
courts whose far corners localise accurately but score a low peak, and amateur
footage where part of the court sits out of frame or behind clutter.

When 2-3 of the four corners clear the peak floor over a scene, we treat those as
bootstrap anchors, pool line evidence across the scene's frames (the camera is
static, so moving players wash out), fit a homography from the official BWF court
geometry to the image, and reproject the 1-2 missing corners through it. Anchors
are bootstrap evidence, not gospel: a hard acceptance gate scores the fitted
homography against the actual detected line evidence AND the anchors, and a poor
fit returns None rather than a poisoned quad.

Court model note: TL/TR/BR/BL map to court metres (0,0)/(6.1,0)/(6.1,13.4)/
(0,13.4) with the far baseline at y=0. On behind-baseline footage the image's
top corners (TL/TR) are the far baseline, which is why the far baseline takes
y=0. cv2 + numpy only; no new dependencies.

A second, video-level pass lives at the bottom of this module: consensus_repair()
takes one video's per-scene quads (from pick_scene_corners above, run once per scene) and
repairs the minority that drifted far from the video's own per-corner median.
pick_scene_corners's per-scene contract is untouched; consensus_repair is an opt-in
post-pass a caller runs across a video's scenes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import cv2
import numpy as np

from .constants import DEFAULT_CORNER_MIN_PEAK_CONF

if TYPE_CHECKING:
    # Type-only, so this module imports without torch (the annotator needs just the
    # court constants). The one runtime wrapper name loads where used: ckn_scene_corners
    # in _ckn_path.
    from .wrapper import CornerDetection

logger = logging.getLogger(__name__)

# Corner slot indices, TL TR BR BL, matching CornerDetection.corners_px order.
TL, TR, BR, BL = 0, 1, 2, 3
CORNER_NAMES = ("TL", "TR", "BR", "BL")

# A detection only qualifies as a bootstrap anchor when its quad is shape-clean.
# low_peak / high_entropy frames stay usable for per-corner evidence; a
# geometry-flagged quad is suspect as a whole (see the trigger policy).
GEOMETRY_FLAGS = frozenset({"non_convex", "bad_area", "bad_corner_order"})

# --- BWF court model, metres. y=0 is the far baseline (see module docstring) ---
COURT_WIDTH_M = 6.10  # doubles sideline to doubles sideline
COURT_LENGTH_M = 13.40  # baseline to baseline

# Court corners in metres, TL TR BR BL, same slot order as the detection quad.
CORNER_COURT_M = np.array(
    [[0.0, 0.0], [COURT_WIDTH_M, 0.0], [COURT_WIDTH_M, COURT_LENGTH_M], [0.0, COURT_LENGTH_M]],
    dtype=np.float32,
)

# Painted lines as metre segments (endpoint pairs), used both to score a fitted
# homography against detected lines and to label each detected line's court
# coordinate. The net is NOT painted on the floor, so it is absent. Centre line
# is split by the service box, hence two segments. Constants: singles sidelines
# 0.46 in from each doubles sideline; short service lines 1.98 either side of the
# net (net at length/2 = 6.70); doubles long service lines 0.76 in from each
# baseline; centre line at width/2 = 3.05 spanning baseline to short service line
# in each half.
_SINGLES_INSET = 0.46
_LONG_SERVICE_INSET = 0.76
_SHORT_SERVICE_OFFSET = 1.98
_NET_Y = COURT_LENGTH_M / 2.0
_CENTRE_X = COURT_WIDTH_M / 2.0
_FAR_SHORT_Y = _NET_Y - _SHORT_SERVICE_OFFSET  # 4.72
_NEAR_SHORT_Y = _NET_Y + _SHORT_SERVICE_OFFSET  # 8.68

# Constant-x court lines (run along the court's length) and constant-y court
# lines (run across it). The intersection algebra and the gate only need each
# line's constant coordinate plus its painted extent.
PAINTED_SEGMENTS_M: tuple[tuple[np.ndarray, np.ndarray], ...] = tuple(
    (np.array(a, dtype=np.float32), np.array(b, dtype=np.float32))
    for a, b in (
        # x-family: constant x, varying y
        ((0.0, 0.0), (0.0, COURT_LENGTH_M)),  # left doubles sideline
        ((_SINGLES_INSET, 0.0), (_SINGLES_INSET, COURT_LENGTH_M)),  # left singles sideline
        ((_CENTRE_X, 0.0), (_CENTRE_X, _FAR_SHORT_Y)),  # centre line, far half
        ((_CENTRE_X, _NEAR_SHORT_Y), (_CENTRE_X, COURT_LENGTH_M)),  # centre line, near half
        ((COURT_WIDTH_M - _SINGLES_INSET, 0.0), (COURT_WIDTH_M - _SINGLES_INSET, COURT_LENGTH_M)),  # right singles
        ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),  # right doubles sideline
        # y-family: constant y, varying x
        ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),  # far baseline
        ((0.0, _LONG_SERVICE_INSET), (COURT_WIDTH_M, _LONG_SERVICE_INSET)),  # far doubles long service
        ((0.0, _FAR_SHORT_Y), (COURT_WIDTH_M, _FAR_SHORT_Y)),  # far short service
        ((0.0, _NEAR_SHORT_Y), (COURT_WIDTH_M, _NEAR_SHORT_Y)),  # near short service
        ((0.0, COURT_LENGTH_M - _LONG_SERVICE_INSET), (COURT_WIDTH_M, COURT_LENGTH_M - _LONG_SERVICE_INSET)),
        ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),  # near baseline
    )
)

# Which two outer bounding lines meet at each corner, keyed by corner slot.
# 'far'/'near' are the y=0 / y=13.4 baselines; 'left'/'right' the x=0 / x=6.1
# doubles sidelines. Every court corner is one of these four intersections.
CORNER_BOUNDING = {TL: ("far", "left"), TR: ("far", "right"), BR: ("near", "right"), BL: ("near", "left")}
# Confident anchors that pin each outer bounding line (the anchor sits on it).
LEFT_CORNERS, RIGHT_CORNERS = (TL, BL), (TR, BR)
FAR_CORNERS, NEAR_CORNERS = (TL, TR), (BL, BR)

# --- Tunable constants (measured/eyeballed on the pilot; re-measure elsewhere) ---
CANNY_LO, CANNY_HI = 50, 150
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_PX = 40
HOUGH_MAX_GAP_PX = 15
# Segment-to-line clustering tolerances: two Hough fragments share a court line
# when their orientations agree to a few degrees and their perpendicular offsets
# to within a modest pixel gap.
ANGLE_CLUSTER_TOL = np.deg2rad(7.0)
# One court line's Hough fragments are collinear, so their offset scatter is a
# few pixels; distinct court lines sit farther apart. 12 px groups the former and
# splits the latter. The far baseline and the far doubles long service line
# (0.76 m apart) can fall inside this under a very oblique view and merge, which
# costs the far baseline a few pixels: an honest limit of pooled Hough evidence.
OFFSET_CLUSTER_TOL_PX = 12.0
# A court line draws Hough support from every frame; a moving player's edge
# appears in a few. Requiring pooled support filters the latter without a
# per-frame median step.
MIN_CLUSTER_SUPPORT_PX = 60.0
# Mat-colour ROI: sample HSV inside the confident-corner region shrunk toward its
# centroid (avoids edge bleed), keep the percentile band, close over the thin
# painted lines so the court becomes one solid blob to detect lines within.
MAT_SHRINK = 0.72
MAT_PERCENTILES = (5.0, 95.0)
MAT_CLOSE_KERNEL_PX = 15
ROI_DILATE_KERNEL_PX = 9
# Assignment: a detected line matches a projected model line when both the
# orientation and the perpendicular offset of its midpoint agree.
ASSIGN_ANGLE_TOL = np.deg2rad(10.0)
ASSIGN_OFFSET_TOL_PX = 25.0
# RANSAC over the correspondence pool. Seeded so the module is deterministic.
RANSAC_SEED = 20260712
RANSAC_ITERS = 300
RANSAC_INLIER_PX = 8.0
# Hard acceptance gate. The winning homography must put the court model onto the
# detected line evidence AND onto the anchors, both within a fraction of the
# frame diagonal. Broadcast CourtKeyNet localises to ~3.5 px; amateur footage is
# looser, so the line arm sits near 1% of the diagonal (~15 px on 720p, ~22 px on
# 1080p) and the anchor arm twice that: enough slack for honest amateur error,
# far under the hundreds-of-pixels a garbage anchor or misassigned line produces.
# The gate is the single arbiter that a fired trigger with bad evidence returns
# None instead of a fabricated quad (Stage-0 saw peaks up to 0.14 on a net post).
GATE_LINE_FRAC = 0.010
GATE_ANCHOR_FRAC = 0.020
# KNOWN FAILURE MODE, measured on ShuttleSet vid 3 (session 17): when neither far
# corner is anchored, the extreme-line rule can hand the far baseline to the
# advertising boards' bottom edge (the green surround apron pulls it into the mat
# ROI), and the resulting fit is self-consistent enough to pass both gate arms
# (9 of 43 rescued scenes, far corners ~220-450 refpx off while anchors stay
# ~4 px). Baseline-candidate enumeration scored by model-line coverage was measured
# as a fix and came out WORSE (vid 3 good rescues 34 -> 24; near-court line spacing
# is too tight for nearest-match scoring to discriminate hypotheses); it is NOT
# shipped. A cross-scene consensus vote (video-level, not scene-level) is: measured
# on the session-18 fb5 eval, good scenes sit <=18.1 refpx from the video's own
# per-corner-median consensus while the 7 boards-aliased vid-3 scenes start at
# >=177.3 refpx, a clean ~10x gap. It ships below as consensus_repair();
# pick_scene_corners above is untouched and can still hit this failure mode on its own,
# which is why consensus_repair runs as a separate video-level post-pass.
# tests/test_courtkeynet_court_corners.py::test_boards_alias_minority_flagged_and_repaired_by_consensus
# drives consensus_repair() over this exact scene shape (reproduced with
# pick_scene_corners, boards line and all) and passes.
# Distortion: the pooled residual of a straight painted line already runs a few
# pixels from line thickness, anti-aliasing and Hough endpoint quantisation, so
# the material-curvature floor sits above that noise. Real barrel distortion bows
# a court-width line by tens of pixels, well clear of this. A division-model
# correction is out of scope (see _measure_sagitta_material); we only measure.
SAGITTA_MATERIAL_PX = 5.0
MIN_FAMILY_LINES = 2  # a court needs at least two lines in each family


@dataclass(frozen=True)
class FallbackDiagnostics:
    """Evidence trail for one fallback fit, so a reviewer can judge the quad."""

    reproj_line_px: float  # median perpendicular distance, model lines vs detected lines
    reproj_anchor_px: float  # largest reprojection error over the confident anchors
    gate_line_frac: float  # reproj_line_px / frame diagonal (the line arm of the gate)
    gate_anchor_frac: float  # reproj_anchor_px / frame diagonal (the anchor arm)
    n_lines_used: int  # detected lines assigned to a court line
    n_correspondences: int  # point pairs the homography was fitted from
    max_sagitta_px: float  # measured line curvature (distortion proxy)


@dataclass(frozen=True)
class CourtQuad:
    """Four court corners plus provenance, the shared output of both paths.

    Downstream callers read corners_px and source without caring which path ran.
    peak carries the model's honest per-corner median peak: the fallback does not
    fabricate a confidence for the corners it recovered from geometry.
    """

    corners_px: np.ndarray  # (4, 2) float32, original-frame pixels, TL TR BR BL
    peak: np.ndarray  # (4,) model's per-corner median peak over the scene
    source: str  # 'model' | 'fallback'
    corner_source: tuple[str, str, str, str]  # per corner: 'model' | 'fallback'
    diagnostics: FallbackDiagnostics | None  # None on the model path


class _Line(NamedTuple):
    """A pooled court-line candidate in one image's pixel space.

    coef is the homogeneous line (a, b, c) normalised so a^2+b^2=1, hence the
    perpendicular distance of a point (x, y) is just |a*x + b*y + c|. angle is the
    line direction in [0, pi). mid is a representative point on the line; support
    is the total Hough segment length behind it.
    """

    coef: np.ndarray  # (3,)
    angle: float
    mid: np.ndarray  # (2,)
    support: float


# --- Line algebra ----------------------------------------------------------

def _circular_diff(angle_a: float, angle_b: float) -> float:
    """:return: smallest angle between two orientations in [0, pi), in radians."""
    diff = abs(angle_a - angle_b) % np.pi
    return float(min(diff, np.pi - diff))


def _normalise_line(coef: np.ndarray) -> np.ndarray:
    """Scale a homogeneous line so a^2+b^2=1 and orient it canonically.

    A consistent orientation (b>0, or a>0 when b==0) makes the offset c directly
    comparable between near-parallel lines, which the segment clustering relies on.

    :param coef: (3,) homogeneous line (a, b, c)
    :return: (3,) float64 normalised, canonically oriented line
    """
    coef = coef.astype(np.float64)
    norm = np.hypot(coef[0], coef[1])
    if norm < 1e-12:
        # Degenerate (a collapsed segment): reachable when a near-singular
        # candidate homography projects a painted line to a point. Returned as-is;
        # the gate scorer detects the ~zero normal and reads the fit as unusable.
        return coef
    coef = coef / norm
    if coef[1] < 0 or (coef[1] == 0 and coef[0] < 0):
        coef = -coef
    return coef


def _line_through(point_a: np.ndarray, point_b: np.ndarray) -> np.ndarray:
    """:return: (3,) normalised homogeneous line through two image points."""
    homog_a = np.array([point_a[0], point_a[1], 1.0])
    homog_b = np.array([point_b[0], point_b[1], 1.0])
    return _normalise_line(np.cross(homog_a, homog_b))


def _point_line_distance(point: np.ndarray, coef: np.ndarray) -> float:
    """:return: perpendicular pixel distance of a point to a normalised line."""
    return float(abs(coef[0] * point[0] + coef[1] * point[1] + coef[2]))


def _intersect(coef_a: np.ndarray, coef_b: np.ndarray) -> np.ndarray | None:
    """Intersect two homogeneous lines via their cross product.

    :param coef_a: (3,) first line
    :param coef_b: (3,) second line
    :return: (2,) intersection xy, or None when the lines are (near) parallel
    """
    homog = np.cross(coef_a, coef_b)
    if abs(homog[2]) < 1e-9:
        return None
    return np.array([homog[0] / homog[2], homog[1] / homog[2]], dtype=np.float64)


def _tls_line(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Total-least-squares line fit: the line minimising perpendicular residuals.

    The principal axis of the centred points is the line direction; the minor
    axis is its normal, which gives the homogeneous coefficients directly.

    :param points: (n, 2) points on one candidate line (n >= 2)
    :return: (normalised coef (3,), direction angle in [0, pi), centroid (2,))
    """
    centroid = points.mean(axis=0)
    _, _, right_vectors = np.linalg.svd(points - centroid)
    direction = right_vectors[0]  # largest singular vector: along the line
    normal = right_vectors[1]  # minor axis: the line normal
    offset = -float(normal @ centroid)
    coef = _normalise_line(np.array([normal[0], normal[1], offset]))
    angle = float(np.arctan2(direction[1], direction[0]) % np.pi)
    return coef, angle, centroid


def _project(homography: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Map court-metre points through a homography to image pixels.

    :param homography: (3, 3) court-metres -> image-pixels
    :param points_m: (n, 2) court coordinates in metres
    :return: (n, 2) float64 image pixels
    """
    reshaped = points_m.reshape(-1, 1, 2).astype(np.float64)
    mapped = cv2.perspectiveTransform(reshaped, homography)
    return mapped.reshape(-1, 2)


# --- Scene trigger evidence ------------------------------------------------

def _geometry_clean(detection: CornerDetection) -> bool:
    """:return: True when the quad carries no shape flag (usable as an anchor)."""
    return GEOMETRY_FLAGS.isdisjoint(detection.flags)


def _scene_peaks(clean: list[CornerDetection]) -> np.ndarray:
    """:return: (4,) per-corner median peak over the geometry-clean detections."""
    return np.median(np.stack([detection.peak for detection in clean]), axis=0)


def _anchor_points(clean: list[CornerDetection], confident: np.ndarray, corner_min_peak_conf: float) -> dict[int, np.ndarray]:
    """Median image position of each confident corner over its own strong frames.

    Per corner the median runs only over the geometry-clean frames where THAT
    corner's peak cleared the floor, so a corner that flickers weak in some frames
    is anchored on the frames where the model actually saw it.

    :param clean: geometry-clean detections for the scene
    :param confident: (4,) bool, corners whose scene-median peak cleared the floor
    :param corner_min_peak_conf: per-corner peak-confidence floor
    :return: {corner slot: (2,) float64 anchor pixel} for each confident corner
    """
    anchors: dict[int, np.ndarray] = {}
    for corner in np.flatnonzero(confident):
        strong = [d.corners_px[corner] for d in clean if d.peak[corner] >= corner_min_peak_conf]
        anchors[int(corner)] = np.median(np.stack(strong), axis=0).astype(np.float64)
    return anchors


def _scene_sample_quad(clean: list[CornerDetection]) -> np.ndarray:
    """Median of ALL four corners over clean frames, a region to sample mat colour.

    Unlike the anchors this uses every corner regardless of confidence: for mat
    sampling we only need a rough region inside the court, and a geometry-clean
    quad is at least convex and correctly ordered. The static-camera assumption
    lets one scene-median quad serve every frame.

    :param clean: geometry-clean detections for the scene
    :return: (4, 2) float32 median quad, TL TR BR BL
    """
    return np.median(np.stack([d.corners_px for d in clean]), axis=0).astype(np.float32)


# --- Mat-colour region of interest -----------------------------------------

def _mat_roi(frame_bgr: np.ndarray, sample_quad: np.ndarray) -> np.ndarray:
    """Court region-of-interest mask sampled from the mat colour, never hardcoded.

    Samples HSV inside the sample quad shrunk toward its centroid (dodging edge
    bleed), keeps the per-channel percentile band, closes over the thin painted
    lines so the playing surface becomes one blob, and returns the blob that
    overlaps the court. Detecting lines only within this ROI drops off-court
    clutter (adjacent courts, crowd, sideline advertising).

    :param frame_bgr: (H, W, 3) uint8 BGR frame
    :param sample_quad: (4, 2) court-region quad to sample the mat colour inside
    :return: (H, W) uint8 mask, 255 on the court ROI
    """
    height, width = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    centroid = sample_quad.mean(axis=0)
    shrunk = (centroid + MAT_SHRINK * (sample_quad - centroid)).astype(np.int32)
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(sample_mask, shrunk, 255)
    sampled = hsv[sample_mask == 255]
    if sampled.shape[0] < 3:
        return np.zeros((height, width), dtype=np.uint8)

    lo = np.percentile(sampled, MAT_PERCENTILES[0], axis=0)
    hi = np.percentile(sampled, MAT_PERCENTILES[1], axis=0)
    band = cv2.inRange(hsv, lo.astype(np.uint8), hi.astype(np.uint8))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MAT_CLOSE_KERNEL_PX, MAT_CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kernel)

    count, labels = cv2.connectedComponents(closed)
    if count <= 1:
        return np.zeros((height, width), dtype=np.uint8)
    # Keep the component covering the most of the sampled region: that is the mat.
    overlaps = []
    for comp in range(1, count):
        in_comp_and_sampled = (labels == comp) & (sample_mask == 255)
        overlaps.append(int(in_comp_and_sampled.sum()))
    best = int(np.argmax(overlaps)) + 1
    roi = np.where(labels == best, np.uint8(255), np.uint8(0))
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ROI_DILATE_KERNEL_PX, ROI_DILATE_KERNEL_PX))
    return cv2.dilate(roi, dilate)


# --- Line evidence ---------------------------------------------------------

def _frame_segments(frame_bgr: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Canny + probabilistic Hough segments inside the court ROI.

    :param frame_bgr: (H, W, 3) uint8 BGR frame
    :param roi: (H, W) uint8 court mask; edges outside it are discarded
    :return: (m, 4) int segments (x1, y1, x2, y2), possibly empty
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    edges = cv2.bitwise_and(edges, edges, mask=roi)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_PX, maxLineGap=HOUGH_MAX_GAP_PX,
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines.reshape(-1, 4)


def _cluster_segments(segments: np.ndarray) -> tuple[list[_Line], float]:
    """Pool Hough segments across the scene into court-line candidates.

    Greedy single pass: a segment joins the first cluster whose running mean
    orientation and offset it matches, else it seeds a new cluster. Each cluster
    is then total-least-squares fitted over all its endpoints and kept only if its
    pooled segment length clears the support floor (which is what makes a
    transient player edge lose to a line present every frame). The max
    perpendicular residual over the kept clusters is returned as a curvature
    (distortion) proxy.

    :param segments: (n, 4) pooled segments (x1, y1, x2, y2) from all frames
    :return: (kept court-line candidates, max sagitta in px)
    """
    starts = segments[:, :2].astype(np.float64)
    ends = segments[:, 2:].astype(np.float64)
    lengths = np.hypot(*(ends - starts).T)
    angles = np.arctan2(*(ends - starts)[:, ::-1].T) % np.pi
    mids = (starts + ends) / 2

    # Each cluster: accumulated endpoints, orientation sum, current line, support.
    # The orientation mean lives in doubled-angle vector space so fragments
    # straddling the 0/pi wrap (one horizontal baseline read as ~0 by one segment
    # and ~pi by another) average correctly instead of snapping to vertical.
    # Membership tests where the segment LIES (midpoint to the cluster's current
    # line), not the global homogeneous offset c: c is the distance from the image
    # origin, which swings with tiny angle changes for lines far from the origin
    # and flips sign outright when a near-vertical normal crosses b=0 under the
    # canonical orientation. Both found by the session-17 cross-model review.
    cluster_points: list[list[np.ndarray]] = []
    angle_vec_sum: list[np.ndarray] = []  # length-weighted (cos 2t, sin 2t) per cluster
    cluster_coef: list[np.ndarray] = []  # TLS line, refreshed on every join
    support: list[float] = []
    for idx in range(segments.shape[0]):
        doubled = lengths[idx] * np.array([np.cos(2 * angles[idx]), np.sin(2 * angles[idx])])
        joined = False
        for cluster in range(len(cluster_points)):
            mean_angle = 0.5 * np.arctan2(angle_vec_sum[cluster][1], angle_vec_sum[cluster][0]) % np.pi
            angle_ok = _circular_diff(angles[idx], float(mean_angle)) < ANGLE_CLUSTER_TOL
            offset_ok = _point_line_distance(mids[idx], cluster_coef[cluster]) < OFFSET_CLUSTER_TOL_PX
            if angle_ok and offset_ok:
                angle_vec_sum[cluster] = angle_vec_sum[cluster] + doubled
                support[cluster] += lengths[idx]
                cluster_points[cluster].extend((starts[idx], ends[idx]))
                # Eager TLS refresh per join; segment counts are small (hundreds)
                # and the refit keeps the membership test honest as the cluster
                # grows. Lazy would let early fragments define the line forever.
                cluster_coef[cluster] = _tls_line(np.stack(cluster_points[cluster]))[0]
                joined = True
                break
        if not joined:
            cluster_points.append([starts[idx], ends[idx]])
            angle_vec_sum.append(doubled)
            cluster_coef.append(_line_through(starts[idx], ends[idx]))
            support.append(float(lengths[idx]))

    lines: list[_Line] = []
    max_sagitta = 0.0
    for cluster in range(len(cluster_points)):
        if support[cluster] < MIN_CLUSTER_SUPPORT_PX:
            continue
        points = np.stack(cluster_points[cluster])
        coef, angle, mid = _tls_line(points)
        residuals = np.abs(points @ coef[:2] + coef[2])
        max_sagitta = max(max_sagitta, float(residuals.max()))
        lines.append(_Line(coef=coef, angle=angle, mid=mid, support=support[cluster]))
    return lines, max_sagitta


def _measure_sagitta_material(max_sagitta: float) -> bool:
    """:return: True when measured line curvature warrants a distortion warning.

    We only MEASURE distortion here. A one-parameter division-model correction is
    out of scope: the pilot's lines read straight by eye, so correcting would risk
    bending good footage for no gain. When curvature is material we log and carry
    on with the identity (uncorrected) geometry.
    """
    return max_sagitta > SAGITTA_MATERIAL_PX


# --- Family split (constant-x vs constant-y court lines) --------------------

def _two_means_angles(angles: np.ndarray) -> np.ndarray:
    """Split line orientations into two pencils with deterministic 2-means.

    Orientations live on a half-circle, so each is embedded as (cos 2t, sin 2t)
    to make the pi wrap-around continuous before clustering. Init seeds the two
    farthest-apart embedded points, so the result is deterministic.

    :param angles: (n,) line orientations in [0, pi)
    :return: (n,) int labels in {0, 1}
    """
    embedded = np.stack([np.cos(2 * angles), np.sin(2 * angles)], axis=1)  # (n, 2)
    seed_far = int(((embedded - embedded[0]) ** 2).sum(axis=1).argmax())
    centres = np.stack([embedded[0], embedded[seed_far]])
    labels = np.zeros(angles.shape[0], dtype=np.int64)
    for _ in range(10):
        distances = ((embedded[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)  # (n, 2)
        new_labels = distances.argmin(axis=1)
        moved = []
        for group in (0, 1):
            members = embedded[new_labels == group]
            moved.append(members.mean(axis=0) if members.shape[0] else centres[group])
        new_centres = np.stack(moved)
        if np.array_equal(new_labels, labels) and np.allclose(new_centres, centres):
            break
        labels, centres = new_labels, new_centres
    return labels


def _baseline_direction(anchors: dict[int, np.ndarray]) -> float | None:
    """:return: orientation of the near/far baseline from anchors, or None.

    Uses a baseline pair (both far, or both near corners) when both are confident.
    Diagonal-only anchor sets give neither pair, hence None.
    """
    for corner_a, corner_b in (FAR_CORNERS, NEAR_CORNERS):
        if corner_a in anchors and corner_b in anchors:
            direction = anchors[corner_b] - anchors[corner_a]
            return float(np.arctan2(direction[1], direction[0]) % np.pi)
    return None


def _split_families(lines: list[_Line], anchors: dict[int, np.ndarray]) -> tuple[list[_Line], list[_Line]]:
    """Split pooled lines into the x-family (along court) and y-family (across).

    The two-means split finds the two pencils; labelling which is which prefers
    the anchor baseline direction (the y-family runs along it). With no baseline
    pair available it falls back to 'baselines read more horizontal than
    sidelines', which holds for the upright, roughly behind-baseline cameras this
    fallback targets.

    :param lines: pooled court-line candidates
    :param anchors: confident-corner anchor points
    :return: (x_family, y_family)
    """
    angles = np.array([line.angle for line in lines])
    labels = _two_means_angles(angles)
    group_mean = []
    for group in (0, 1):
        members = angles[labels == group]
        embedded = np.array([np.cos(2 * members).mean(), np.sin(2 * members).mean()])
        group_mean.append(0.5 * np.arctan2(embedded[1], embedded[0]) % np.pi)

    baseline_dir = _baseline_direction(anchors)
    if baseline_dir is not None:
        y_group = 0 if _circular_diff(group_mean[0], baseline_dir) < _circular_diff(group_mean[1], baseline_dir) else 1
    else:
        # More-horizontal pencil (smaller |sin(angle)|) reads as the baselines.
        y_group = 0 if abs(np.sin(group_mean[0])) < abs(np.sin(group_mean[1])) else 1
    y_family = [line for line, label in zip(lines, labels) if label == y_group]
    x_family = [line for line, label in zip(lines, labels) if label != y_group]
    return x_family, y_family


# --- Outer bounding lines --------------------------------------------------

def _nearest_line(point: np.ndarray, family: list[_Line]) -> _Line | None:
    """:return: the family line whose infinite extent passes closest to a point."""
    if not family:
        return None
    return min(family, key=lambda line: _point_line_distance(point, line.coef))


def _extreme_line(family: list[_Line], reference: _Line) -> _Line | None:
    """The family line whose midpoint sits farthest from a reference line.

    Used only for a baseline whose two corners are both withheld (the 2-adjacent
    case): the opposite baseline is the extreme parallel line, since perspective
    preserves the court's line ordering.

    :param family: the y-family lines
    :param reference: the already-identified opposite baseline
    :return: the farthest line, or None if the family is empty
    """
    candidates = [line for line in family if line is not reference]
    if not candidates:
        return None
    return max(candidates, key=lambda line: _point_line_distance(line.mid, reference.coef))


def _outer_lines(
    x_family: list[_Line], y_family: list[_Line], anchors: dict[int, np.ndarray]
) -> dict[str, _Line] | None:
    """Identify the four outer bounding lines (both sidelines, both baselines).

    Each sideline and each baseline is pinned by any confident anchor that sits on
    it (the anchor is a court corner, so it lies on its own two bounding lines). A
    baseline with both corners withheld falls back to the extreme-line rule. Fails
    (None) if a bounding line cannot be found or two of them collapse to the same
    detected line (too little evidence to define the court rectangle).

    :return: {'left','right','far','near': _Line}, or None
    """
    def pin(corners: tuple[int, int], family: list[_Line]) -> _Line | None:
        for corner in corners:
            if corner in anchors:
                return _nearest_line(anchors[corner], family)
        return None

    left = pin(LEFT_CORNERS, x_family)
    right = pin(RIGHT_CORNERS, x_family)
    far = pin(FAR_CORNERS, y_family)
    near = pin(NEAR_CORNERS, y_family)
    if far is None and near is not None:
        far = _extreme_line(y_family, near)
    if near is None and far is not None:
        near = _extreme_line(y_family, far)

    outer = {"left": left, "right": right, "far": far, "near": near}
    if any(line is None for line in outer.values()):
        return None
    if outer["left"] is outer["right"] or outer["far"] is outer["near"]:
        return None
    return {name: line for name, line in outer.items() if line is not None}


# --- Homography fit and line assignment ------------------------------------

def _bootstrap_corners(outer: dict[str, _Line], anchors: dict[int, np.ndarray]) -> np.ndarray | None:
    """Four image corners from the outer bounding lines, seeding the homography.

    A confident corner takes its anchor position; a withheld corner is the
    intersection of its two bounding lines (which stays finite and valid even when
    it lands off the frame). None if any intersection is degenerate.

    :return: (4, 2) image corners TL TR BR BL, or None
    """
    corners = np.zeros((4, 2), dtype=np.float64)
    for slot in (TL, TR, BR, BL):
        if slot in anchors:
            corners[slot] = anchors[slot]
            continue
        baseline_name, sideline_name = CORNER_BOUNDING[slot]
        crossing = _intersect(outer[baseline_name].coef, outer[sideline_name].coef)
        if crossing is None:
            return None
        corners[slot] = crossing
    return corners


def _model_image_lines(homography: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Project every painted court line into the image under a homography.

    :param homography: (3, 3) court-metres -> image-pixels
    :return: per painted segment, (image line coef (3,), image endpoint a, endpoint b)
    """
    projected = []
    for end_a_m, end_b_m in PAINTED_SEGMENTS_M:
        ends_px = _project(homography, np.stack([end_a_m, end_b_m]))
        projected.append((_line_through(ends_px[0], ends_px[1]), ends_px[0], ends_px[1]))
    return projected


def _assign_lines(
    homography: np.ndarray, detected: list[_Line]
) -> list[tuple[_Line, int]]:
    """Match each detected line to the nearest projected painted line.

    A detected line is assigned to a painted line when their orientations agree
    and the detected midpoint lies close to the projected line. Assignment is the
    court evidence the gate scores against, and its painted-segment index recovers
    the line's court coordinate.

    :param homography: (3, 3) current fit, court-metres -> image-pixels
    :param detected: pooled court-line candidates
    :return: list of (detected line, painted-segment index)
    """
    model = _model_image_lines(homography)
    assignments: list[tuple[_Line, int]] = []
    for line in detected:
        best_idx, best_dist = -1, np.inf
        for seg_idx, (coef, _, _) in enumerate(model):
            model_angle = float(np.arctan2(-coef[0], coef[1]) % np.pi)
            if _circular_diff(line.angle, model_angle) > ASSIGN_ANGLE_TOL:
                continue
            dist = _point_line_distance(line.mid, coef)
            if dist < best_dist:
                best_idx, best_dist = seg_idx, dist
        if best_idx >= 0 and best_dist < ASSIGN_OFFSET_TOL_PX:
            assignments.append((line, best_idx))
    return assignments


def _correspondences(
    assignments: list[tuple[_Line, int]], anchors: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Court-to-image point pairs from assigned-line intersections plus anchors.

    Each assigned line contributes its court constant coordinate; intersecting an
    assigned x-family line with an assigned y-family line yields an image point
    whose court metres are known. Anchors add their corner correspondences.

    :return: (court points (n, 2), image points (n, 2)), n may be < 4
    """
    x_lines: list[tuple[_Line, float]] = []
    y_lines: list[tuple[_Line, float]] = []
    for line, seg_idx in assignments:
        end_a, end_b = PAINTED_SEGMENTS_M[seg_idx]
        if end_a[0] == end_b[0]:  # constant x -> x-family
            x_lines.append((line, float(end_a[0])))
        else:  # constant y -> y-family
            y_lines.append((line, float(end_a[1])))

    court_pts: list[list[float]] = []
    image_pts: list[list[float]] = []
    for x_line, x_coord in x_lines:
        for y_line, y_coord in y_lines:
            crossing = _intersect(x_line.coef, y_line.coef)
            if crossing is None:
                continue
            court_pts.append([x_coord, y_coord])
            image_pts.append([crossing[0], crossing[1]])
    for slot, point in anchors.items():
        court_pts.append([float(CORNER_COURT_M[slot][0]), float(CORNER_COURT_M[slot][1])])
        image_pts.append([point[0], point[1]])
    return np.array(court_pts, dtype=np.float64), np.array(image_pts, dtype=np.float64)


def _ransac_homography(court_pts: np.ndarray, image_pts: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Seeded RANSAC homography over court-to-image point pairs.

    Minimal 4-point samples propose a homography; the largest inlier set (by image
    reprojection) is refit with least squares. Seeded so the whole module is
    deterministic. None when fewer than four pairs or no sample is well-posed.

    :param court_pts: (n, 2) court metres
    :param image_pts: (n, 2) image pixels, paired with court_pts
    :param rng: seeded generator for the minimal-sample draws
    :return: (3, 3) homography, or None
    """
    n = court_pts.shape[0]
    if n < 4:
        return None
    court32 = court_pts.astype(np.float32)
    image32 = image_pts.astype(np.float32)
    best_inliers: np.ndarray | None = None
    best_count = 0
    for _ in range(RANSAC_ITERS):
        pick = rng.choice(n, 4, replace=False)
        try:
            candidate = cv2.getPerspectiveTransform(court32[pick], image32[pick])
        except cv2.error:
            continue
        errors = np.linalg.norm(_project(candidate, court_pts) - image_pts, axis=1)
        inliers = errors < RANSAC_INLIER_PX
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_count < 4:
        return None
    homography, _ = cv2.findHomography(court32[best_inliers], image32[best_inliers], method=0)
    return homography


# --- Acceptance gate -------------------------------------------------------

def _line_reproj_error(homography: np.ndarray, assignments: list[tuple[_Line, int]]) -> float:
    """Median perpendicular distance between assigned detected lines and the model.

    Scores over ALL assigned line evidence, not just the intersections the fit
    used. Each detected line is sampled along its own extent and measured against
    its painted line as projected by the homography.

    :return: median distance in pixels, or inf when nothing is assigned
    """
    if not assignments:
        return float("inf")
    model = _model_image_lines(homography)
    distances: list[float] = []
    for line, seg_idx in assignments:
        coef = model[seg_idx][0]
        # A near-singular homography collapses a painted line to a point; its
        # coef is ~(0, 0, 0) and would score ~zero distance, letting a fully
        # degenerate fit WIN the candidate comparison. Normalised lines carry a
        # unit normal, so anything far under 1 is the collapsed case: unusable.
        if np.hypot(coef[0], coef[1]) < 0.5:
            return float("inf")
        direction = np.array([np.cos(line.angle), np.sin(line.angle)])
        samples = line.mid + np.linspace(-line.support / 2, line.support / 2, 5)[:, None] * direction
        distances.extend(abs(samples @ coef[:2] + coef[2]))
    return float(np.median(distances))


def _anchor_reproj_error(homography: np.ndarray, anchors: dict[int, np.ndarray]) -> float:
    """:return: largest reprojection error over the confident anchors, in pixels.

    The anchors are fitted alongside the line evidence, so a homography that
    cannot honour them (a garbage anchor RANSAC rejected, or lines that contradict
    a displaced anchor) shows a large error here and fails the gate. This is what
    keeps a bad anchor from poisoning the output.
    """
    slots = sorted(anchors)
    projected = _project(homography, CORNER_COURT_M[slots])
    errors = [float(np.linalg.norm(projected[i] - anchors[slot])) for i, slot in enumerate(slots)]
    return max(errors)


def _assemble_corners(
    homography: np.ndarray, anchors: dict[int, np.ndarray]
) -> tuple[np.ndarray, tuple[str, str, str, str]]:
    """Final quad: anchors keep their model position, missing corners come from H.

    :return: ((4, 2) float32 corners TL TR BR BL, per-corner source tuple)
    """
    projected = _project(homography, CORNER_COURT_M)
    corners = np.zeros((4, 2), dtype=np.float32)
    sources: list[str] = []
    for slot in (TL, TR, BR, BL):
        if slot in anchors:
            corners[slot] = anchors[slot]
            sources.append("model")
        else:
            corners[slot] = projected[slot]
            sources.append("fallback")
    return corners, (sources[0], sources[1], sources[2], sources[3])


# --- Entry point -----------------------------------------------------------

def _ckn_path(detections: list[CornerDetection], scene_peaks: np.ndarray) -> CourtQuad | None:
    """All four corners confident: reuse the wrapper's per-scene median verbatim."""
    from .wrapper import ckn_scene_corners  # local import keeps this module torch-free

    corners = ckn_scene_corners(detections)
    if corners is None:
        logger.info("court fallback: 4 confident corners but no fully-passing frame; failing closed")
        return None
    return CourtQuad(
        corners_px=corners,
        peak=scene_peaks.astype(np.float32),
        source="model",
        corner_source=("model", "model", "model", "model"),
        diagnostics=None,
    )


def _cv2_path(
    frames_bgr: list[np.ndarray],
    clean: list[CornerDetection],
    scene_peaks: np.ndarray,
    confident: np.ndarray,
    corner_min_peak_conf: float,
) -> CourtQuad | None:
    """Recover 1-2 withheld corners from pooled line evidence and the BWF model."""
    anchors = _anchor_points(clean, confident, corner_min_peak_conf)
    sample_quad = _scene_sample_quad(clean)

    segments = [_frame_segments(frame, _mat_roi(frame, sample_quad)) for frame in frames_bgr]
    pooled = np.concatenate(segments) if any(s.shape[0] for s in segments) else np.empty((0, 4), dtype=np.int32)
    lines, max_sagitta = _cluster_segments(pooled)
    if _measure_sagitta_material(max_sagitta):
        logger.info("court fallback: line curvature %.1f px looks material; leaving geometry uncorrected", max_sagitta)
    if len(lines) < 2 * MIN_FAMILY_LINES:
        logger.info("court fallback: only %d pooled court lines; too little evidence", len(lines))
        return None

    x_family, y_family = _split_families(lines, anchors)
    if len(x_family) < MIN_FAMILY_LINES or len(y_family) < MIN_FAMILY_LINES:
        logger.info("court fallback: a line family is under-populated (x=%d y=%d)", len(x_family), len(y_family))
        return None

    outer = _outer_lines(x_family, y_family, anchors)
    if outer is None:
        logger.info("court fallback: could not identify the four outer court lines")
        return None
    bootstrap = _bootstrap_corners(outer, anchors)
    if bootstrap is None:
        logger.info("court fallback: outer court lines do not intersect to a valid quad")
        return None

    try:
        homography0 = cv2.getPerspectiveTransform(CORNER_COURT_M, bootstrap.astype(np.float32))
    except cv2.error:
        logger.info("court fallback: bootstrap homography is degenerate")
        return None

    court_pts, image_pts = _correspondences(_assign_lines(homography0, lines), anchors)
    refined = _ransac_homography(court_pts, image_pts, np.random.default_rng(RANSAC_SEED))
    # Keep whichever of the bootstrap and the refined fit scores lower against the
    # line evidence; on clean footage the bootstrap already fits, on noisy footage
    # the refit over pooled intersections wins.
    candidates = [homography0] + ([refined] if refined is not None else [])
    homography = min(candidates, key=lambda h: _line_reproj_error(h, _assign_lines(h, lines)))

    assignments = _assign_lines(homography, lines)
    reproj_line = _line_reproj_error(homography, assignments)
    reproj_anchor = _anchor_reproj_error(homography, anchors)
    diagonal = float(np.hypot(*frames_bgr[0].shape[:2]))
    gate_line = reproj_line / diagonal
    gate_anchor = reproj_anchor / diagonal
    if gate_line > GATE_LINE_FRAC or gate_anchor > GATE_ANCHOR_FRAC:
        logger.info(
            "court fallback: acceptance gate failed (line %.4f > %.4f or anchor %.4f > %.4f); failing closed",
            gate_line, GATE_LINE_FRAC, gate_anchor, GATE_ANCHOR_FRAC,
        )
        return None

    corners, corner_source = _assemble_corners(homography, anchors)
    diagnostics = FallbackDiagnostics(
        reproj_line_px=reproj_line,
        reproj_anchor_px=reproj_anchor,
        gate_line_frac=gate_line,
        gate_anchor_frac=gate_anchor,
        n_lines_used=len(assignments),
        n_correspondences=court_pts.shape[0],
        max_sagitta_px=max_sagitta,
    )
    return CourtQuad(
        corners_px=corners,
        peak=scene_peaks.astype(np.float32),
        source="fallback",
        corner_source=corner_source,
        diagnostics=diagnostics,
    )


def pick_scene_corners(
    frames_bgr: list[np.ndarray],
    detections: list[CornerDetection],
    *,
    corner_min_peak_conf: float = DEFAULT_CORNER_MIN_PEAK_CONF,
) -> CourtQuad | None:
    """One scene in, four court corners with provenance out, or None (fail closed).

    Trigger over the scene's geometry-clean frames only: a corner is confident
    when its per-corner median peak clears the floor. Four confident corners take
    the model path (the wrapper's median); two or three take the classical-CV
    fallback; zero or one, or no geometry-clean frame at all, fails closed, since
    from-zero detection is out of scope.

    :param frames_bgr: the scene's sampled BGR frames (static-camera assumption)
    :param detections: the matching CornerDetection per frame, same order/length
    :param corner_min_peak_conf: per-corner peak-confidence floor defining a
        confident corner
    :return: the recovered court quad, or None when the scene fails closed
    """
    clean = [detection for detection in detections if _geometry_clean(detection)]
    if not clean:
        logger.info("court fallback: no geometry-clean frame in the scene; failing closed")
        return None

    scene_peaks = _scene_peaks(clean)
    confident = scene_peaks >= corner_min_peak_conf
    n_confident = int(confident.sum())

    if n_confident == 4:
        return _ckn_path(detections, scene_peaks)
    if n_confident <= 1:
        logger.info("court fallback: only %d confident corner(s); from-zero detection is out of scope", n_confident)
        return None

    clean_frames = [frame for frame, detection in zip(frames_bgr, detections) if _geometry_clean(detection)]
    return _cv2_path(clean_frames, clean, scene_peaks, confident, corner_min_peak_conf)


# --- Cross-scene consensus repair (video-level post-pass) ------------------

# A fixed broadcast camera keeps every scene's true corners nearly still, so a
# scene that latched onto a spurious line (the KNOWN FAILURE MODE above
# GATE_LINE_FRAC) sits far from where the rest of the video agrees. Measured on
# the recorded 720p eval (46 + 44 fallback scenes over the two videos): good
# scenes drift at most 18.1 px from the per-corner-median consensus, the 7
# boards-aliased scenes start at 177.3 px, and the clean control maxes at 8.2 px
# with zero false flags anywhere inside that ~10x gap. 55 sits at ~3x the
# good-scene max. Absolute px in the caller's frame space: at 1080p the gap
# reads 1.5x these numbers so 55 still fits; 4K needs a re-measure first.
CONSENSUS_FLAG_THRESHOLD_PX = 55.0


@dataclass(frozen=True)
class ConsensusRepair:
    """Video-level consensus over one video's per-scene quads, outliers repaired.

    pick_scene_corners's per-scene output stays primary; this is an opt-in post-pass a
    caller runs across a video's own scenes. A flagged scene is replaced
    WHOLE-QUAD by the consensus, not just its offending corner or corners: the
    boards-alias failure this targets fits a self-consistent-looking homography
    through the whole quad (both far corners move together), so per-corner
    patching would leave a mismatched blend of real and aliased geometry. A
    corners-only variant was measured beside this and is not shipped. See
    ``docs/courtkeynet/fallback_evaluation/README.md`` for the evaluation and
    reproducible check.
    """

    consensus_quad: np.ndarray  # (4, 2) float64, per-corner median over all scenes
    distances_px: np.ndarray  # (n_scenes,) worst-corner distance of each scene's quad to the consensus
    flagged: np.ndarray  # (n_scenes,) bool, True where distances_px exceeds CONSENSUS_FLAG_THRESHOLD_PX
    repaired_quads: np.ndarray  # (n_scenes, 4, 2) float64, flagged scenes replaced by consensus_quad; others unchanged


def consensus_repair(quads: np.ndarray) -> ConsensusRepair:
    """Per-corner median across a video's scene quads, with far-flung scenes repaired.

    Pure and torch-free: operates on whatever pixel space the caller's quads
    already share (native-frame px throughout pick_scene_corners's contract; every quad
    passed in must be from the SAME video, since the consensus assumes one fixed
    camera). Distance is maxed over the 4 corners, mirroring the scoping's
    dist_to_quad: a homography-level fault moves corners together, and one
    strayed corner registers the full displacement. A single-scene video is its
    own consensus and comes back unflagged.

    :param quads: (n_scenes, 4, 2) one video's scene quads, TL TR BR BL corner
        order (matching CourtQuad.corners_px), all in the same pixel space
    :return: ConsensusRepair
    :raises ValueError: malformed shape, no scenes, or half or more of the
        scenes disagree with the median. That last case means there is no
        trustworthy majority to repair from (an even 50/50 split puts the median
        at a hallucinated midpoint that NO scene produced, so everything flags
        and a silent repair would overwrite the good scenes with it); the caller
        keeps the per-scene answers instead.
    """
    quads = np.asarray(quads, dtype=np.float64)
    if quads.ndim != 3 or quads.shape[1:] != (4, 2):
        raise ValueError(f"consensus_repair: expected (n_scenes, 4, 2) quads, got shape {quads.shape}")
    if quads.shape[0] == 0:
        raise ValueError("consensus_repair: at least one scene quad is required")

    consensus = np.median(quads, axis=0)  # marginal per-axis median, matching the scoping's consensus_quad
    per_corner_dist = np.linalg.norm(quads - consensus, axis=2)  # (n_scenes, 4)
    distances = per_corner_dist.max(axis=1)  # (n_scenes,), worst corner per scene
    flagged = distances > CONSENSUS_FLAG_THRESHOLD_PX
    if flagged.sum() * 2 >= quads.shape[0]:
        raise ValueError(
            f"consensus_repair: {int(flagged.sum())} of {quads.shape[0]} scenes sit beyond "
            f"{CONSENSUS_FLAG_THRESHOLD_PX} px of the median; no trustworthy majority exists, "
            "keep the per-scene answers"
        )
    repaired_quads = np.where(flagged[:, None, None], consensus, quads)

    return ConsensusRepair(
        consensus_quad=consensus,
        distances_px=distances,
        flagged=flagged,
        repaired_quads=repaired_quads,
    )
