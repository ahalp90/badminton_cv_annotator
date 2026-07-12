"""Tests for the court-recovery fallback (src/courtkeynet/fallback.py).

CPU-only and model-free: no CourtKeyNet weights load here. The recovery tests
render a synthetic court onto a mat-coloured canvas under a known homography,
synthesise CornerDetection objects with chosen peaks and flags, run the fallback,
and check the recovered corners against the homography's own corners. The unit
tests exercise the line algebra and clustering on hand-built geometry.
"""

import cv2
import numpy as np
import pytest

from courtkeynet import fallback as fb
from courtkeynet.fallback import (
    CORNER_COURT_M,
    BL,
    BR,
    TL,
    TR,
    _anchor_points,
    _cluster_segments,
    _circular_diff,
    _intersect,
    _line_through,
    _point_line_distance,
    _project,
    _split_families,
    _tls_line,
    scene_court,
)
from courtkeynet.wrapper import CornerDetection

FRAME_H, FRAME_W = 720, 1280
# A mild, roughly-behind-baseline perspective. Foreshortening is gentle so the
# far baseline and the far service lines keep real pixel separation (see the
# clustering-tolerance note in the module).
IMG_CORNERS = np.array([[300, 150], [980, 150], [1050, 650], [230, 650]], dtype=np.float32)  # TL TR BR BL
# A homography that drops the near-left corner off the bottom-left of the frame.
IMG_CORNERS_OFF = np.array([[350, 140], [1000, 140], [1120, 690], [-40, 735]], dtype=np.float32)

GREEN_MAT, YELLOW_LINE = (40, 110, 45), (0, 220, 230)
BLUE_MAT, WHITE_LINE = (150, 70, 20), (255, 255, 255)


def _homography(img_corners: np.ndarray) -> np.ndarray:
    """:return: (3, 3) court-metres -> image-pixels for the given image corners."""
    return cv2.getPerspectiveTransform(CORNER_COURT_M, img_corners.astype(np.float32))


def _render_court(homography: np.ndarray, mat_bgr: tuple, line_bgr: tuple) -> np.ndarray:
    """Paint the BWF court lines onto a mat-coloured canvas under a homography."""
    frame = np.full((FRAME_H, FRAME_W, 3), mat_bgr, dtype=np.uint8)
    for end_a_m, end_b_m in fb.PAINTED_SEGMENTS_M:
        ends = _project(homography, np.stack([end_a_m, end_b_m]))
        cv2.line(frame, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), line_bgr, 2, cv2.LINE_AA)
    return frame


def _detection(homography: np.ndarray, confident: set, withheld_offset: float = 0.0) -> CornerDetection:
    """A geometry-clean detection: confident corners strong, the rest under the floor.

    Withheld corners can be nudged toward the centroid so the recovery has to lean
    on the line evidence rather than the model's own (deliberately wrong) corner.
    """
    corners = _project(homography, CORNER_COURT_M).astype(np.float32)
    peak = np.full(4, 0.005, dtype=np.float32)
    centroid = corners.mean(axis=0)
    for slot in range(4):
        if slot in confident:
            peak[slot] = 0.5
        elif withheld_offset:
            toward = centroid - corners[slot]
            corners[slot] = corners[slot] + withheld_offset * toward / np.linalg.norm(toward)
    return CornerDetection(
        corners_px=corners, peak=peak, entropy=np.full(4, 0.3, dtype=np.float32), flags=()
    )


def _scene(img_corners: np.ndarray, confident: set, mat=GREEN_MAT, line=YELLOW_LINE, offset=25.0):
    """Build a four-frame static-camera scene and its detections for one config."""
    homography = _homography(img_corners)
    frames = [_render_court(homography, mat, line) for _ in range(4)]
    detections = [_detection(homography, confident, offset) for _ in range(4)]
    true_corners = _project(homography, CORNER_COURT_M)
    return frames, detections, true_corners


# --- Line algebra ----------------------------------------------------------

def test_line_through_and_distance() -> None:
    """A horizontal line: points on it score zero distance, off it score the gap."""
    coef = _line_through(np.array([0.0, 100.0]), np.array([50.0, 100.0]))
    assert _point_line_distance(np.array([25.0, 100.0]), coef) < 1e-9
    assert _point_line_distance(np.array([25.0, 130.0]), coef) == pytest.approx(30.0)


def test_intersect_crossing_lines() -> None:
    """Two hand-built lines cross at their shared point."""
    horizontal = _line_through(np.array([0.0, 200.0]), np.array([100.0, 200.0]))
    vertical = _line_through(np.array([300.0, 0.0]), np.array([300.0, 100.0]))
    crossing = _intersect(horizontal, vertical)
    assert crossing is not None
    assert crossing == pytest.approx([300.0, 200.0])


def test_intersect_parallel_returns_none() -> None:
    """Parallel lines have no finite intersection."""
    line_a = _line_through(np.array([0.0, 100.0]), np.array([100.0, 100.0]))
    line_b = _line_through(np.array([0.0, 200.0]), np.array([100.0, 200.0]))
    assert _intersect(line_a, line_b) is None


def test_tls_line_recovers_slope() -> None:
    """Total-least-squares recovers a noisy-free set of collinear points."""
    points = np.array([[0.0, 10.0], [10.0, 20.0], [20.0, 30.0], [30.0, 40.0]])  # slope 1
    coef, angle, _ = _tls_line(points)
    assert _point_line_distance(np.array([40.0, 50.0]), coef) < 1e-6
    assert angle == pytest.approx(np.pi / 4)


def test_circular_diff_wraps_at_pi() -> None:
    """Orientation difference is taken modulo pi (a line has no direction sign)."""
    assert _circular_diff(0.05, np.pi - 0.05) == pytest.approx(0.10)
    assert _circular_diff(0.1, 0.4) == pytest.approx(0.3)


# --- Segment clustering ----------------------------------------------------

def test_cluster_segments_groups_two_lines() -> None:
    """Fragments of one horizontal and one vertical line pool into two candidates."""
    segments = np.array(
        [
            [0, 100, 60, 100], [70, 100, 130, 100], [140, 100, 200, 100],  # horizontal, y=100
            [400, 0, 400, 70], [400, 80, 400, 160],  # vertical, x=400
        ],
        dtype=np.int32,
    )
    lines, max_sagitta = _cluster_segments(segments)
    assert len(lines) == 2
    assert max_sagitta < 1.0  # straight fragments have no curvature
    angles = sorted(line.angle for line in lines)
    assert angles[0] == pytest.approx(0.0, abs=1e-6)  # horizontal
    assert angles[1] == pytest.approx(np.pi / 2, abs=1e-6)  # vertical


def test_cluster_segments_drops_weak_support() -> None:
    """A stray short fragment falls below the support floor and is discarded."""
    segments = np.array(
        [[0, 100, 80, 100], [80, 100, 160, 100], [500, 500, 510, 505]],  # long line + a nub
        dtype=np.int32,
    )
    lines, _ = _cluster_segments(segments)
    assert len(lines) == 1


# --- Family split ----------------------------------------------------------

def test_split_families_labels_by_baseline_direction() -> None:
    """With a near-baseline anchor pair, the horizontal pencil is the y-family."""
    homography = _homography(IMG_CORNERS)
    frames = [_render_court(homography, GREEN_MAT, YELLOW_LINE) for _ in range(2)]
    detections = [_detection(homography, {BL, BR}) for _ in range(2)]
    sample_quad = fb._scene_sample_quad(detections)
    pooled = np.concatenate([fb._frame_segments(f, fb._mat_roi(f, sample_quad)) for f in frames])
    lines, _ = _cluster_segments(pooled)
    anchors = _anchor_points(detections, np.array([False, False, True, True]), 0.02)
    x_family, y_family = _split_families(lines, anchors)
    # Six across-court lines (baselines, service lines), five along-court (sidelines, centre).
    assert len(y_family) == 6
    assert len(x_family) == 5


# --- Trigger branches ------------------------------------------------------

def test_trigger_four_confident_takes_model_path() -> None:
    """Four confident corners use the wrapper median, no fallback fit."""
    frames, detections, true_corners = _scene(IMG_CORNERS, {TL, TR, BR, BL}, offset=0.0)
    result = scene_court(frames, detections)
    assert result is not None
    assert result.source == "model"
    assert result.corner_source == ("model", "model", "model", "model")
    assert result.diagnostics is None
    assert np.allclose(result.corners_px, true_corners, atol=1e-3)


def test_trigger_one_confident_fails_closed() -> None:
    """A single confident corner is out of scope: None."""
    frames, detections, _ = _scene(IMG_CORNERS, {BL})
    assert scene_court(frames, detections) is None


def test_trigger_zero_confident_fails_closed() -> None:
    """No confident corner: None."""
    frames, detections, _ = _scene(IMG_CORNERS, set())
    assert scene_court(frames, detections) is None


def test_trigger_no_geometry_clean_fails_closed() -> None:
    """Every frame geometry-flagged: the whole quad is suspect, so None."""
    homography = _homography(IMG_CORNERS)
    frames = [_render_court(homography, GREEN_MAT, YELLOW_LINE) for _ in range(3)]
    corners = _project(homography, CORNER_COURT_M).astype(np.float32)
    flagged = [
        CornerDetection(
            corners_px=corners, peak=np.full(4, 0.5, dtype=np.float32),
            entropy=np.full(4, 0.3, dtype=np.float32), flags=("non_convex",),
        )
        for _ in range(3)
    ]
    assert scene_court(frames, flagged) is None


# --- Anchor medians --------------------------------------------------------

def test_anchor_points_ignore_low_peak_frames() -> None:
    """A corner's anchor is the median over only the frames where it cleared the floor.

    Bottom-right flickers weak in two frames at a displaced position; those frames
    must not drag its anchor away from where the model actually saw it.
    """
    strong_pos = np.array([500.0, 400.0], dtype=np.float32)
    weak_pos = np.array([900.0, 900.0], dtype=np.float32)
    detections = []
    for frame_idx in range(4):
        corners = np.array([[100, 100], [500, 100], strong_pos, [100, 400]], dtype=np.float32)
        peak = np.full(4, 0.5, dtype=np.float32)
        if frame_idx >= 2:  # BR weak (and displaced) in the last two frames
            corners[BR] = weak_pos
            peak[BR] = 0.005
        detections.append(
            CornerDetection(corners_px=corners, peak=peak, entropy=np.full(4, 0.3, dtype=np.float32), flags=())
        )
    confident = np.array([True, True, True, True])  # BR median peak over 4 frames still clears the floor
    anchors = _anchor_points(detections, confident, 0.02)
    assert np.allclose(anchors[BR], strong_pos)


# --- Corner recovery -------------------------------------------------------

@pytest.fixture(scope="module")
def recovered_two_adjacent_green():
    """Fallback result for BL+BR confident on a yellow-on-green court."""
    frames, detections, true_corners = _scene(IMG_CORNERS, {BL, BR})
    return scene_court(frames, detections), true_corners


@pytest.fixture(scope="module")
def recovered_two_adjacent_blue():
    """Fallback result for BL+BR confident on a white-on-blue court."""
    frames, detections, true_corners = _scene(IMG_CORNERS, {BL, BR}, mat=BLUE_MAT, line=WHITE_LINE)
    return scene_court(frames, detections), true_corners


@pytest.fixture(scope="module")
def recovered_three_anchor():
    """Fallback result for TL+TR+BL confident (BR withheld)."""
    frames, detections, true_corners = _scene(IMG_CORNERS, {TL, TR, BL})
    return scene_court(frames, detections), true_corners


@pytest.fixture(scope="module")
def recovered_two_diagonal():
    """Fallback result for TL+BR confident (the deprioritised diagonal case)."""
    frames, detections, true_corners = _scene(IMG_CORNERS, {TL, BR})
    return scene_court(frames, detections), true_corners


def test_two_adjacent_recovers_far_corners(recovered_two_adjacent_green) -> None:
    """BL+BR confident: the withheld far corners are recovered from lines, not the model.

    The model's TL/TR were nudged 25 px off; landing within a few px of the true
    corners proves the recovery came from the line evidence.
    """
    result, true_corners = recovered_two_adjacent_green
    assert result is not None
    assert result.source == "fallback"
    assert result.corner_source == ("fallback", "fallback", "model", "model")
    assert np.linalg.norm(result.corners_px[TL] - true_corners[TL]) < 4.0
    assert np.linalg.norm(result.corners_px[TR] - true_corners[TR]) < 4.0


def test_two_adjacent_gate_diagnostics(recovered_two_adjacent_green) -> None:
    """A clean fit sits well inside both arms of the acceptance gate."""
    result, _ = recovered_two_adjacent_green
    assert result is not None
    assert result.diagnostics.gate_line_frac < fb.GATE_LINE_FRAC
    assert result.diagnostics.gate_anchor_frac < fb.GATE_ANCHOR_FRAC
    assert result.diagnostics.n_lines_used >= 8
    assert result.diagnostics.max_sagitta_px < fb.SAGITTA_MATERIAL_PX


def test_two_adjacent_blue_court(recovered_two_adjacent_blue) -> None:
    """Mat colour is sampled, not hardcoded: white-on-blue recovers just as well."""
    result, true_corners = recovered_two_adjacent_blue
    assert result is not None
    assert np.linalg.norm(result.corners_px[TL] - true_corners[TL]) < 4.0
    assert np.linalg.norm(result.corners_px[TR] - true_corners[TR]) < 4.0


def test_three_anchor_recovers_missing_corner(recovered_three_anchor) -> None:
    """TL+TR+BL confident: the withheld BR is recovered within a few px."""
    result, true_corners = recovered_three_anchor
    assert result is not None
    assert result.corner_source == ("model", "model", "fallback", "model")
    assert np.linalg.norm(result.corners_px[BR] - true_corners[BR]) < 4.0


def test_two_diagonal_recovers_both_corners(recovered_two_diagonal) -> None:
    """TL+BR confident: both withheld corners recover within a few px."""
    result, true_corners = recovered_two_diagonal
    assert result is not None
    assert result.corner_source == ("model", "fallback", "model", "fallback")
    assert np.linalg.norm(result.corners_px[TR] - true_corners[TR]) < 4.0
    assert np.linalg.norm(result.corners_px[BL] - true_corners[BL]) < 4.0


def test_off_frame_corner_is_finite_and_out_of_bounds() -> None:
    """A near corner projecting below the frame is recovered, negative/out-of-frame."""
    frames, detections, true_corners = _scene(IMG_CORNERS_OFF, {TL, TR, BR}, offset=0.0)
    result = scene_court(frames, detections)
    assert result is not None
    recovered_bl = result.corners_px[BL]
    assert np.all(np.isfinite(recovered_bl))
    assert recovered_bl[0] < 0 or recovered_bl[1] > FRAME_H  # genuinely off the frame
    assert np.linalg.norm(recovered_bl - true_corners[BL]) < 4.0


# --- Fail-closed and determinism -------------------------------------------

def test_garbage_anchors_fail_the_gate() -> None:
    """Confident but positionally garbage anchors must yield None, not a poisoned quad.

    The anchors sit mid-court, far from any painted line; no homography reconciles
    them with the line evidence, so the acceptance gate fires.
    """
    homography = _homography(IMG_CORNERS)
    frames = [_render_court(homography, GREEN_MAT, YELLOW_LINE) for _ in range(4)]
    corners = _project(homography, CORNER_COURT_M).astype(np.float32)
    corners[BL] = [560, 440]
    corners[BR] = [720, 470]
    peak = np.array([0.005, 0.005, 0.5, 0.5], dtype=np.float32)  # BL, BR confident but garbage
    detections = [
        CornerDetection(corners_px=corners, peak=peak, entropy=np.full(4, 0.3, dtype=np.float32), flags=())
        for _ in range(4)
    ]
    assert scene_court(frames, detections) is None


def test_degenerate_too_few_lines_fails_closed() -> None:
    """A blank mat with no painted lines has no evidence to fit: None, no crash."""
    frames = [np.full((FRAME_H, FRAME_W, 3), GREEN_MAT, dtype=np.uint8) for _ in range(4)]
    detections = [_detection(_homography(IMG_CORNERS), {BL, BR}) for _ in range(4)]
    assert scene_court(frames, detections) is None


def test_fallback_is_deterministic() -> None:
    """Seeded RANSAC and no wall-clock: identical inputs give identical corners."""
    frames, detections, _ = _scene(IMG_CORNERS, {BL, BR})
    first = scene_court(frames, detections)
    second = scene_court(frames, detections)
    assert first is not None and second is not None
    assert np.array_equal(first.corners_px, second.corners_px)


@pytest.mark.xfail(
    strict=True,
    reason="known failure mode, measured on ShuttleSet vid 3 (session 17): a boards-like "
    "line beyond the far baseline steals the extreme-line far-baseline pick and the "
    "aliased fit passes the gate. Candidate fixes measured worse or are parked; see the "
    "KNOWN FAILURE MODE note beside the gate constants in fallback.py.",
)
def test_boards_alias_line_is_overridden() -> None:
    """A strong spurious line beyond the far baseline should not steal it.

    Encodes the measured ShuttleSet vid-3 failure: the advertising boards' bottom
    edge sits on mat colour beyond the far baseline and wins the extreme-line rule.
    A fix should recover the true far baseline (or fail closed) and flip this xfail
    to a pass.
    """
    homography = _homography(IMG_CORNERS)
    frames = []
    for _ in range(4):
        frame = _render_court(homography, GREEN_MAT, YELLOW_LINE)
        # The boards edge: bright, well-supported, parallel to and above the far
        # baseline (y=150 at the corners), still on the all-mat canvas so it lands
        # inside the ROI exactly as vid 3's surround apron put it there.
        cv2.line(frame, (250, 90), (1050, 90), WHITE_LINE, 2, cv2.LINE_AA)
        frames.append(frame)
    detections = [_detection(homography, {BL, BR}, 25.0) for _ in range(4)]
    true_corners = _project(homography, CORNER_COURT_M)

    result = scene_court(frames, detections)
    assert result is not None
    assert result.source == "fallback"
    for slot in (TL, TR):
        assert np.linalg.norm(result.corners_px[slot] - true_corners[slot]) < 4.0


def test_cluster_segments_joins_across_angle_wrap() -> None:
    """Fragments of one horizontal line read as ~0 and ~pi share a cluster.

    A linear average of the two orientations snaps to vertical and shatters the
    cluster (session-17 review find); the doubled-angle mean must not.
    """
    segments = np.array([
        [0, 100, 200, 101],  # angle just above 0
        [300, 101, 500, 100],  # the same line, angle just under pi
        [600, 100, 800, 101],
    ])
    lines, _ = _cluster_segments(segments)
    assert len(lines) == 1
    assert lines[0].support == pytest.approx(600.0, rel=0.01)


def test_cluster_segments_joins_near_vertical_fragments() -> None:
    """Fragments of one near-vertical line either side of 90 degrees share a cluster.

    The old membership test compared global homogeneous offsets, which flip sign
    as the normal crosses b=0 (session-17 review find); midpoint-to-line distance
    is orientation-stable.
    """
    segments = np.array([
        [640, 0, 638, 200],  # just past vertical one way
        [641, 400, 644, 600],  # just past vertical the other way
    ])
    lines, _ = _cluster_segments(segments)
    assert len(lines) == 1
