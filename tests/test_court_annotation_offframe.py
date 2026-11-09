"""Tests for the off-frame corner annotation tool and the court landmark maths.

Same loading trick as ``test_court_annotation.py``: the modules are loaded
straight from their file paths so the pure helpers run without importing torch,
opening a cv2 window, or decoding video. ``court_landmarks`` is registered under
its canonical name first, so the tool's own import resolves
to the same module object.
"""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relpath: str) -> ModuleType:
    """Load a module from a repo-relative path, bypassing the package __init__.

    :param name: the module name to register it under
    :param relpath: path to the .py file relative to the repo root
    :return: the executed module
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so machinery that resolves the module by name (e.g.
    # dataclasses inspecting annotations) finds it while the module body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


court = _load_module("court_annotation.court_landmarks", "scripts/court_annotation/court_landmarks.py")
offframe = _load_module(
    "annotate_court_corners_offframe_under_test",
    "scripts/court_annotation/annotate_court_corners_offframe.py",
)
# Loading the tool imported the base clicker; its loupe maths drives click_at.
image_helpers = sys.modules["court_annotation.image_helpers"]
check = _load_module(
    "check_extrapolation_under_test",
    "scripts/court_annotation/check_extrapolation.py",
)

WIDTH, HEIGHT = 1280, 720
# Same shape as the fallback tests' off-frame case: BL sits beyond the left and
# bottom edges of a 1280x720 frame.
IMG_CORNERS_OFF = np.array([[350.0, 140.0], [1000.0, 140.0], [1120.0, 690.0], [-40.0, 735.0]], dtype=np.float32)
H_TRUE = cv2.getPerspectiveTransform(np.asarray(court.CORNER_COURT_M, dtype=np.float32), IMG_CORNERS_OFF)

# Landmarks spread across both court axes, so any prefix of 4+ pins the fit.
# Picked to dodge the court's hidden collinear runs: evenly spaced line
# positions put e.g. right-doubles at the long service line, centre at the
# short service line, and left-doubles at the other short service line on one
# straight diagonal. The catalogue's own order is y-major, so ITS prefixes are
# collinear; the degenerate-spread tests below lean on both traps.
GOOD_NAMES = [
    "far_baseline_x_left_singles",
    "far_long_service_x_right_doubles",
    "near_short_service_x_left_doubles",
    "near_baseline_x_right_singles",
    "far_short_service_x_centre",
    "near_long_service_x_left_singles",
    "far_baseline_x_right_doubles",
    "near_baseline_x_centre",
]


def landmark_pixels(names: list[str], homography: np.ndarray) -> np.ndarray:
    """:return: (n, 2) image pixels of the named landmarks under ``homography``."""
    court_pts = np.array([court.LANDMARKS[name] for name in names], dtype=np.float64)
    return court.project_points(homography, court_pts)


# --- Landmark catalogue -----------------------------------------------------

def test_catalogue_covers_every_painted_crossing() -> None:
    """5 x-lines by 6 y-lines gives 30 distinct named points."""
    assert len(court.LANDMARKS) == 30
    assert len(set(court.LANDMARKS.values())) == 30


def test_catalogue_spot_values() -> None:
    """Known intersections sit where the BWF court diagram puts them."""
    assert court.LANDMARKS["far_baseline_x_left_doubles"] == (0.0, 0.0)
    assert court.LANDMARKS["near_baseline_x_right_doubles"] == (6.10, 13.40)
    assert court.LANDMARKS["far_short_service_x_centre"] == (3.05, pytest.approx(4.72))
    assert court.LANDMARKS["near_long_service_x_left_singles"] == (0.46, pytest.approx(12.64))


def test_catalogue_contains_the_four_corners() -> None:
    """The outer corners are catalogue points, matching CORNER_COURT_M slot order."""
    corner_names = (
        "far_baseline_x_left_doubles", "far_baseline_x_right_doubles",
        "near_baseline_x_right_doubles", "near_baseline_x_left_doubles",
    )
    for slot, name in enumerate(corner_names):
        assert court.LANDMARKS[name] == pytest.approx(tuple(court.CORNER_COURT_M[slot]))


# --- Homography fit and projection ------------------------------------------

def test_fit_recovers_offframe_corner_exactly() -> None:
    """Exact landmark pixels pin the homography; the off-frame corner projects true."""
    names = GOOD_NAMES[:6]
    homography, residuals = court.fit_homography(
        np.array([court.LANDMARKS[name] for name in names]), landmark_pixels(names, H_TRUE)
    )
    # Tolerances allow the float32 noise baked into H_TRUE's construction.
    assert residuals == pytest.approx(np.zeros(6), abs=1e-3)
    corners = court.project_corners(homography)
    assert corners == pytest.approx(IMG_CORNERS_OFF, abs=1e-2)
    # The BL slot genuinely lies outside the frame and stays unclamped.
    assert corners[3, 0] < 0 and corners[3, 1] > HEIGHT


def test_fit_with_click_noise_stays_close() -> None:
    """With +-1 px noise on 8 spread landmarks, every corner lands within a few px."""
    names = [
        "far_baseline_x_left_singles", "far_baseline_x_right_singles",
        "far_short_service_x_left_doubles", "far_short_service_x_right_doubles",
        "near_short_service_x_left_doubles", "near_short_service_x_right_doubles",
        "near_baseline_x_left_singles", "near_baseline_x_right_singles",
    ]
    rng = np.random.default_rng(20260712)
    noisy = landmark_pixels(names, H_TRUE) + rng.uniform(-1.0, 1.0, size=(len(names), 2))
    homography, _ = court.fit_homography(np.array([court.LANDMARKS[name] for name in names]), noisy)
    errors = np.linalg.norm(court.project_corners(homography) - IMG_CORNERS_OFF, axis=1)
    assert errors.max() < 8.0


def test_fit_floor_refuses_exact_four_point_fits() -> None:
    """4 pairs fit exactly, zero residual even on a mislabelled point, so the
    default floor is 5; callers that gate correctness another way opt down."""
    names = GOOD_NAMES[:4]
    court_pts = np.array([court.LANDMARKS[name] for name in names])
    pixels = landmark_pixels(names, H_TRUE)
    with pytest.raises(ValueError, match="at least 5"):
        court.fit_homography(court_pts, pixels)
    # The blindness the floor exists for: an exact fit shows no residual at all.
    _homography, residuals = court.fit_homography(court_pts, pixels, min_points=4)
    assert residuals == pytest.approx(np.zeros(4), abs=1e-3)


def test_fit_rejects_collinear_landmarks() -> None:
    """Five points along one line cannot pin a homography and must not pretend to."""
    names = [f"far_baseline_x_{x_name}"
             for x_name in ("left_doubles", "left_singles", "centre", "right_singles", "right_doubles")]
    with pytest.raises(ValueError, match="degenerate"):
        court.fit_homography(np.array([court.LANDMARKS[name] for name in names]), landmark_pixels(names, H_TRUE))


def test_fit_rejects_mostly_collinear_landmarks() -> None:
    """Five baseline points plus one stray leave the map underdetermined.

    This set is invertible and fits its own inputs with tiny residuals, so only
    the rank check can catch it; a silent pass here would hand back a corner
    hundreds of px wrong.
    """
    names = list(court.LANDMARK_NAMES[:6])  # five on the far baseline, one on the long service line
    with pytest.raises(ValueError, match="degenerate"):
        court.fit_homography(np.array([court.LANDMARKS[name] for name in names]), landmark_pixels(names, H_TRUE))


def test_fit_rejects_the_hidden_court_diagonal() -> None:
    """Right-doubles/long-service, centre/short-service, and left-doubles/other
    short-service lie on one straight diagonal, so this innocent-looking pick of
    four is really three collinear points plus one."""
    names = [
        "far_baseline_x_left_singles", "far_long_service_x_right_doubles",
        "far_short_service_x_centre", "near_short_service_x_left_doubles",
    ]
    with pytest.raises(ValueError, match="degenerate"):
        # min_points=4 isolates the conditioning guard on the minimal sneaky pick.
        court.fit_homography(np.array([court.LANDMARKS[name] for name in names]),
                             landmark_pixels(names, H_TRUE), min_points=4)


def test_extrapolation_errors_vanish_under_the_true_homography() -> None:
    assert court.extrapolation_errors(IMG_CORNERS_OFF, H_TRUE) == pytest.approx(np.zeros(4), abs=1e-3)


def test_check_fit_low_count_catches_a_misclick_the_mean_would_pass() -> None:
    """Port of the tour tool's spread-points case: one 30 px misidentification
    among six clicks smears into a low mean while the worst point sticks out."""
    names = GOOD_NAMES[:6]
    pixels = landmark_pixels(names, H_TRUE)
    pixels[2] += 30.0  # one wrong crossing
    _homography, residuals = court.fit_homography(np.array([court.LANDMARKS[n] for n in names]), pixels)
    verdict = court.check_fit(residuals, WIDTH)
    assert verdict.level == "fail" and "mislabelled" in verdict.reason
    assert verdict.rms_px < verdict.worst_px  # the smear the worst-point statistic defeats


def test_check_fit_bands() -> None:
    """3 px native floor at low counts; width-scaled warn/fail bands above."""
    low = np.array([0.5, 1.0, 2.9, 1.2, 0.8])
    assert court.check_fit(low, 1280).level == "ok"
    low_bad = low.copy()
    low_bad[2] = 3.2
    assert court.check_fit(low_bad, 1280).level == "fail"

    high = np.full(12, 2.0)
    assert court.check_fit(high, 1280).level == "ok"
    high[0] = 8.0  # inside the 1280-wide warn band (6.7 to 10)
    assert court.check_fit(high, 1280).level == "warn"
    assert court.check_fit(high, 1920).level == "ok"  # the same error is fine on wider footage
    high[0] = 12.0  # over the 1280 ceiling, inside the 1920 warn band (10 to 15)
    assert court.check_fit(high, 1280).level == "fail"
    assert court.check_fit(high, 1920).level == "warn"


def test_quad_camera_validity() -> None:
    """Convex behind-baseline quads pass; crossed and upside-down ones do not."""
    assert court.quad_is_camera_valid(IMG_CORNERS_OFF)
    crossed = IMG_CORNERS_OFF[[0, 1, 3, 2]]  # BR and BL swapped: a bowtie
    assert not court.quad_is_camera_valid(crossed)
    upside_down = IMG_CORNERS_OFF[[3, 2, 1, 0]]  # near baseline above the far one
    assert not court.quad_is_camera_valid(upside_down)


def test_shipped_ground_truth_passes_the_gates() -> None:
    """The calibration must accept the accepted: all 11 committed frames refit
    inside the gates on their native 1920-wide footage."""
    landmarks_csv = REPO_ROOT / "data/amateur_court_corners/hand_corners_landmarks.csv"
    by_frame: dict[tuple[str, str], list[dict[str, str]]] = {}
    with landmarks_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            by_frame.setdefault((row["video"], row["frame"]), []).append(row)
    assert len(by_frame) == 11
    for rows in by_frame.values():
        court_pts = np.array([[float(r["court_x_m"]), float(r["court_y_m"])] for r in rows])
        image_pts = np.array([[float(r["x_px"]), float(r["y_px"])] for r in rows])
        homography, residuals = court.fit_homography(court_pts, image_pts)
        verdict = court.check_fit(residuals, 1920)
        assert verdict.level == "ok", verdict
        assert court.quad_is_camera_valid(court.project_corners(homography))


# --- Capture session ---------------------------------------------------------

def make_session(tmp_path: Path) -> object:
    return offframe.OffframeSession("vid.mp4", WIDTH, HEIGHT, "landscape", tmp_path / "corners.csv", 20, 8)


def place_landmark(session: object, name: str, dx: float = 0.0) -> object:
    session.select_point(name)
    point = landmark_pixels([name], H_TRUE)[0]
    target = (float(point[0]) + dx, float(point[1]))
    action = session.coarse_click(round(target[0]), round(target[1]))
    lx, ly = image_helpers.source_to_loupe(*target, action.loupe_origin, session.zoom)
    session.reposition_from_loupe(lx, ly)
    return session.confirm_point()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def test_session_fit_gates_and_pending_point_invalidation(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(3)
    names = list(court.CORNER_LANDMARK_NAMES[:3]) + GOOD_NAMES[:4]
    for name in names[:4]:
        place_landmark(session, name)
    assert "at least 5" in session.do_fit().message
    for name in names[4:]:
        place_landmark(session, name)
    session.do_fit()
    assert session.fit is not None
    session.select_point(GOOD_NAMES[4])
    session.coarse_click(500, 300)
    assert session.fit is not None
    assert not session.can_save
    session.cancel_aim()
    assert session.can_save
    place_landmark(session, GOOD_NAMES[4], dx=30)
    assert session.fit is None
    assert "fit failed" in session.do_fit().message
    place_landmark(session, GOOD_NAMES[4], dx=12)
    assert "WARN" in session.do_fit().message
    assert session.fit is not None
    assert session.commit().kind == offframe.ActionKind.COMMITTED
    saved = read_rows(session.csv_path)
    assert [row['source'] for row in saved] == ['click', 'click', 'click', 'extrapolated']
    assert saved[3]['visible'] == '0'


def test_session_rejects_degenerate_point_spread(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(0)
    for name in court.LANDMARK_NAMES[:5]:
        place_landmark(session, name)
    assert "fit failed" in session.do_fit().message
    assert session.fit is None
    assert not session.can_save
    session.commit()
    assert not session.csv_path.exists()


def test_recommit_replaces_corner_and_landmark_rows(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(3)
    for name in GOOD_NAMES:
        place_landmark(session, name)
    session.do_fit()
    session.commit()
    assert len(read_rows(session.landmark_csv_path)) == len(GOOD_NAMES)
    session.begin_capture(3)
    for name in court.CORNER_LANDMARK_NAMES:
        place_landmark(session, name)
    session.commit()
    assert len(read_rows(session.csv_path)) == 4
    assert len(read_rows(session.landmark_csv_path)) == 4
    assert {row['source'] for row in read_rows(session.csv_path)} == {'click'}
    resumed = make_session(tmp_path)
    assert resumed.annotated_frames == {3}
    assert 'already annotated' in resumed.begin_capture(3).message


def test_abort_and_foreign_header(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(3)
    place_landmark(session, GOOD_NAMES[0])
    session.coarse_click(500, 300)
    session.abort()
    assert session.landmarks == ()
    assert session.pending_point is None
    assert session.fit is None
    assert not session.csv_path.exists()
    session.csv_path.write_text('video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation\n')
    with pytest.raises(ValueError, match='different tool or version'):
        make_session(tmp_path)


def test_load_prefill_picks_nearest_saved_frame(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    assert offframe.load_prefill(session.csv_path, session.landmark_csv_path, 'vid.mp4', 5) is None
    for frame in (10, 400):
        session.begin_capture(frame)
        for name in court.CORNER_LANDMARK_NAMES:
            place_landmark(session, name)
        session.commit()
    prefill = offframe.load_prefill(session.csv_path, session.landmark_csv_path, 'vid.mp4', 350)
    assert prefill is not None and prefill[0] == 400
    assert offframe.load_prefill(session.csv_path, session.landmark_csv_path, 'other.mp4', 350) is None


def test_offframe_tool_imports_without_torch() -> None:
    """The whole annotator chain runs with torch blocked, pinning the decouple
    that keeps the geometry independent of model dependencies. A subprocess, so torch is blocked
    from interpreter start; exit 0 is the proof."""
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # any `import torch` on the chain now raises
        "sys.path.insert(0, 'scripts/court_annotation')\n"
        "import annotate_court_corners_offframe\n"
        "import court_landmarks\n"
        "assert len(court_landmarks.LANDMARK_NAMES) == 30\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"torch-free import failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.startswith("OK"), result.stdout


# --- Row builders ------------------------------------------------------------

def test_build_slot_rows_guards() -> None:
    with pytest.raises(ValueError, match="4 resolved slots"):
        offframe.build_slot_rows("v", 0, [(1.0, 2.0)], None, WIDTH, HEIGHT, "landscape")
    with pytest.raises(ValueError, match="landmark fit"):
        offframe.build_slot_rows("v", 0, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), None], None, WIDTH, HEIGHT, "landscape")


# --- check_extrapolation ------------------------------------------------------

def test_frame_errors_recover_exact_clicks() -> None:
    """Corners clicked exactly where the homography puts them read as zero error."""
    names = GOOD_NAMES[:6]
    pixels = landmark_pixels(names, H_TRUE)
    landmark_rows = pd.DataFrame({
        "video": "vid.mp4", "frame": 3, "landmark": names,
        "court_x_m": [court.LANDMARKS[name][0] for name in names],
        "court_y_m": [court.LANDMARKS[name][1] for name in names],
        "x_px": pixels[:, 0], "y_px": pixels[:, 1],
    })
    corner_rows = pd.DataFrame({
        "video": "vid.mp4", "frame": 3, "corner_idx": [0, 1, 2, 3],
        "corner_label": ["tl", "tr", "br", "bl"],
        "x_px": IMG_CORNERS_OFF[:, 0], "y_px": IMG_CORNERS_OFF[:, 1],
        "x_norm": IMG_CORNERS_OFF[:, 0] / WIDTH,  # the gate recovers frame width from this
        "source": ["click", "click", "click", "extrapolated"],
    })
    errors, in_fit, verdict = check.frame_errors(corner_rows, landmark_rows)
    assert set(errors) == {"tl", "tr", "br"}  # the extrapolated row is not ground truth
    assert in_fit == []  # none of these landmarks are corner crossings
    assert list(errors.values()) == pytest.approx([0.0, 0.0, 0.0], abs=1e-2)
    assert verdict.level == "ok"
    assert verdict.rms_px == pytest.approx(0.0, abs=1e-3) and verdict.worst_px == pytest.approx(0.0, abs=1e-3)
