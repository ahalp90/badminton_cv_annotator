"""Tests for the off-frame corner annotation tool and the court landmark maths.

Same loading trick as ``test_courtkeynet_annotation.py``: the modules are loaded
straight from their file paths so the pure helpers run without importing torch,
opening a cv2 window, or decoding video. ``court_landmarks`` is registered under
its canonical name first, so the tool's own ``import court_landmarks`` resolves
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


court = _load_module("court_landmarks", "src/courtkeynet/validation_scripts/court_landmarks.py")
offframe = _load_module(
    "annotate_court_corners_offframe_under_test",
    "src/courtkeynet/validation_scripts/annotate_court_corners_offframe.py",
)
# Loading the tool imported the base clicker; its loupe maths drives click_at.
base_clicker = sys.modules["annotate_court_corners"]
check = _load_module(
    "check_extrapolation_under_test",
    "src/courtkeynet/validation_scripts/check_extrapolation.py",
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

def make_session(tmp_path: Path, policy: str = "auto") -> object:
    return offframe.OffframeSession(
        "vid.mp4", WIDTH, HEIGHT, "landscape", tmp_path / "corners.csv", half=64, zoom=8, landmark_policy=policy
    )


def click_at(session: object, target_xy: tuple[float, float]) -> object:
    """Aim near the target, reposition to it exactly via the loupe maths, confirm."""
    action = session.coarse_click(round(target_xy[0]), round(target_xy[1]))
    assert action.kind is offframe.ActionKind.OPEN_LOUPE
    lx, ly = base_clicker.source_to_loupe(target_xy[0], target_xy[1], action.loupe_origin, session.zoom)
    assert session.reposition_from_loupe(lx, ly).kind is offframe.ActionKind.ADJUSTED
    return session.confirm_point()


def place_landmark(session: object, name: str) -> object:
    """Cycle the cursor to ``name`` and click it where H_TRUE puts it."""
    while session.cursor_name != name:
        session.next_landmark()
    point = landmark_pixels([name], H_TRUE)[0]
    return click_at(session, (float(point[0]), float(point[1])))


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_all_visible_flow_commits_on_fourth_corner(tmp_path: Path) -> None:
    """With every corner clicked, the session commits without a landmark phase."""
    session = make_session(tmp_path)
    session.begin_capture(7)
    visible = [(350.0, 140.0), (1000.0, 140.0), (1120.0, 690.0), (60.0, 700.0)]
    for target in visible[:3]:
        assert click_at(session, target).kind is offframe.ActionKind.REDRAW
    action = click_at(session, visible[3])
    assert action.kind is offframe.ActionKind.COMMITTED
    assert session.state is offframe.CaptureState.SCRUB

    rows = read_rows(tmp_path / "corners.csv")
    assert [row["corner_label"] for row in rows] == ["tl", "tr", "br", "bl"]
    assert all(row["visible"] == "1" and row["source"] == "click" and row["fit_rms_px"] == "" for row in rows)
    assert float(rows[0]["x_norm"]) == pytest.approx(350.0 / WIDTH)
    assert not (tmp_path / "corners_landmarks.csv").exists()


def test_offframe_flow_extrapolates_the_missing_corner(tmp_path: Path) -> None:
    """Three clicks, one x, four landmarks, fit, commit: BL comes back off-frame."""
    session = make_session(tmp_path)
    session.begin_capture(3)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    action = session.mark_offframe()
    assert action.kind is offframe.ActionKind.REDRAW
    assert session.state is offframe.CaptureState.LANDMARK_COARSE
    assert session.offframe_slots == (3,)

    # The three clicked corners seeded themselves into the landmark set.
    seeds = list(court.CORNER_LANDMARK_NAMES[:3])
    assert [name for name, _x, _y in session.landmarks] == seeds

    # Enter and fit both refuse while the landmark set is under the floor.
    assert "press f" in session.commit().message
    assert "at least" in session.do_fit().message

    names = GOOD_NAMES[:4]
    for name in names:
        place_landmark(session, name)
    assert [name for name, _x, _y in session.landmarks] == seeds + names

    assert "fit:" in session.do_fit().message
    action = session.commit()
    assert action.kind is offframe.ActionKind.COMMITTED
    assert session.state is offframe.CaptureState.SCRUB

    rows = read_rows(tmp_path / "corners.csv")
    bl = rows[3]
    assert bl["corner_label"] == "bl" and bl["visible"] == "0" and bl["source"] == "extrapolated"
    assert float(bl["x_px"]) == pytest.approx(-40.0, abs=1e-2)
    assert float(bl["y_px"]) == pytest.approx(735.0, abs=1e-2)
    assert float(bl["x_norm"]) < 0 and float(bl["y_norm"]) > 1  # off-frame by design
    assert float(bl["fit_rms_px"]) == pytest.approx(0.0, abs=1e-3)
    assert rows[0]["source"] == "click" and rows[0]["fit_rms_px"] == ""

    landmark_rows = read_rows(tmp_path / "corners_landmarks.csv")
    assert [row["landmark"] for row in landmark_rows] == seeds + names
    first = landmark_rows[0]
    expected_x, expected_y = court.LANDMARKS[seeds[0]]
    assert float(first["court_x_m"]) == pytest.approx(expected_x)
    assert float(first["court_y_m"]) == pytest.approx(expected_y)


def test_mark_offframe_is_a_corner_phase_event(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    assert session.mark_offframe().kind is offframe.ActionKind.NOOP  # scrub
    session.begin_capture(0)
    for _ in range(4):
        session.mark_offframe()
    assert session.state is offframe.CaptureState.LANDMARK_COARSE
    assert session.mark_offframe().kind is offframe.ActionKind.NOOP  # landmark mode


def test_cursor_cycles_and_undo_restores(tmp_path: Path) -> None:
    """n/p wrap around the catalogue; undo drops the click and points back at it."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for _ in range(4):
        session.mark_offframe()

    session.prev_landmark()
    assert session.cursor == len(court.LANDMARK_NAMES) - 1
    session.next_landmark()
    assert session.cursor == 0

    click_at(session, (100.0, 100.0))
    assert session.cursor == 1  # advanced to the next unplaced entry
    action = session.undo_last()
    assert "undid" in action.message
    assert session.landmarks == ()
    assert session.cursor == 0


def test_new_click_invalidates_the_fit(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    for name in GOOD_NAMES[:4]:
        place_landmark(session, name)
    session.do_fit()
    assert session.fit is not None
    place_landmark(session, GOOD_NAMES[4])
    assert session.fit is None  # stale against the changed landmark set
    assert "press f" in session.commit().message


def test_session_reports_a_degenerate_landmark_spread(tmp_path: Path) -> None:
    """Four clicks along the far baseline refuse to fit instead of lying."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    # Catalogue entries 0-3 all sit on the far baseline.
    for name in court.LANDMARK_NAMES[:4]:
        place_landmark(session, name)
    action = session.do_fit()
    assert "fit failed" in action.message
    assert session.fit is None
    assert "press f" in session.commit().message


def test_do_fit_applies_the_reprojection_gates(tmp_path: Path) -> None:
    """Floor, hard fail, and warn all surface through do_fit; only warn and ok
    leave a usable fit behind."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()  # three seeded corner landmarks ride in

    place_landmark(session, GOOD_NAMES[0])  # 4 total: under the floor
    action = session.do_fit()
    assert "at least 5" in action.message and session.fit is None

    for name in GOOD_NAMES[1:4]:
        place_landmark(session, name)  # 7 total, all honest
    assert "fit:" in session.do_fit().message and session.fit is not None

    def place_offset(name: str, dx: float) -> None:
        while session.cursor_name != name:
            session.next_landmark()
        point = landmark_pixels([name], H_TRUE)[0]
        click_at(session, (float(point[0]) + dx, float(point[1])))

    place_offset(GOOD_NAMES[4], 30.0)  # 8 total, one mislabel-sized error
    action = session.do_fit()
    assert "fit failed" in action.message and session.fit is None

    place_offset(GOOD_NAMES[4], 12.0)  # replaced with a warn-band error (worst ~7.5 px)
    action = session.do_fit()
    assert "WARN" in action.message and "Enter commits anyway" in action.message
    assert session.fit is not None
    assert session.commit().kind is offframe.ActionKind.COMMITTED


def test_undo_cancels_a_provisional_point_first(tmp_path: Path) -> None:
    """One-key undo parity with the tour tool: u drops the pending cross first,
    then starts popping confirmed points."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    click_at(session, (350.0, 140.0))
    session.coarse_click(700, 300)  # aiming TR
    action = session.undo_last()
    assert "cancelled" in action.message
    assert session.pending_point is None
    assert session.state is offframe.CaptureState.CORNER_COARSE
    assert len(session.slots) == 1  # the confirmed corner survived


def test_reclicking_a_landmark_replaces_it(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(0)
    for _ in range(4):
        session.mark_offframe()
    click_at(session, (100.0, 100.0))
    session.prev_landmark()  # cursor moved to 1 after placing; step back to entry 0
    assert session.cursor == 0
    action = click_at(session, (120.0, 130.0))
    assert "replaced" in action.message
    assert len(session.landmarks) == 1
    assert session.landmarks[0][1:] == pytest.approx((120.0, 130.0))


def test_abort_clears_everything_and_writes_nothing(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(5)
    click_at(session, (350.0, 140.0))
    session.mark_offframe()
    session.mark_offframe()
    session.mark_offframe()
    click_at(session, (100.0, 100.0))
    action = session.abort()
    assert action.kind is offframe.ActionKind.ABORTED
    assert session.state is offframe.CaptureState.SCRUB
    assert session.slots == () and session.landmarks == ()
    assert not (tmp_path / "corners.csv").exists()
    assert not (tmp_path / "corners_landmarks.csv").exists()


def test_always_policy_collects_landmarks_on_full_court_frames(tmp_path: Path) -> None:
    """The leave-one-out route: all corners clicked, landmarks still collected."""
    session = make_session(tmp_path, policy="always")
    session.begin_capture(2)
    for slot in range(4):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    assert session.state is offframe.CaptureState.LANDMARK_COARSE
    names = list(court.LANDMARK_NAMES[:4])
    for point in landmark_pixels(names, H_TRUE):
        click_at(session, (float(point[0]), float(point[1])))
    action = session.commit()  # no fit required: every corner was clicked
    assert action.kind is offframe.ActionKind.COMMITTED
    rows = read_rows(tmp_path / "corners.csv")
    assert all(row["source"] == "click" for row in rows)
    assert len(read_rows(tmp_path / "corners_landmarks.csv")) == 4


def test_cancel_aim_drops_the_point_without_losing_work(tmp_path: Path) -> None:
    """ESC on a provisional point backs out one step, keeping every confirmed corner."""
    session = make_session(tmp_path)
    assert session.cancel_aim().kind is offframe.ActionKind.NOOP  # scrub
    session.begin_capture(0)
    click_at(session, (350.0, 140.0))
    assert session.coarse_click(500, 500).kind is offframe.ActionKind.OPEN_LOUPE  # aimed wrong
    action = session.cancel_aim()
    assert "cancelled" in action.message and "TR" in action.message
    assert session.state is offframe.CaptureState.CORNER_COARSE
    assert session.pending_point is None
    assert len(session.slots) == 1  # TL survived
    click_at(session, (1000.0, 140.0))
    assert len(session.slots) == 2


def test_undo_last_steps_back_through_corners(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.begin_capture(0)
    assert "nothing to undo" in session.undo_last().message
    click_at(session, (350.0, 140.0))
    session.mark_offframe()
    action = session.undo_last()
    assert "off-frame mark" in action.message and "TR" in action.message
    assert session.slots == ((350.0, 140.0),)
    action = session.undo_last()
    assert "undid corner" in action.message and "TL" in action.message
    assert session.slots == ()


def test_undo_last_backs_out_of_landmark_mode(tmp_path: Path) -> None:
    """u first eats landmarks, then the off-frame mark that opened landmark mode."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    place_landmark(session, GOOD_NAMES[0])
    for _ in range(4):  # the placed crossing plus the three seeded corners
        session.undo_last()
    assert session.landmarks == ()
    action = session.undo_last()
    assert "off-frame mark" in action.message
    assert session.state is offframe.CaptureState.CORNER_COARSE
    assert session.offframe_slots == () and len(session.slots) == 3


def test_adjust_flow_nudge_type_reaim_then_confirm(tmp_path: Path) -> None:
    """Nothing locks until confirmed: nudges, typed coords, and re-aims all apply."""
    session = make_session(tmp_path)
    assert session.nudge(0.5, 0.0).kind is offframe.ActionKind.NOOP  # nothing pending
    assert session.confirm_point().kind is offframe.ActionKind.NOOP
    session.begin_capture(0)
    session.coarse_click(340, 150)
    action = session.nudge(0.5, -0.5)
    assert action.kind is offframe.ActionKind.ADJUSTED
    assert "340.50, 149.50" in action.message
    action = session.set_point(352.25, 141.75)  # typed coordinates win outright
    assert "352.25, 141.75" in action.message
    action = session.coarse_click(350, 140)  # a main-window click re-aims wholesale
    assert action.kind is offframe.ActionKind.OPEN_LOUPE
    assert session.pending_point == pytest.approx((350.0, 140.0))
    assert session.slots == ()  # still nothing confirmed
    assert session.confirm_point().kind is offframe.ActionKind.REDRAW
    assert session.pending_point is None
    assert session.slots[0] == pytest.approx((350.0, 140.0))


def test_fit_survives_adjusting_but_not_confirming_a_landmark(tmp_path: Path) -> None:
    """A provisional point leaves the fit alone; only a confirmed landmark stales it."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    for name in GOOD_NAMES[:4]:
        place_landmark(session, name)
    session.do_fit()
    assert session.fit is not None
    session.coarse_click(400, 400)  # aiming a fifth landmark
    session.nudge(1.0, 0.0)
    assert session.fit is not None  # nothing recorded yet
    session.confirm_point()
    assert session.fit is None


def test_recommitting_a_frame_replaces_its_rows(tmp_path: Path) -> None:
    """Fix a bad frame by redoing it: rows are replaced, stale landmarks cleared."""
    session = make_session(tmp_path)
    session.begin_capture(3)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    for name in GOOD_NAMES[:4]:
        place_landmark(session, name)
    session.do_fit()
    session.commit()
    assert len(read_rows(tmp_path / "corners_landmarks.csv")) == 7  # 3 seeded corners + 4 placed

    # Second pass over the same frame, this time all visible: auto-commit.
    session.begin_capture(3)
    for slot in range(4):
        click_at(session, (100.0 + slot, 100.0))
    rows = read_rows(tmp_path / "corners.csv")
    assert len(rows) == 4  # replaced, not appended
    assert all(row["source"] == "click" for row in rows)
    assert float(rows[0]["x_px"]) == pytest.approx(100.0)
    assert read_rows(tmp_path / "corners_landmarks.csv") == []  # stale sidecar rows cleared


def test_resume_reads_annotated_frames_and_warns(tmp_path: Path) -> None:
    """A fresh session against an existing CSV knows what is already done."""
    first = make_session(tmp_path)
    first.begin_capture(7)
    for slot in range(4):
        click_at(first, (200.0 + slot, 300.0))

    second = make_session(tmp_path)
    assert second.annotated_frames == {7}
    action = second.begin_capture(7)
    assert "already annotated" in action.message
    second.abort()
    assert "already annotated" not in second.begin_capture(8).message


def test_foreign_csv_header_is_refused(tmp_path: Path) -> None:
    """The original clicker's 8-column CSV must not be silently extended."""
    legacy = tmp_path / "corners.csv"
    legacy.write_text("video,frame,corner_idx,x_px,y_px,x_norm,y_norm,orientation\nv.mp4,1,0,1,2,0.1,0.2,portrait\n")
    with pytest.raises(ValueError, match="different tool or version"):
        make_session(tmp_path)


def test_copy_forward_prefills_and_reextrapolates(tmp_path: Path) -> None:
    """v on a static camera: corners and landmarks arrive pre-aimed, off-frame
    marks re-apply, and the new frame gets its own fit from confirmed points."""
    session = make_session(tmp_path)
    session.begin_capture(100)
    for slot in range(3):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()
    for name in GOOD_NAMES[:4]:
        place_landmark(session, name)
    session.do_fit()
    session.commit()

    prefill = offframe.load_prefill(session.csv_path, session.landmark_csv_path, "vid.mp4", 500)
    assert prefill is not None
    source_frame, slots, landmarks = prefill
    assert source_frame == 100
    assert slots[3] is None  # the extrapolated corner comes back as off-frame
    assert slots[0] == pytest.approx(tuple(IMG_CORNERS_OFF[0]))
    copied_names = list(court.CORNER_LANDMARK_NAMES[:3]) + GOOD_NAMES[:4]  # seeds ride along
    assert [name for name, _x, _y in landmarks] == copied_names

    action = session.begin_capture_prefilled(500, slots, landmarks, source_frame)
    assert action.kind is offframe.ActionKind.OPEN_LOUPE
    assert "copied from frame 100" in action.message
    assert session.pending_point == pytest.approx(tuple(IMG_CORNERS_OFF[0]))
    for _ in range(3):
        session.confirm_point()  # three visible corners, Enter each
    # The off-frame mark re-applied by itself; we are now on the first landmark.
    assert session.state is offframe.CaptureState.LANDMARK_ADJUST
    assert session.offframe_slots == (3,)
    for _ in range(len(copied_names)):
        session.confirm_point()  # every copied landmark, Enter each
    assert "fit:" in session.do_fit().message
    assert session.commit().kind is offframe.ActionKind.COMMITTED

    rows = [row for row in read_rows(tmp_path / "corners.csv") if row["frame"] == "500"]
    assert len(rows) == 4
    assert rows[3]["source"] == "extrapolated"
    assert float(rows[3]["x_px"]) == pytest.approx(-40.0, abs=1e-2)


def test_clicked_corners_seed_the_fit_on_offframe_frames(tmp_path: Path) -> None:
    """Two clicked corners plus three crossings reach the floor: the corners
    auto-join the fit under their crossing names, so nothing is asked twice."""
    session = make_session(tmp_path)
    session.begin_capture(0)
    for slot in range(2):
        click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    session.mark_offframe()  # br
    action = session.mark_offframe()  # bl; enters landmark mode
    assert "2 clicked corners" in action.message
    assert [name for name, _x, _y in session.landmarks] == list(court.CORNER_LANDMARK_NAMES[:2])
    place_landmark(session, GOOD_NAMES[2])
    place_landmark(session, GOOD_NAMES[3])
    place_landmark(session, GOOD_NAMES[4])
    assert "fit:" in session.do_fit().message
    session.commit()
    rows = read_rows(tmp_path / "corners.csv")
    assert [row["source"] for row in rows] == ["click", "click", "extrapolated", "extrapolated"]
    assert float(rows[3]["x_px"]) == pytest.approx(-40.0, abs=1e-2)
    assert float(rows[2]["x_px"]) == pytest.approx(1120.0, abs=1e-2)


def test_load_prefill_picks_the_nearest_frame(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    assert offframe.load_prefill(session.csv_path, session.landmark_csv_path, "vid.mp4", 5) is None
    for frame in (10, 400):
        session.begin_capture(frame)
        for slot in range(4):
            click_at(session, tuple(IMG_CORNERS_OFF[slot]))
    near = offframe.load_prefill(session.csv_path, session.landmark_csv_path, "vid.mp4", 350)
    assert near is not None and near[0] == 400
    assert offframe.load_prefill(session.csv_path, session.landmark_csv_path, "other.mp4", 350) is None


def test_overlay_renders_in_every_state(tmp_path: Path) -> None:
    """The drawing helpers run in every capture state, headless.

    Regression: a stale state name inside the shell's status composer crashed
    only at runtime because the session tests never render.
    """
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    session = make_session(tmp_path, policy="always")

    def render() -> None:
        canvas = offframe.draw_overlay(frame, session, 5, 100, "status message", show_help=True)
        assert canvas.shape == frame.shape

    render()  # scrub
    session.begin_capture(5)
    render()  # corner prompt
    session.coarse_click(350, 140)
    render()  # corner adjust, pending cross drawn
    session.confirm_point()
    for target in [(1000.0, 140.0), (1120.0, 690.0), (60.0, 700.0)]:
        click_at(session, target)
    render()  # landmark prompt (always policy)
    session.coarse_click(400, 300)
    render()  # landmark adjust
    session.confirm_point()
    for name in GOOD_NAMES[:4]:
        place_landmark(session, name)
    session.do_fit()
    render()  # fitted quad drawn, possibly wild but clamped


def test_offframe_tool_imports_without_torch() -> None:
    """The whole annotator chain runs with torch blocked, pinning the decouple
    that lets the GUI venv stay opencv-only. A subprocess, so torch is blocked
    from interpreter start; exit 0 is the proof."""
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # any `import torch` on the chain now raises
        "sys.path.insert(0, 'src/courtkeynet/validation_scripts')\n"
        "import annotate_court_corners_offframe\n"
        "import court_landmarks\n"
        "assert len(court_landmarks.LANDMARK_NAMES) == 30\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True)
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
