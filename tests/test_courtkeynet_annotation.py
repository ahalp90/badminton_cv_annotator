"""Tests for the CourtKeyNet hand-annotation tool and its scoring adapter.

Both scripts live under the ``courtkeynet`` package, whose ``__init__`` imports
the wrapper (and so torch). These tests load the two modules straight from their
file paths instead, so the pure helpers are exercised without importing torch,
loading the model, opening a cv2 window, or decoding any video. Frames, when
needed, are tiny numpy arrays.
"""

import importlib.util
import itertools
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

    :param name: the standalone module name to register it under
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


annotate = _load_module(
    "annotate_court_corners_under_test",
    "src/courtkeynet/validation_scripts/annotate_court_corners.py",
)
score = _load_module(
    "score_hand_corners_under_test",
    "src/courtkeynet/validation_scripts/score_hand_corners.py",
)


# --- Loupe mapping round-trip ----------------------------------------------

@pytest.mark.parametrize(
    "true_x,true_y,coarse_x,coarse_y,half,zoom,width,height",
    [
        (300.4, 150.7, 305, 148, 64, 8, 512, 288),  # comfortably interior
        (12.3, 9.6, 15, 11, 32, 8, 200, 120),        # top-left corner: origin clamps to (0, 0)
        (198.6, 118.2, 190, 112, 32, 8, 200, 120),   # bottom-right corner: origin clamps inward
    ],
)
def test_loupe_round_trip(true_x, true_y, coarse_x, coarse_y, half, zoom, width, height) -> None:
    """A true source pixel -> loupe coords -> back recovers the exact source pixel."""
    origin = annotate.loupe_origin(coarse_x, coarse_y, half, width, height)
    loupe_x, loupe_y = annotate.source_to_loupe(true_x, true_y, origin, zoom)
    recovered = annotate.loupe_to_source(loupe_x, loupe_y, origin, zoom)
    assert recovered == pytest.approx((true_x, true_y), abs=1e-9)


def test_loupe_origin_clamps_at_edges() -> None:
    """Origin never leaves the frame: a corner click clamps to (0, 0) and inward."""
    assert annotate.loupe_origin(2, 3, 32, 200, 120) == (0, 0)
    # Right/bottom edge: origin caps at dim - 2*half so the crop stays on-frame.
    assert annotate.loupe_origin(199, 119, 32, 200, 120) == (200 - 64, 120 - 64)


def test_loupe_click_is_sub_pixel() -> None:
    """A loupe click between two zoomed pixels maps to a fractional source pixel."""
    origin = (100, 50)
    # Half a source pixel across at zoom 8 == 4 zoomed pixels from the origin.
    src_x, src_y = annotate.loupe_to_source(4, 12, origin, 8)
    assert src_x == pytest.approx(100.5)
    assert src_y == pytest.approx(51.5)


# --- Normalisation ---------------------------------------------------------

def test_normalise_xy() -> None:
    """Pixel xy divides by the frame extent to land in [0, 1]."""
    assert annotate.normalise_xy(256, 72, 512, 288) == pytest.approx((0.5, 0.25))


# --- CSV writer: header once, append, schema -------------------------------

def test_csv_header_written_once_and_appends(tmp_path: Path) -> None:
    """First write lays the header; a second write appends rows without repeating it."""
    csv_path = tmp_path / "hand.csv"
    corners = [(10.0, 20.0), (110.0, 22.0), (108.0, 120.0), (12.0, 118.0)]
    rows = annotate.build_corner_rows("clip.mp4", 55, corners, 512, 288, "portrait")
    annotate.append_rows(csv_path, rows)
    annotate.append_rows(csv_path, annotate.build_corner_rows("clip.mp4", 99, corners, 512, 288, "portrait"))

    table = pd.read_csv(csv_path)
    assert list(table.columns) == list(annotate.CSV_HEADER)
    assert len(table) == 8  # two frames, four corners each, header not duplicated as a row
    assert set(table["frame"]) == {55, 99}


def test_build_corner_rows_schema_and_norms() -> None:
    """Rows keep click order in corner_idx and carry correct normalised xy."""
    corners = [(256.0, 72.0), (300.0, 80.0), (290.0, 200.0), (200.0, 210.0)]
    rows = annotate.build_corner_rows("clip.mp4", 7, corners, 512, 288, "portrait")
    assert [row["corner_idx"] for row in rows] == [0, 1, 2, 3]
    assert list(rows[0].keys()) == list(annotate.CSV_HEADER)
    assert rows[0]["x_norm"] == pytest.approx(0.5)
    assert rows[0]["y_norm"] == pytest.approx(0.25)
    assert all(row["orientation"] == "portrait" for row in rows)


# --- Capture state machine -------------------------------------------------

def _make_session(tmp_path: Path) -> "annotate.CaptureSession":
    return annotate.CaptureSession(
        video="clip.mp4", width=512, height=288, orientation="portrait",
        csv_path=tmp_path / "hand.csv", half=64, zoom=8,
    )


def test_full_capture_writes_four_rows_and_returns_to_scrub(tmp_path: Path) -> None:
    """Four coarse+refine cycles append exactly four rows and land back in scrub."""
    session = _make_session(tmp_path)
    begin = session.begin_capture(frame_idx=42)
    assert begin.kind is annotate.ActionKind.AWAIT_COARSE
    assert session.state is annotate.CaptureState.AWAITING_COARSE

    coarse_points = [(100, 90), (400, 90), (410, 250), (95, 255)]
    for corner_i, (coarse_x, coarse_y) in enumerate(coarse_points):
        open_loupe = session.coarse_click(coarse_x, coarse_y)
        assert open_loupe.kind is annotate.ActionKind.OPEN_LOUPE
        assert session.state is annotate.CaptureState.AWAITING_REFINE
        # Click the loupe centre: refines to the coarse point (origin + half*zoom / zoom).
        placed = session.refine_click(64 * 8, 64 * 8)
        expected_kind = annotate.ActionKind.COMMITTED if corner_i == 3 else annotate.ActionKind.CORNER_PLACED
        assert placed.kind is expected_kind

    assert session.state is annotate.CaptureState.SCRUB
    table = pd.read_csv(tmp_path / "hand.csv")
    assert len(table) == 4
    assert list(table["corner_idx"]) == [0, 1, 2, 3]
    assert set(table["frame"]) == {42}


def test_abort_writes_nothing_and_resets(tmp_path: Path) -> None:
    """ESC mid-capture drops placed corners, writes no CSV, and returns to scrub."""
    session = _make_session(tmp_path)
    session.begin_capture(frame_idx=3)
    session.coarse_click(100, 90)
    session.refine_click(64 * 8, 64 * 8)  # one corner placed
    assert len(session.placed_corners) == 1

    aborted = session.abort()
    assert aborted.kind is annotate.ActionKind.ABORTED
    assert session.state is annotate.CaptureState.SCRUB
    assert session.placed_corners == ()
    assert not (tmp_path / "hand.csv").exists()


def test_events_out_of_state_are_noops(tmp_path: Path) -> None:
    """A refine click with no loupe open, or a second begin, is ignored."""
    session = _make_session(tmp_path)
    assert session.refine_click(10, 10).kind is annotate.ActionKind.NOOP  # nothing to refine yet
    session.begin_capture(frame_idx=1)
    assert session.begin_capture(frame_idx=2).kind is annotate.ActionKind.NOOP  # already capturing
    assert session.coarse_click(50, 50).kind is annotate.ActionKind.OPEN_LOUPE
    assert session.coarse_click(60, 60).kind is annotate.ActionKind.NOOP  # awaiting refine, not coarse


# --- Canonicalisation ------------------------------------------------------

# A court-like perspective trapezium already in TL, TR, BR, BL order: narrow top,
# wide bottom, in native-ish pixels. TL is the corner nearest the image origin.
TRAPEZIUM = np.array([[150, 80], [360, 80], [470, 250], [40, 250]], dtype=np.float64)


def test_canonicalise_reproduces_tl_tr_br_bl() -> None:
    """The rule leaves an already-ordered court trapezium unchanged."""
    assert np.allclose(score.canonicalise_quad(TRAPEZIUM), TRAPEZIUM)


def test_canonicalise_all_permutations_agree() -> None:
    """All 24 orderings of the same four points canonicalise to one TL TR BR BL ring."""
    for permuted in itertools.permutations(range(4)):
        result = score.canonicalise_quad(TRAPEZIUM[list(permuted)])
        assert np.allclose(result, TRAPEZIUM), f"permutation {permuted} broke the ring"


# --- Adapter error maths ---------------------------------------------------

def test_per_corner_error_zero_on_identical() -> None:
    """Identical quads give zero error at every corner."""
    assert np.allclose(score.per_corner_error(TRAPEZIUM, TRAPEZIUM), np.zeros(4))


def test_per_corner_error_recovers_known_offset() -> None:
    """A known per-corner shift is recovered as its Euclidean magnitude."""
    shifted = TRAPEZIUM.copy()
    shifted[0] += [3.0, 4.0]   # 3-4-5 triangle -> 5 px
    shifted[2] += [0.0, -2.0]  # 2 px straight up
    err = score.per_corner_error(TRAPEZIUM, shifted)
    assert err == pytest.approx([5.0, 0.0, 2.0, 0.0])


def test_canonicalise_then_error_survives_scrambled_order() -> None:
    """Scrambled detected vs hand quads still score zero once both are canonicalised."""
    hand = TRAPEZIUM[[2, 0, 3, 1]]       # some click order
    detected = TRAPEZIUM[[1, 3, 0, 2]]   # the detector's own order
    err = score.per_corner_error(score.canonicalise_quad(detected), score.canonicalise_quad(hand))
    assert np.allclose(err, np.zeros(4))


# --- hand_quad_px + row filtering ------------------------------------------

def test_hand_quad_px_from_normalised() -> None:
    """Normalised rows rebuild to native pixels in corner_idx order."""
    frame_rows = pd.DataFrame({
        "corner_idx": [2, 0, 3, 1],  # deliberately unsorted
        "x_norm": [0.9, 0.1, 0.05, 0.7],
        "y_norm": [0.8, 0.3, 0.8, 0.3],
    })
    quad = score.hand_quad_px(frame_rows, width=512, height=288)
    # sorted by corner_idx: 0,1,2,3 -> the (0.1,0.3),(0.7,0.3),(0.9,0.8),(0.05,0.8) rows
    assert quad[0] == pytest.approx([0.1 * 512, 0.3 * 288])
    assert quad[3] == pytest.approx([0.05 * 512, 0.8 * 288])


def test_hand_quad_px_requires_four_rows() -> None:
    """Fewer than four corner rows for a frame fails loudly."""
    frame_rows = pd.DataFrame({"corner_idx": [0, 1], "x_norm": [0.1, 0.2], "y_norm": [0.3, 0.4]})
    with pytest.raises(ValueError):
        score.hand_quad_px(frame_rows, width=512, height=288)


def test_rows_for_video_matches_basename() -> None:
    """Rows are matched by video basename, tolerating a different path prefix."""
    annotations = pd.DataFrame({
        "video": ["/some/host/clip.mp4", "/some/host/clip.mp4", "other.mp4"],
        "frame": [1, 1, 5],
    })
    matched = score.rows_for_video(annotations, Path("/a/different/prefix/clip.mp4"))
    assert len(matched) == 2
    assert set(matched["frame"]) == {1}


# --- Intersection tour: point table ----------------------------------------

def test_point_table_count_and_corner_order() -> None:
    """The plan yields 30 clickable intersections; the four corners come first."""
    table = annotate.build_point_table()
    assert len(table) == 30
    assert [point.name for point in table[:4]] == [
        "top-left corner (far baseline meets left doubles sideline)",
        "top-right corner (far baseline meets right doubles sideline)",
        "bottom-right corner (near baseline meets right doubles sideline)",
        "bottom-left corner (near baseline meets left doubles sideline)",
    ]
    # Corner coords come straight from CORNER_COURT_M, in TL TR BR BL slot order.
    for slot in range(4):
        assert table[slot].court_xy == pytest.approx(tuple(annotate.CORNER_COURT_M[slot]))


def test_point_table_spot_check_intersections() -> None:
    """Split-centre-line cases: reaches the short/near lines but skips the net gap."""
    table = annotate.build_point_table()
    names = {point.name for point in table}
    assert "centre line meets far short service line" in names
    # The centre line reaches the near baseline through its near-half segment.
    assert "centre line meets near baseline" in names
    # It is split across the net, so no centre-line point sits strictly between the
    # two short service lines (4.72 < y < 8.68).
    mid_centre = [p for p in table if abs(p.court_xy[0] - 3.05) < 0.01 and 4.72 < p.court_xy[1] < 8.68]
    assert mid_centre == []


def test_point_table_fails_loud_on_unknown_constant(monkeypatch) -> None:
    """A painted segment whose constant is not in the lookup raises at build time."""
    doctored = ((np.array([1.23, 0.0], dtype=np.float32), np.array([1.23, 13.4], dtype=np.float32)),)
    monkeypatch.setattr(annotate, "PAINTED_SEGMENTS_M", doctored)
    with pytest.raises(ValueError, match="no name"):
        annotate.build_point_table()


# --- Intersection tour: driving helpers ------------------------------------

# A homography that drops both near corners (BL, BR) below the 720 px frame
# bottom, so the tour's off-frame corner projection is actually exercised.
TOUR_W, TOUR_H = 1280, 720
IMG_CORNERS_OFF = np.array([[350, 140], [1000, 140], [1500, 900], [-150, 900]], dtype=np.float32)


def _tour_session(tmp_path: Path, table, csv_name: str = "hand.csv") -> "annotate.IntersectionSession":
    return annotate.IntersectionSession(
        video="clip.mp4", width=TOUR_W, height=TOUR_H, orientation="portrait",
        csv_path=tmp_path / csv_name, half=64, zoom=8, point_table=table,
    )


def _click_point(session, src_x: float, src_y: float):
    """Two-click flow landing exactly on (src_x, src_y): coarse, then the loupe
    coord that maps back to that source pixel (the round-trip the loupe guarantees).
    """
    open_loupe = session.coarse_click(src_x, src_y)
    lx, ly = annotate.source_to_loupe(src_x, src_y, open_loupe.loupe_origin, session.zoom)
    return session.refine_click(lx, ly)


def _run_tour(session, frame_idx: int, clicked_px: dict, table):
    """Drive a whole tour: click the mapped points, skip the rest, declare done
    right after the last clicked point. Returns the final action.
    """
    session.begin_tour(frame_idx)
    last_index = max(clicked_px)
    action = None
    for index in range(len(table)):
        action = _click_point(session, *clicked_px[index]) if index in clicked_px else session.skip()
        if index == last_index:
            action = session.declare_done()
            break
    return action


def _homography_off() -> np.ndarray:
    return cv2.getPerspectiveTransform(annotate.CORNER_COURT_M, IMG_CORNERS_OFF)


def _project_plan(homography: np.ndarray, court_xy: tuple[float, float]) -> tuple[float, float]:
    point = cv2.perspectiveTransform(np.array([[court_xy]], dtype=np.float64), homography)
    return float(point[0, 0, 0]), float(point[0, 0, 1])


def _index_of(table, court_xy: tuple[float, float]) -> int:
    for index, point in enumerate(table):
        if round(point.court_xy[0], 2) == round(court_xy[0], 2) and round(point.court_xy[1], 2) == round(court_xy[1], 2):
            return index
    raise KeyError(court_xy)


# --- Intersection tour: state machine --------------------------------------

def test_tour_state_machine_skip_and_abort(tmp_path: Path) -> None:
    """Skip advances and counts; a second begin and a stray refine are no-ops; ESC resets."""
    table = annotate.build_point_table()
    session = _tour_session(tmp_path, table)

    assert session.refine_click(10, 10).kind is annotate.ActionKind.NOOP  # nothing to refine yet
    begin = session.begin_tour(frame_idx=4)
    assert begin.kind is annotate.ActionKind.AWAIT_COARSE
    assert begin.point_name == table[0].name
    assert session.begin_tour(frame_idx=9).kind is annotate.ActionKind.NOOP  # already touring

    skipped = session.skip()
    assert skipped.kind is annotate.ActionKind.POINT_SKIPPED
    assert session.cursor == 1 and session.skipped_count == 1

    assert session.abort().kind is annotate.ActionKind.ABORTED
    assert session.state is annotate.CaptureState.SCRUB
    assert not (tmp_path / "hand.csv").exists()


def test_tour_all_corners_clicked_matches_c_mode(tmp_path: Path) -> None:
    """Clicking the four corner points writes rows byte-identical to the 'c' mode's."""
    table = annotate.build_point_table()
    corners = [(100.0, 90.0), (400.0, 90.0), (410.0, 250.0), (95.0, 255.0)]

    corner_session = annotate.CaptureSession(
        video="clip.mp4", width=TOUR_W, height=TOUR_H, orientation="portrait",
        csv_path=tmp_path / "corner.csv", half=64, zoom=8,
    )
    corner_session.begin_capture(frame_idx=7)
    for src_x, src_y in corners:
        _click_point(corner_session, src_x, src_y)

    tour_session = _tour_session(tmp_path, table, csv_name="tour.csv")
    action = _run_tour(tour_session, 7, {index: corners[index] for index in range(4)}, table)
    assert action.kind is annotate.ActionKind.COMMITTED

    assert (tmp_path / "corner.csv").read_bytes() == (tmp_path / "tour.csv").read_bytes()
    # The sidecar still records all four clicked corners, with an empty rms (no fit).
    sidecar = pd.read_csv(tmp_path / "tour_points.csv")
    assert len(sidecar) == 4
    assert sidecar["rms_px"].isna().all()


def test_tour_done_early_with_too_few_points_writes_nothing(tmp_path: Path) -> None:
    """'Done' with three clicks (no corners, under the 5-point floor) aborts cleanly."""
    table = annotate.build_point_table()
    homography = _homography_off()
    visible = [(0, 0.76), (0, 4.72), (0.46, 4.72)]
    clicked_px = {_index_of(table, xy): _project_plan(homography, xy) for xy in visible}
    session = _tour_session(tmp_path, table)
    action = _run_tour(session, 6, clicked_px, table)
    assert action.kind is annotate.ActionKind.ABORTED
    assert not (tmp_path / "hand.csv").exists()
    assert not (tmp_path / "hand_points.csv").exists()


# --- Intersection tour: homography fit -------------------------------------

def test_tour_fit_recovers_offframe_corners(tmp_path: Path) -> None:
    """Six visible far-half clicks fit a homography that recovers all four corners,
    the near two landing below the frame bottom.
    """
    table = annotate.build_point_table()
    homography = _homography_off()
    # Spread over three rows and both sidelines, so a 4-subset is in general position.
    visible = [(0, 0), (6.1, 0), (0, 0.76), (6.1, 0.76), (0, 4.72), (6.1, 4.72)]
    clicked_px = {_index_of(table, xy): _project_plan(homography, xy) for xy in visible}
    session = _tour_session(tmp_path, table)
    action = _run_tour(session, 12, clicked_px, table)
    assert action.kind is annotate.ActionKind.COMMITTED

    written = pd.read_csv(tmp_path / "hand.csv").sort_values("corner_idx")
    got = written[["x_px", "y_px"]].to_numpy()
    truth = cv2.perspectiveTransform(
        annotate.CORNER_COURT_M.reshape(-1, 1, 2).astype(np.float64), homography
    ).reshape(-1, 2)
    assert np.max(np.linalg.norm(got - truth, axis=1)) < 0.5
    # corner_idx 2 (BR) and 3 (BL) are the near corners and land off the frame bottom.
    assert got[2, 1] > TOUR_H and got[3, 1] > TOUR_H


def test_tour_fit_rejects_a_misclick(tmp_path: Path) -> None:
    """A single 30 px misclick on a minimal 5-point fit trips the worst-point gate."""
    table = annotate.build_point_table()
    homography = _homography_off()
    # Minimal general-position set, the 5-point boundary the fit accepts: the
    # worst-point gate must fire here just as it does with more clicks.
    visible = [(0, 0), (6.1, 0), (0, 4.72), (6.1, 4.72), (3.05, 0.76)]
    clicked_px = {_index_of(table, xy): _project_plan(homography, xy) for xy in visible}
    centre_index = _index_of(table, (3.05, 0.76))
    off_x, off_y = clicked_px[centre_index]
    clicked_px[centre_index] = (off_x + 30.0, off_y + 30.0)
    session = _tour_session(tmp_path, table)
    action = _run_tour(session, 12, clicked_px, table)
    assert action.kind is annotate.ActionKind.ABORTED
    assert not (tmp_path / "hand.csv").exists()
    assert not (tmp_path / "hand_points.csv").exists()


def test_tour_fit_rejects_a_misclick_among_six_spread_points(tmp_path: Path) -> None:
    """The worst-point gate catches a 30 px misclick among six well-spread points,
    where least squares smears the mean RMS to ~1.8 px and a mean gate would pass.
    """
    table = annotate.build_point_table()
    homography = _homography_off()
    visible = [(0, 0), (6.1, 0), (0, 0.76), (6.1, 0.76), (0, 4.72), (6.1, 4.72)]
    clicked_px = {_index_of(table, xy): _project_plan(homography, xy) for xy in visible}
    # Perturb the (0, 4.72) click: worst residual 3.22 px, mean RMS 1.77 px, so the
    # max gate aborts where a mean gate would not. The (6.1, 4.72) click is NOT a
    # usable case: with six points the DLT nearly interpolates that high-leverage
    # click (worst residual 2.31 px), so no residual gate at any statistic can see
    # a misclick the fit absorbs. A leverage limit, not a gate-metric choice.
    bad_index = _index_of(table, (0, 4.72))
    off_x, off_y = clicked_px[bad_index]
    clicked_px[bad_index] = (off_x + 30.0, off_y + 30.0)
    session = _tour_session(tmp_path, table)
    action = _run_tour(session, 12, clicked_px, table)
    assert action.kind is annotate.ActionKind.ABORTED
    assert not (tmp_path / "hand.csv").exists()
    assert not (tmp_path / "hand_points.csv").exists()


def test_collinear_catches_diagonal_grid_triples() -> None:
    """The court grid's even spacings put diagonal triples on one line; float fuzz
    in the cross product must not hide them from the general-position guard.
    """
    diagonal = np.array([(0.46, 0.76), (3.05, 4.72), (5.64, 8.68)], dtype=np.float64)
    assert annotate._collinear(diagonal[0], diagonal[1], diagonal[2])


def test_tour_fit_rejects_degenerate_points(tmp_path: Path) -> None:
    """Four clicks on one sideline plus one off it have no general-position quad."""
    table = annotate.build_point_table()
    # Four points on the left doubles sideline (x=0) plus one off it: every 4-subset
    # keeps three collinear, so no homography can be fitted.
    degenerate = [(0, 0), (0, 0.76), (0, 4.72), (0, 8.68), (0.46, 0.76)]
    clicked_px = {_index_of(table, xy): (100.0 + 10 * i, 120.0 + 5 * i) for i, xy in enumerate(degenerate)}
    session = _tour_session(tmp_path, table)
    action = _run_tour(session, 3, clicked_px, table)
    assert action.kind is annotate.ActionKind.ABORTED
    assert not (tmp_path / "hand.csv").exists()
    assert not (tmp_path / "hand_points.csv").exists()


def test_tour_sidecar_records_rms_only_when_fitted(tmp_path: Path) -> None:
    """A fitted commit fills rms_px on every sidecar row (the direct path leaves it empty)."""
    table = annotate.build_point_table()
    homography = _homography_off()
    visible = [(0, 0), (6.1, 0), (0, 0.76), (6.1, 0.76), (0, 4.72), (6.1, 4.72)]
    clicked_px = {_index_of(table, xy): _project_plan(homography, xy) for xy in visible}
    session = _tour_session(tmp_path, table, csv_name="fit.csv")
    _run_tour(session, 5, clicked_px, table)

    sidecar = pd.read_csv(tmp_path / "fit_points.csv")
    assert len(sidecar) == len(visible)
    assert sidecar["rms_px"].notna().all()
    assert (sidecar["rms_px"] >= 0).all()


# --- CSV header cross-guard --------------------------------------------------

def test_append_rows_refuses_mismatched_header(tmp_path: Path) -> None:
    """Appending main corner rows onto a sidecar-headed file raises instead of
    silently mixing the two schemas in one file.
    """
    csv_path = tmp_path / "hand_points.csv"
    csv_path.write_text(",".join(annotate.POINT_CSV_HEADER) + "\n")
    rows = annotate.build_corner_rows("clip.mp4", 1, [(1.0, 2.0)] * 4, 512, 288, "portrait")
    with pytest.raises(ValueError, match="header"):
        annotate.append_rows(csv_path, rows)


# --- Torch decouple: the annotator runs without torch installed ------------

def test_annotator_imports_without_torch() -> None:
    """fallback.py and the annotator's build_point_table import and run with torch
    blocked, pinning the decouple so a future import can't re-couple torch onto the
    annotator's path (the GUI-capable OpenCV venv has no torch).

    A subprocess, so torch is blocked from interpreter start: this pytest process
    has already imported torch (via the wrapper tests), so an in-process block
    would not model torch being absent. The proof is the subprocess exiting 0: a
    blocked ``import torch`` anywhere on the chain raises and crashes it.
    """
    script = (
        "import sys\n"
        "sys.modules['torch'] = None\n"  # None in sys.modules makes any `import torch` raise ImportError
        "import src.courtkeynet.fallback\n"
        "from src.courtkeynet.validation_scripts.annotate_court_corners import build_point_table\n"
        "table = build_point_table()\n"
        "assert len(table) == 30, f'expected 30 tour points, got {len(table)}'\n"
        "print('OK', len(table))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, f"torch-free import failed\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert result.stdout.startswith("OK"), result.stdout
