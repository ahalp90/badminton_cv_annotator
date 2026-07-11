"""Tests for the CourtKeyNet hand-annotation tool and its scoring adapter.

Both scripts live under the ``courtkeynet`` package, whose ``__init__`` imports
the wrapper (and so torch). These tests load the two modules straight from their
file paths instead, so the pure helpers are exercised without importing torch,
loading the model, opening a cv2 window, or decoding any video. Frames, when
needed, are tiny numpy arrays.
"""

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

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
