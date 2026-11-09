"""Coordinate mapping and image rendering used by the native annotator."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from court_annotation import image_helpers as annotate


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


def test_normalise_xy() -> None:
    """Pixel xy divides by the frame extent to land in [0, 1]."""
    assert annotate.normalise_xy(256, 72, 512, 288) == pytest.approx((0.5, 0.25))


def test_render_loupe_magnifies_crop_and_marks_source_point() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[12, 15] = (30, 60, 90)
    original = frame.copy()
    crop = annotate.render_loupe(frame, (10, 10), None, half=10, zoom=4)
    assert crop.shape == (80, 80, 3)
    np.testing.assert_array_equal(crop[8:12, 20:24], np.full((4, 4, 3), (30, 60, 90)))

    marked = annotate.render_loupe(frame, (10, 10), (22.5, 20), half=10, zoom=4)
    changed_y, changed_x = np.where(np.any(marked != crop, axis=2))
    assert changed_x.min() >= 36 and changed_x.max() <= 64
    assert changed_y.min() >= 26 and changed_y.max() <= 54
    assert marked[40, 50].any()
    np.testing.assert_array_equal(frame, original)


def test_loupe_outline_is_drawn_after_magnification_in_same_colour() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    outline = np.asarray([(15, 15), (25, 15), (25, 25), (15, 25)], dtype=float)
    image = annotate.render_loupe(frame, (10, 10), None, half=10, zoom=8, outline=outline)
    assert tuple(image[40, 60]) == annotate.CORNER_COLOUR
    # A two-pixel coloured line remains narrow; it does not become 16 pixels thick.
    assert not image[55, 60].any()
    assert not frame.any()


def test_projected_inner_lines_and_open_centre_in_main_view_and_loupe() -> None:
    from court_annotation import court_landmarks as court

    # Known perspective: x = (30*x_m + 100)/(1 + .03*y_m), likewise for y.
    transform = np.array([[30, 0, 100], [0, 30, 50], [0, .03, 1]], dtype=float)
    corners = court.project_corners(transform)
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    annotate.draw_outline(image, corners)
    for point in [(0.46, 6.7), (5.64, 6.7), (2, .76), (2, 4.72),
                  (2, 8.68), (2, 12.64), (3.05, 2), (3.05, 11)]:
        x, y = court.project_points(transform, np.array([point]))[0]
        patch = image[round(y) - 1:round(y) + 2, round(x) - 1:round(x) + 2]
        assert np.any(np.all(patch == annotate.CORNER_COLOUR, axis=2))
    x, y = court.project_points(transform, np.array([[3.05, 6.7]]))[0]
    assert not image[round(y), round(x)].any()
    x, y = court.project_points(transform, np.array([[3.05, 2]]))[0]
    origin = (round(x) - 20, round(y) - 20)
    zoomed = annotate.render_loupe(np.zeros_like(image), origin, None, 20, 8, corners)
    zx, zy = annotate.source_to_loupe(x, y, origin, 8)
    patch = zoomed[round(zy) - 1:round(zy) + 2, round(zx) - 1:round(zx) + 2]
    assert np.any(np.all(patch == annotate.CORNER_COLOUR, axis=2))
