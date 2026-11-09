"""Pixel coordinates and magnified crops for the native court annotator."""

import cv2
import numpy as np
from court_annotation import court_landmarks as court

# Fluorescent pink, in OpenCV BGR order; black outlines preserve contrast.
CORNER_COLOUR = (180, 30, 255)
TEXT_COLOUR = (255, 255, 255)


def normalise_xy(x_px: float, y_px: float, width: int, height: int) -> tuple[float, float]:
    """:return: (x_norm, y_norm) pixel coords divided by the frame extent.

    :param x_px: x in source pixels
    :param y_px: y in source pixels
    :param width: frame width in pixels
    :param height: frame height in pixels
    """
    return x_px / width, y_px / height


def loupe_origin(coarse_x: float, coarse_y: float, half: int, width: int, height: int) -> tuple[int, int]:
    """Top-left source pixel of the loupe crop, clamped so the window stays on-frame.

    The crop is a ``2*half`` square centred on the coarse click. Near a frame edge
    the origin is clamped inward, so the click no longer sits dead-centre in the
    loupe; that is fine because the refine click, not the coarse one, sets the
    corner. Origin is integer because the crop is taken by array slicing.

    :param coarse_x: coarse click x in source pixels
    :param coarse_y: coarse click y in source pixels
    :param half: half-window of the crop in source pixels (crop side is ``2*half``)
    :param width: frame width in pixels
    :param height: frame height in pixels
    :return: (origin_x, origin_y) integer top-left of the crop
    """
    win = 2 * half
    origin_x = round(coarse_x) - half
    origin_y = round(coarse_y) - half
    # Clamp into [0, dim - win]; max(dim - win, 0) guards a frame smaller than the crop.
    origin_x = min(max(origin_x, 0), max(width - win, 0))
    origin_y = min(max(origin_y, 0), max(height - win, 0))
    return origin_x, origin_y


def loupe_to_source(lx: float, ly: float, origin_xy: tuple[int, int], zoom: int) -> tuple[float, float]:
    """Map a click in the zoomed loupe image back to source pixels.

    The loupe shows the ``[origin, origin + 2*half]`` source window scaled up by
    ``zoom``, so a loupe click at ``(lx, ly)`` sits at ``origin + click/zoom`` in
    source pixels. Sub-source-pixel because ``lx/zoom`` is fractional.

    :param lx: click x in the zoomed loupe image
    :param ly: click y in the zoomed loupe image
    :param origin_xy: the loupe crop's source-pixel top-left
    :param zoom: integer magnification the loupe was drawn at
    :return: (x, y) in source pixels
    """
    origin_x, origin_y = origin_xy
    return origin_x + lx / zoom, origin_y + ly / zoom


def source_to_loupe(src_x: float, src_y: float, origin_xy: tuple[int, int], zoom: int) -> tuple[float, float]:
    """Inverse of :func:`loupe_to_source`: source pixel to zoomed loupe coords.

    Used to draw the coarse point inside the loupe. Exact inverse, so a source
    pixel round-trips through both functions unchanged.

    :param src_x: x in source pixels
    :param src_y: y in source pixels
    :param origin_xy: the loupe crop's source-pixel top-left
    :param zoom: integer magnification the loupe was drawn at
    :return: (x, y) in the zoomed loupe image
    """
    origin_x, origin_y = origin_xy
    return (src_x - origin_x) * zoom, (src_y - origin_y) * zoom


def render_loupe(
    frame: np.ndarray,
    origin: tuple[int, int],
    point: tuple[float, float] | None,
    half: int,
    zoom: int,
    outline: np.ndarray | None = None,
) -> np.ndarray:
    """Magnify a BGR crop, optionally marking a source point with a crosshair."""
    origin_x, origin_y = origin
    win = 2 * half
    crop = frame[origin_y : origin_y + win, origin_x : origin_x + win]
    zoomed = cv2.resize(crop, (win * zoom, win * zoom), interpolation=cv2.INTER_NEAREST)
    if outline is not None:
        draw_outline(zoomed, (outline - np.asarray(origin)) * zoom)
    if point is not None:
        loupe_x, loupe_y = source_to_loupe(point[0], point[1], origin, zoom)
        cx, cy = round(loupe_x), round(loupe_y)
        cv2.drawMarker(zoomed, (cx, cy), (0, 0, 0), cv2.MARKER_CROSS, 25, 3)
        cv2.drawMarker(zoomed, (cx, cy), CORNER_COLOUR, cv2.MARKER_CROSS, 25, 1)
    return zoomed


def draw_outline(image: np.ndarray, points: np.ndarray) -> None:
    """Project all painted court lines from TL/TR/BR/BL at either display scale."""
    transform = cv2.getPerspectiveTransform(court.CORNER_COURT_M, np.asarray(points, dtype=np.float32))
    segments = []
    for name, x in court.X_LINES.items():
        if name == 'centre':
            segments.extend([[(x, 0), (x, court.Y_LINES['far_short_service'])],
                             [(x, court.Y_LINES['near_short_service']), (x, court.COURT_LENGTH_M)]])
        else:
            segments.append([(x, 0), (x, court.COURT_LENGTH_M)])
    for y in court.Y_LINES.values():
        segments.append([(0, y), (court.COURT_WIDTH_M, y)])
    projected = court.project_points(transform, np.asarray(segments)).reshape(-1, 2, 2)
    lines = np.int32(np.rint(np.clip(projected, -100000, 100000)))
    cv2.polylines(image, list(lines), False, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.polylines(image, list(lines), False, CORNER_COLOUR, 2, cv2.LINE_AA)
