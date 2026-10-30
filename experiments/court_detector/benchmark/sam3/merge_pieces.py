"""Merge SAM 3's court-line pieces into one court region per frame and render them.

SAM 3 returns a court's painted lines as several pieces, and in multi-court halls it
also returns the neighbouring courts' lines. Each method is one fixed rule that
turns the pieces into a single region, so SAM 3 can be compared like-for-like with
the court detector's outline.
"""

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

OPACITY = 0.3
RED_BGR = np.array([0, 0, 255], dtype=np.float32)
# Dilation radius used to join line pieces into networks, as a fraction of frame
# width (about 6 px at 1920 wide). Dilation grows both sides, so it bridges breaks
# along thin lines of up to twice this.
BRIDGE_FRACTION = 0.003
GRID_CELL_WIDTH = 480
# Hull simplification tolerances, tried in order, as fractions of the hull perimeter.
QUAD_TOLERANCES = np.linspace(0.01, 0.05, 9)
# A hull this close to the frame edge is treated as a partly visible court.
BORDER_MARGIN_PX = 2


def bbox_region(lines: np.ndarray) -> np.ndarray:
    """:return: the filled axis-aligned bounding box of the line pixels, (height, width) bool."""
    ys, xs = np.nonzero(lines)
    region = np.zeros_like(lines)
    region[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True
    return region


def hull_region(lines: np.ndarray) -> np.ndarray:
    """:return: the filled convex hull of the line pixels, (height, width) bool."""
    ys, xs = np.nonzero(lines)
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    region = np.zeros(lines.shape, dtype=np.uint8)
    cv2.fillPoly(region, [hull], 1)
    return region.astype(bool)


def line_networks(lines: np.ndarray) -> np.ndarray:
    """Label connected line networks after bridging small breaks.

    One court's lines form a single network; neighbouring courts sit across a gap of
    floor, so each network approximates one court.

    :return: (height, width) int labels, 0 for background.
    """
    bridge_px = max(1, round(lines.shape[1] * BRIDGE_FRACTION))
    networks, _ = ndimage.label(ndimage.binary_dilation(lines, iterations=bridge_px))
    return networks


def most_central_network(lines: np.ndarray) -> np.ndarray:
    """:return: the line pixels of the network holding the line pixel nearest the frame centre."""
    height, width = lines.shape
    networks = line_networks(lines)
    ys, xs = np.nonzero(lines)
    nearest = np.argmin((xs - width / 2) ** 2 + (ys - height / 2) ** 2)
    return lines & (networks == networks[ys[nearest], xs[nearest]])


def largest_network(lines: np.ndarray) -> np.ndarray:
    """:return: the line pixels of the network with the most line pixels."""
    networks = line_networks(lines)
    line_pixels_per_network = np.bincount(networks[lines])  # index 0 (background) is always 0 here
    return lines & (networks == line_pixels_per_network.argmax())


def quadrilateral_region(hull: np.ndarray) -> np.ndarray:
    """Simplify a fully visible court hull to a quadrilateral where the hull supports one.

    A hull touching the frame edge is a partly visible court. It is kept as it is,
    because the visible lines can't pin down a missing side. A hull that no tolerance
    reduces to four convex corners is also kept.

    :return: (height, width) bool region.
    """
    contours, _ = cv2.findContours(hull.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    height, width = hull.shape
    xs, ys = contour[:, 0].T
    is_near_left_or_right = (xs <= BORDER_MARGIN_PX) | (xs >= width - 1 - BORDER_MARGIN_PX)
    is_near_top_or_bottom = (ys <= BORDER_MARGIN_PX) | (ys >= height - 1 - BORDER_MARGIN_PX)
    if (is_near_left_or_right | is_near_top_or_bottom).any():
        return hull
    perimeter = cv2.arcLength(contour, True)
    for tolerance in QUAD_TOLERANCES:
        corners = cv2.approxPolyDP(contour, tolerance * perimeter, True)
        if len(corners) == 4 and cv2.isContourConvex(corners):
            region = np.zeros(hull.shape, dtype=np.uint8)
            cv2.fillConvexPoly(region, corners, 1)
            return region.astype(bool)
    return hull


def top_piece_hull(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return hull_region(masks[scores.argmax()])


def union_bbox(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return bbox_region(masks.any(axis=0))


def union_hull(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return hull_region(masks.any(axis=0))


def central_bbox(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return bbox_region(most_central_network(masks.any(axis=0)))


def central_hull(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return hull_region(most_central_network(masks.any(axis=0)))


def largest_network_hull(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return hull_region(largest_network(masks.any(axis=0)))


def largest_network_quad(masks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return quadrilateral_region(largest_network_hull(masks, scores))


# Each method takes masks (pieces, height, width) bool and scores (pieces,) and
# returns one (height, width) bool region.
METHODS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "top_piece_hull": top_piece_hull,
    "union_bbox": union_bbox,
    "union_hull": union_hull,
    "central_bbox": central_bbox,
    "central_hull": central_hull,
    "largest_network_hull": largest_network_hull,
    "largest_network_quad": largest_network_quad,
}


def render(frame: np.ndarray, region: np.ndarray, label: str) -> np.ndarray:
    """:return: the frame with the region tinted and outlined in red and a label in the corner."""
    rendered = frame.astype(np.float32)
    rendered[region] = (1 - OPACITY) * rendered[region] + OPACITY * RED_BGR
    rendered = rendered.astype(np.uint8)
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rendered, contours, -1, (0, 0, 255), thickness=max(2, frame.shape[1] // 640))
    return labelled(rendered, label)


def labelled(image: np.ndarray, label: str) -> np.ndarray:
    """:return: a copy of the image with white text on a black strip in the top-left corner."""
    image = image.copy()
    scale = image.shape[1] / 1600
    thickness = max(1, round(scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (0, 0), (text_width + 16, text_height + baseline + 16), (0, 0, 0), thickness=-1)
    cv2.putText(image, label, (8, text_height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections_dir", type=Path, help="folder of .npz files from segment_courts.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=0.5, help="keep pieces scoring above this (SAM 3 default 0.5)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    grid_rows_by_frame: dict[str, list[np.ndarray]] = {}
    detections_paths = sorted(args.detections_dir.glob("*.npz"))
    if not detections_paths:
        raise SystemExit(f"no .npz files in {args.detections_dir}")
    for detections_path in detections_paths:
        detections = np.load(detections_path)
        # Frame paths are stored as given to segment_courts.py; run both from the repo root.
        frame = cv2.imread(str(detections["frame_path"]))
        # SAM 3 can score a piece whose thresholded mask is empty; such pieces carry no lines.
        has_pixels = detections["masks"].any(axis=(1, 2))
        is_kept = (detections["scores"] > args.cutoff) & has_pixels
        masks = detections["masks"][is_kept]  # (pieces, height, width) bool
        scores = detections["scores"][is_kept]  # (pieces,)
        frame_name, prompt_slug = detections_path.stem.rsplit("__", 1)
        has_pieces = len(scores) > 0
        top_score = float(scores.max()) if has_pieces else 0.0

        pieces = masks.any(axis=0) if has_pieces else np.zeros(frame.shape[:2], dtype=bool)
        cells = [render(frame, pieces, f"{detections['prompt']} | pieces: {len(scores)}, top {top_score:.2f}")]
        for method_name, method in METHODS.items():
            region = method(masks, scores) if has_pieces else np.zeros(frame.shape[:2], dtype=bool)
            rendered = render(frame, region, f"{detections['prompt']} | {method_name}")
            cv2.imwrite(str(args.out_dir / f"{frame_name}__{prompt_slug}__{method_name}.jpg"), rendered)
            cells.append(rendered)
            summary_rows.append(
                {
                    "frame": frame_name,
                    "prompt": str(detections["prompt"]),
                    "method": method_name,
                    "cutoff": args.cutoff,
                    "pieces": len(scores),
                    "top_score": round(top_score, 3),
                    "region_area_percent": round(float(region.mean() * 100), 1),
                }
            )

        cell_height = round(GRID_CELL_WIDTH * frame.shape[0] / frame.shape[1])
        grid_row = np.hstack([cv2.resize(cell, (GRID_CELL_WIDTH, cell_height)) for cell in cells])
        grid_rows_by_frame.setdefault(frame_name, []).append(grid_row)

    # One grid per frame: a row per prompt, a column for the raw pieces and then each method.
    for frame_name, grid_rows in grid_rows_by_frame.items():
        cv2.imwrite(str(args.out_dir / f"{frame_name}__00_grid.jpg"), np.vstack(grid_rows))

    with open(args.out_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"{len(summary_rows)} frame-prompt-method rows, {len(grid_rows_by_frame)} grids in {args.out_dir}")


if __name__ == "__main__":
    main()
