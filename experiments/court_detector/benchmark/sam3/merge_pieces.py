"""Fit one enclosing quadrilateral to SAM 3 court-marking masks and render it."""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np


def largest_group(masks: np.ndarray) -> np.ndarray:
    """Keep the group with most pixels, joining whole masks that overlap or touch."""
    union = masks.any(axis=0)
    _, labels = cv2.connectedComponents(union.astype(np.uint8), connectivity=8)
    remaining = [set(np.unique(labels[mask])) for mask in masks if mask.any()]
    groups = []
    # A single SAM mask can contain disconnected strokes; keep them together.
    while remaining:
        group = remaining.pop()
        while True:
            neighbours = [other for other in remaining if other & group]
            if not neighbours:
                break
            group.update(*neighbours)
            remaining = [other for other in remaining if not other & group]
        groups.append(union & np.isin(labels, list(group)))
    return max(groups, key=np.count_nonzero) if groups else union


def fit_quadrilateral(mask: np.ndarray) -> np.ndarray | None:
    """Return four cyclic (x, y) corners, or None if the hull has fewer than four."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None:
        return None
    try:
        return cv2.approxPolyN(points.astype(np.float32), 4).reshape(4, 2)
    except cv2.error:
        # approxPolyN cannot fit four sides to a point, line or triangular hull.
        if len(cv2.convexHull(points)) < 4:
            return None
        raise


def render_comparison(frame: np.ndarray, mask: np.ndarray, corners: np.ndarray | None) -> np.ndarray:
    """Show retained strokes and the fitted polygon at the same scale and crop."""
    height, width = frame.shape[:2]
    thickness = max(2, round(width / 400))
    cyan, orange = (255, 235, 0), (0, 185, 255)
    markings = frame.copy()
    # Thicken only the display so thin SAM strokes remain visible in the preview.
    visible = cv2.dilate(mask.astype(np.uint8), np.ones((thickness, thickness), np.uint8)) > 0
    markings[visible] = cyan
    fitted = frame.copy()
    if corners is not None:
        polygon = np.rint(corners).astype(np.int32)
        tint = frame.copy()
        cv2.fillPoly(tint, [polygon], (200, 85, 20))
        fitted = cv2.addWeighted(frame, 0.8, tint, 0.2, 0)
        cv2.polylines(fitted, [polygon], True, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.polylines(fitted, [polygon], True, orange, thickness, cv2.LINE_AA)
        for index, (x, y) in enumerate(polygon, start=1):
            if 0 <= x < width and 0 <= y < height:
                radius = max(12, round(width / 60))
                cv2.circle(fitted, (int(x), int(y)), radius, orange, -1)
                cv2.putText(fitted, str(index), (int(x - radius / 2), int(y + radius / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, width / 1400, (0, 0, 0), 2, cv2.LINE_AA)
    panels = []
    title = "Enclosing quadrilateral" if corners is not None else "No quadrilateral"
    for panel, label in [(markings, "Retained SAM 3 markings"), (fitted, title)]:
        panel = cv2.copyMakeBorder(panel, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 28, 28))
        cv2.putText(panel, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        panels.append(panel)
    return np.hstack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections_dir", type=Path, help="folder of .npz files from segment_courts.py")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    detections_paths = sorted(args.detections_dir.glob("*__court_markings.npz"))
    if not detections_paths:
        raise SystemExit(f"no court_markings detections in {args.detections_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for path in detections_paths:
        with np.load(path) as detections:
            if str(detections["prompt"]) != "court markings":
                raise ValueError(f"unexpected prompt in {path.name}")
            # Also accepts earlier caches containing lower-confidence predictions.
            masks = detections["masks"][detections["scores"] > 0.5]
            frame = cv2.imread(str(detections["frame_path"]))
        if frame is None:
            raise FileNotFoundError(f"cannot read the frame recorded in {path.name}")
        mask = largest_group(masks)
        corners = fit_quadrilateral(mask)
        comparison = render_comparison(frame, mask, corners)
        output_path = args.out_dir / f"{path.stem}.jpg"
        if not cv2.imwrite(str(output_path), comparison):
            raise OSError(f"could not write {output_path}")
        results[path.stem] = {
            "image_size": [frame.shape[1], frame.shape[0]],
            "retained_pixels": int(mask.sum()),
            "corners": None if corners is None else corners.tolist(),
        }
    with gzip.open(args.out_dir / "quadrilaterals.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    print(f"{len(results)} frames written to {args.out_dir}")


if __name__ == "__main__":
    main()
