"""Compare short OpenCV conversions of the same retained SAM3 markings."""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.court_detector.benchmark.sam3.merge_pieces import largest_group

WORK_WIDTH = 640
CLOSING_KERNEL = np.ones((5, 5), np.uint8)
CONTOUR_EPSILON_FRACTION = 0.02
METHODS = {
    'mask_polyn': 'Mask / approxPolyN',
    'mask_enclosing': 'Mask / minimum enclosing quad',
    'mask_polydp': 'Mask hull / approxPolyDP to <=4',
    'hough_polyn': 'Hough / approxPolyN',
    'hough_enclosing': 'Hough / minimum enclosing quad',
    'hough_polydp': 'Hough hull / approxPolyDP to <=4',
    'contour_polydp': 'Largest contour / approxPolyDP 2%',
    'closed_contour_polydp': 'Close 5x5 / contour / approxPolyDP 2%',
    'contour_polydp_four': 'Largest contour / approxPolyDP to <=4',
    'closed_contour_polydp_four': 'Close 5x5 / contour / approxPolyDP to <=4',
}


def approximate_four(contour: np.ndarray) -> np.ndarray:
    """Use the first tolerance yielding at most four vertices; expose triangles."""
    perimeter = cv2.arcLength(contour, True)
    for fraction in np.arange(0.001, 0.301, 0.001):
        polygon = cv2.approxPolyDP(contour, fraction * perimeter, True)
        if len(polygon) <= 4:
            return polygon.reshape(-1, 2)
    raise RuntimeError('The tolerance sweep did not reach four vertices')


def fit_options(mask: np.ndarray) -> dict[str, np.ndarray | None]:
    results: dict[str, np.ndarray | None] = dict.fromkeys(METHODS)
    if not mask.any():
        return results
    points = cv2.findNonZero(mask)
    segments = cv2.HoughLinesP(mask, 1, np.pi / 180, 64, minLineLength=64, maxLineGap=19)
    inputs = {'mask': points, 'hough': None if segments is None else segments.reshape(-1, 1, 2)}
    for source, source_points in inputs.items():
        if source_points is None:
            continue
        results[f'{source}_polyn'] = cv2.approxPolyN(source_points.astype(np.float32), 4).reshape(-1, 2)
        hull = cv2.convexHull(source_points).astype(np.float32)
        _, enclosing = cv2.minEnclosingConvexPolygon(hull, 4)
        results[f'{source}_enclosing'] = enclosing.reshape(-1, 2)
        results[f'{source}_polydp'] = approximate_four(hull)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSING_KERNEL)
    for method, contour_mask in [('contour_polydp', mask), ('closed_contour_polydp', closed)]:
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        epsilon = CONTOUR_EPSILON_FRACTION * cv2.arcLength(contour, True)
        results[method] = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        results[method + '_four'] = approximate_four(contour)
    return results


def render_panel(frame: np.ndarray, title: str, polygon: np.ndarray | None, status: str) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(888 / width, 510 / height)
    view = cv2.resize(frame, (round(width * scale), round(height * scale)))
    offscreen = []
    if polygon is not None:
        drawn = np.rint(polygon * scale).astype(np.int32)
        fill = view.copy()
        cv2.fillPoly(fill, [drawn], (200, 85, 20))
        view = cv2.addWeighted(view, 0.8, fill, 0.2, 0)
        for colour, thickness in [((0, 0, 0), 10), ((255, 255, 255), 7), ((0, 185, 255), 4)]:
            cv2.polylines(view, [drawn], True, colour, thickness, cv2.LINE_AA)
        if len(polygon) == 4:
            for index, point in enumerate(drawn, start=1):
                if not (0 <= point[0] < view.shape[1] and 0 <= point[1] < view.shape[0]):
                    offscreen.append(str(index))
                    continue
                cv2.circle(view, tuple(point), 15, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(view, tuple(point), 13, (0, 185, 255), -1, cv2.LINE_AA)
                cv2.putText(view, str(index), tuple(point + [-7, 7]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 0, 0), 2, cv2.LINE_AA)
    panel = np.full((660, 960, 3), 28, np.uint8)
    panel[95:95 + view.shape[0], 36:36 + view.shape[1]] = view
    for label, position, size in [(title, (24, 35), 0.7), (status, (24, 65), 0.55)]:
        cv2.putText(panel, label, position, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 2, cv2.LINE_AA)
    if offscreen:
        cv2.putText(panel, 'Corners outside image: ' + ', '.join(offscreen), (24, 638),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detections_dir", type=Path, help="saved court_markings SAM 3 masks")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.detections_dir.glob("*__court_markings.npz"))
    if not paths:
        raise SystemExit(f"no court_markings detections in {args.detections_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for path in paths:
        name = path.stem.split("__", 1)[0]
        if name in records:
            raise ValueError(f"each input needs a distinct name before '__': {name}")
        with np.load(path) as data:
            if str(data["prompt"]) != "court markings":
                raise ValueError(f"unexpected prompt in {path.name}")
            frame = cv2.imread(str(data["frame_path"]))
            masks = data["masks"][data["scores"] > 0.5]
            strokes = largest_group(masks)
        if frame is None:
            raise FileNotFoundError(f"cannot read the frame recorded in {path.name}")
        height, width = strokes.shape
        size = (WORK_WIDTH, round(height * WORK_WIDTH / width))
        mask = (cv2.resize(strokes.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        scale = np.array([width / size[0], height / size[1]])
        markings = frame.copy()
        markings[cv2.dilate(strokes.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0] = 0
        markings[cv2.dilate(strokes.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = (255, 235, 0)
        raw = render_panel(markings, name + " | retained SAM3 markings", None,
                           "Cyan thickened for display only. Same scale and crop in every panel.")
        panels = [raw]
        records[name] = {}
        for method, polygon in fit_options(mask).items():
            status = "No detection above 0.5" if not mask.any() else "No Hough segments"
            if polygon is not None:
                polygon = polygon * scale
                if not np.isfinite(polygon).all():
                    raise ValueError(f"non-finite polygon for {name}: {method}")
                convex = cv2.isContourConvex(polygon.astype(np.float32))
                valid_quad = len(polygon) == 4 and convex
                status = ("4-sided convex output" if valid_quad
                          else f"{len(polygon)} vertices; NOT a convex quadrilateral")
            panels.append(render_panel(frame, name + " | " + METHODS[method], polygon, status))
            records[name][method] = {"status": status, "polygon": None if polygon is None else polygon.tolist()}
        panels.append(np.full_like(raw, 28))
        sheet = np.vstack([np.hstack(panels[start:start + 3]) for start in range(0, 12, 3)])
        output = args.out_dir / f"{name}__all_methods.jpg"
        if not cv2.imwrite(str(output), sheet):
            raise OSError(f"could not write {output}")
        print(f"{name}: {len(METHODS)} fits rendered", flush=True)
    with gzip.open(args.out_dir / "results.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
