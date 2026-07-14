#!/usr/bin/env python3
"""Render an annotated frame's ground truth over the actual video frame.

Refits the homography from the frame's landmarks, then draws every painted
court line through it (green), the outer boundary (orange), the clicked
landmarks (cyan) and the corners (orange cross = clicked, red = extrapolated).
The visual audit: the green grid should hug the real painted lines.

Usage::

    python src/courtkeynet/validation_scripts/render_ground_truth.py \\
        --corners-csv hand_corners.csv --video clip.mkv --frame 150 --out gt.jpg
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import court_landmarks as court  # noqa: E402

BOUNDARY = (0, 94, 213)  # orange, Wong palette
GRID = (80, 200, 80)
LANDMARK = (233, 180, 86)  # sky blue
EXTRAPOLATED = (0, 0, 255)
CLAMP = 100_000  # keep projected points inside int32 for cv2 drawing


def _rows_for(csv_path: Path, video_basename: str, frame: int) -> list[dict[str, str]]:
    with Path(csv_path).open(newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if Path(str(row["video"])).name == video_basename and int(row["frame"]) == frame
        ]


def _clip_point(point: np.ndarray) -> tuple[int, int]:
    x_px, y_px = np.clip(point, -CLAMP, CLAMP).round().astype(int)
    return int(x_px), int(y_px)


def render(image: np.ndarray, corner_rows: list[dict[str, str]], landmark_rows: list[dict[str, str]]) -> np.ndarray:
    """Draw the ground truth onto ``image`` (modified in place and returned)."""
    if len(landmark_rows) >= 4:
        court_pts = np.array([[float(r["court_x_m"]), float(r["court_y_m"])] for r in landmark_rows])
        image_pts = np.array([[float(r["x_px"]), float(r["y_px"])] for r in landmark_rows])
        homography, _ = court.fit_homography(court_pts, image_pts)
        segments = [((0.0, y_m), (court.COURT_WIDTH_M, y_m)) for y_m in court.Y_LINES.values()]
        for name, x_m in court.X_LINES.items():
            if name == "centre":
                # Not painted between the short service lines.
                segments.append(((x_m, 0.0), (x_m, court.Y_LINES["far_short_service"])))
                segments.append(((x_m, court.Y_LINES["near_short_service"]), (x_m, court.COURT_LENGTH_M)))
            else:
                segments.append(((x_m, 0.0), (x_m, court.COURT_LENGTH_M)))
        for start_m, end_m in segments:
            start, end = court.project_points(homography, np.array([start_m, end_m]))
            cv2.line(image, _clip_point(start), _clip_point(end), GRID, 1, cv2.LINE_AA)
        quad = np.clip(court.project_corners(homography), -CLAMP, CLAMP).round().astype(np.int32)
        cv2.polylines(image, [quad.reshape(-1, 1, 2)], True, BOUNDARY, 3, cv2.LINE_AA)

    for row in landmark_rows:
        centre = (int(round(float(row["x_px"]))), int(round(float(row["y_px"]))))
        cv2.circle(image, centre, 5, LANDMARK, 2, cv2.LINE_AA)
    for row in sorted(corner_rows, key=lambda r: int(r["corner_idx"])):
        point = (int(round(float(row["x_px"]))), int(round(float(row["y_px"]))))
        if not (0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]):
            continue  # off-frame corners have nowhere to draw
        colour = BOUNDARY if row["source"] == "click" else EXTRAPOLATED
        cv2.drawMarker(image, point, colour, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.putText(image, str(row["corner_label"]), (point[0] + 10, point[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corners-csv", type=Path, required=True)
    parser.add_argument("--landmarks-csv", type=Path, default=None,
                        help="Defaults to <corners-csv stem>_landmarks.csv beside the corners CSV.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output image; .jpg keeps the repo light.")
    args = parser.parse_args()

    landmarks_csv = args.landmarks_csv
    if landmarks_csv is None:
        landmarks_csv = args.corners_csv.with_name(args.corners_csv.stem + "_landmarks.csv")
    corner_rows = _rows_for(args.corners_csv, args.video.name, args.frame)
    if not corner_rows:
        sys.exit(f"ERROR: no corner rows for {args.video.name} frame {args.frame}")
    landmark_rows = _rows_for(landmarks_csv, args.video.name, args.frame) if landmarks_csv.exists() else []

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"ERROR: could not decode frame {args.frame} of {args.video}")
    render(image, corner_rows, landmark_rows)
    if not cv2.imwrite(str(args.out), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        sys.exit(f"ERROR: could not write {args.out}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
