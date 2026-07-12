#!/usr/bin/env python3
"""Measure how far landmark extrapolation lands from actually clicked corners.

The off-frame method recovers an unseen corner by fitting a homography to
clicked line intersections and projecting the corner through it. Before
trusting that on footage where the corner truly is invisible, run this on a
frame where all four corners ARE visible: annotate the corners and 4+ landmarks
with ``annotate_court_corners_offframe.py --landmarks always``, then this
script fits from the landmarks alone and reports the distance between each
projected corner and its click. Those distances are the method's error.

Reads the corners CSV and its ``<stem>_landmarks.csv`` sidecar. Only labelled
corner rows (``corner_label`` column) are accepted; extrapolated rows are
skipped, since they are outputs of the same fit rather than ground truth.

Usage::

    python src/courtkeynet/validation_scripts/check_extrapolation.py \\
        --corners-csv hand_corners.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import court_landmarks as court  # noqa: E402


def frame_errors(
    corner_rows: pd.DataFrame, landmark_rows: pd.DataFrame
) -> tuple[dict[str, float], list[str], "court.FitCheck"]:
    """Fit from one frame's landmarks and compare against its clicked corners.

    Clicked corners whose own crossing sits in the landmark set (the annotation
    tool seeds them on off-frame frames) are not independent measurements, so
    they are listed separately instead of scored.

    :param corner_rows: the frame's corner rows; only ``source == "click"`` rows
        are compared. Frame width for the gate is recovered from x_px / x_norm
    :param landmark_rows: the frame's landmark sidecar rows, 5+ of them
    :return: (per-label error in px for independent clicked corners, labels of
        clicked corners that were inside the fit, the reprojection gate verdict)
    :raises ValueError: propagated from a degenerate or under-floor landmark fit
    """
    court_pts = landmark_rows[["court_x_m", "court_y_m"]].to_numpy(dtype=np.float64)  # (n, 2)
    image_pts = landmark_rows[["x_px", "y_px"]].to_numpy(dtype=np.float64)  # (n, 2)
    homography, residuals = court.fit_homography(court_pts, image_pts)
    projected = court.project_corners(homography)  # (slot=4, xy=2)

    # The corners CSV stores both px and normalised coords, so the frame width
    # the gate needs falls out of their ratio.
    first = corner_rows.iloc[0]
    frame_width = int(round(float(first["x_px"]) / float(first["x_norm"]))) if float(first["x_norm"]) else 1920
    verdict = court.check_fit(residuals, frame_width)

    landmark_names = set(landmark_rows["landmark"].astype(str))
    errors: dict[str, float] = {}
    in_fit: list[str] = []
    clicked = corner_rows[corner_rows["source"] == "click"]
    for _, row in clicked.iterrows():
        slot = int(row["corner_idx"])
        if court.CORNER_LANDMARK_NAMES[slot] in landmark_names:
            in_fit.append(str(row["corner_label"]))
            continue
        offset = projected[slot] - np.array([row["x_px"], row["y_px"]], dtype=np.float64)
        errors[str(row["corner_label"])] = float(np.linalg.norm(offset))
    return errors, in_fit, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corners-csv", type=Path, required=True)
    parser.add_argument("--landmarks-csv", type=Path, default=None,
                        help="Defaults to <corners-csv stem>_landmarks.csv beside the corners CSV.")
    parser.add_argument("--video", type=str, default=None,
                        help="Only check rows whose video basename matches (default: all).")
    args = parser.parse_args()

    landmarks_csv = args.landmarks_csv
    if landmarks_csv is None:
        landmarks_csv = args.corners_csv.with_name(args.corners_csv.stem + "_landmarks.csv")
    for path in (args.corners_csv, landmarks_csv):
        if not path.exists():
            sys.exit(f"ERROR: {path} does not exist")

    corners = pd.read_csv(args.corners_csv)
    landmarks = pd.read_csv(landmarks_csv)
    if "corner_label" not in corners.columns:
        sys.exit("ERROR: corners CSV has no corner_label column; annotate with the off-frame tool")
    if args.video is not None:
        video_names = corners["video"].map(lambda value: Path(str(value)).name)
        corners = corners[video_names == Path(args.video).name]
        landmark_names = landmarks["video"].map(lambda value: Path(str(value)).name)
        landmarks = landmarks[landmark_names == Path(args.video).name]

    all_errors: list[float] = []
    frames_checked = 0
    for (video, frame), landmark_rows in landmarks.groupby(["video", "frame"]):
        if len(landmark_rows) < court.FIT_MIN_POINTS:
            print(f"{Path(str(video)).name} frame {frame}: only {len(landmark_rows)} landmarks "
                  f"(floor {court.FIT_MIN_POINTS}), skipped")
            continue
        corner_rows = corners[(corners["video"] == video) & (corners["frame"] == frame)]
        if corner_rows.empty:
            print(f"{Path(str(video)).name} frame {frame}: landmarks but no corner rows, skipped")
            continue
        try:
            errors, in_fit, verdict = frame_errors(corner_rows, landmark_rows)
        except ValueError as error:
            print(f"{Path(str(video)).name} frame {frame}: fit failed ({error})")
            continue
        frames_checked += 1
        print(f"{Path(str(video)).name} frame {frame}: {len(landmark_rows)} landmarks, "
              f"fit rms {verdict.rms_px:.2f} px, max {verdict.worst_px:.2f} px [{verdict.level}]")
        if verdict.reason:
            print(f"  gate: {verdict.reason}")
        skipped = [str(label) for label in corner_rows.loc[corner_rows["source"] != "click", "corner_label"]]
        parts = [f"{label} {error:.1f} px" for label, error in errors.items()]
        if in_fit:
            parts.append(f"({', '.join(in_fit)} fed the fit, not scored)")
        if skipped:
            parts.append(f"({', '.join(skipped)} extrapolated, skipped)")
        print(f"  {'  |  '.join(parts)}")
        all_errors.extend(errors.values())

    if not frames_checked:
        sys.exit(f"ERROR: no frame had both {court.FIT_MIN_POINTS}+ landmarks and corner rows to compare")
    if all_errors:
        pooled = np.array(all_errors, dtype=np.float64)
        print(f"\n{frames_checked} frame(s), {pooled.size} clicked corners: "
              f"median {np.median(pooled):.1f} px, worst {pooled.max():.1f} px")
    else:
        # Every clicked corner's own crossing fed its fit (the tool seeds them),
        # so there is nothing independent to pool; the gate levels above stand.
        print(f"\n{frames_checked} frame(s); no independent clicked corners to pool")
    return 0


if __name__ == "__main__":
    sys.exit(main())
