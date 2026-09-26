"""Average grey profile across each painted line, and across bare floor, on the hand-marked views.

A court fitted to each view's landmark hand marks gives the true place of every painted line. Along each
line, every 1 cm of floor gives one sample: the grey level at a row of points across the line, from 0.36 m
on one side to 0.36 m on the other, measured on the floor. Averaging those rows along the line shows whether
a faint line is still brighter than the floor beside it when single samples are too noisy to tell. The same
runs along the floor midway between neighbouring parallel lines, as a no-paint comparison.

The far end is the end whose baseline and long-service line sit closer together in the image.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.check_20260926_line_paint.line_profiles
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court.paint_geometry import CENTRE_SEGMENTS_M
from scratch.court_det_fix.court_detector.run_views import pack_sources
from scratch.court_det_fix.w5_holistic import verifier

COURT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

OFFSETS_M = np.round(np.arange(-0.36, 0.361, 0.06), 2)
STEP_M = 0.01
LENGTHWISE_CENTRES_M = np.array([0.02, 0.48, 3.05, 5.62, 6.08])
TRANSVERSE_CENTRES_M = np.array([0.02, 0.78, 4.70, 8.70, 12.62, 13.38])
# Segment indices in CENTRE_SEGMENTS_M: baseline and long-service line at each end.
END_LINES = {"y0": (6, 7), "y13": (11, 10)}


def project(homography: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(points_m.reshape(-1, 1, 2).astype(np.float64), homography).reshape(
        points_m.shape)


def profiles(grey: np.ndarray, homography: np.ndarray, line_m: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Grey levels across one floor line: one row per usable 1 cm sample, one column per offset."""
    start_m, end_m = line_m
    length_m = np.linalg.norm(end_m - start_m)
    along = (end_m - start_m) / length_m
    normal = np.array([-along[1], along[0]])
    centres_m = start_m + np.arange(0, length_m, STEP_M)[:, None] * along
    points_m = centres_m[:, None, :] + OFFSETS_M[None, :, None] * normal
    points_px = project(homography, points_m)
    usable = verifier.observable_points(points_px, (grey.shape[1], grey.shape[0]), boxes).all(axis=1)
    points_px = points_px[usable].astype(np.float32)
    if not len(points_px):
        return np.empty((0, len(OFFSETS_M)))
    return cv2.remap(grey, points_px[..., 0], points_px[..., 1], cv2.INTER_LINEAR)


def midway_lines_m() -> list[np.ndarray]:
    lengthwise = [np.array([[x, 0.0], [x, 13.4]]) for x in (LENGTHWISE_CENTRES_M[1:] + LENGTHWISE_CENTRES_M[:-1]) / 2]
    transverse = [np.array([[0.0, y], [6.1, y]]) for y in (TRANSVERSE_CENTRES_M[1:] + TRANSVERSE_CENTRES_M[:-1]) / 2]
    return lengthwise + transverse


def image_gap(homography: np.ndarray, first: int, second: int) -> float:
    """Image distance between two transverse lines at the court's centre line, in native px."""
    first_m = np.array([[3.05, CENTRE_SEGMENTS_M[first][0][1]]])
    second_m = np.array([[3.05, CENTRE_SEGMENTS_M[second][0][1]]])
    return float(np.linalg.norm(project(homography, first_m) - project(homography, second_m)))


def described(rows: np.ndarray) -> str:
    """The mean profile, less the mean of its two outermost offsets, with its sample count."""
    if not len(rows):
        return "no samples"
    mean = rows.mean(axis=0)
    relative = mean - (mean[0] + mean[-1]) / 2
    return f"n {len(rows):5d}  " + " ".join(f"{value:6.1f}" for value in relative)


def main() -> None:
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    sources, provenances, _ = pack_sources(verifier.CASE_PACKS)
    print("mean grey profile across each line, less its two outer offsets, in grey levels")
    print("offsets across the line, floor metres: " + " ".join(f"{offset:6.2f}" for offset in OFFSETS_M))
    for row in manifest["cases"]:
        view = row["case_id"]
        landmarks = references.get(view, {}).get("landmarks")
        if not landmarks:
            continue
        court_m = np.array([landmark["court_m"] for landmark in landmarks], dtype=float)
        marked_px = np.array([landmark["image_px"] for landmark in landmarks], dtype=float)
        homography, _ = cv2.findHomography(court_m, marked_px)
        frame = cv2.imread(str(COURT_ROOT / row["image"]))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        boxes = np.asarray(sources[view]["bbox_px"], dtype=float).reshape(-1, 4)
        if not provenances[view].has_same_image_boxes:
            boxes = np.empty((0, 4))
        gaps = {end: image_gap(homography, *lines) for end, lines in END_LINES.items()}
        far_end = min(gaps, key=lambda end: gaps[end])
        far_baseline, far_long_service = END_LINES[far_end]
        print(f"\n{view}: far baseline to far long-service line {gaps[far_end]:.1f} native px apart "
              f"(near end {max(gaps.values()):.1f})")
        print(f"  far baseline         {described(profiles(grey, homography, CENTRE_SEGMENTS_M[far_baseline], boxes))}")
        print(f"  far long-service     "
              f"{described(profiles(grey, homography, CENTRE_SEGMENTS_M[far_long_service], boxes))}")
        other = [profiles(grey, homography, line_m, boxes) for index, line_m in enumerate(CENTRE_SEGMENTS_M)
                 if index not in (far_baseline, far_long_service)]
        print(f"  other lines          {described(np.concatenate(other))}")
        floor = [profiles(grey, homography, line_m, boxes) for line_m in midway_lines_m()]
        print(f"  midway floor         {described(np.concatenate(floor))}")


if __name__ == "__main__":
    main()
