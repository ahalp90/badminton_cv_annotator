"""Choose the line-averaged paint test's pass bar from bare floor on the hand-marked views.

A court fitted to each view's landmark hand marks gives the true place of every painted line. The test
runs along each true line, and along the floor midway between each pair of neighbouring parallel lines,
with side points as far out as a court line between that pair would get. The pass bar is the contrast,
in steps of 0.5 grey levels, that maximises the line pass rate minus the floor pass rate over all
hand-marked views together; ties go to the higher bar. This rule was set before the measurement.

It also lists each view's two far lines against its floor lines, where the far end is the end whose
baseline and long-service line sit closer together in the image.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.check_20260926_line_paint.line_bar
"""

import sys
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import line_paint
from scratch.court_det_fix.court_detector.line_paint import (
    CENTRE_SEGMENTS_M,
    MARKING_INTERVALS,
)
from scratch.court_det_fix.court_detector.run_views import pack_sources
from scratch.court_det_fix.w5_holistic import verifier

COURT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

BARS = np.arange(0, 20.5, 0.5)
# Marking indices of the baseline and long-service line at each end of the court model.
END_MARKINGS = {"y0": (5, 6), "y13": (10, 9)}


def midway_lines_m() -> list[tuple[np.ndarray, float]]:
    """Floor lines halfway between neighbouring parallel painted lines, each with its side distance."""
    lines = []
    for axis in (0, 1):
        positions = sorted({segment[0, axis] for segment in CENTRE_SEGMENTS_M
                            if (segment[0, 0] == segment[1, 0]) == (axis == 0)})
        for first, second in pairwise(positions):
            middle = (first + second) / 2
            line_m = np.array([[middle, 0.0], [middle, 13.4]]) if axis == 0 else np.array([[0.0, middle], [6.1, middle]])
            lines.append((line_m, min(line_paint.MAX_SIDE_M, line_paint.SIDE_SHARE_OF_GAP * (second - first))))
    return lines


def marking_contrast(grey: np.ndarray, homography: np.ndarray, marking: int, boxes: np.ndarray) -> float | None:
    samples = np.concatenate([line_paint.line_samples(grey, homography, CENTRE_SEGMENTS_M[segment],
                                                      line_paint.SIDE_DISTANCES_M[segment], boxes)
                              for segment in MARKING_INTERVALS[marking]])
    return line_paint.line_contrast(samples)


def far_end(homography: np.ndarray) -> str:
    """The end whose baseline and long-service line sit closer together in the image, at the centre line."""
    gaps = {}
    for end, (baseline, long_service) in END_MARKINGS.items():
        ends_m = np.array([[3.05, CENTRE_SEGMENTS_M[MARKING_INTERVALS[marking][0]][0, 1]]
                           for marking in (baseline, long_service)])
        points = cv2.perspectiveTransform(ends_m.reshape(-1, 1, 2), homography).reshape(2, 2)
        gaps[end] = np.linalg.norm(points[0] - points[1])
    return min(gaps, key=lambda end: gaps[end])


def main() -> None:
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    sources, provenances, _ = pack_sources(verifier.CASE_PACKS)
    line_values, floor_values, far_rows = [], [], []
    views = [row for row in manifest["cases"] if references.get(row["case_id"], {}).get("landmarks")]
    for row in views:
        view = row["case_id"]
        landmarks = references[view]["landmarks"]
        court_m = np.array([landmark["court_m"] for landmark in landmarks], dtype=float)
        marked_px = np.array([landmark["image_px"] for landmark in landmarks], dtype=float)
        homography, _ = cv2.findHomography(court_m, marked_px)
        frame = cv2.imread(str(COURT_ROOT / row["image"]))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        boxes = np.asarray(sources[view]["bbox_px"], dtype=float).reshape(-1, 4)
        if not provenances[view].has_same_image_boxes:
            boxes = np.empty((0, 4))
        lines = {marking: marking_contrast(grey, homography, marking, boxes) for marking in range(11)}
        floor = [line_paint.line_contrast(line_paint.line_samples(grey, homography, line_m, side_m, boxes))
                 for line_m, side_m in midway_lines_m()]
        line_values += [value for value in lines.values() if value is not None]
        floor_values += [value for value in floor if value is not None]
        baseline, long_service = END_MARKINGS[far_end(homography)]
        known_floor = sorted(value for value in floor if value is not None)
        far_rows.append(f"{view}\t{lines[baseline]:.1f}\t{lines[long_service]:.1f}\t"
                        f"{min(value for value in lines.values() if value is not None):.1f}\t"
                        + " ".join(f"{value:.1f}" for value in known_floor))
    lines_array, floor_array = np.array(line_values), np.array(floor_values)
    print(f"{len(views)} views; {len(lines_array)} true lines, {len(floor_array)} floor lines midway between them")
    print("\nbar (grey levels)\tlines pass\tfloor passes\tlines minus floor")
    differences = []
    for bar in BARS:
        line_rate, floor_rate = (lines_array >= bar).mean(), (floor_array >= bar).mean()
        differences.append(line_rate - floor_rate)
        print(f"{bar:.1f}\t{line_rate:.3f}\t{floor_rate:.3f}\t{line_rate - floor_rate:.3f}")
    best = max(range(len(BARS)), key=lambda index: (round(differences[index], 6), BARS[index]))
    print(f"\nchosen bar: {BARS[best]:.1f} grey levels")
    print("\nline contrast in grey levels: far baseline, far long-service line, the weakest true line, "
          "then every floor line, lowest first")
    print("view\tfar baseline\tfar long-service\tweakest line\tfloor lines")
    print("\n".join(far_rows))


if __name__ == "__main__":
    main()
