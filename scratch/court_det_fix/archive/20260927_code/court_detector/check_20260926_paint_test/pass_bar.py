"""Choose the gap-bounded paint test's pass bar from bare floor on the hand-marked views.

A court fitted to each view's landmark hand marks gives the true place of every painted line.
The test runs along each true line, and along the floor midway between each pair of neighbouring
parallel lines. The pass bar is the contrast that maximises the line pass rate minus the floor
pass rate, over all hand-marked views together; ties go to the higher bar.

Usage, from the repository root: python -m scratch.court_det_fix.court_detector.check_20260926_paint_test.pass_bar
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import detector
from experiments.annotator.independent_court.paint_geometry import CENTRE_SEGMENTS_M
from scratch.court_det_fix.court_detector.run_views import pack_sources
from scratch.court_det_fix.w5_holistic import verifier

COURT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

BARS = np.arange(0, 21)
SAMPLES_PER_LINE = 64
FAR_LINES = (6, 7)  # far baseline and far long-service line, where slips happen


def midway_lines_m() -> list[np.ndarray]:
    """Floor lines halfway between neighbouring parallel painted lines, across the whole court."""
    lengthwise = [np.array([[x, 0.0], [x, 13.4]]) for x in (verifier.LENGTHWISE_CENTRES_M[1:]
                                                          + verifier.LENGTHWISE_CENTRES_M[:-1]) / 2]
    transverse = [np.array([[0.0, y], [6.1, y]]) for y in (verifier.TRANSVERSE_CENTRES_M[1:]
                                                          + verifier.TRANSVERSE_CENTRES_M[:-1]) / 2]
    return lengthwise + transverse


def contrasts(grey: np.ndarray, homography: np.ndarray, line_m: np.ndarray, neighbours_m: np.ndarray,
              boxes: np.ndarray, native_per_working: float) -> np.ndarray:
    """Best ridge contrast at each known sample along the visible part of one line."""
    endpoints = detector.project(homography[None], line_m)[0]
    samples, visible = detector._visible_samples(endpoints[None], (grey.shape[1], grey.shape[0]), SAMPLES_PER_LINE)
    if not visible[0, 0]:
        return np.empty(0)
    best, _ = verifier.gap_bounded_photometry(grey, homography, samples[0, 0], line_m, neighbours_m, boxes,
                                              native_per_working)
    return best[np.isfinite(best)]


def main() -> None:
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    sources, provenances, _ = pack_sources(verifier.CASE_PACKS)
    line_values, far_line_values, floor_values = [], [], []
    views = [row for row in manifest["cases"] if references.get(row["case_id"], {}).get("landmarks")]
    for row in views:
        view = row["case_id"]
        landmarks = references[view]["landmarks"]
        court_m = np.array([landmark["court_m"] for landmark in landmarks], dtype=float)
        marked_px = np.array([landmark["image_px"] for landmark in landmarks], dtype=float)
        homography, _ = cv2.findHomography(court_m, marked_px)
        frame = cv2.imread(str(COURT_ROOT / row["image"]))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        native_per_working = max(1.0, max(frame.shape[:2]) / 960)
        boxes = np.asarray(sources[view]["bbox_px"], dtype=float).reshape(-1, 4)
        if not provenances[view].has_same_image_boxes:
            boxes = np.empty((0, 4))
        for segment, line_m in enumerate(CENTRE_SEGMENTS_M):
            values = contrasts(grey, homography, line_m, verifier.parallel_neighbours_m(line_m), boxes,
                               native_per_working)
            line_values.append(values)
            if segment in FAR_LINES:
                far_line_values.append(values)
        for line_m in midway_lines_m():
            lengthwise = line_m[0, 0] == line_m[1, 0]
            neighbours_m = verifier.LENGTHWISE_CENTRES_M if lengthwise else verifier.TRANSVERSE_CENTRES_M
            floor_values.append(contrasts(grey, homography, line_m, neighbours_m, boxes, native_per_working))
    lines, far_lines, floor = (np.concatenate(values) for values in (line_values, far_line_values, floor_values))
    print(f"{len(views)} views; known samples: {len(lines)} on lines, {len(far_lines)} on the two far lines, "
          f"{len(floor)} on floor midway between lines")
    print("\nbar (grey levels)\tlines pass\tfar lines pass\tfloor passes\tlines minus floor")
    differences = []
    for bar in BARS:
        line_rate, far_rate, floor_rate = ((values >= bar).mean() for values in (lines, far_lines, floor))
        differences.append(line_rate - floor_rate)
        print(f"{bar}\t{line_rate:.3f}\t{far_rate:.3f}\t{floor_rate:.3f}\t{line_rate - floor_rate:.3f}")
    best = max(range(len(BARS)), key=lambda index: (round(differences[index], 6), BARS[index]))
    print(f"\nchosen bar: {BARS[best]} grey levels")


if __name__ == "__main__":
    main()
