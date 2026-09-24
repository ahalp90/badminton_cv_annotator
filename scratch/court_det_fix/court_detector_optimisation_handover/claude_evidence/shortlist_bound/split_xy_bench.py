"""continuous_support with x and y sample coordinates as separate arrays: timing and bit-identity (throwaway)."""
import sys
import time

import cv2
import numpy as np

sys.path[:0] = [".", "src", "scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis",
                "scratch/court_det_fix/frozen_helpers_20260914/legacy", "scratch/court_det_fix/wider_evaluation",
                "scratch/court_det_fix/w5_holistic"]
from experiments.annotator.independent_court import assignment, detector

import scan_population  # noqa: E402


def split_support(homographies, maps, size):
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower, upper, visible = detector._visible_fractions(endpoints, size)
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, 64)
    sample_x = starts[..., 0, None] + fractions * vectors[..., 0, None]
    sample_y = starts[..., 1, None] + fractions * vectors[..., 1, None]
    pixel_x = np.clip(sample_x, 0, size[0] - 1).astype(int)
    pixel_y = np.clip(sample_y, 0, size[1] - 1).astype(int)
    family = np.repeat([0, 1], 6)[None, :, None]
    distance = maps[family, pixel_y, pixel_x]
    response = np.exp(-.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX)).mean(axis=2)
    response *= visible
    return scan_population.marking_score(response, visible)


rng = np.random.default_rng(1)
size = (960, 540)
base = np.array([[250, 450], [710, 450], [620, 120], [340, 120]], dtype=float)
# Include courts partly off-image, so clipping and hidden intervals are exercised.
homographies = np.asarray([cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (base + rng.normal(0, 120, base.shape)).astype(np.float32))
                           for _ in range(40000)])
segments = np.vstack([detector.project(homographies[:1], detector.SEGMENTS_M)[0].reshape(-1, 4), rng.uniform(0, 540, (300, 4))])
maps = detector._distance_maps((segments, segments), size)
for batch in (256, 1024):
    for name, function in (("original", scan_population.continuous_support), ("split x/y", split_support)):
        started = time.perf_counter()
        scores = np.concatenate([function(homographies[start:start + batch], maps, size)
                                 for start in range(0, len(homographies), batch)])
        seconds = time.perf_counter() - started
        if name == "original":
            reference = scores
        print(f"batch {batch} {name}: {seconds:.2f} s, bit-identical: {np.array_equal(scores, reference, equal_nan=True)}, "
              f"hidden intervals present: {np.isnan(scores).sum()} NaN")
