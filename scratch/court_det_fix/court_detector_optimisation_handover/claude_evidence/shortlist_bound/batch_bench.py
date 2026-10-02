"""Time continuous_support at several batch sizes on synthetic courts and check bit-identity (throwaway)."""
import sys
import time

import cv2
import numpy as np

sys.path[:0] = [".", "src", "scratch/court_det_fix/frozen_helpers_20260914/marking_diagnosis",
                "scratch/court_det_fix/frozen_helpers_20260914/legacy", "scratch/court_det_fix/wider_evaluation",
                "scratch/court_det_fix/w5_holistic"]
from experiments.annotator.independent_court import detector

import scan_population  # noqa: E402

rng = np.random.default_rng(0)
size = (960, 540)
base = np.array([[250, 450], [710, 450], [620, 120], [340, 120]], dtype=float)
homographies = np.asarray([cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (base + rng.normal(0, 15, base.shape)).astype(np.float32))
                           for _ in range(40000)])
true_segments = detector.project(homographies[:1], detector.SEGMENTS_M)[0].reshape(-1, 4)
noise = rng.uniform(0, 960, (300, 4)) * [1, 540 / 960, 1, 540 / 960]
segments = np.vstack([true_segments + rng.normal(0, 1.5, true_segments.shape), noise])
maps = detector._distance_maps((segments, segments), size)
reference = None
for batch in (256, 1024, 4096):
    started = time.perf_counter()
    scores = np.concatenate([scan_population.continuous_support(homographies[start:start + batch], maps, size)
                             for start in range(0, len(homographies), batch)])
    seconds = time.perf_counter() - started
    reference = scores if reference is None else reference
    print(f"batch {batch}: {seconds:.2f} s, bit-identical to 256: {np.array_equal(scores, reference)}")
shuffled = rng.permutation(len(homographies))
scores = np.empty(len(homographies))
for start in range(0, len(homographies), 256):
    rows = shuffled[start:start + 256]
    scores[rows] = scan_population.continuous_support(homographies[rows], maps, size)
print("shuffled batches bit-identical:", np.array_equal(scores, reference))
