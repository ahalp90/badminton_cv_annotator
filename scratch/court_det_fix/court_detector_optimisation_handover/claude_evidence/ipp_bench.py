"""Interleaved timing of the precise distance transform with and without IPP (throwaway)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from statistics import median
from time import perf_counter

import cv2
import numpy as np

cv2.setNumThreads(1)
rng = np.random.default_rng(0)
masks = []
for line_count in (100, 300, 1000):
    mask = np.full((540, 960), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in rng.integers(0, [960, 540, 960, 540], size=(line_count, 4)):
        cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
    masks.append((line_count, mask))
print("opencv", cv2.__version__, "|", cv2.ipp.getIppVersion())
for line_count, mask in masks:
    rounds = {True: [], False: []}
    for _ in range(15):
        for use_ipp in (True, False):
            cv2.ipp.setUseIPP(use_ipp)
            started = perf_counter()
            for _ in range(20):
                cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            rounds[use_ipp].append((perf_counter() - started) * 50)
    print(f"{line_count:4d} lines: IPP on median {median(rounds[True]):.2f} ms (range {min(rounds[True]):.2f}-{max(rounds[True]):.2f}),"
          f" IPP off median {median(rounds[False]):.2f} ms (range {min(rounds[False]):.2f}-{max(rounds[False]):.2f})")
