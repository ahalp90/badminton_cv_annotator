"""Time the precise distance transform with and without IPP, and count output variants by buffer offset (throwaway)."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from time import perf_counter

import cv2
import numpy as np
from scipy import ndimage

cv2.setNumThreads(1)
rng = np.random.default_rng(0)
width, height = 960, 540
mask = np.full((height, width), 255, dtype=np.uint8)
for x1, y1, x2, y2 in rng.integers(0, [width, height, width, height], size=(300, 4)):
    cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
exact = ndimage.distance_transform_edt(mask != 0).astype(np.float32)


def at_offset(shape, offset):
    buffer = np.empty(int(np.prod(shape)) * 4 + 128, dtype=np.uint8)
    start = (-buffer.ctypes.data) % 64 + offset
    return np.frombuffer(buffer[start:start + int(np.prod(shape)) * 4], dtype=np.float32).reshape(shape)


print("opencv", cv2.__version__, "| numpy", np.__version__, "|", cv2.ipp.getIppVersion())
for use_ipp in (True, False):
    cv2.ipp.setUseIPP(use_ipp)
    variants = {cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=at_offset(mask.shape, offset)).tobytes()
                for offset in range(0, 64, 4)}
    started = perf_counter()
    for _ in range(100):
        result = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    per_call_ms = (perf_counter() - started) * 10
    print(f"IPP {use_ipp}: {per_call_ms:.2f} ms per 960x540 map | variants over 16 buffer offsets {len(variants)}"
          f" | equals rounded exact distances {np.array_equal(result, exact)}")
started = perf_counter()
for _ in range(20):
    ndimage.distance_transform_edt(mask != 0)
print(f"scipy distance_transform_edt: {(perf_counter() - started) * 50:.2f} ms per map")
