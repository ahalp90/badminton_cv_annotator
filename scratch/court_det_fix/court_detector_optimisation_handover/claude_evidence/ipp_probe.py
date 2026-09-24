"""Check whether finite_scores gives different bits for identical inputs at different memory offsets (throwaway)."""

from __future__ import annotations

import os

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
import sys
from itertools import permutations
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--court-root", type=Path, required=True)
parser.add_argument("--case", required=True)
parser.add_argument("--pair", type=int, required=True)
args = parser.parse_args()

COURT_ROOT = args.court_root.resolve()
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic"), str(COURT_ROOT / "wider_evaluation")]
cv2.setNumThreads(1)
import run_cases

run_w5, verifier_module, runtime = run_cases.load_runtime(COURT_ROOT)
import generation
import run_automatic

run_given = sys.modules["run_given"]
scan_population = sys.modules["scan_population"]
detector = scan_population.detector
print("numpy", np.__version__, "run_given", run_given.__file__, flush=True)

verifier = runtime["verifier"]
context = verifier_module.prepare_view(COURT_ROOT, args.case)
direction = generation._new_direction(COURT_ROOT, context, verifier, sys.modules["vp_pruning"])
source = context.source
segments, families, size = run_automatic.prepare(source)
points = np.asarray(direction["estimator"]["points_working"])
native_size = (source["dimensions"]["width"], source["dimensions"]["height"])
scale = np.asarray(native_size) / size
feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                   for frame in source["all_feet_px"]], dtype=float) / scale
observations = run_automatic.assignment.prepare_observations(segments, size)
settings = run_automatic.Settings(keep_axes=512)

captured = {}
original_finite_scores = run_given.finite_scores


def capturing_finite_scores(homographies, observations, axes, size):
    captured.update(homographies=homographies.copy(), axes=axes)
    return original_finite_scores(homographies, observations, axes, size)


run_given.finite_scores = capturing_finite_scores
pencil_ids = list(permutations(range(len(points)), 2))[args.pair]
run_automatic.propose_role(points[list(pencil_ids)], observations, feet, size, settings, runtime["zone"])
run_given.finite_scores = original_finite_scores
homographies = captured["homographies"]
print("usable courts", len(homographies), flush=True)


def at_offset(array: np.ndarray, offset: int) -> np.ndarray:
    """Copy an array so its data starts `offset` bytes past a 64-byte boundary."""
    buffer = np.empty(array.nbytes + 128, dtype=np.uint8)
    start = (-buffer.ctypes.data) % 64 + offset
    moved = np.frombuffer(buffer[start:start + array.nbytes], dtype=array.dtype).reshape(array.shape)
    moved[...] = array
    assert moved.ctypes.data % 64 == offset
    return moved


families = []
for matched in captured["axes"]:
    groups = matched.diagnostics["retained_group_ids"]
    members = np.concatenate([observations.groups[index] for index in groups])
    families.append(observations.segments[members].reshape(-1, 4))
width, height = size
mask = np.full((height, width), 255, dtype=np.uint8)
for x1, y1, x2, y2 in np.rint(families[0]).astype(int):
    cv2.line(mask, (x1, y1), (x2, y2), 0, 1)
print("opencv", cv2.__version__, "| working size", size, "| IPP available", cv2.ipp.useIPP(), cv2.ipp.getIppVersion())


def transform_at(offset: int) -> np.ndarray:
    """Run the precise distance transform with its output buffer `offset` bytes past a 64-byte boundary."""
    out = at_offset(np.zeros((height, width), dtype=np.float32), offset)
    result = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE, dst=out)
    assert result.ctypes.data == out.ctypes.data
    return result


for use_ipp in (True, False):
    cv2.ipp.setUseIPP(use_ipp)
    outputs = {offset: transform_at(offset) for offset in range(0, 64, 4)}
    fresh = [cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) for _ in range(6)]
    reference = outputs[0]
    print(f"IPP {use_ipp}: pixels differing from +0 per output offset:", [int((outputs[o] != reference).sum()) for o in outputs])
    print(f"IPP {use_ipp}: distinct results over 6 fresh calls:", len({f.tobytes() for f in fresh}),
          "| offset outputs distinct:", len({o.tobytes() for o in outputs.values()}))
    if use_ipp:
        ipp_variants = {o.tobytes(): o for o in outputs.values()}
    else:
        plain = reference
        exact = np.sqrt(np.square(np.float64(0)))  # placeholder so the name exists
for variant in ipp_variants.values():
    print("IPP variant vs OpenCV-only: pixels differing", int((variant != plain).sum()), f"max {np.abs(variant - plain).max():.2e}")

# Exact Euclidean distances (float64) for judging which version rounds correctly.
from scipy import ndimage
exact = ndimage.distance_transform_edt(mask != 0)
print("OpenCV-only vs exact: max error", f"{np.abs(plain - exact).max():.2e}",
      "| equals float32(exact) everywhere:", bool(np.array_equal(plain, exact.astype(np.float32))))
for variant in ipp_variants.values():
    print("IPP variant vs exact: max error", f"{np.abs(variant - exact).max():.2e}",
          "| equals float32(exact):", bool(np.array_equal(variant, exact.astype(np.float32))))
