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


reference = original_finite_scores(homographies, observations, captured["axes"], size)
repeat = original_finite_scores(homographies, observations, captured["axes"], size)
print("repeat identical:", np.array_equal(reference, repeat))
for offset in range(0, 64, 8):
    scores = original_finite_scores(at_offset(homographies, offset), observations, captured["axes"], size)
    changed = scores != reference
    print(f"homographies at +{offset:2d} bytes: changed {changed.sum()} of {len(scores)}, max delta {np.abs(scores - reference).max():.2e}")

# Isolate the projection step, which runs np.einsum over each court's 3x3 homography.
reference_pixels, _ = detector.project(homographies[:256], detector.SEGMENTS_M)
for offset in range(0, 64, 8):
    pixels, _ = detector.project(at_offset(homographies[:256], offset), detector.SEGMENTS_M)
    print(f"project at +{offset:2d} bytes: pixels changed {(pixels != reference_pixels).sum()}")
