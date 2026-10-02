"""Profile one direction pair of the fresh G0 generator (throwaway harness)."""

from __future__ import annotations

import os

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
import sys
from itertools import permutations
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--court-root", type=Path, required=True)
parser.add_argument("--case", required=True)
parser.add_argument("--pairs", type=int, nargs="+", required=True)


args = parser.parse_args()

COURT_ROOT = args.court_root.resolve()
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic"), str(COURT_ROOT / "wider_evaluation")]
cv2.setNumThreads(1)
import run_cases

run_w5, verifier_module, runtime = run_cases.load_runtime(COURT_ROOT)
import generation
import run_automatic

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
zone = runtime["zone"]
pair_list = list(permutations(range(len(points)), 2))
run_given = sys.modules["run_given"]
seed = sys.modules["projective_seed"]
detector = run_given.detector


def band_masks(basis: np.ndarray, matched, axis: int, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per retained axis hypothesis: which foot samples fall inside this axis's court band, and in its far half."""
    homogeneous_feet = np.concatenate((feet, np.ones((*feet.shape[:-1], 1))), axis=2)
    mapped = homogeneous_feet @ np.linalg.inv(basis).T
    with np.errstate(divide="ignore", invalid="ignore"):
        rectified = mapped[..., axis] / mapped[..., 2]
        scale, shift = matched.parameters[matched.retained].T
        position = (rectified[None] - shift[:, None, None]) / scale[:, None, None] / float(coordinates.max())
    inside = np.isfinite(position) & (position >= -.15) & (position <= 1.15)
    return inside, inside & (position < .5)


for pair_id in args.pairs:
    pencil_ids = pair_list[pair_id]
    basis, _ = run_given.basis_for(points[list(pencil_ids)], size, settings)
    horizontal = seed.match_axis(basis, 0, detector.X_COORDS, observations, size, settings, feet)
    vertical = seed.match_axis(basis, 1, detector.Y_COORDS, observations, size, settings, feet)
    transforms, _ = run_given.combine(basis, horizontal, vertical)
    transforms, _ = run_given.canonicalise(transforms)
    valid, _ = run_given.geometry(transforms, size)
    valid_ids = np.flatnonzero(valid)

    started = perf_counter()
    one_old, two_old = zone.player_fractions(transforms[valid_ids], feet)
    old_s = perf_counter() - started

    started = perf_counter()
    inside_x, _ = band_masks(basis, horizontal, 0, detector.X_COORDS)
    inside_y, far_y = band_masks(basis, vertical, 1, detector.Y_COORDS)
    near_y = inside_y & ~far_y
    # One matrix product per frame counts the foot samples inside both bands for every (horizontal, vertical) pair.
    by_frame_x = inside_x.transpose(1, 0, 2).astype(np.float32)  # (frames, horizontal hypotheses, foot samples)
    anyone = (by_frame_x @ inside_y.transpose(1, 2, 0).astype(np.float32)) > 0
    far = (by_frame_x @ far_y.transpose(1, 2, 0).astype(np.float32)) > 0
    near = (by_frame_x @ near_y.transpose(1, 2, 0).astype(np.float32)) > 0
    one_new = anyone.mean(axis=0).reshape(-1)[valid_ids]
    two_new = (far & near).mean(axis=0).reshape(-1)[valid_ids]
    new_s = perf_counter() - started

    usable_old = (one_old == 1) & (two_old >= .5)
    usable_new = (one_new == 1) & (two_new >= .5)
    print(f"{args.case} pair {pair_id}: combined {len(transforms)}, valid {len(valid_ids)}, feet slots {feet.shape[0] * feet.shape[1]}"
          f" | one mismatches {int((one_old != one_new).sum())}, two mismatches {int((two_old != two_new).sum())},"
          f" usable mismatches {int((usable_old != usable_new).sum())} of {int(usable_old.sum())} usable"
          f" | current {old_s:.2f} s, matrix product {new_s:.3f} s", flush=True)

    # Invalid courts only feed diagnostics; check a random sample against the current test.
    invalid_sample = np.random.default_rng(0).choice(np.flatnonzero(~valid), size=min(20000, int((~valid).sum())), replace=False)
    one_check, two_check = zone.player_fractions(transforms[invalid_sample], feet)
    one_all = anyone.mean(axis=0).reshape(-1)[invalid_sample]
    two_all = (far & near).mean(axis=0).reshape(-1)[invalid_sample]
    for name, matched in (("horizontal", horizontal), ("vertical", vertical)):
        enumerated = len(matched.parameters)
        pattern = matched.supported >= settings.minimum_matches
        print(f"   {name}: enumerated {enumerated}, player-compatible {matched.player_compatible.mean():.0%},"
              f" pattern-supported {pattern.mean():.0%}, both {np.mean(pattern & matched.player_compatible):.0%}")
    print(f"   invalid-court sample {len(invalid_sample)}: one mismatches {int((one_check != one_all).sum())},"
          f" two mismatches {int((two_check != two_all).sum())}", flush=True)
