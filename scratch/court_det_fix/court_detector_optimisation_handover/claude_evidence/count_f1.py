"""Count axis hypotheses that pass the necessary player test, per axis (throwaway)."""

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
"""Count axis hypotheses the player gate would let F1 skip (throwaway)."""
import sys
from pathlib import Path

exec(Path(sys.argv[1]).read_text().split("pair_list = ")[0].replace("args = parser.parse_args()", "args = parser.parse_args(sys.argv[2:])"))

pair_list = list(permutations(range(len(points)), 2))
for pair_id in args.pairs:
    pair_points = points[list(pair_list[pair_id])]
    basis, _ = run_automatic.run_given.basis_for(pair_points, size, settings)
    for axis, coords in ((0, run_automatic.detector.X_COORDS), (1, run_automatic.detector.Y_COORDS)):
        matches = run_automatic.run_given.match_axis(basis, axis, coords, observations, size, settings, feet)
        total = len(matches.parameters)
        compatible = int(matches.player_compatible.sum())
        print(args.case, pair_id, "axis", axis, "hypotheses", total, "player-compatible", compatible,
              f"({compatible / max(total, 1):.0%})", flush=True)
