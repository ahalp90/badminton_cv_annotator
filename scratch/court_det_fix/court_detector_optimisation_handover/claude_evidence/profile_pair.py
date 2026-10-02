"""Profile one direction pair of the fresh G0 generator (throwaway harness)."""

from __future__ import annotations

import os

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
import cProfile
import pstats
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
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--no-profile", action="store_true")
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


def one_pair(pair_id: int) -> dict:
    pencil_ids = pair_list[pair_id]
    proposed = run_automatic.propose_role(points[list(pencil_ids)], observations, feet, size, settings, zone)
    local_details = {}
    for index, (candidate, details) in enumerate(zip(proposed.candidates, proposed.details, strict=True)):
        local_details[id(candidate)] = {"candidate_id": f"{pair_id}:{index}", "pair_id": pair_id, **details}
    retained = run_automatic.select_pool(proposed.candidates)
    return {"combined": proposed.record.get("combined"), "usable": len(proposed.candidates),
            "retained": [(local_details[id(c)]["candidate_id"], c.score, c.corners_px.tolist()) for c in retained]}


args.out.mkdir(parents=True, exist_ok=True)
for pair_id in args.pairs:
    started = perf_counter()
    if args.no_profile:
        result = one_pair(pair_id)
    else:
        profiler = cProfile.Profile()
        result = profiler.runcall(one_pair, pair_id)
        profiler.dump_stats(str(args.out / f"{args.case}_pair{pair_id}.prof"))
    elapsed = perf_counter() - started
    print(args.case, "pair", pair_id, "combined", result["combined"], "usable", result["usable"],
          "retained", len(result["retained"]), "seconds", round(elapsed, 2), flush=True)
    np.savez(args.out / f"{args.case}_pair{pair_id}_retained.npz",
             ids=np.asarray([row[0] for row in result["retained"]]),
             scores=np.asarray([row[1] for row in result["retained"]], dtype=np.float64),
             corners=np.asarray([row[2] for row in result["retained"]], dtype=np.float64),
             usable=np.asarray([result["usable"]]))
    if not args.no_profile:
        stats = pstats.Stats(str(args.out / f"{args.case}_pair{pair_id}.prof"))
        stats.sort_stats("tottime").print_stats(18)
