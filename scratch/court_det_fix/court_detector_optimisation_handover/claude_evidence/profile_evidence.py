"""Profile pool evidence and the W5 refit stage from a saved generation record (throwaway)."""

from __future__ import annotations

import os

for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_variable] = "1"

import argparse
import cProfile
import gzip
import json
import pickle
import pstats
import sys
from pathlib import Path
from time import perf_counter

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--court-root", type=Path, required=True)
parser.add_argument("--case", required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()

COURT_ROOT = args.court_root.resolve()
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic"), str(COURT_ROOT / "wider_evaluation")]
cv2.setNumThreads(1)
import run_cases

run_w5, verifier_module, runtime = run_cases.load_runtime(COURT_ROOT)
import run_automatic
from measurement import prepared_measurements

verifier = runtime["verifier"]
context = verifier_module.prepare_view(COURT_ROOT, args.case)
run_automatic.frame_path = lambda source, _root: verifier_module.frame_path(COURT_ROOT, source, context.provenance)
# This file sits in scratch/court_det_fix/court_detector_optimisation_handover/claude_evidence/.
saved_path = Path(__file__).resolve().parents[2] / "svd_search" / "run_20260923" / "generation" / "baseline" / f"{args.case}.json.gz"
with gzip.open(saved_path, "rt", encoding="utf-8") as stream:
    saved = json.load(stream)
shortlist = [{key: value for key, value in entry.items() if key not in ("stripe", "gates", "profile")}
             for entry in saved["entries"]]
source = context.source
segments, families, size = run_automatic.prepare(source)
observations = run_automatic.assignment.prepare_observations(segments, size)


def evidence() -> list[dict]:
    return run_automatic.evaluate_pool(source, shortlist, observations, size, segments, families,
                                       runtime["zone"], COURT_ROOT)


def refit(entries: list[dict]) -> dict:
    cache: dict = {}
    parents, children = [], []
    with prepared_measurements(verifier_module):
        for index, entry in enumerate(entries):
            parent, _ = run_w5.make_parent_record(context, entry, "G0", 0, index, runtime, cache)
            parents.append(parent)
        for parent in parents:
            _row, child, _ = run_w5.attempt_refit(context, parent, runtime, cache)
            if child is not None:
                children.append(child)
        candidates = [parent for parent in parents if parent.get("hard_valid") and "evidence" in parent] + children
        return {"ranking": verifier["rank_candidates"](candidates), "parents": parents, "children": children}


args.out.mkdir(parents=True, exist_ok=True)
profiler = cProfile.Profile()
started = perf_counter()
entries = profiler.runcall(evidence)
evidence_s = perf_counter() - started
profiler.dump_stats(str(args.out / f"{args.case}_evidence.prof"))
print("evidence seconds", round(evidence_s, 1), "entries", len(entries), flush=True)
same_gates = all(new["gates"] == old["gates"] for new, old in zip(entries, saved["entries"], strict=True))
print("gates match saved record:", same_gates, flush=True)

profiler = cProfile.Profile()
started = perf_counter()
refit_result = profiler.runcall(refit, saved["entries"])
ranking = refit_result["ranking"]
refit_s = perf_counter() - started
profiler.dump_stats(str(args.out / f"{args.case}_refit.prof"))
print("refit seconds", round(refit_s, 1), "top", next(iter(ranking["provisional_rank"]), None), flush=True)
with open(args.out / f"{args.case}_outputs.pkl", "wb") as stream:
    pickle.dump({"entries": entries, **refit_result}, stream)
for name in ("evidence", "refit"):
    print("=====", name)
    stats = pstats.Stats(str(args.out / f"{args.case}_{name}.prof"))
    stats.sort_stats("tottime").print_stats(22)
