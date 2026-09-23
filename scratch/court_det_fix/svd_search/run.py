"""Run isolated SVD12 G0 search-depth arms on frozen views."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from time import perf_counter, process_time

for thread_variable in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[thread_variable] = "1"

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
COURT_ROOT = BASE.parent
REPO = COURT_ROOT.parents[1]
CASES = (
    "shuttleset_03_scene_0019", "shuttleset_03_scene_0034",
    "gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5",
    "am1_window_00_frame_54", "am2_window_01_frame_28019",
)
ARMS = {
    "baseline": (512, 256, 256),
    "deeper": (640, 256, 256),
    "shortlist": (512, 512, 512),
}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(json.dumps(value, allow_nan=False, separators=(",", ":")).encode(), mtime=0))
    temporary.replace(path)


def best_agreement(candidates: list[dict], selectable_keys: set[str], reference: np.ndarray, verifier: dict) -> dict | None:
    selectable = [candidate for candidate in candidates if candidate["origin_key"] in selectable_keys]
    if not selectable:
        return None
    rated = []
    for candidate in selectable:
        error = verifier["reference_corner_error"](np.asarray(candidate["corners_px"]), reference)
        rated.append((error["maximum"], candidate["origin_key"], error))
    maximum, key, error = min(rated)
    return {"origin_key": key, "maximum_native_px": maximum, "error": error}


def run_one(case_id: str, arm: str, output: Path, max_matched_pairs: int | None) -> dict:
    cv2.setNumThreads(1)
    sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic"),
                    str(COURT_ROOT / "wider_evaluation")]
    import run_cases

    run_w5, verifier_module, runtime = run_cases.load_runtime(COURT_ROOT)
    import automatic_generation
    import generation
    import run_automatic
    from measurement import prepared_measurements

    total_wall = perf_counter()
    total_cpu = process_time()
    verifier = runtime["verifier"]
    context = verifier_module.prepare_view(COURT_ROOT, case_id)
    direction = generation._new_direction(COURT_ROOT, context, verifier, sys.modules["vp_pruning"])
    preparation_timing = {"wall_s": perf_counter() - total_wall, "cpu_s": process_time() - total_cpu}
    caps = ARMS[arm]
    run_automatic.frame_path = lambda source, _root: verifier_module.frame_path(COURT_ROOT, source, context.provenance)
    generation_wall = perf_counter()
    generation_cpu = process_time()
    log_path = output / "generation_logs" / arm / f"{case_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log):
        record = automatic_generation.generate(
            context.source, direction, runtime["zone"], COURT_ROOT, run_automatic,
            direction_budget=12, keep_axes=caps[0], keep_per_pair=caps[1], keep_global=caps[2],
            max_matched_pairs=max_matched_pairs,
        )
    generation_timing = {"wall_s": perf_counter() - generation_wall, "cpu_s": process_time() - generation_cpu}
    refit_wall = perf_counter()
    refit_cpu = process_time()
    cache = {}
    parents, children, fit_rows = [], [], []
    with prepared_measurements(verifier_module):
        for index, entry in enumerate(record["entries"]):
            parent, _ = run_w5.make_parent_record(context, entry, "G0", 0, index, runtime, cache)
            parents.append(parent)
        for parent in parents:
            row, child, _ = run_w5.attempt_refit(context, parent, runtime, cache)
            fit_rows.append(row)
            if child is not None:
                children.append(child)
        candidates = [parent for parent in parents if parent.get("hard_valid") and "evidence" in parent] + children
        ranking = verifier["rank_candidates"](candidates)
    refit_timing = {"wall_s": perf_counter() - refit_wall, "cpu_s": process_time() - refit_cpu}
    total_timing = {"total_wall_s": perf_counter() - total_wall, "total_cpu_s": process_time() - total_cpu}
    reference = run_w5.load_reference(COURT_ROOT, case_id, verifier)
    reference_corners = np.asarray(reference["corners_px"], dtype=float)
    by_key = {candidate["origin_key"]: candidate for candidate in parents + children}
    selected_key = next(iter(ranking["provisional_rank"]), None)
    selected = by_key.get(selected_key)
    selected_error = (
        verifier["reference_corner_error"](np.asarray(selected["corners_px"]), reference_corners)
        if selected is not None else None
    )
    with gzip.open(BASE / "saved_selected.json.gz", "rt", encoding="utf-8") as stream:
        saved_selected = json.load(stream)[case_id]
    selected_saved_error = (
        verifier["reference_corner_error"](
            np.asarray(selected["corners_px"]), np.asarray(saved_selected["corners_native_px"])
        ) if selected is not None and saved_selected["corners_native_px"] is not None else None
    )
    matched = [pair for pair in record["pairs"] if pair["status"] == "matched"]
    selectable_keys = set(ranking["provisional_rank"])
    summary = {
        "schema": "svd-search-case/1", "case_id": case_id, "arm": arm,
        "scope": "automatic G0 only; SVD12; W5 4,3 refit and ranker",
        "caps": {"keep_axes": caps[0], "keep_per_pair": caps[1], "keep_global": caps[2]},
        "max_matched_pairs": max_matched_pairs,
        "timing": {"preparation": preparation_timing, "generation": generation_timing,
                   "refit_and_scoring": refit_timing,
                   **total_timing},
        "counts": {"matched_pairs": len(matched),
                   "combined_courts": sum(pair["role"]["combined"] for pair in matched),
                   "geometry_player_courts": sum(pair["role"]["geometry_players"] for pair in matched),
                   "raw_parents": sum(pair["raw_parent_count"] for pair in matched),
                   "per_pair_retained": record["pooled_candidates"],
                   "generated": len(record["entries"]), "fitted_children": len(children),
                   "fit_attempts": len(fit_rows),
                   "axis_cap_exclusions": sum(
                       axis["diagnostics"]["axis_cap_excluded"]
                       for pair in matched for axis in pair["role"]["axes"]
                   ),
                   "per_pair_cap_reached": sum(pair["per_pair_cap_reached"] for pair in matched),
                   "global_cap_reached": record["global_cap_reached"]},
        "reference_corners_native_px": reference_corners.tolist(),
        "selected_key": selected_key, "selected_reference_error": selected_error,
        "selected_saved_error": selected_saved_error,
        "oracle_generated": best_agreement(parents, selectable_keys, reference_corners, verifier),
        "oracle_refitted": best_agreement(children, selectable_keys, reference_corners, verifier),
        "ranking_status": ranking["status"],
        "generation_record": f"generation/{arm}/{case_id}.json.gz",
        "generation_log": f"generation_logs/{arm}/{case_id}.log",
        "candidates": [
            {field: candidate.get(field) for field in
             ("origin_key", "candidate_id", "kind", "parent_origin_key", "corners_px", "hard_valid", "camera_eligible")}
            for candidate in parents + children
        ],
        "fit_rows": fit_rows,
    }
    write(output / summary["generation_record"], record)
    write(output / "cases" / arm / f"{case_id}.json.gz", summary)
    return {"case_id": case_id, "arm": arm, "counts": summary["counts"], "timing": summary["timing"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--max-matched-pairs", type=int, help="Local smoke only; omit for the experiment")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Fresh output directory required: {output}")
    cases = args.case or list(CASES)
    arms = args.arm or list(ARMS)
    jobs = [(case_id, arm) for case_id in cases for arm in arms]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), max_tasks_per_child=1) as executor:
        futures = {executor.submit(run_one, case_id, arm, output, args.max_matched_pairs): (case_id, arm)
                   for case_id, arm in jobs}
        for future in as_completed(futures):
            case_id, arm = futures[future]
            result = future.result()
            results.append(result)
            print(f"{case_id} {arm}: generated {result['counts']['generated']}, "
                  f"refitted {result['counts']['fitted_children']}", flush=True)
    write(output / "completion.json.gz", {"schema": "svd-search-completion/1", "results": results})


if __name__ == "__main__":
    main()
