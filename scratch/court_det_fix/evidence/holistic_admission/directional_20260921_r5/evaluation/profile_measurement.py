"""Time unused junction diagnostics without changing the saved W5 implementation."""

from __future__ import annotations

import cProfile
import gzip
import json
import pstats
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[6]
ROOT = REPO / "scratch/court_det_fix"
HERE = Path(__file__).resolve().parent
CASES = ("gxBQ_window_00_frame_0", "am2_window_00_frame_150", "shuttleset_03_scene_0016")


def main() -> None:
    sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "w5_holistic")]
    import run_w5

    run_w5.load_runtime(ROOT)
    import verifier

    cv2.setNumThreads(1)
    review_path = HERE.parent / "w5_directional_20260921_r5_43/review_candidates.json"
    review = json.loads(review_path.read_text())
    contexts = {case_id: verifier.prepare_view(ROOT, case_id) for case_id in CASES}
    original_junctions = verifier.raw_junctions
    outputs: dict[tuple[str, str, int], tuple[dict, dict]] = {}
    timings = []
    profiler = cProfile.Profile()
    try:
        for repetition in range(3):
            # Alternate order to reduce warm-cache bias. Each measurement gets
            # an empty candidate cache; frame preparation stays outside timing.
            modes = ("full", "without_junctions") if repetition % 2 == 0 else ("without_junctions", "full")
            for mode in modes:
                verifier.raw_junctions = original_junctions if mode == "full" else lambda *_args: {}
                for case_id in CASES:
                    selected = review[case_id]["selected"]["C"]
                    entry = review[case_id]["candidates"][selected]
                    started = perf_counter()
                    outputs[(mode, case_id, repetition)] = verifier.measure_candidate(contexts[case_id], entry, {})
                    timings.append({"case_id": case_id, "mode": mode, "repetition": repetition,
                                    "elapsed_s": perf_counter() - started})
        verifier.raw_junctions = original_junctions
        profiler.enable()
        for case_id in CASES:
            entry = review[case_id]["candidates"][review[case_id]["selected"]["C"]]
            verifier.measure_candidate(contexts[case_id], entry, {})
        profiler.disable()
    finally:
        verifier.raw_junctions = original_junctions

    for case_id in CASES:
        for repetition in range(3):
            full, full_arrays = outputs[("full", case_id, repetition)]
            reduced, reduced_arrays = outputs[("without_junctions", case_id, repetition)]
            full.pop("junctions")
            reduced.pop("junctions")
            if verifier.jsonable(full) != verifier.jsonable(reduced):
                raise AssertionError(f"non-junction evidence changed: {case_id}")
            if full_arrays.keys() != reduced_arrays.keys():
                raise AssertionError(f"array keys changed: {case_id}")
            for name in full_arrays:
                np.testing.assert_array_equal(full_arrays[name], reduced_arrays[name])

    stats = pstats.Stats(profiler)
    functions = []
    for (filename, line, function), (primitive_calls, calls, own, cumulative, _callers) in stats.stats.items():
        if "raw_junctions" in function or "cvtColor" in function or "grayscale_sample" in function:
            functions.append({"filename": filename, "line": line, "function": function,
                              "calls": calls, "primitive_calls": primitive_calls,
                              "own_s": own, "cumulative_s": cumulative})
    totals = {}
    for mode in ("full", "without_junctions"):
        totals[mode] = sum(row["elapsed_s"] for row in timings if row["mode"] == mode)
    result: dict[str, Any] = {
        "schema": "w5-measurement-timing/1", "cases": list(CASES), "repetitions": 3,
        "comparison_measurements_per_mode": 9, "totals_s": totals,
        "non_junction_evidence_and_arrays_equal": True,
        "profile_measurements": 3, "profile_total_s": stats.total_tt,
        "profile_functions": functions, "timings": timings,
        "scope": "Selected candidates only; excludes frame preparation, proposal generation and refitting.",
        "verifier_module_path": verifier.__file__,
    }
    with gzip.open(HERE / "measurement_timing.json.gz", "wt") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"totals_s": totals, "equal": True, "profile_functions": functions}, indent=2))


if __name__ == "__main__":
    main()
