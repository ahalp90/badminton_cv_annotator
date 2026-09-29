"""Check frozen W5 inputs and equality when unused junction diagnostics are omitted."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from measurement import prepared_measurements

CASES = ("shuttleset_03_scene_0016", "gxBQ_window_00_frame_5", "shuttleset_21_scene_0010")
BASELINE = "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root.parents[1]), str(root.parents[1] / "src"), str(root / "w5_holistic")]
    import run_w5

    run_w5.load_runtime(root)
    import verifier

    cv2.setNumThreads(1)
    rows = []
    for case_id in CASES:
        context = verifier.prepare_view(root, case_id)
        record = verifier.read_json_gz(root / BASELINE / "case_records" / f"{case_id}.json.gz")
        selected = record["rankings"]["C"]["selected_origin_key"]
        candidates = record["parents"] + record["valid_children"]
        candidate = next(candidate for candidate in candidates if candidate["origin_key"] == selected)
        elapsed = {}
        started = perf_counter()
        full, full_arrays = verifier.measure_candidate(context, candidate, {})
        elapsed["full"] = perf_counter() - started
        with prepared_measurements(verifier) as counts:
            started = perf_counter()
            reduced, reduced_arrays = verifier.measure_candidate(context, candidate, {})
            elapsed["prepared_without_junctions"] = perf_counter() - started
        if counts["greyscale_conversions"] != 1:
            raise AssertionError(f"{case_id}: expected one greyscale conversion: {counts}")
        full.pop("junctions")
        reduced.pop("junctions")
        if verifier.jsonable(full) != verifier.jsonable(reduced):
            raise AssertionError(f"{case_id}: retained evidence changed")
        if full_arrays.keys() != reduced_arrays.keys():
            raise AssertionError(f"{case_id}: array keys changed")
        for name in full_arrays:
            np.testing.assert_array_equal(full_arrays[name], reduced_arrays[name])
        row = {
            "case_id": case_id, "candidate": selected,
            "source_memberships": candidate["source_memberships"],
            "frame": context.frame_relative_path, "image_kind": context.image_kind,
            "same_image_mask_available": context.same_image_mask_available,
            "mask_unavailable_reason": context.person_mask_unavailable_reason,
            "player_observed_frames": len(context.source["all_feet_px"]),
            "retained_evidence_and_arrays_equal": True, "elapsed_s": elapsed,
            "conversion_counts": counts,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump({"schema": "wider-w5-smoke/1", "cases": rows}, stream, indent=2)


if __name__ == "__main__":
    main()
