"""Compare completed fits with references that never enter the polarity decision."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from run_probe import BASE, ROOT, read


def main() -> None:
    from numeric_fit import EDGE_NAMES, reference_disagreement

    results = read(Path(__file__).with_name("results.json.gz"))
    manifest = {case["case_id"]: case for case in read(BASE / "manifest.json.gz")["cases"]}
    cases = []
    for case in results["cases"]:
        reference = read(ROOT / manifest[case["case_id"]]["source_pack"])["references"][case["case_id"]]
        scale = np.asarray(case["working_size"]) / case["native_size"]
        arms = {}
        for name in ("baseline", "polarity", "selected"):
            measurement = case["selected"] if name == "selected" else case["fits"][name]["measurement"]
            arms[name] = reference_disagreement(measurement["corners_native_px"], reference, scale)
        cases.append({"case_id": case["case_id"], "reference": reference, "arms": arms})
    broadcast = [case for case in cases if case["case_id"].startswith("shuttleset_")]
    medians = {}
    for arm in ("baseline", "polarity", "selected"):
        medians[arm] = {}
        for edge in EDGE_NAMES:
            distances = [case["arms"][arm]["edge_disagreement_working"][edge]["mean_signed_distance"]
                         for case in broadcast]
            medians[arm][edge] = float(np.median(distances))
    result = {
        "reference_limit": "Six scenes share one static video grid; this is agreement, not independent scene accuracy.",
        "edge_sign": "Positive towards reference court interior, in working pixels.",
        "broadcast_case_count": len(broadcast), "broadcast_median_signed_edges": medians, "cases": cases,
    }
    output = Path(__file__).with_name("reference_comparison.json.gz")
    output.write_bytes(gzip.compress(json.dumps(result, allow_nan=False, indent=2).encode(), mtime=0))
    print(json.dumps(medians, indent=2))


if __name__ == "__main__":
    main()
