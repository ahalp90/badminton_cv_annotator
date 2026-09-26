"""Compare fresh D17 runs at direction budgets 16 and 12 (SVD12), and with the accepted gallery.

Per view: run time and stage split for each budget; whether the bounded choice and the
corrected court agree between budgets; reference errors where the frame has a reference,
using the D17 statistics' measure in working pixels; and the corrected-corner difference
from the accepted 20-case gallery. Non-court controls report whether each budget picked a
court at all.

Usage: compare_d17.py RUN_DIR ACCEPTED_RESULTS OUTPUT_JSON
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (16, 12)

sys.path.insert(0, str(ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def corner_difference_working_px(first: list, second: list, scale: np.ndarray) -> float:
    """Largest corner distance between two courts, allowing the 180-degree corner-order swap."""
    first_corners = np.asarray(first, dtype=float) / scale
    second_corners = np.asarray(second, dtype=float) / scale
    direct = np.linalg.norm(first_corners - second_corners, axis=1).max()
    rotated = np.linalg.norm(first_corners[[2, 3, 0, 1]] - second_corners, axis=1).max()
    return float(min(direct, rotated))


def corrected_geometry(polarity: dict | None) -> dict | None:
    """The corrected court when the stripe-polarity refit ran and gave a valid court."""
    if polarity is None or "error" in polarity or not polarity["corrected"]["valid"]:
        return None
    return polarity["corrected"]


def measurable_reference(case_id: str, manifest_rows: dict, references: dict) -> dict | None:
    """The reference the D17 statistics would measure against, or None (paired_reference_analysis.main)."""
    reference = references.get(case_id, {})
    status = reference.get("reference_status", manifest_rows[case_id].get("reference_status"))
    if status == "view_unverified" or not (reference.get("landmarks") or reference.get("corners_px")):
        return None
    return reference


def reference_summary(sizes: dict, geometry: dict | None, reference: dict | None) -> dict | None:
    if geometry is None or reference is None:
        return None
    metric, distances = statistics.reference_errors(sizes, geometry, reference)
    return {"metric": metric, "median_working_px": float(np.median(distances)),
            "max_working_px": float(np.max(distances))}


def arm_row(summary: dict, reference: dict | None) -> dict:
    stages = summary["stages"]
    selection = summary["selection"]
    corrected = corrected_geometry(selection["polarity_refit"])
    sizes = {"native_size_wh": summary["native_size_wh"], "working_size_wh": summary["working_size_wh"]}
    return {
        "wall_s": summary["run_wall_s"] + summary["startup_s"].get("interpreter_start", 0.0),
        "startup_s": sum(summary["startup_s"].values()),
        "phases_s": summary["phases_s"],
        "generation_s": sum(stages[f"populations/{name}"]["seconds"] for name in ("G0_generation", "G1_generation")),
        "matched_pairs": {name: value["pair_statuses"].get("matched", 0) for name, value in summary["populations"].items()},
        "union_candidates": summary["population_counts"]["union"],
        "peak_rss_mb": summary["peak_rss_mb"],
        "bounded": selection["bounded"],
        "refit_error": None if selection["polarity_refit"] is None else selection["polarity_refit"].get("error"),
        "refit_split_s": None if selection["polarity_refit"] is None
        else selection["polarity_refit"].get("timings_seconds"),
        "corrected_valid": corrected is not None,
        "corrected_corners_native_px": None if corrected is None else corrected["corners_native_px"],
        "reference": reference_summary(sizes, corrected, reference),
        "smoke": summary["max_matched_pairs"] is not None,
    }


def main() -> None:
    run_dir, accepted_path, output_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    manifest = statistics.read(statistics.MANIFEST)
    manifest_rows = {row["case_id"]: row for row in manifest["cases"]}
    references = statistics.load_references(manifest)
    accepted = {row["case_id"]: row for row in read(accepted_path)["selections"]}
    case_ids = sorted({path.name.removesuffix(".json.gz")
                       for budget in BUDGETS for path in (run_dir / f"budget{budget}/d17").glob("*.json.gz")})

    rows = []
    stage_totals = {budget: defaultdict(lambda: [0.0, 0.0]) for budget in BUDGETS}
    for case_id in case_ids:
        reference = measurable_reference(case_id, manifest_rows, references)
        non_court = manifest_rows[case_id].get("reference_status") == "non_court"
        row = {"case_id": case_id, "non_court": non_court, "arms": {}}
        summaries = {}
        for budget in BUDGETS:
            path = run_dir / f"budget{budget}/d17/{case_id}.json.gz"
            if not path.exists():
                row["arms"][budget] = None
                continue
            summaries[budget] = read(path)
            row["arms"][budget] = arm_row(summaries[budget], reference)
            for stage, value in summaries[budget]["stages"].items():
                stage_totals[budget][stage][0] += value["seconds"]
                stage_totals[budget][stage][1] += value["self_seconds"]
        if len(summaries) == len(BUDGETS):
            scale = (np.asarray(summaries[16]["native_size_wh"], dtype=float)
                     / np.asarray(summaries[16]["working_size_wh"], dtype=float))
            full16, svd12 = row["arms"][16], row["arms"][12]
            row["same_bounded_choice"] = full16["bounded"] == svd12["bounded"]
            if full16["corrected_valid"] and svd12["corrected_valid"]:
                row["corrected_difference_working_px"] = corner_difference_working_px(
                    full16["corrected_corners_native_px"], svd12["corrected_corners_native_px"], scale)
            if case_id in accepted:
                accepted_geometry = corrected_geometry(accepted[case_id])
                sizes = {"native_size_wh": summaries[16]["native_size_wh"],
                         "working_size_wh": summaries[16]["working_size_wh"]}
                row["accepted"] = {"bounded": accepted[case_id]["selected_origin_key"],
                                   "reference": reference_summary(sizes, accepted_geometry, reference)}
                for budget in BUDGETS:
                    corners = row["arms"][budget]["corrected_corners_native_px"]
                    if accepted_geometry is not None and corners is not None:
                        row["accepted"][f"difference_budget{budget}_working_px"] = corner_difference_working_px(
                            corners, accepted_geometry["corners_native_px"], scale)
        rows.append(row)

    totals = {}
    for budget in BUDGETS:
        arms = [row["arms"][budget] for row in rows if row["arms"].get(budget)]
        totals[budget] = {"views": len(arms), "wall_s": sum(arm["wall_s"] for arm in arms),
                          "generation_s": sum(arm["generation_s"] for arm in arms),
                          "smoke_views": sum(arm["smoke"] for arm in arms),
                          "stages": {stage: {"seconds": seconds, "self_seconds": self_seconds}
                                     for stage, (seconds, self_seconds) in sorted(stage_totals[budget].items())}}
    report = {"schema": "d17-timing-comparison/1", "run_dir": str(run_dir), "totals": totals, "cases": rows}
    output_path.write_text(json.dumps(report, indent=1))

    for budget in BUDGETS:
        total = totals[budget]
        print(f"budget {budget}: {total['views']} views, wall {total['wall_s']:.0f} s, "
              f"G0+G1 generation {total['generation_s']:.0f} s, smoke views {total['smoke_views']}")
    print("\ncase | wall 16 / 12 s | picked 16 / 12 | bounded same | corrected diff px | ref median 16 / 12 / accepted | vs accepted 16 / 12 px")
    for row in rows:
        full16, svd12 = row["arms"].get(16), row["arms"].get(12)
        if full16 is None or svd12 is None:
            print(row["case_id"], "| missing arm")
            continue

        def median(arm_or_accepted: dict | None) -> str:
            reference = arm_or_accepted and arm_or_accepted.get("reference")
            return "-" if not reference else f"{reference['median_working_px']:.2f}"

        accepted_row = row.get("accepted", {})
        print(f"{row['case_id']}{' (non-court)' if row['non_court'] else ''} | "
              f"{full16['wall_s']:.0f} / {svd12['wall_s']:.0f} | "
              f"{full16['bounded'] is not None} / {svd12['bounded'] is not None} | {row['same_bounded_choice']} | "
              f"{row.get('corrected_difference_working_px', '-')} | "
              f"{median(full16)} / {median(svd12)} / {median(accepted_row)} | "
              f"{accepted_row.get('difference_budget16_working_px', '-')} / "
              f"{accepted_row.get('difference_budget12_working_px', '-')}")
        for budget, arm in ((16, full16), (12, svd12)):
            if arm["refit_error"] is not None:
                print(f"    budget {budget} stripe-polarity refit failed: {arm['refit_error']}")


if __name__ == "__main__":
    main()
