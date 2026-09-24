"""Summarise a D17 timing run into plain-language stage buckets per budget and scene group (throwaway).

Usage: d17_report.py RUN_DIR
Reads RUN_DIR/budget{16,12}/d17/*.json.gz and RUN_DIR/comparison.json.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BUDGETS = ("16", "12")

# Each bucket: (plain description, deployment role, stage paths whose seconds it sums).
# "needed" = a deployed detector does this work; "research" = only the evaluation pipeline does;
# "likely avoidable" = pool evidence that never reaches the accepted choice (FOLLOWUPS item 7).
BUCKETS = [
    ("startup", "start Python, import numpy/OpenCV, load detector code", "needed once per process", None),
    ("prepare_view", "load the frozen line fragments and frame for the view", "needed", ["prepare_view"]),
    ("direction_estimate", "estimate vanishing directions, shared by G0 and G1", "needed",
     ["populations/direction_estimate"]),
    ("G0_search", "G0: try direction pairs and keep the best 256 courts", "needed", None),
    ("G0_pool_evidence", "G0: legacy evidence on the kept 256 courts", "likely avoidable",
     ["populations/G0_generation/pool_evidence"]),
    ("G1_search", "G1: same search on paint-filtered fragments", "needed", None),
    ("G1_pool_evidence", "G1: legacy evidence on the kept 256 courts", "likely avoidable",
     ["populations/G1_generation/pool_evidence"]),
    ("paint_filter_and_other", "paint mask for G1 and other population glue", "needed", None),
    ("population_io", "write and re-read G0/G1 population files", "research",
     ["populations/write_population", "populations/read_json_gz"]),
    ("line_templates", "seeded line templates: candidate courts built from line-crossing rectangles",
     "needed", ["w5_record/line_templates"]),
    ("w5_measure", "W5: measure paint and stripe evidence for every merged candidate", "needed",
     ["w5_record/make_parent_record"]),
    ("w5_refit", "W5: refit every valid candidate to its stripes and re-measure the result", "needed",
     ["w5_record/attempt_refit"]),
    ("w5_rank_and_merge", "W5: merge duplicates, gate and rank", "needed", None),
    ("w5_research", "W5: determinism re-check, control entry, candidate review, sensitivity rankings",
     "research", ["w5_record/permutation_determinism", "w5_record/load_control_entry",
                  "w5_record/candidate_review", "w5_record/rank_sensitivity"]),
    ("w5_io", "W5: re-prepare view, write full record and arrays", "research",
     ["w5_record/prepare_view", "w5_record/write_json_gz", "w5_record/write_arrays",
      "w5_record/read_json_gz", "w5_record/load_runtime"]),
    ("net_choice", "bounded net choice: post evidence for full-court candidates, then choose", "needed", None),
    ("net_choice_io", "re-read the W5 record for the net choice", "research", ["net_choice/read_json_gz"]),
    ("refit_replay", "polarity refit: reload record and view, check the saved fit replays", "research", None),
    ("refit_correct", "polarity refit: infer stripe polarity, refit, validate", "needed", None),
]


def seconds(stages: dict, path: str) -> float:
    return stages.get(path, {}).get("seconds", 0.0)


def buckets_for(summary: dict) -> dict[str, float]:
    stages = summary["stages"]
    values = {name: sum(seconds(stages, path) for path in paths) for name, _, _, paths in BUCKETS if paths}
    values["startup"] = sum(summary["startup_s"].values())
    for label, search, pool in (("G0_generation", "G0_search", "G0_pool_evidence"),
                                ("G1_generation", "G1_search", "G1_pool_evidence")):
        values[search] = seconds(stages, f"populations/{label}") - values[pool]
    values["paint_filter_and_other"] = (
        seconds(stages, "populations") - values["direction_estimate"] - values["G0_search"]
        - values["G0_pool_evidence"] - values["G1_search"] - values["G1_pool_evidence"] - values["population_io"]
    )
    values["w5_rank_and_merge"] = (
        seconds(stages, "w5_record") - values["line_templates"] - values["w5_measure"] - values["w5_refit"]
        - values["w5_research"] - values["w5_io"]
    )
    values["net_choice"] = seconds(stages, "net_choice") - values["net_choice_io"]
    polarity = summary["selection"]["polarity_refit"]
    split = None if polarity is None else polarity.get("timings_seconds")
    refit_total = seconds(stages, "stripe_polarity_refit")
    values["refit_replay"] = 0.0 if split is None else refit_total - split["automatic_and_corrected"]
    values["refit_correct"] = 0.0 if split is None else split["automatic_and_corrected"]
    return values


def group_of(case_id: str, non_court: bool) -> str:
    if non_court:
        return "non-court"
    if case_id.startswith("gxBQ"):
        return "GX"
    if case_id.startswith(("am", "letterboxed")):
        return "amateur"
    return "broadcast"


def main(run_dir: Path) -> None:
    comparison = json.loads((run_dir / "comparison.json").read_text())
    non_court = {row["case_id"]: row["non_court"] for row in comparison["cases"]}
    per_view: dict[str, dict[str, dict]] = defaultdict(dict)
    for budget in BUDGETS:
        for path in sorted((run_dir / f"budget{budget}/d17").glob("*.json.gz")):
            with gzip.open(path, "rt") as stream:
                summary = json.load(stream)
            case_id = summary["case_id"]
            per_view[case_id][budget] = {
                "buckets": buckets_for(summary),
                "wall": summary["run_wall_s"] + summary["startup_s"].get("interpreter_start", 0.0),
                "cpu": summary["run_cpu_s"],
                "rss": summary["peak_rss_mb"],
                "matched": {name: value["pair_statuses"].get("matched", 0) for name, value in summary["populations"].items()},
            }

    groups = sorted({group_of(case_id, non_court[case_id]) for case_id in per_view})
    print("## Wall seconds per view (startup included), budget 16 / budget 12")
    for group in groups:
        cases = [case_id for case_id in per_view if group_of(case_id, non_court[case_id]) == group]
        for budget in BUDGETS:
            walls = [per_view[case_id][budget]["wall"] for case_id in cases if budget in per_view[case_id]]
            print(f"{group:10s} budget {budget}: views {len(walls)}, total {sum(walls):7.0f}, "
                  f"median {statistics.median(walls):6.0f}, max {max(walls):6.0f}")

    print("\n## Stage buckets, seconds summed over court views (non-court shown separately)")
    header = "bucket | role | " + " | ".join(f"{group} {budget}" for group in groups for budget in BUDGETS)
    print(header)
    for name, description, role, _ in BUCKETS:
        cells = []
        for group in groups:
            for budget in BUDGETS:
                total = sum(per_view[case_id][budget]["buckets"][name] for case_id in per_view
                            if group_of(case_id, non_court[case_id]) == group and budget in per_view[case_id])
                cells.append(f"{total:7.1f}")
        print(f"{name} ({description}) | {role} | " + " | ".join(cells))

    print("\n## Role totals per view (median over court views), budget 16 / 12")
    for budget in BUDGETS:
        by_role: dict[str, list[float]] = defaultdict(list)
        for case_id, arms in per_view.items():
            if non_court[case_id] or budget not in arms:
                continue
            role_sums: dict[str, float] = defaultdict(float)
            for name, _, role, _ in BUCKETS:
                role_sums[role] += arms[budget]["buckets"][name]
            for role, total in role_sums.items():
                by_role[role].append(total)
        print(f"budget {budget}: " + ", ".join(
            f"{role} median {statistics.median(values):.0f} s (total {sum(values):.0f})" for role, values in by_role.items()))

    print("\n## Per view: wall 16 / 12, G0+G1 search 16 / 12, matched pairs G0 16 / 12, peak RSS MB")
    for case_id in sorted(per_view, key=lambda key: (group_of(key, non_court[key]), key)):
        arms = per_view[case_id]
        if set(arms) != set(BUDGETS):
            print(case_id, "missing arm", sorted(arms))
            continue
        search = {budget: arms[budget]["buckets"]["G0_search"] + arms[budget]["buckets"]["G1_search"] for budget in BUDGETS}
        print(f"{case_id} | {arms['16']['wall']:.0f} / {arms['12']['wall']:.0f} | {search['16']:.0f} / {search['12']:.0f} | "
              f"{arms['16']['matched'].get('G0')} / {arms['12']['matched'].get('G0')} | "
              f"{arms['16']['rss']:.0f} / {arms['12']['rss']:.0f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
