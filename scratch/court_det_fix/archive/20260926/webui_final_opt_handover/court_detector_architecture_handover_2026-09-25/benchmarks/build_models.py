#!/usr/bin/env python3
"""Build the committed-data performance tables and simple latency models."""
from __future__ import annotations

import csv
from pathlib import Path
import argparse

import matplotlib.pyplot as plt

CURRENT_TOTAL_S = 8250.0
CURRENT_VIEWS = 28
CURRENT_SLOWEST_S = 666.0
FINITE_SCORES_S = 2054.0


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def amdahl(total: float, parallel_fraction: float, workers: int,
           parallel_overhead: float = 0.10, fixed_overhead_s: float = 10.0) -> float:
    return total * ((1.0 - parallel_fraction) + parallel_fraction * (1.0 + parallel_overhead) / workers) + fixed_overhead_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    data = args.root / "data"
    figures = args.root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    top_level = [
        {"stage": "search_populations", "seconds": 5319, "share_of_total": 5319 / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"},
        {"stage": "w5_record", "seconds": 2754, "share_of_total": 2754 / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"},
        {"stage": "stripe_polarity_refit", "seconds": 48, "share_of_total": 48 / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"},
        {"stage": "net_choice", "seconds": 47, "share_of_total": 47 / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"},
        {"stage": "prepare_view", "seconds": 27, "share_of_total": 27 / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"},
        {"stage": "startup_and_unlisted", "seconds": CURRENT_TOTAL_S - (5319 + 2754 + 48 + 47 + 27),
         "share_of_total": (CURRENT_TOTAL_S - (5319 + 2754 + 48 + 47 + 27)) / CURRENT_TOTAL_S,
         "source": "derived"},
    ]
    write_csv(data / "current_top_level_stage_times.csv", top_level)

    detailed = [
        ("finite_scores", 2054), ("match_axis", 1384), ("stripe_measure", 1156),
        ("geometry", 403), ("physical_marking_evidence", 392), ("write_json_gz", 334),
        ("refit_refine", 325), ("combine", 286), ("canonicalise", 230),
        ("geometry_and_support", 216), ("pair_own_time", 175), ("stripe_score_model", 172),
        ("vp_select", 127), ("write_population", 120), ("player_fractions", 94),
        ("read_json_gz", 95), ("template_camera_check", 92), ("retain", 86),
        ("distance_maps", 83), ("prepare_observations", 21),
    ]
    write_csv(data / "current_detailed_stage_times.csv", [
        {"stage": name, "seconds": seconds, "share_of_total": seconds / CURRENT_TOTAL_S,
         "source": "claude_evidence/w5_savings/carmack_stage_times_vs_item12.txt"}
        for name, seconds in detailed
    ])

    prefilter = [
        {"samples_per_marking": 4, "fixed_k": 1024, "shortlists_identical": 1102, "calls": 3369, "cost_ratio": 0.32},
        {"samples_per_marking": 4, "fixed_k": 2048, "shortlists_identical": 1552, "calls": 3369, "cost_ratio": 0.36},
        {"samples_per_marking": 4, "fixed_k": 4096, "shortlists_identical": 2089, "calls": 3369, "cost_ratio": 0.41},
        {"samples_per_marking": 4, "fixed_k": 8192, "shortlists_identical": 2624, "calls": 3369, "cost_ratio": 0.51},
        {"samples_per_marking": 8, "fixed_k": 1024, "shortlists_identical": 1444, "calls": 3369, "cost_ratio": 0.38},
        {"samples_per_marking": 8, "fixed_k": 2048, "shortlists_identical": 2445, "calls": 3369, "cost_ratio": 0.41},
        {"samples_per_marking": 8, "fixed_k": 4096, "shortlists_identical": 3099, "calls": 3369, "cost_ratio": 0.47},
        {"samples_per_marking": 8, "fixed_k": 8192, "shortlists_identical": 3321, "calls": 3369, "cost_ratio": 0.57},
        {"samples_per_marking": 16, "fixed_k": 1024, "shortlists_identical": 2471, "calls": 3369, "cost_ratio": 0.47},
        {"samples_per_marking": 16, "fixed_k": 2048, "shortlists_identical": 3291, "calls": 3369, "cost_ratio": 0.51},
        {"samples_per_marking": 16, "fixed_k": 4096, "shortlists_identical": 3368, "calls": 3369, "cost_ratio": 0.57},
        {"samples_per_marking": 16, "fixed_k": 8192, "shortlists_identical": 3369, "calls": 3369, "cost_ratio": 0.66},
    ]
    for row in prefilter:
        row["replay_rate"] = row["shortlists_identical"] / row["calls"]
        row["projected_28_view_total_s_if_only_finite_scores_changes"] = (
            CURRENT_TOTAL_S - FINITE_SCORES_S + FINITE_SCORES_S * row["cost_ratio"]
        )
        row["projected_whole_run_saving_fraction"] = 1 - row["projected_28_view_total_s_if_only_finite_scores_changes"] / CURRENT_TOTAL_S
        row["source"] = "claude_evidence/prefilter/summary.txt"
    write_csv(data / "committed_prefilter_tradeoff.csv", prefilter)

    bound_rows = [
        (94524, 4.881180486998346, 73472, 2.7297523069973977),
        (104538, 4.201157962001162, 32768, 2.5159046949993353),
        (20378, 1.0651822479994735, 17920, 0.5287496330020076),
        (13402, 0.7679420029999164, 8192, 0.33427899399976013),
        (53408, 2.9777525050012628, 5888, 1.5889151160008623),
        (990, 0.05931183099892223, 990, 0.026220087998808594),
    ]
    bound_output = []
    for courts, exact_s, must_score, bound_s in bound_rows:
        model_s = bound_s + exact_s * must_score / courts
        bound_output.append({
            "courts": courts, "full_exact_s": exact_s, "bound_s": bound_s,
            "courts_must_score_exact": must_score, "must_score_fraction": must_score / courts,
            "modeled_bound_plus_tail_s": model_s, "modeled_ratio_to_full": model_s / exact_s,
            "source": "claude_evidence/shortlist_bound/gap_bound_rows.jsonl",
        })
    write_csv(data / "committed_shortlist_bound_rows.csv", bound_output)

    cascade_scenarios = []
    for k, ratio, matches in ((1024, 0.47, 2471), (2048, 0.51, 3291), (4096, 0.57, 3368), (8192, 0.66, 3369)):
        total = CURRENT_TOTAL_S - FINITE_SCORES_S + FINITE_SCORES_S * ratio
        cascade_scenarios.append({
            "scenario": f"16-sample fixed K={k}", "finite_scores_cost_ratio": ratio,
            "assumed_fallback_rate": 0.0, "projected_28_view_total_s": total,
            "projected_saving_fraction": 1 - total / CURRENT_TOTAL_S,
            "pair_shortlists_identical": matches, "pair_calls": 3369,
            "quality_note": "empirical pair-shortlist replay; not an unseen-view guarantee",
        })
    # For K=4096, a full fallback completes the unscored tail after paying the coarse pass.
    # The measured two-pass cost is 0.57. Completing all remaining exact work is modelled as 1.43.
    for fallback in (0.05, 0.10, 0.25, 0.50):
        ratio = 0.57 + (1.43 - 0.57) * fallback
        total = CURRENT_TOTAL_S - FINITE_SCORES_S + FINITE_SCORES_S * ratio
        cascade_scenarios.append({
            "scenario": "16-sample K=4096 with completion fallback",
            "finite_scores_cost_ratio": ratio, "assumed_fallback_rate": fallback,
            "projected_28_view_total_s": total, "projected_saving_fraction": 1 - total / CURRENT_TOTAL_S,
            "pair_shortlists_identical": "depends on fallback trigger", "pair_calls": 3369,
            "quality_note": "model only; fallback trigger not evaluated in committed data",
        })
    write_csv(data / "cascade_scenario_model.csv", cascade_scenarios)

    amdahl_rows = []
    average = CURRENT_TOTAL_S / CURRENT_VIEWS
    for parallel_fraction in (0.75, 0.85, 0.92):
        for workers in (4, 8, 16, 24):
            amdahl_rows.append({
                "parallel_fraction": parallel_fraction, "workers": workers,
                "parallel_overhead_fraction": 0.10, "fixed_scheduler_overhead_s": 10,
                "modeled_average_scene_s": amdahl(average, parallel_fraction, workers),
                "modeled_slowest_scene_s": amdahl(CURRENT_SLOWEST_S, parallel_fraction, workers),
                "baseline_average_scene_s": average, "baseline_slowest_scene_s": CURRENT_SLOWEST_S,
                "note": "Amdahl model, not measured parallel execution",
            })
    write_csv(data / "parallel_latency_model.csv", amdahl_rows)

    combined_rows = []
    for algorithmic_ratio in (0.90, 0.85, 0.80):
        for workers in (8, 16, 24):
            combined_rows.append({
                "algorithmic_runtime_ratio": algorithmic_ratio,
                "parallel_fraction": 0.92,
                "workers": workers,
                "modeled_slowest_scene_s": amdahl(CURRENT_SLOWEST_S * algorithmic_ratio, 0.92, workers),
                "note": "optimistic parallel fraction; target-sizing model, not measurement",
            })
    write_csv(data / "combined_target_sizing_model.csv", combined_rows)

    # Figure 1: current top-level stage share.
    stages = [row["stage"] for row in top_level]
    seconds = [row["seconds"] for row in top_level]
    plt.figure(figsize=(9, 4.8))
    plt.barh(stages[::-1], seconds[::-1])
    plt.xlabel("Summed seconds over 28 views")
    plt.title("Current D17 top-level runtime (8,250 s total)")
    plt.tight_layout()
    plt.savefig(figures / "current_top_level_runtime.png", dpi=160)
    plt.close()

    # Figure 2: fixed-K coarse cascade cost and replay rate.
    selected = [row for row in prefilter if row["samples_per_marking"] == 16]
    ks = [row["fixed_k"] for row in selected]
    projected = [row["projected_28_view_total_s_if_only_finite_scores_changes"] for row in selected]
    plt.figure(figsize=(8, 4.8))
    plt.plot(ks, projected, marker="o")
    plt.axhline(CURRENT_TOTAL_S, linestyle="--", label="current 8,250 s")
    plt.xscale("log", base=2)
    plt.xlabel("Exact-score proposal budget K per pair")
    plt.ylabel("Projected 28-view total seconds")
    plt.title("16-sample proposal cascade: runtime projection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "cascade_runtime_projection.png", dpi=160)
    plt.close()

    replay = [100 * row["replay_rate"] for row in selected]
    plt.figure(figsize=(8, 4.8))
    plt.plot(ks, replay, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Exact-score proposal budget K per pair")
    plt.ylabel("Pair shortlists reproduced (%)")
    plt.ylim(70, 100.2)
    plt.title("Committed 28-view replay: proposal recall")
    plt.tight_layout()
    plt.savefig(figures / "cascade_shortlist_replay.png", dpi=160)
    plt.close()

    # Figure 3: modeled slowest latency under deterministic task parallelism.
    plt.figure(figsize=(8, 4.8))
    for parallel_fraction in (0.75, 0.85, 0.92):
        subset = [row for row in amdahl_rows if row["parallel_fraction"] == parallel_fraction]
        plt.plot([row["workers"] for row in subset], [row["modeled_slowest_scene_s"] for row in subset],
                 marker="o", label=f"{int(parallel_fraction * 100)}% parallel")
    plt.axhline(90, linestyle="--", label="90 s target")
    plt.xlabel("Worker processes")
    plt.ylabel("Modeled slowest-scene seconds")
    plt.title("Amdahl target-sizing model (current slowest = 666 s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "modeled_slowest_latency.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
