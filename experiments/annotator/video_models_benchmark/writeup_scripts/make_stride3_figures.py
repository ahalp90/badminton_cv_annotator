"""Draw the completed stride-three validation and test comparisons."""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "stride3_20261105"
FIGURES = ROOT / "writeup_figures"
BASELINE = ROOT.parent / "monotrack_benchmark/benchmark_results/hp_search/primary/contact_summary.json.gz"
METRICS = ("contact_f1", "correct_player_f1", "fully_correct_percent")
TITLES = ("Contact F1 (%)", "Correct-player F1 (%)", "Fully correct rallies (%)")
COLOURS = ("#0072B2", "#E69F00", "#8B5DA8", "#555555")


def draw_comparison(groups: list[tuple[str, list[list[float]], str]], filename: str, title: str) -> None:
    """Show each seed, its mean and the observed range for each metric."""
    figure, axes = plt.subplots(1, len(METRICS), figsize=(13, max(3.2, len(groups) * 0.55)))
    for metric_index, axis in enumerate(axes):
        for row_index, (_label, metrics, colour) in enumerate(groups):
            values = metrics[metric_index]
            axis.plot([min(values), max(values)], [row_index, row_index], color=colour, linewidth=2)
            axis.scatter(values, [row_index] * len(values), color=colour, marker="|", s=100, zorder=3)
            axis.scatter(mean(values), row_index, color=colour, marker="D", s=30, zorder=4)
        axis.set_title(TITLES[metric_index])
        axis.set_yticks(range(len(groups)), [group[0] for group in groups] if metric_index == 0 else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
    figure.subplots_adjust(left=0.27, right=0.99, bottom=0.13, top=0.82, wspace=0.12)
    figure.suptitle(title)
    figure.savefig(FIGURES / filename, dpi=180)
    plt.close(figure)


def main() -> None:
    with gzip.open(RESULTS / "scores.csv.gz", "rt") as stream:
        scores = list(csv.DictReader(stream))
    with gzip.open(RESULTS / "selection.json.gz", "rt") as stream:
        selection = json.load(stream)
    with gzip.open(RESULTS / "runs.json.gz", "rt") as stream:
        runs = json.load(stream)
    with gzip.open(BASELINE, "rt") as stream:
        baselines = json.load(stream)["summary"]
    FIGURES.mkdir(exist_ok=True)
    for model, name in (("x3d", "X3D-M"), ("videomae", "VideoMAE Base")):
        run_lookup = {run["name"]: run for run in runs[model]["runs"]}
        groups = []
        for config in selection[model]["ranking"]:
            selected_metrics = []
            for selected in config["runs"]:
                history = run_lookup[selected["name"]]["history"]
                best = next(check for check in history if check["steps"] == selected["best_step"])
                selected_metrics.append(best["events"]["10"])
            metrics = [[result[key] for result in selected_metrics] for key in METRICS]
            label = f"LR {config['learning_rate']:g}"
            if model == "x3d":
                radius = 1 if config["scheme"] == "A" else 2
                label = f"±{radius}, {config['contact_fraction']:.0%} contacts, {label}"
            groups.append((label, metrics, COLOURS[0]))
        draw_comparison(groups, f"{model}_stride3_validation.png", f"{name}: validation, forced alternation, ±10 frames")
        groups = []
        for split in ("validation", "test"):
            for rule, label, colour in (("raw", "raw hitters", COLOURS[0]),
                                        ("forced_alternation", "forced alternation", COLOURS[1])):
                rows = [row for row in scores if row["model"] == model and row["split"] == split
                        and row["hitter_rule"] == rule and row["tolerance"] == "10"]
                metrics = [[float(row[key]) for row in rows] for key in METRICS]
                groups.append((f"{split.title()}: {label}", metrics, colour))
        draw_comparison(groups, f"{model}_stride3_hitters.png", f"{name}: selected configuration, ±10 frames")
    groups = []
    for model, name, colour in (("x3d", "X3D-M", COLOURS[0]), ("videomae", "VideoMAE", COLOURS[1])):
        for rule, label in (("raw", "raw"), ("forced_alternation", "forced")):
            rows = [row for row in scores if row["model"] == model and row["split"] == "test"
                    and row["hitter_rule"] == rule and row["tolerance"] == "10"]
            groups.append((f"{name}: {label}", [[float(row[key]) for row in rows] for key in METRICS], colour))
    for key, name, colour in (("annotator_our_courts", "Our annotator", COLOURS[2]),
                              ("monotrack_own_courts", "MonoTrack", COLOURS[3])):
        metrics = [baselines[key]["10"][metric]["runs"] for metric in ("f1", "player_f1", "fully_correct")]
        groups.append((name, metrics, colour))
    draw_comparison(groups, "stride3_test_comparison.png", "Test results: supplied rally bounds, ±10 frames")


if __name__ == "__main__":
    main()
