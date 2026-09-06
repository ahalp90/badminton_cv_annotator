import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# This script regenerates the simple report figures.
# Final comparisons read measured counts; development-only charts use saved summaries.
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)

def simple_bar(labels, values, title, ylabel, name, percent=False):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for b, v in zip(bars, values):
        text = f"{v:.1f}%" if percent else f"{int(v):,}"
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), text,
                ha="center", va="bottom")
    save(fig, name)

def comparison_chart(
    panels: list[tuple[str, list[str], list[float], list[float]]],
    title: str,
    ylabel: str,
    name: str,
    out: Path,
    percent: bool = True,
) -> None:
    """Use the same population colours and labels across every comparison."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.8), squeeze=False)
    for ax, (panel_title, labels, trusted, all_gt) in zip(axes[0], panels):
        positions = np.arange(len(labels))
        for offset, values, label, colour in (
            (-0.19, trusted, "Trusted GT", "#1f77b4"),
            (0.19, all_gt, "All GT", "#ff7f0e"),
        ):
            bars = ax.bar(positions + offset, values, width=0.38, label=label, color=colour)
            labels_on_bars = [f"{value:.1f}%" if percent else f"{value:,.0f}" for value in values]
            ax.bar_label(bars, labels=labels_on_bars, padding=3, fontsize=9)
        ax.set_xticks(positions, labels)
        ax.set_title(panel_title)
        ax.set_ylim(0, 112 if percent else max(trusted + all_gt) * 1.2)
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    fig.savefig(out / f"{name}.png", dpi=180)
    svg = out / f"{name}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def regenerate_metric_figures(result: Mapping[str, Any], out: Path = OUT) -> None:
    """Build final comparisons from the same measured counts as the reference table."""
    counts = [result["contacts"][population]["10"] for population in ("retained", "all_gt")]
    panels = []
    for title, key in (("Timing only", "matched"), ("Timing + correct player", "side_correct")):
        values = []
        for population in counts:
            correct, predicted, labelled = population[key], population["predicted"], population["labelled"]
            values.append([100 * correct / predicted, 100 * correct / labelled, 200 * correct / (predicted + labelled)])
        panels.append((title, ["Precision", "Recall", "F1"], *values))
    comparison_chart(panels, "Final detector: contact performance at ±10", "Percent", "contact_prf", out)

    panels = []
    for title, contact_key, serve_key in (
        ("Timing only", "matched", "serve_matched"),
        ("Timing + correct player", "side_correct", "serve_side_correct"),
    ):
        values = []
        for population in counts:
            nonserve_labels = population["labelled"] - population["labelled_serves"]
            nonserve = 100 * (population[contact_key] - population[serve_key]) / nonserve_labels
            serve = 100 * population[serve_key] / population["labelled_serves"]
            values.append([nonserve, serve])
        panels.append((title, ["Non-serve", "Serve"], *values))
    comparison_chart(panels, "Final detector: contact recall at ±10", "Recall (%)", "contact_recovery", out)

    serve_values = [[100 * population[key] / population["labelled_serves"]
                     for key in ("serve_matched", "serve_side_correct")] for population in counts]
    comparison_chart([("", ["Serve found", "Serve + correct server"], *serve_values)],
                     "Final detector: serve recall at ±10", "Recall (%)", "serve_summary", out)

    selection_values = []
    for population in ("retained", "all_gt"):
        selected = result["selected"][population]["10"]
        judgeable = selected["proposals"] - (selected["unknown"] if population == "retained" else 0)
        values = []
        for key in ("unique_complete", "unique_contained"):
            values.extend([100 * selected[key] / selected["labelled_rallies"], 100 * selected[key] / judgeable])
        selection_values.append(values)
    comparison_chart([("", ["Fully correct\nrecall", "Fully correct\nprecision",
                            "Whole-rally\nrecall", "Whole-rally\nprecision"], *selection_values)],
                     "Selected rallies at ±10", "Percent", "automatic_use", out)

    for name, title, stages, labels in (
        ("system_progression", "Detector progression", ["original", "combined", "later", "boundaries", "recommended"],
         ["Original", "Whole-sequence\nselection", "+ later contact", "Boundary fix\nonly", "Final detector"]),
        ("broader_gain", "47-video whole-sequence result", ["original", "opening", "combined"],
         ["Original detector", "First-contact\nrepair only", "Whole-sequence\nmodel"]),
        ("final_followup", "Final detector refinements", ["later", "local", "boundaries", "recommended", "early"],
         ["Starting point", "Local inserted-\ncontact score", "Boundary fix\nonly",
          "Final detector", "More serve\ncandidates"]),
    ):
        values = [[result["stages"][stage][population]["10"]["unique_complete"] for stage in stages]
                  for population in ("retained", "all_gt")]
        comparison_chart([("", labels, *values)], title, "Fully correct rallies at ±10", name, out, percent=False)


def run():
    with gzip.open(ROOT / "results/metric_summary.json.gz", "rt") as source:
        regenerate_metric_figures(json.load(source))
    simple_bar(
        ["Original", "Rally summaries", "+ first contact\n+ player evidence",
         "+ physical\nmeasurements"],
        [182, 191, 233, 235],
        "Whole-sequence experiment — trusted GT only",
        "Perfect rallies at ±10", "whole_sequence_comparison")
    labels = ["1–5", "6–10", "11–20", "21+"]
    before = np.array([462, 441, 394, 138])
    after = np.array([465, 480, 462, 190])
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    w = 0.36
    ax.bar(x-w/2, before, w, label="Before")
    ax.bar(x+w/2, after, w, label="After")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Contacts in labelled rally")
    ax.set_ylabel("Perfect rallies")
    ax.set_title("Later-contact repair by rally length — trusted GT only")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    save(fig, "later_by_length")

    simple_bar(
        ["Extra contact(s)", "Misses serve", "Misses later\ncontact",
         "Player error", "Partial or\nmerged rally"],
        [92, 43, 39, 10, 12],
        "Why 124 selected proposals fail strict scoring — trusted GT only\n(categories overlap)",
        "Selected proposals", "selected_errors")

    # Promising-leads: scale of the open backlog.
    labels = [
        "Later contacts absent\nfrom saved candidate data",
        "Missed serves where a useful\ncandidate was already available",
        "Right-rally selections needing\ncontact cleanup",
        "Selected outputs needing\nGT review",
    ]
    values = [1072, 243, 112, 44]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cases identified in the saved diagnostics")
    ax.set_title("Where the strongest remaining opportunities are")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y()+bar.get_height()/2, f" {v:,}", va="center", ha="left")
    ax.text(0, -0.18,
            "Counts come from different diagnostic populations; compare their scale, not their rates.",
            transform=ax.transAxes, fontsize=9)
    save(fig, "promising_opportunities")

    # Promising-leads: overlapping contact-level failure modes.
    labels = ["Extra predicted contact(s)", "Misses the serve",
              "Misses a later contact", "Wrong/missing player"]
    values = [80, 31, 32, 7]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Selected near-misses affected")
    ax.set_title("Why 112 correct-rally selections still fail exact scoring")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, values):
        ax.text(v, bar.get_y()+bar.get_height()/2, f" {v}", va="center", ha="left")
    ax.text(0, -0.18,
            "Categories overlap: one rally can miss a real contact and also contain an extra prediction.",
            transform=ax.transAxes, fontsize=9)
    save(fig, "near_miss_errors")

    # Contact-level progression.
    stages = ["Whole-sequence\nmodel", "+ later-contact\nrepair", "Final detector"]
    timing_prog = np.array([87.3, 88.0, 88.2])
    side_prog = np.array([80.8, 85.0, 85.5])
    x = np.arange(len(stages))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5))
    b1 = ax.bar(x-w/2, timing_prog, w, label="Timing recall")
    b2 = ax.bar(x+w/2, side_prog, w, label="Timing + correct-player recall")
    ax.set_xticks(x, stages)
    ax.set_ylim(70, 92)
    ax.set_ylabel("Recall against trusted GT (%)")
    ax.set_title("Contact-level progress during the closing pass")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    save(fig, "contact_progression")

if __name__ == "__main__":
    run()
