# Metrics note: contact_prf/contact_recovery/contact_progression were regenerated in the metric-normalisation pass.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# This script regenerates the simple report figures.
# The numbers are the saved experiment results summarised in the Markdown reports.
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

def run():
    simple_bar(
        ["Original", "Whole-sequence\nselection", "+ later contact",
         "Boundary fix\nonly", "Final detector"],
        [995, 1435, 1597, 1732, 1763],
        "Detector progression — trusted GT only",
        "Perfect rallies at ±10", "system_progression")
    simple_bar(
        ["Original", "Rally summaries", "+ first contact\n+ player evidence",
         "+ physical\nmeasurements"],
        [182, 191, 233, 235],
        "Whole-sequence experiment — trusted GT only",
        "Perfect rallies at ±10", "whole_sequence_comparison")
    simple_bar(
        ["Original detector", "First-contact\nrepair only", "Whole-sequence\nmodel"],
        [995, 1105, 1435],
        "47-video whole-sequence result — trusted GT only",
        "Perfect rallies at ±10", "broader_gain")
    simple_bar(
        ["Starting point", "Local inserted-\ncontact score", "Boundary fix\nonly",
         "Final detector", "More serve\ncandidates"],
        [1597, 1622, 1732, 1763, 1767],
        "Final refinements — trusted GT only",
        "Perfect rallies at ±10", "final_followup")
    simple_bar(
        ["Serve found", "Serve +\ncorrect server"], [81.3, 77.4],
        "Final detector serve performance — trusted GT only\n(all-GT score was not measured)",
        "Percent of 3,422 trusted-GT rallies", "serve_summary", percent=True)

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

    metrics = ["Exact\nrecall", "Exact\nprecision", "Whole-rally\nrecall", "Whole-rally\nprecision"]
    trusted = np.array([18.0, 83.2, 21.3, 98.4])
    allgt = np.array([15.5, 78.4, 18.6, 94.3])
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    w = 0.36
    b1 = ax.bar(x-w/2, trusted, w, label="Trusted GT only")
    b2 = ax.bar(x+w/2, allgt, w, label="All GT included")
    ax.set_xticks(x, metrics)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 105)
    ax.set_title("Automatic use: same two reads for every measure")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x()+b.get_width()/2, v, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=9)
    save(fig, "automatic_use")

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

    # Final individual-contact recovery.
    cats = ["Any contact\n(incl. serves)", "Non-serve\ncontact", "Serve"]
    timing = np.array([88.2, 88.9, 81.3])
    side = np.array([85.5, 86.3, 77.4])
    x = np.arange(len(cats))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 5))
    b1 = ax.bar(x-w/2, timing, w, label="Timing correct")
    b2 = ax.bar(x+w/2, side, w, label="Timing + correct player")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall against trusted GT (%)")
    ax.set_title("Final detector: individual contact recovery at ±10")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                    f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=9)
    save(fig, "contact_recovery")

    # Contact-level progression.
    stages = ["Whole-sequence\nmodel", "+ later-contact\nrepair", "Final detector"]
    timing_prog = np.array([87.3, 88.0, 88.2])
    side_prog = np.array([80.8, 85.0, 85.5])
    x = np.arange(len(stages))
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
