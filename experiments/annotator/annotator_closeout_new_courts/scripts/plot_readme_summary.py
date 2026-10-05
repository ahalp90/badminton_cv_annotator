"""Draw contact metrics, selected clips and historical development figures.

Current counts are checked against results/ when the saved tables are present.
Otherwise the published snapshot supports rendering this report on its own.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Published current closeout counts. These also make the plotting script usable
# from a documentation-only handover pack that does not carry the large result
# tables. When results/ exists, current_metrics() recomputes and checks them.
SNAPSHOT = {
    "labelled_rallies": 3327,
    "labelled_contacts": 37184,
    "predicted_contacts": 40246,
    "matched_contacts": 34200,
    "correct_side_contacts": 33156,
    "fully_correct_rallies": 1744,
    "selected": 725,
    "selected_judgeable": 708,
    "selected_exact": 591,
    "selected_whole_rally": 706,
}

# Historical architecture progression from scratch/contact_det_closing_pass.
# This is a different, earlier population: 3,422 trusted rallies across the
# then-used 47 ShuttleSet22 videos. It explains where the final architecture
# came from; it is not a new-court ablation.
HISTORICAL_PROGRESSION = [
    ("Previous model", 995),
    ("Repair a missing serve", 1105),
    ("Score whole contact sequences", 1435),
    ("Allow one later contact insertion", 1597),
    ("Check the added contact separately", 1622),
    ("Adjust clip start and end", 1763),
]
HISTORICAL_RALLIES = 3422


def f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall)


def as_bool(series: pd.Series) -> pd.Series:
    """Read CSV boolean columns safely whether pandas inferred bool or strings."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def current_metrics() -> dict[str, int]:
    """Recompute the README counts from the saved closeout tables when present."""
    required = [RESULTS / "contacts.csv.gz", RESULTS / "predictions.csv.gz", RESULTS / "rallies.csv.gz",
                RESULTS / "proposals.csv.gz"]
    if not all(path.exists() for path in required):
        return dict(SNAPSHOT)

    contacts = pd.read_csv(RESULTS / "contacts.csv.gz", dtype={"video": str})
    predictions = pd.read_csv(RESULTS / "predictions.csv.gz", dtype={"video": str})
    rallies = pd.read_csv(RESULTS / "rallies.csv.gz", dtype={"video": str})
    proposals = pd.read_csv(RESULTS / "proposals.csv.gz", dtype={"video": str})

    contacts = contacts[(contacts.split == "test") & (contacts.tolerance_base30 == 10)]
    predictions = predictions[(predictions.split == "test") & (predictions.tolerance_base30 == 10)]
    rallies = rallies[(rallies.split == "test") & (rallies.tolerance_base30 == 10)]
    proposals = proposals[(proposals.split == "test") & (proposals.tolerance_base30 == 10)]
    selected = proposals[as_bool(proposals.selected)].copy()

    exact_judgeable = selected[selected.outcome != "unknown"]
    whole_judgeable = selected[selected.overlapping_rallies > 0]
    whole = whole_judgeable[(whole_judgeable.overlapping_rallies == 1) & ~as_bool(whole_judgeable.boundary_error)]
    whole_unique = whole[["video", "rally_id"]].drop_duplicates()

    metrics = {
        "labelled_rallies": len(rallies),
        "labelled_contacts": len(contacts),
        "predicted_contacts": len(predictions),
        "matched_contacts": int(as_bool(contacts.matched).sum()),
        "correct_side_contacts": int(as_bool(contacts.player_correct).sum()),
        "fully_correct_rallies": int(as_bool(rallies.fully_correct).sum()),
        "selected": len(selected),
        "selected_judgeable": len(exact_judgeable),
        "selected_exact": int((exact_judgeable.outcome == "correct").sum()),
        "selected_whole_rally": len(whole_unique),
    }
    assert metrics == SNAPSHOT, {key: (metrics[key], SNAPSHOT[key]) for key in metrics if metrics[key] != SNAPSHOT[key]}
    return metrics


BLUE = "#4477AA"
SAND = "#DDCC77"
GREY = "#999999"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str, note: str) -> None:
    fig.text(0.04, 0.03, note, fontsize=11, va="bottom", color="#454545")
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def plot_results_at_a_glance(metrics: dict[str, int]) -> None:
    matched = [metrics["matched_contacts"], metrics["correct_side_contacts"]]
    precision = [100 * count / metrics["predicted_contacts"] for count in matched]
    recall = [100 * count / metrics["labelled_contacts"] for count in matched]
    f1s = [f1(prec, rec) for prec, rec in zip(precision, recall, strict=True)]
    positions = np.arange(2)
    width = 0.27
    fig, ax = plt.subplots(figsize=(11.5, 5.1))
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.23, top=0.80)
    fig.suptitle("Contact recovery and false predictions", x=0.04,
                 ha="left", fontsize=16, weight="bold")
    for offset, values, name, colour in [
        (-width, precision, "Precision", BLUE),
        (0, recall, "Recall", SAND),
        (width, f1s, "F1", "#777777"),
    ]:
        bars = ax.bar(positions + offset, values, width, label=name, color=colour)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4)
    ax.set_xticks(positions, ["Correct timing", "Correct timing and player"])
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Percent")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    save(fig, "results_at_a_glance",
         "46 ShuttleSet22 videos · 37,184 labels · 40,246 predictions · ±10 frames at 30 fps\n"
         "Precision: share of predictions matching labels. Recall: share of labels recovered.")


def plot_performance_overview(metrics: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.20, top=0.70, wspace=0.18)
    fig.suptitle("The annotator finds most contacts and selects reliable rally clips",
                 x=0.04, ha="left", fontsize=16, weight="bold")
    panels = [
        (
            "All output: contact accuracy",
            f'{metrics["labelled_contacts"]:,} labels · {metrics["predicted_contacts"]:,} predictions',
            metrics["predicted_contacts"],
            metrics["labelled_contacts"],
            [
                ("Correct timing", metrics["matched_contacts"], BLUE),
                ("Correct timing and player", metrics["correct_side_contacts"], SAND),
            ],
        ),
        (
            "Selected clips: confidence ≥ 0.757",
            f'{metrics["selected"]} selected clips · {metrics["selected_judgeable"]} assessed',
            metrics["selected_judgeable"],
            metrics["labelled_rallies"],
            [
                ("Rally boundaries correct", metrics["selected_whole_rally"], BLUE),
                ("Fully correct rallies (including all contacts)", metrics["selected_exact"], SAND),
            ],
        ),
    ]
    positions = np.arange(3)
    width = 0.36
    for ax, (title, population, precision_total, recall_total, series) in zip(axes, panels, strict=True):
        ax.set_title(f"{title}\n{population}", loc="left", fontsize=12, pad=51)
        for series_index, (label, count, colour) in enumerate(series):
            precision = 100 * count / precision_total
            recall = 100 * count / recall_total
            values = [precision, recall, f1(precision, recall)]
            offset = (series_index - 0.5) * width
            bars = ax.bar(positions + offset, values, width, color=colour, label=label)
            ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4, fontsize=10)
        ax.set_xticks(positions, ["Precision", "Recall", "F1"])
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=10)
    axes[0].set_ylabel("Percent")
    fully_correct = metrics["fully_correct_rallies"]
    rally_total = metrics["labelled_rallies"]
    fig.text(0.06, 0.09,
             f'Fully correct rally annotations: {fully_correct:,} / {rally_total:,} ({100 * fully_correct / rally_total:.1f}%)',
             fontsize=11, weight="bold")
    save(fig, "performance_overview",
         "46 ShuttleSet22 videos · timing allowance ±10 frames at 30 fps")


def plot_historical_progression() -> None:
    labels, counts = zip(*HISTORICAL_PROGRESSION, strict=True)
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.22, top=0.83)
    fig.suptitle("Earlier development improved complete rally annotations", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.90, "Historical benchmark: 3,422 rallies in 47 videos, including video 15")
    bars = ax.barh(labels, counts, color=BLUE, height=0.68)
    ax.invert_yaxis()
    for bar, count in zip(bars, counts, strict=True):
        ax.text(count + 25, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({100 * count / HISTORICAL_RALLIES:.1f}%)", va="center", fontsize=11)
    ax.set_xlim(0, 2_300)
    ax.set_xticks([0, 500, 1_000, 1_500, 2_000])
    ax.set_xlabel("Fully correct rallies")
    save(fig, "system_progression_trusted",
         "Stages are cumulative. Each row includes the changes above it.\n"
         "These stages were not rerun with new courts; the current evaluation uses a different population.")


def plot_high_confidence_selection(metrics: dict[str, int]) -> None:
    total = metrics["selected"]
    exact = metrics["selected_exact"]
    corrections = metrics["selected_whole_rally"] - exact
    incomplete = metrics["selected_judgeable"] - metrics["selected_whole_rally"]
    unassessed = total - metrics["selected_judgeable"]
    parts = [
        ("Exact annotation", exact, BLUE, ""),
        ("Complete rally; corrections needed", corrections, SAND, ""),
        ("Incomplete rally", incomplete, GREY, ""),
        ("Labels cannot assess", unassessed, GREY, "///"),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.50, top=0.78)
    fig.suptitle("Most high-confidence clips contain one complete rally", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.865, f"{total} selected clips: {exact} exact + {corrections} complete rallies needing corrections")
    left = 0
    for label, count, colour, hatch in parts:
        ax.barh(0, count, left=left, color=colour, hatch=hatch, height=0.72,
                label=f"{label}: {count}", edgecolor="white", linewidth=0.6)
        if count > 50:
            ax.text(left + count / 2, 0, str(count), ha="center", va="center",
                    color="white" if colour == BLUE else "#222222", weight="bold", fontsize=15)
        left += count
    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 200, 400, 600, total])
    ax.set_xlabel("Selected clips")
    ax.spines["left"].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(0.06, 0.31), fontsize=11)
    save(fig, "high_confidence_selection",
         "Confidence cutoff 0.757 · 4,021 proposals in total\n"
         "706 of 708 assessed clips contain a complete rally (99.7%), covering 21.2% of labelled rallies.")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = current_metrics()
    plot_performance_overview(metrics)
    plot_results_at_a_glance(metrics)
    plot_historical_progression()
    plot_high_confidence_selection(metrics)


if __name__ == "__main__":
    main()
