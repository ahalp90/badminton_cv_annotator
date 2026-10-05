"""Draw the performance overview with a whole-rally panel for all output.

The middle panel shows how many labelled rallies the annotator gets fully right
without confidence selection. It replaces the bold footer line in
performance_overview.png, so all-output and selected-clip rally scores sit side by side.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_readme_summary import BLUE, RESULTS, SAND, as_bool, current_metrics, f1, save

# Published count, used when results/ is absent, as in plot_readme_summary.SNAPSHOT.
EXACT_SEQUENCE_RALLIES = 1767


def exact_sequence_rallies() -> int:
    """Count rallies whose contacts all match, ignoring player assignment."""
    path = RESULTS / "rallies.csv.gz"
    if not path.exists():
        return EXACT_SEQUENCE_RALLIES
    rallies = pd.read_csv(path, dtype={"video": str})
    rallies = rallies[(rallies.split == "test") & (rallies.tolerance_base30 == 10)]
    count = int(as_bool(rallies.exact_sequence).sum())
    assert count == EXACT_SEQUENCE_RALLIES, count
    return count


def style_panel(ax: plt.Axes, title: str, population: str) -> None:
    ax.set_title(f"{title}\n{population}", loc="left", fontsize=12, pad=51)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=10)


def draw_metric_panel(ax: plt.Axes, precision_total: int, recall_total: int,
                      series: list[tuple[str, int, str]]) -> None:
    positions = np.arange(3)
    width = 0.36
    for series_index, (label, count, colour) in enumerate(series):
        precision = 100 * count / precision_total
        recall = 100 * count / recall_total
        values = [precision, recall, f1(precision, recall)]
        offset = (series_index - 0.5) * width
        bars = ax.bar(positions + offset, values, width, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4, fontsize=10)
    ax.set_xticks(positions, ["Precision", "Recall", "F1"])


def draw_rally_panel(ax: plt.Axes, metrics: dict[str, int], exact_sequence: int) -> None:
    width = 0.36
    series = [
        ("Exact contact sequence", exact_sequence, BLUE),
        ("Exact sequence and correct players", metrics["fully_correct_rallies"], SAND),
    ]
    for series_index, (label, count, colour) in enumerate(series):
        share = 100 * count / metrics["labelled_rallies"]
        offset = (series_index - 0.5) * width
        bars = ax.bar([offset], [share], width, color=colour, label=label)
        ax.bar_label(bars, labels=[f"{share:.1f}%"], padding=4, fontsize=10)
    ax.set_xticks([0], ["Labelled rallies"])
    # Matches the bar scale of the three-group panels, given their width ratio.
    ax.set_xlim(-0.75, 0.75)


def plot_performance_overview_rallies(metrics: dict[str, int], exact_sequence: int) -> None:
    # The empty third column keeps the selected-clip panel apart from the two all-output panels.
    fig, (contact_ax, rally_ax, spacer_ax, selected_ax) = plt.subplots(
        1, 4, figsize=(17, 5.5), sharey=True, gridspec_kw={"width_ratios": [3.1, 1.5, 0.25, 3.1]})
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.17, top=0.70, wspace=0.06)
    spacer_ax.axis("off")
    fig.suptitle("The annotator finds most contacts and selects reliable rally clips",
                 x=0.04, ha="left", fontsize=16, weight="bold")

    draw_metric_panel(contact_ax, metrics["predicted_contacts"], metrics["labelled_contacts"], [
        ("Correct timing", metrics["matched_contacts"], BLUE),
        ("Correct timing and player", metrics["correct_side_contacts"], SAND),
    ])
    style_panel(contact_ax, "All output: contact accuracy",
                f'{metrics["labelled_contacts"]:,} labels · {metrics["predicted_contacts"]:,} predictions')

    draw_rally_panel(rally_ax, metrics, exact_sequence)
    style_panel(rally_ax, "All output: whole rallies", f'{metrics["labelled_rallies"]:,} labelled rallies')

    draw_metric_panel(selected_ax, metrics["selected_judgeable"], metrics["labelled_rallies"], [
        ("Rally boundaries correct", metrics["selected_whole_rally"], BLUE),
        ("Fully correct rallies (including all contacts)", metrics["selected_exact"], SAND),
    ])
    style_panel(selected_ax, "Selected clips: confidence ≥ 0.757",
                f'{metrics["selected"]} selected clips · {metrics["selected_judgeable"]} assessed')

    contact_ax.set_ylabel("Percent")
    save(fig, "performance_overview_rallies",
         "46 ShuttleSet22 videos · timing allowance ±10 frames at 30 fps")


def main() -> None:
    plot_performance_overview_rallies(current_metrics(), exact_sequence_rallies())


if __name__ == "__main__":
    main()
