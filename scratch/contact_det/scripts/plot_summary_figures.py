"""Generate the compact contact-detector summary figures.

The figures use only the retained pooled measurements documented in the parent
directory. By default, output goes to scratch/contact_det/figures.
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

SCOPE = "Pooled sset_01, sset_15, sset_21 · ±10 base-30 frames"


def scoped_title(figure: Figure, title: str, scope: str = SCOPE) -> None:
    figure.suptitle(title, y=0.98, fontsize=13)
    figure.text(0.5, 0.89, scope, ha="center", fontsize=10)


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5.8)

    ax.text(0.35, 5.3, "Old standalone heuristic path", fontsize=13, fontweight="bold")
    old = [
        (0.45, 4.25, 2.05, 0.72, "Raw contact\nproposals"),
        (2.85, 4.25, 2.15, 0.72, "Wrist gate +\nsuppression"),
        (5.35, 4.25, 1.95, 0.72, "Contact\nevent list"),
        (7.65, 4.25, 1.95, 0.72, "Top / Bottom\nside"),
    ]
    for x, y, w, h, label in old:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.8)
    for current, following in pairwise(old):
        ax.annotate(
            "",
            xy=(following[0] - 0.07, following[1] + following[3] / 2),
            xytext=(current[0] + current[2] + 0.07, current[1] + current[3] / 2),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        )

    ax.text(0.35, 3.05, "Experimental path", fontsize=13, fontweight="bold")
    new = [
        (0.45, 1.88, 2.05, 0.82, "Raw proposals +\nrelaxed vision cues"),
        (2.85, 1.88, 2.15, 0.82, "Region v2\nsearch space only"),
        (5.35, 1.88, 1.95, 0.82, "Region-v2\nHGB / RF"),
        (7.65, 1.88, 1.95, 0.82, "Contact\nevent list"),
        (9.95, 1.88, 0.95, 0.82, "Player\nside"),
    ]
    for x, y, w, h, label in new:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
    for current, following in pairwise(new):
        ax.annotate(
            "",
            xy=(following[0] - 0.07, following[1] + following[3] / 2),
            xytext=(current[0] + current[2] + 0.07, current[1] + current[3] / 2),
            arrowprops={"arrowstyle": "->", "linewidth": 1.0},
        )

    ax.text(
        3.92,
        1.25,
        "31.9% of video searched · 98.3% contact coverage at ±10",
        ha="center",
        fontsize=8.7,
    )
    ax.text(
        5.9,
        0.46,
        "Region v2 decides where to look; HGB/RF decides whether there was a contact; "
        "Top/Bottom side is scored afterwards.",
        ha="center",
        fontsize=9.2,
        fontweight="bold",
    )

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def region_tradeoff(path: Path) -> None:
    labels = [
        "Video inspected by region v2",
        "All labelled contacts reachable",
        "Labelled serves reachable",
    ]
    values = [31.9, 98.3, 97.9]
    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    y = list(range(len(labels)))
    ax.barh(y, values)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 103)
    ax.set_xlabel("Percent")
    scoped_title(fig, "Region v2 search cost versus reachability")
    for yy, value in zip(y, values):
        ax.text(value + 0.8, yy, f"{value:.1f}%", va="center", fontsize=9.3)
    fig.subplots_adjust(left=0.34, right=0.95, top=0.76, bottom=0.18)
    fig.text(
        0.34,
        0.055,
        "Different denominators: video inspected uses source frames; reachability uses labelled contacts.",
        ha="left",
        fontsize=8.7,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def contact_output(path: Path) -> None:
    labels = ["Current final", "Region-v2 HGB", "Region-v2 RF"]
    found = [79.3, 90.5, 85.2]
    found_side = [70.6, 75.7, 73.1]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    ax.bar([i - width / 2 for i in x], found, width, label="Contact found")
    ax.bar(
        [i + width / 2 for i in x],
        found_side,
        width,
        label="Contact found + side correct",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of all 3,128 labelled contacts")
    scoped_title(fig, "Contact output: timing versus timing + correct side")
    ax.legend()
    for xs, vals in (
        ([i - width / 2 for i in x], found),
        ([i + width / 2 for i in x], found_side),
    ):
        for xx, value in zip(xs, vals):
            ax.text(xx, value + 1.0, f"{value:.1f}", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.76, bottom=0.13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def serve_output(path: Path) -> None:
    labels = ["Current final", "Region-v2 HGB", "Region-v2 RF"]
    found = [61.0, 67.5, 42.8]
    found_side = [46.2, 56.2, 37.0]
    x = list(range(len(labels)))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    ax.bar([i - width / 2 for i in x], found, width, label="Serve found")
    ax.bar(
        [i + width / 2 for i in x],
        found_side,
        width,
        label="Serve found + serving side correct",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of all 292 labelled serves")
    scoped_title(fig, "Serve recall: timing and correct serving side")
    ax.legend()
    for xs, vals in (
        ([i - width / 2 for i in x], found),
        ([i + width / 2 for i in x], found_side),
    ):
        for xx, value in zip(xs, vals):
            ax.text(xx, value + 1.0, f"{value:.1f}", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.76, bottom=0.13)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def selected_side_and_serve_summary(path: Path) -> None:
    """Compare the selected HGB stream with the old heuristic output."""
    labels = [
        "Timing + correct-\nside recall",
        "Serve timing\nrecall",
        "Serve timing +\ncorrect-side recall",
    ]
    old_heuristics = [70.6, 61.0, 46.2]
    selected_hgb = [75.2, 67.1, 56.2]
    positions = list(range(len(labels)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.0, 5.9))
    old_positions = [position - width / 2 for position in positions]
    selected_positions = [position + width / 2 for position in positions]
    ax.bar(old_positions, old_heuristics, width, label="Old final heuristics")
    ax.bar(
        selected_positions,
        selected_hgb,
        width,
        label="Selected HGB event stream",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Score at ±10 base-30 frames (%)")
    ax.set_xlabel("Metric")
    ax.set_title(
        "The selected HGB stream improves timing-and-side and serve output",
        pad=14,
    )
    ax.legend()
    for bar_positions, values in (
        (old_positions, old_heuristics),
        (selected_positions, selected_hgb),
    ):
        for position, value in zip(bar_positions, values):
            ax.text(position, value + 1.0, f"{value:.1f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def cleanup_headroom(path: Path) -> None:
    """Show the simple repair upper bounds in the original HGB spans."""
    labels = [
        "Current output",
        "Remove extra\nevents perfectly",
        "Also repair the separate\none-missing rallies",
    ]
    values = [21, 38, 51]
    positions = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(11.0, 6.1))
    bars = ax.bar(positions, values)
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 58)
    ax.set_ylabel("Potential fully correct rallies (count)")
    ax.set_title(
        "How much could simple repair improve the rally output?",
        pad=14,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.0,
            str(value),
            ha="center",
            fontsize=10,
        )
    fig.subplots_adjust(bottom=0.26)
    fig.text(
        0.5,
        0.045,
        "Original learned-model rally records using the ±10 timing tolerance. "
        "The 38 and 51 values are upper bounds, not achieved model results.\n"
        "The 51 case assumes the separate one-missing rallies receive a correct new event and player side.",
        ha="center",
        fontsize=8.8,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rally_quality(path: Path) -> None:
    labels = ["1 s", "2 s", "3 s", "5 s", "No cap"]
    values = [1.7, 15.6, 45.8, 64.3, 77.3]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(9.3, 5.5))
    ax.plot(x, values, marker="o")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Clean one-rally F1 (%)")
    ax.set_xlabel("Maximum extra padding beyond the first / last labelled contact")
    fig.suptitle("Rally-span quality versus extra padding cap", y=0.98, fontsize=13)
    fig.text(
        0.5,
        0.91,
        "Pooled sset_01, sset_15, sset_21\n"
        "Clean match contains every contact from exactly one rally and no contact from another",
        ha="center",
        va="top",
        fontsize=9.0,
    )
    for xx, value in zip(x, values):
        ax.text(xx, value + 2.0, f"{value:.1f}", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.74, bottom=0.14)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def dense_scorecard(path: Path) -> None:
    setups = ["Current final", "Region-v2 HGB", "Region-v2 RF"]
    keys = ["RF1", "CP", "CR", "C+SR", "SA", "SR", "S+SR", "SSA"]
    values = {
        "Current final": [77.3, 66.9, 79.3, 70.6, 89.0, 61.0, 46.2, 75.8],
        "Region-v2 HGB": [None, 84.5, 90.5, 75.7, 83.7, 67.5, 56.2, 84.1],
        "Region-v2 RF": [None, 84.1, 85.2, 73.1, 85.8, 42.8, 37.0, 86.4],
    }
    fig, ax = plt.subplots(figsize=(11.7, 6.2))
    bar_w, gap, group_gap = 0.35, 0.09, 0.48
    positions, labels, centers, boundaries = [], [], [], []
    cursor = 0.0
    for index, setup in enumerate(setups):
        group_positions = []
        for key, value in zip(keys, values[setup]):
            group_positions.append(cursor)
            positions.append(cursor)
            labels.append(key)
            if value is None:
                ax.text(cursor, 4, "n/a", ha="center", fontsize=8.2, fontstyle="italic")
            else:
                ax.bar(cursor, value, width=bar_w)
                ax.text(cursor, value + 0.9, f"{value:.1f}", ha="center", fontsize=8.0)
            cursor += bar_w + gap
        centers.append((group_positions[0] + group_positions[-1]) / 2)
        if index < len(setups) - 1:
            boundaries.append(cursor + group_gap / 2 - (bar_w + gap) / 2)
        cursor += group_gap
    ax.set_xticks(positions, labels, fontsize=8.1)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Percent")
    scoped_title(fig, "Supplemental dense scorecard")
    for center, setup in zip(centers, setups):
        ax.text(center, 101.0, setup, ha="center", fontsize=9.4, fontweight="bold")
    for boundary in boundaries:
        ax.axvline(boundary, linewidth=0.8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.70, bottom=0.27)
    fig.text(
        0.07,
        0.135,
        "RF1 rally-span F1 · CP contact precision · CR contact recall · C+SR contact + correct-side recall",
        ha="left",
        fontsize=8.6,
    )
    fig.text(
        0.07,
        0.095,
        "SA side accuracy when matched · SR serve recall · S+SR serve + correct-side recall · SSA serving-side accuracy when matched",
        ha="left",
        fontsize=8.6,
    )
    fig.text(
        0.07,
        0.05,
        "RF1 is n/a for HGB/RF because rally spans were not rerun. SA/SSA use timing-matched events; recall bars use all labelled contacts or serves.",
        ha="left",
        fontsize=8.3,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "figures"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    architecture(args.output_dir / "contact_pipeline_architecture.png")
    region_tradeoff(args.output_dir / "region_v2_search_tradeoff.png")
    contact_output(args.output_dir / "contact_output_recall.png")
    serve_output(args.output_dir / "serve_output_recall.png")
    selected_side_and_serve_summary(
        args.output_dir / "followup_side_and_serve_summary.png"
    )
    cleanup_headroom(args.output_dir / "followup_cleanup_headroom.png")
    rally_quality(args.output_dir / "rally_segmentation_quality.png")
    dense_scorecard(args.output_dir / "dense_scorecard.png")


if __name__ == "__main__":
    main()
