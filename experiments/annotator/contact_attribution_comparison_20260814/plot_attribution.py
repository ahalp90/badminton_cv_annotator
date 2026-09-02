"""Plot exact-contact accuracy and predicted-side share by method.

Reads ``contact_rows.csv.gz`` from this file's own directory and writes
``figures/attribution_by_side.png`` beside it. No network access, no
randomness: re-running produces the same image byte-for-byte given the
same input CSV.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "contact_rows.csv.gz"
OUTPUT_PNG = HERE / "figures" / "attribution_by_side.png"

METHODS = ("body_height", "image_pixels", "court_projection")
METHOD_LABELS = {
    "body_height": "Body height",
    "image_pixels": "Image pixels",
    "court_projection": "Court projection",
}

# Colours: validated colour-blind-safe categorical pair (blue / orange) for the
# Top / Bottom sides, used consistently in both panels; a neutral grey marks the
# "Overall" summary bar in panel (a) so it reads as an aggregate, not a third side.
COLOR_TOP = "#2a78d6"
COLOR_BOTTOM = "#eb6834"
COLOR_OVERALL = "#8f8d86"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_TEXT_MUTED = "#898781"
SURFACE = "#fcfcfb"


def side_accuracy(df: pd.DataFrame, method: str, side: str | None) -> tuple[int, int]:
    """Return (correct, eligible) for one method, optionally restricted to one ground-truth side."""
    pred_col = f"{method}_pred"
    rows = df if side is None else df[df["gt_side"] == side]
    eligible = rows[rows[pred_col].notna()]
    correct = int((eligible[pred_col] == eligible["gt_side"]).sum())
    return correct, len(eligible)


def predicted_share(df: pd.DataFrame, method: str) -> tuple[float, float]:
    """Return (top share, bottom share) of eligible predictions for one method, as percentages."""
    pred_col = f"{method}_pred"
    eligible = df[df[pred_col].notna()]
    n = len(eligible)
    top_share = 100.0 * (eligible[pred_col] == "Top").sum() / n
    bottom_share = 100.0 * (eligible[pred_col] == "Bot").sum() / n
    return top_share, bottom_share


def label_bar(ax: plt.Axes, bar, text: str) -> None:
    """Place a value label just above one bar's tip, in text ink (never the fill colour)."""
    height = bar.get_height()
    ax.annotate(
        text,
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=COLOR_TEXT_PRIMARY,
    )


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylabel(ylabel, color=COLOR_TEXT_SECONDARY, fontsize=10)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, color=COLOR_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_AXIS)
    ax.tick_params(axis="x", colors=COLOR_TEXT_SECONDARY, labelsize=9.5, length=0)
    ax.tick_params(axis="y", colors=COLOR_TEXT_MUTED, labelsize=8.5, length=0)
    ax.set_yticks([0, 20, 40, 60, 80, 100])


def plot_accuracy(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel (a): exact-contact accuracy per method, split Top / Bottom / Overall."""
    bar_width = 0.26
    group_positions = range(len(METHODS))
    offsets = {"Top": -bar_width, "Bot": 0.0, "Overall": bar_width}
    colors = {"Top": COLOR_TOP, "Bot": COLOR_BOTTOM, "Overall": COLOR_OVERALL}
    legend_labels = {"Top": "Top", "Bot": "Bottom", "Overall": "Overall"}
    handles = {}
    for method_index, method in enumerate(METHODS):
        for side_key in ("Top", "Bot", "Overall"):
            side_arg = None if side_key == "Overall" else side_key
            correct, eligible = side_accuracy(df, method, side_arg)
            accuracy_pct = 100.0 * correct / eligible
            bar = ax.bar(
                method_index + offsets[side_key],
                accuracy_pct,
                width=bar_width,
                color=colors[side_key],
                zorder=3,
            )[0]
            label_bar(ax, bar, f"{accuracy_pct:.1f}%")
            handles.setdefault(side_key, bar)
    ax.set_xticks(list(group_positions))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    style_axes(ax, "Exact-contact accuracy (%)")
    ax.set_title(
        "(a) Accuracy by ground-truth side", fontsize=11, color=COLOR_TEXT_PRIMARY, loc="left"
    )
    ax.legend(
        [handles["Top"], handles["Bot"], handles["Overall"]],
        [legend_labels["Top"], legend_labels["Bot"], legend_labels["Overall"]],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=COLOR_TEXT_SECONDARY,
        ncol=1,
    )


def plot_predicted_share(ax: plt.Axes, df: pd.DataFrame) -> None:
    """Panel (b): predicted side share per method, versus the ground-truth share."""
    categories = list(METHODS) + ["ground_truth"]
    category_labels = [METHOD_LABELS[m] for m in METHODS] + ["Ground truth"]
    bar_width = 0.32
    positions = range(len(categories))

    top_shares = []
    bottom_shares = []
    for method in METHODS:
        top_share, bottom_share = predicted_share(df, method)
        top_shares.append(top_share)
        bottom_shares.append(bottom_share)
    gt_top_share = 100.0 * (df["gt_side"] == "Top").sum() / len(df)
    gt_bottom_share = 100.0 * (df["gt_side"] == "Bot").sum() / len(df)
    top_shares.append(gt_top_share)
    bottom_shares.append(gt_bottom_share)

    top_bars = ax.bar(
        [p - bar_width / 2 for p in positions],
        top_shares,
        width=bar_width,
        color=COLOR_TOP,
        zorder=3,
    )
    bottom_bars = ax.bar(
        [p + bar_width / 2 for p in positions],
        bottom_shares,
        width=bar_width,
        color=COLOR_BOTTOM,
        zorder=3,
    )
    for bar, value in zip(top_bars, top_shares):
        label_bar(ax, bar, f"{value:.1f}%")
    for bar, value in zip(bottom_bars, bottom_shares):
        label_bar(ax, bar, f"{value:.1f}%")

    ax.set_xticks(list(positions))
    ax.set_xticklabels(category_labels)
    style_axes(ax, "Share of contacts (%)")
    ax.set_title(
        "(b) Predicted side share vs. ground truth", fontsize=11, color=COLOR_TEXT_PRIMARY, loc="left"
    )
    ax.legend(
        [top_bars, bottom_bars],
        ["Top", "Bottom"],
        loc="upper right",
        frameon=False,
        fontsize=8.5,
        labelcolor=COLOR_TEXT_SECONDARY,
        ncol=1,
    )


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    plot_accuracy(ax_left, df)
    plot_predicted_share(ax_right, df)

    fig.suptitle(
        "Contact-side accuracy is much lower for Bottom-side contacts, for every method",
        fontsize=12.5,
        color=COLOR_TEXT_PRIMARY,
        x=0.02,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.02,
        0.93,
        "3,128 labelled contacts across sset_01, sset_15, sset_21. Accuracy uses only contacts where "
        "the method returned a player.",
        fontsize=8.5,
        color=COLOR_TEXT_SECONDARY,
        ha="left",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
