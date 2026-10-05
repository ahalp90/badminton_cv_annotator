"""Draw contact and error diagnostics from counts in evaluation_tables.md."""

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parents[1] / "figures"
BLUE = "#4477AA"
SAND = "#DDCC77"
GREY = "#999999"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(fig: plt.Figure, name: str, note: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.text(0.04, 0.03, note, fontsize=11, va="bottom", color="#454545")
    fig.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def plot_selected_errors() -> None:
    rows = [
        ("Extra contacts only", 65),
        ("Missed contacts only", 22),
        ("Missed + extra contacts", 21),
        ("Missed contacts + wrong player", 5),
        ("Missed + extra + incomplete clip", 1),
        ("Missed + extra + wrong player", 1),
        ("Missed contacts + incomplete clip", 1),
        ("Wrong player only", 1),
    ]
    labels, values = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    fig.subplots_adjust(left=0.34, right=0.97, bottom=0.19, top=0.80)
    fig.suptitle("Extra contacts are the most common high-confidence error", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.91,
             "117 of 708 assessed high-confidence clips have incorrect annotations.\n"
             "115 of these still contain exactly one complete rally; only two are incomplete.",
             va="top", linespacing=1.4)
    bars = ax.barh(labels, values, color=SAND, height=0.68)
    ax.bar_label(bars, padding=5)
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xlabel("Clips")
    save(fig, "selected_errors",
         "46 ShuttleSet22 videos · confidence cutoff 0.757 · ±10 frames at 30 fps\n"
         "Each clip appears in one row. An incomplete clip cuts off part of the labelled rally.")


def plot_timing_accuracy() -> None:
    labels = ["Exact frame", "Within 2 frames", "Within 5 frames"]
    counts = [9_033, 28_460, 33_504]
    total = 34_200
    rates = [100 * count / total for count in counts]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.26, top=0.81)
    fig.suptitle("98% of matched contacts are within five frames", x=0.04,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.04, 0.895, "Cumulative windows: each wider window includes the narrower ones")
    bars = ax.barh(labels, rates, color=BLUE, height=0.68)
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Share of matched contacts (%)")
    for bar, rate, count in zip(bars, rates, counts, strict=True):
        ax.text(rate + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1f}% ({count:,})", va="center", fontsize=11)
    save(fig, "timing_offsets",
         "34,200 timing matches · 30 fps clock\n"
         "The 2,984 missed contacts are excluded from this distribution.")


def plot_contact_position() -> None:
    labels = ["Serve", "Middle contact", "Final contact"]
    missed = [293, 721, 350]
    totals = [2_826, 29_441, 3_082]
    rates = [100 * count / total for count, total in zip(missed, totals, strict=True)]
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.24, top=0.82)
    fig.suptitle("Serves and final contacts are missed more often", x=0.04,
                 ha="left", fontsize=16, weight="bold")
    fig.text(0.04, 0.895, "Court-accepted frames only · contact timing within ±10 frames at 30 fps")
    bars = ax.bar(labels, rates, color=SAND, width=0.75)
    for bar, rate, count, total in zip(bars, rates, missed, totals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.4,
                f"{rate:.1f}%\n{count:,} / {total:,}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 14.5)
    ax.set_ylabel("Labelled contacts missed (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.15)
    save(fig, "contact_position",
         "Counts show misses / labels. Accepted frames include those with a missing player.\n"
         "Single-contact rallies count as serves only.")


def plot_upstream_context() -> None:
    labels = ["Court rejected", "Court accepted;\none or both players missing", "Court accepted;\nboth players available"]
    counts = [1_620, 29, 1_335]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(11.5, 4.9))
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.25, top=0.82)
    fig.suptitle("Court rejection accounts for 54.3% of missed contacts", x=0.04,
                 ha="left", fontsize=15, weight="bold")
    fig.text(0.04, 0.895, f"Court and player availability at each of the {total:,} missed labelled contacts")
    bars = ax.barh(labels, counts, color=[SAND, GREY, BLUE], height=0.68)
    for bar, count in zip(bars, counts, strict=True):
        ax.text(count + 25, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({100 * count / total:.1f}%)", va="center", fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 2_150)
    ax.set_xticks([0, 500, 1_000, 1_500, 2_000])
    ax.set_xlabel("Missed labelled contacts")
    save(fig, "upstream_context",
         "46 ShuttleSet22 videos · ±10 frames at 30 fps\n"
         "The three groups are mutually exclusive and sum to all 2,984 misses.")


def main() -> None:
    plot_selected_errors()
    plot_timing_accuracy()
    plot_contact_position()
    plot_upstream_context()


if __name__ == "__main__":
    main()
