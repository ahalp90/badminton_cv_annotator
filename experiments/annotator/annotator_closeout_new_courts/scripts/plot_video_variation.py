"""Draw the README's video comparisons from results/per_video.csv.gz.

Use --results to read saved tables outside this report folder while drafting.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
BLUE = "#4477AA"
SAND = "#DDCC77"
GREY = "#999999"
LABEL_OFFSETS = {18: (-130, 18), 17: (-150, -5), 42: (-70, -65), 53: (-150, 12)}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def save(figure: plt.Figure, name: str, note: str) -> None:
    figure.text(0.04, 0.025, note, fontsize=11, va="bottom", color="#454545")
    figure.savefig(FIGURES / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=0.25 if name == "video_variation" else 0.12, facecolor="white")
    plt.close(figure)


def plot_video_variation(per_video: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.8))
    figure.subplots_adjust(left=0.12, right=0.95, bottom=0.19, top=0.80)
    figure.suptitle("Fully correct rallies vary from 25.9% to 75.0% by video",
                   x=0.04, ha="left", fontsize=15, weight="bold")
    figure.text(0.04, 0.88, "One point per video · 46 ShuttleSet22 videos · ±10 frames at 30 fps")
    axis.scatter(per_video.contact_percent, per_video.rally_percent,
                 color=BLUE, alpha=0.8, s=52, zorder=3)
    for video, offset in LABEL_OFFSETS.items():
        row = per_video.loc[video]
        axis.annotate(f"Video {video}\n{row.correct_rallies:.0f}/{row.labelled_rallies:.0f} rallies",
                      (row.contact_percent, row.rally_percent), xytext=offset,
                      textcoords="offset points", fontsize=11,
                      arrowprops={"arrowstyle": "-", "color": GREY})
    axis.set(xlim=(0, 100), ylim=(0, 100), xticks=range(0, 101, 20), yticks=range(0, 101, 20),
             xlabel="Labelled contacts recovered within the timing allowance (%)",
             ylabel="Labelled rallies with a fully correct annotation (%)")
    axis.grid(alpha=0.15)
    save(figure, "video_variation",
         "Every video recovers at least 81.6% of contacts. Both axes show the full 0–100% range.\n"
         "A fully correct rally requires all contacts and players to be correct, with no extra contacts.")


def plot_video_pages(per_video: pd.DataFrame) -> None:
    ordered = per_video.sort_values(["rally_percent", "correct_rallies"], ascending=False)
    pages = [ordered.iloc[:23], ordered.iloc[23:]]
    colours = [BLUE, SAND, GREY]
    legend_labels = ["Timing + player correct", "Timing only", "Missed"]
    handles = [Patch(facecolor=colour, label=label)
               for colour, label in zip(colours, legend_labels, strict=True)]
    for page, rows in enumerate(pages, start=1):
        figure, axis = plt.subplots(figsize=(12, 10.5))
        figure.subplots_adjust(left=0.11, right=0.80, bottom=0.15, top=0.875)
        figure.suptitle(f"Contacts and complete rallies by video ({page}/2)",
                       x=0.04, ha="left", fontsize=16, weight="bold")
        figure.text(0.04, 0.925, "46 ShuttleSet22 videos · ±10 frames at 30 fps\n"
                    "Sorted by fully correct rally rate, highest first", fontsize=11)
        axis.text(1.025, 1.025, "Fully correct rallies", transform=axis.transAxes,
                  fontsize=11, weight="bold")
        positions = list(range(len(rows)))
        for position, row in zip(positions, rows.itertuples(), strict=True):
            counts = [row.player_correct, row.matched_contacts - row.player_correct,
                      row.labelled_contacts - row.matched_contacts]
            shares = [100 * count / row.labelled_contacts for count in counts]
            left = 0
            for share, count, colour in zip(shares, counts, colours, strict=True):
                axis.barh(position, share, left=left, color=colour, height=0.68)
                if share > 8:
                    axis.text(left + share / 2, position, f"{count:,}", ha="center",
                              va="center", color="white" if colour == BLUE else "#222222", fontsize=11)
                left += share
            axis.text(102.5, position,
                      f"{row.correct_rallies}/{row.labelled_rallies} ({row.rally_percent:.1f}%)",
                      va="center", fontsize=11)
        axis.set_yticks(positions, [f"Video {video}" for video in rows.index])
        axis.set(xlim=(0, 100), ylim=(len(rows) - 0.4, -0.7),
                 xlabel="Share of labelled contacts (%) · each bar sums to 100%")
        axis.set_axisbelow(True)
        axis.grid(axis="x", alpha=0.15)
        figure.legend(handles=handles, ncol=3, frameon=False,
                      loc="lower left", bbox_to_anchor=(0.10, 0.067), fontsize=11)
        save(figure, f"video_outcome_breakdown_{page}",
             "Counts inside larger bar segments refer to contacts; the right column counts whole rallies.\n"
             "Timing only: the contact matches in time, without a confirmed correct player.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    per_video = pd.read_csv(args.results / "per_video.csv.gz", index_col="video")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_video_variation(per_video)
    plot_video_pages(per_video)


if __name__ == "__main__":
    main()
