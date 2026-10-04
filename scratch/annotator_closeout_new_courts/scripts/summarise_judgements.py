"""Merge the footage judgements for the sampled misses and plot them by court decision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ("a", "b", "c")
BLUE = "#0072B2"
ORANGE = "#D55E00"
LIGHT_ORANGE = "#E69F00"
RESULT_NAMES = {"hit and player agree": "Hit and player agree", "timing label wrong": "Timing label wrong",
                "clear footage disagreement": "Clear footage disagreement", "unclear": "Unclear"}
COURT_STATES = (("Accepted", BLUE, "Court accepted"), ("No court found", ORANGE, "No court found"),
                ("Court found; player check failed", LIGHT_ORANGE, "Court found; player check failed"))


def merged_judgements() -> pd.DataFrame:
    judgements = []
    for batch in BATCHES:
        with open(ROOT / f"worklog/judgements_{batch}.json") as source:
            judgements.extend(json.load(source))
    sample = pd.read_csv(ROOT / "results/miss_sample.csv.gz")
    merged = sample.merge(pd.DataFrame(judgements), on="sample_id", validate="one_to_one")
    assert len(merged) == len(sample) == 24
    assert set(merged.judgement) <= set(RESULT_NAMES)
    return merged


def plot(rows: pd.DataFrame) -> None:
    positions = np.arange(len(RESULT_NAMES))
    figure, axis = plt.subplots(figsize=(12, 6.5))
    left = np.zeros(len(RESULT_NAMES), dtype=int)
    for state, colour, label in COURT_STATES:
        counts = np.array([int(((rows.judgement == result) & (rows.court_rejection == state)).sum())
                           for result in RESULT_NAMES])
        axis.barh(positions, counts, left=left, height=0.62, color=colour, label=label)
        for position, start, count in zip(positions, left, counts, strict=True):
            if count:
                axis.text(start + count / 2, position, str(count), va="center", ha="center", color="white",
                          weight="bold", fontsize=12)
        left += counts
    for position, total in zip(positions, left, strict=True):
        axis.text(int(total) + 0.25, position, str(int(total)), va="center", fontsize=13)
    axis.set_yticks(positions, list(RESULT_NAMES.values()))
    axis.invert_yaxis()
    axis.set_xlim(0, int(left.max()) + 2)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("Number of sampled missed contacts")
    axis.set_ylabel("Direct frame check")
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("24 randomly sampled missed contacts — new-court model", x=0.08, y=0.97, ha="left",
                    weight="bold", fontsize=17)
    figure.text(0.08, 0.90, "Timing check: ±10 frames at 30 fps. Court state is saved at the labelled frame.",
                fontsize=13)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=12)
    figure.subplots_adjust(left=0.30, right=0.96, top=0.78, bottom=0.22)
    figure.savefig(ROOT / "figures/contact_sample_results.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> None:
    rows = merged_judgements()
    rows.to_csv(ROOT / "results/miss_judgements.csv.gz", index=False)
    plot(rows)
    print(pd.crosstab(rows.judgement, rows.court_rejection, margins=True).to_string())
    columns = ["sample_id", "video", "position", "target_side", "court_rejection", "view", "court_visible",
               "players_visible", "judgement", "source_frame", "estimated_hit_frame"]
    print(rows[columns].to_string(index=False))


if __name__ == "__main__":
    run()
