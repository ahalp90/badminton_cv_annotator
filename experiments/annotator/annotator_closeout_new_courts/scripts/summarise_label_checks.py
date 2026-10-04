"""Turn the blind footage judgements of the label-check cases into label verdicts and plot them.

Judges saw only a query frame and stills around it. The verdict says whether the official label
holds up against what they saw:

- extra final hit: no contact means the label is right and the extra hit is the model's; a
  contact by the predicted player means the labels miss a hit;
- swapped side: a contact by the labelled player means the label is right and the model has the
  side wrong; a contact by the predicted player means the label side is wrong;
- shifted rally: no labelled-side contact within 10 frames of the label, with a contact just
  outside, means the label timing is wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BATCHES = ("a", "b", "c")
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#999999"
CONTACT_SIDES = {"far player contact": "Top", "near player contact": "Bot"}
GROUP_NAMES = {"tail": "Extra hit after the last label\n16 of 262 rallies", "side": "Rally with swapped sides\nall 10 rallies",
               "drift": "Video 12 shifted rally\n2 of 4 rallies"}
VERDICTS = (("Label right", BLUE), ("Label wrong", ORANGE), ("Unclear", GREY))
TOLERANCE_FRAMES = 10


def merged_judgements() -> pd.DataFrame:
    judgements = []
    for batch in BATCHES:
        with open(ROOT / f"worklog/label_judgements_{batch}.json") as source:
            judgements.extend(json.load(source))
    cases = pd.read_csv(ROOT / "results/label_check_cases.csv.gz", dtype={"video": str})
    trace = pd.read_csv(ROOT / "results/label_trace.csv.gz", dtype={"video": str})
    merged = cases.merge(pd.DataFrame(judgements), on="case_id", validate="one_to_one")
    merged = merged.merge(trace[["case_id", "type", "ending"]], on="case_id", validate="one_to_one")
    assert len(merged) == len(cases) == 28
    return merged


def verdict(case: pd.Series) -> str:
    contact_side = CONTACT_SIDES.get(case.contact)
    if case.group == "tail":
        if case.contact == "no contact":
            return "Label right"
        if contact_side == case.predicted_side:
            return "Label wrong"
    if case.group == "side":
        if contact_side == case.label_side:
            return "Label right"
        if contact_side == case.predicted_side:
            return "Label wrong"
    if case.group == "drift" and case.contact == "contact outside window":
        offset = case.estimated_contact_frame - case.query_frame
        if abs(offset) > TOLERANCE_FRAMES:
            return "Label wrong"
    return "Unclear"


def plot(rows: pd.DataFrame) -> None:
    groups = list(GROUP_NAMES)
    positions = np.arange(len(groups))
    figure, axis = plt.subplots(figsize=(12, 5.8))
    left = np.zeros(len(groups), dtype=int)
    for name, colour in VERDICTS:
        counts = np.array([int(((rows.group == group) & (rows.verdict == name)).sum()) for group in groups])
        axis.barh(positions, counts, left=left, height=0.62, color=colour, label=name)
        for position, start, count in zip(positions, left, counts, strict=True):
            if count:
                axis.text(start + count / 2, position, str(count), va="center", ha="center", color="white",
                          weight="bold", fontsize=12)
        left += counts
    for position, total in zip(positions, left, strict=True):
        axis.text(int(total) + 0.25, position, str(int(total)), va="center", fontsize=13)
    axis.set_yticks(positions, [GROUP_NAMES[group] for group in groups])
    axis.invert_yaxis()
    axis.set_xlim(0, int(left.max()) + 2)
    axis.set_xlabel("Number of footage cases")
    axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Bulk failures checked against the footage — new-court model", x=0.08, y=0.97, ha="left",
                    weight="bold", fontsize=17)
    figure.text(0.08, 0.89, "Blind judges saw stills around one frame per case. Timing check: ±10 frames at 30 fps.",
                fontsize=13)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=12)
    figure.subplots_adjust(left=0.30, right=0.96, top=0.78, bottom=0.24)
    figure.savefig(ROOT / "figures/label_check_results.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> None:
    rows = merged_judgements()
    rows["verdict"] = rows.apply(verdict, axis=1)
    rows.to_csv(ROOT / "results/label_check_judgements.csv.gz", index=False)
    plot(rows)
    print(pd.crosstab(rows.group, rows.verdict, margins=True).to_string())
    columns = ["case_id", "group", "video", "rally_id", "label_side", "predicted_side", "contact",
               "estimated_contact_frame", "query_frame", "confidence", "ending", "verdict"]
    print(rows[columns].sort_values(["group", "case_id"]).to_string(index=False))


if __name__ == "__main__":
    run()
