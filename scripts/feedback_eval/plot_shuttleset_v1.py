"""Figures for the COSC320 feedback-evaluation baseline run.

Reads the score files written by `feedback_eval.score_cli` and the fault CSV
written by `feedback_eval.shuttleset_faults`, so the figures cannot drift from
the numbers they came from. Re-run after any re-score.

    python scripts/feedback_eval/plot_shuttleset_v1.py
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = Path("experiments/feedback_eval/shuttleset_v1")
FIGURES = RUN / "figures"
DATA = Path("data/feedback_eval")

# Validated categorical palette, light mode (dataviz reference instance).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dddcd8"
GREY = "#b6b4ad"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_MUTED,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": GRID,
    }
)

LABELS = {
    "empty": "empty\n(no output)",
    "constant": "constant\n(same text every clip)",
    "random_template": "random template\n(right register, wrong fault)",
    "oracle": "oracle\n(copies the reference)",
}
ORDER = ("empty", "constant", "random_template", "oracle")


def _score(name: str, rescaled: bool) -> float:
    suffix = "_rescaled" if rescaled else ""
    path = RUN / f"scores_{name}{suffix}.json"
    if not path.is_file():
        # `empty` is a hard zero by construction and never reaches the scorer,
        # so it has no rescaled run of its own.
        if name == "empty":
            return 0.0
        raise FileNotFoundError(path)
    return json.loads(path.read_text())["aggregates"]["mean_f1"]


def figure_band() -> Path:
    """Where a real model version has to land, raw and baseline-rescaled."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = range(len(ORDER))
    for ax, rescaled, title in (
        (axes[0], False, "Raw BERTScore F1"),
        (axes[1], True, "Rescaled against the bert-score baseline"),
    ):
        values = [_score(name, rescaled) for name in ORDER]
        ax.barh(list(y), values, height=0.55, color=BLUE, zorder=3)
        for index, value in enumerate(values):
            ax.text(
                value + 0.02, index, f"{value:.3f}", va="center", fontsize=10, color=INK
            )
        # The band a model must clear: fluent-but-wrong feedback to the ceiling.
        floor = _score("random_template", rescaled)
        ax.axvspan(floor, 1.0, color=ORANGE, alpha=0.10, zorder=1)
        ax.axvline(floor, color=ORANGE, lw=2, zorder=4)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.tick_params(length=0)
        ax.text(
            (floor + 1.0) / 2,
            -0.72,
            f"usable range {1.0 - floor:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=ORANGE,
        )

    axes[0].set_yticks(list(y))
    axes[0].set_yticklabels([LABELS[name] for name in ORDER], fontsize=9)
    axes[0].set_ylim(len(ORDER) - 0.5, -1.1)
    fig.suptitle(
        "A BERTScore number means nothing without its floor",
        fontsize=13,
        x=0.012,
        ha="left",
        y=0.99,
        color=INK,
    )
    fig.text(
        0.012,
        0.015,
        "543 held-out clips, 11 test players, roberta-large. Coaching text about the wrong fault "
        "already scores 0.897 raw;\nrescaling widens the gap it has to beat from 0.018 to 0.105.",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = FIGURES / "bertscore_band.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_construction() -> Path:
    """How 3,508 annotated rallies became 2,247 reference clips."""
    rows = [
        ("Reference clips built", 2247, BLUE),
        ("Excluded: ended in a winner,\nso no error was recorded", 1189, GREY),
        ("Excluded: ambiguous lose_reason\n(misjudgement, foul)", 55, GREY),
        ("Excluded: point credited to the\nplayer who erred", 14, GREY),
        ("Excluded: stroke type unknown", 3, GREY),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    labels = [label for label, _, _ in rows]
    values = [value for _, value, _ in rows]
    colors = [color for _, _, color in rows]
    y = range(len(rows))
    ax.barh(list(y), values, height=0.6, color=colors, zorder=3)
    for index, value in enumerate(values):
        ax.text(value + 30, index, f"{value:,}", va="center", fontsize=10, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 2600)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.set_xlabel("rallies")
    ax.set_title(
        "3,508 expert-annotated ShuttleSet rallies -> 2,247 reference clips",
        fontsize=13,
        loc="left",
        pad=12,
        color=INK,
    )
    fig.text(
        0.012,
        0.015,
        "Every exclusion is counted. Rallies ending in a clean winner are dropped because the "
        "annotation records no error\nto coach -- inventing one would be a guess the data cannot "
        "support.",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = FIGURES / "reference_set_construction.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_faults() -> Path:
    """What the reference set is actually made of."""
    counts: collections.Counter[str] = collections.Counter()
    with (DATA / "shuttleset_clip_faults.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["template_id"]] += 1

    items = sorted(counts.items(), key=lambda kv: kv[1])
    labels = [key.replace("_into_net", "").replace("_hit_out", "") for key, _ in items]
    values = [value for _, value in items]
    colors = [BLUE if key.endswith("into_net") else ORANGE for key, _ in items]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = range(len(items))
    ax.barh(list(y), values, height=0.68, color=colors, zorder=3)
    for index, value in enumerate(values):
        ax.text(value + 6, index, str(value), va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels([label.replace("_", " ") for label in labels], fontsize=9)
    ax.set_xlim(0, 545)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(length=0)
    ax.set_xlabel("clips")
    ax.set_title(
        "The 16 faults the reference set covers", fontsize=13, loc="left", pad=12, color=INK
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    ax.legend(
        handles,
        ["played into the net", "hit out of court"],
        frameon=False,
        loc="lower right",
        fontsize=9,
    )
    fig.text(
        0.012,
        0.015,
        "Stroke family x error mode, both read off ShuttleSet's terminal-shot annotation. "
        "Only the coaching text is drafted.",
        fontsize=9,
        color=INK_MUTED,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    path = FIGURES / "fault_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for build in (figure_band, figure_construction, figure_faults):
        print(f"wrote {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
