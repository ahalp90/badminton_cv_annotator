"""Render the report charts from the checked development records."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch

POPULATION = 239
BLUE = "#5a7a9a"
PALE_BLUE = "#c8dde8"
SAND = "#c8a060"
PALE_SAND = "#f5ead0"
PURPLE = "#9070a0"
GREY = "#d8d8d8"
DARK_GREY = "#555555"

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def read_csv_gz(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(figure: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES_DIR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style_axis(axis: Axes) -> None:
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#eeeeee", linewidth=0.8)
    axis.set_axisbelow(True)


def render_server_attribution() -> None:
    labels = [
        "Released alternating fit",
        "Earliest accepted contact",
        "PR #82 local motion",
        "Narrow high-shot correction",
        "Preferred layered rule",
        "Rank-1 fallback sensitivity",
    ]
    correct = [124, 152, 163, 164, 170, 171]
    incorrect = [POPULATION - value for value in correct]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = list(range(len(labels)))
    axis.barh(positions, correct, color=BLUE, label="Correct server side")
    axis.barh(positions, incorrect, left=correct, color=GREY, label="Incorrect server side")
    for position, value in zip(positions, correct, strict=True):
        delta = value - 163
        delta_text = "PR #82" if delta == 0 else f"{delta:+d} vs PR #82"
        axis.text(value - 3, position, f"{value}", ha="right", va="center", color="white", fontweight="bold")
        axis.text(POPULATION + 3, position, delta_text, ha="left", va="center", color=DARK_GREY)

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 270)
    axis.set_xlabel(f"Rallies in the fixed {POPULATION}-rally development set")
    axis.set_title("Correct server attribution improves as local motion is used more carefully", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    style_axis(axis)
    save_figure(figure, "server_attribution.png")


def render_visible_start_attribution(metrics: dict[str, Any]) -> None:
    strict = metrics["original_followup"]["strict_outgoing_first"]["tolerance_10"]
    relaxed = metrics["original_followup"]["relaxed_trajectory_evidence"]["sequential_tolerance_10"]
    labels = [
        "PR #82 local motion",
        "Strict outgoing search",
        "Less brittle outgoing search",
        "Narrow high-shot correction",
        "Preferred layered rule",
        "Rank-1 fallback sensitivity",
    ]
    correct = [125, strict["final_correct"], relaxed["final_correct"], 127, 132, 131]
    unresolved = [
        0,
        strict["transitions"]["pre_contact_unknown"] + strict["transitions"]["no_credible_contact"],
        relaxed["transitions"]["pre_contact_unavailable"] + relaxed["transitions"]["no_credible_contact"],
        0,
        0,
        0,
    ]
    incorrect = [POPULATION - right - unknown for right, unknown in zip(correct, unresolved, strict=True)]

    figure, axis = plt.subplots(figsize=(10, 5.5))
    positions = list(range(len(labels)))
    axis.barh(positions, correct, color=BLUE, label="Correct visible-start attribution")
    axis.barh(positions, unresolved, left=correct, color=SAND, label="Trajectory could not answer")
    left = [right + unknown for right, unknown in zip(correct, unresolved, strict=True)]
    axis.barh(positions, incorrect, left=left, color=GREY, label="Incorrect attribution")
    for position, value in zip(positions, correct, strict=True):
        axis.text(value - 3, position, f"{value}", ha="right", va="center", color="white", fontweight="bold")
    for position, value, start in zip(positions, unresolved, correct, strict=True):
        if value:
            axis.text(start + value / 2, position, f"{value}\nno answer", ha="center", va="center", fontsize=8)

    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, POPULATION)
    axis.set_xlabel(f"Rallies in the fixed {POPULATION}-rally development set")
    axis.set_title("A useful server-side rule does not automatically find the right visible start", loc="left", fontweight="bold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    style_axis(axis)
    save_figure(figure, "visible_start_attribution.png")


def paired_timing_changes(result_path: Path, baseline_by_key: dict[tuple[str, ...], bool]) -> tuple[int, int]:
    fixes = 0
    damages = 0
    for row in read_csv_gz(result_path):
        key = tuple(row[name] for name in ("fixture", "video_id", "set_id", "rally"))
        baseline_correct = baseline_by_key[key]
        current_correct = row["temporal_slot_correct"] == "True"
        fixes += current_correct and not baseline_correct
        damages += baseline_correct and not current_correct
    return fixes, damages


def render_gains_and_losses(metrics: dict[str, Any]) -> None:
    baseline_rows = read_csv_gz(DATA_DIR / "high_shot_correction_results.csv.gz")
    baseline_by_key = {
        tuple(row[name] for name in ("fixture", "video_id", "set_id", "rally")): row["baseline_start_correct"] == "True"
        for row in baseline_rows
    }
    preferred_timing = paired_timing_changes(RESULTS_DIR / "preferred_server_rule.csv.gz", baseline_by_key)
    sensitivity_timing = paired_timing_changes(RESULTS_DIR / "rank1_fallback_sensitivity.csv.gz", baseline_by_key)

    server_labels = ["High-shot correction", "Preferred layered rule", "Rank-1 sensitivity"]
    server_fixes = [1, 20, 21]
    server_damages = [0, 13, 13]
    timing_labels = [
        "Strict outgoing search",
        "Less brittle outgoing search",
        "High-shot correction",
        "Preferred layered rule",
        "Rank-1 sensitivity",
    ]
    timing_fixes = [16, 26, 2, preferred_timing[0], sensitivity_timing[0]]
    timing_damages = [34, 13, 0, preferred_timing[1], sensitivity_timing[1]]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    panels = (
        (axes[0], server_labels, server_fixes, server_damages, "Server-side changes"),
        (axes[1], timing_labels, timing_fixes, timing_damages, "Visible-start changes"),
    )
    for axis, labels, fixes, damages, title in panels:
        positions = list(range(len(labels)))
        axis.barh(positions, fixes, color=BLUE, label="Fixes")
        axis.barh(positions, [-value for value in damages], color=PURPLE, label="Damages")
        for position, value in zip(positions, fixes, strict=True):
            axis.text(value + 0.6, position, f"+{value}", va="center", color=BLUE, fontweight="bold")
        for position, value in zip(positions, damages, strict=True):
            if value:
                axis.text(-value - 0.6, position, f"−{value}", ha="right", va="center", color=PURPLE, fontweight="bold")
        axis.axvline(0, color=DARK_GREY, linewidth=1)
        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Rallies changed relative to PR #82")
        axis.spines[["top", "right", "left", "bottom"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#eeeeee", linewidth=0.8)
        axis.set_axisbelow(True)

    axes[0].set_xlim(-18, 25)
    axes[1].set_xlim(-40, 29)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("Every extra rule trades some repairs for new mistakes", x=0.02, ha="left", fontweight="bold")
    save_figure(figure, "gains_and_losses.png")


def add_box(axis: Axes, xy: tuple[float, float], width: float, height: float, text: str, colour: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#5a6a78",
        facecolor=colour,
    )
    axis.add_patch(box)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def add_arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "-|>", "color": "#5a6a78", "lw": 1.7})
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.02, label, ha="center", fontsize=9)


def render_layered_rule_infographic() -> None:
    figure, axis = plt.subplots(figsize=(14, 7.6))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title("How the layered server rule turns a noisy shuttle path into a cautious answer", loc="left", fontweight="bold", fontsize=16)
    axis.text(
        0,
        0.94,
        "Each layer answers one small question. When the path cannot support an answer, the rule falls back instead of guessing.",
        fontsize=11,
        color=DARK_GREY,
    )

    add_box(
        axis,
        (0.02, 0.65),
        0.18,
        0.2,
        "1. Keep a usable path\n\nSeveral continuous frames\nnear the contact\nNo giant one-frame leap",
        PALE_BLUE,
    )
    add_box(
        axis,
        (0.27, 0.65),
        0.18,
        0.2,
        "2. Confirm a real hit\n\nThe shuttle moves away\nafter the contact",
        PALE_SAND,
    )
    add_box(
        axis,
        (0.52, 0.65),
        0.18,
        0.2,
        "3. Read the approach\n\nMoves towards player first:\nreturn-like\nOtherwise: serve-like",
        PALE_BLUE,
    )
    add_box(
        axis,
        (0.77, 0.65),
        0.2,
        0.2,
        "4. Avoid guessing\n\n91 direct answers\n148 PR #82 fallbacks",
        PALE_SAND,
    )
    add_arrow(axis, (0.2, 0.75), (0.27, 0.75))
    add_arrow(axis, (0.45, 0.75), (0.52, 0.75))
    add_arrow(axis, (0.70, 0.75), (0.77, 0.75))

    add_box(axis, (0.16, 0.29), 0.2, 0.17, "68 return-like hits\nChoose the other player", PALE_BLUE)
    add_box(axis, (0.40, 0.29), 0.2, 0.17, "23 serve-like hits\nChoose the contact player", PALE_BLUE)
    add_box(axis, (0.64, 0.29), 0.2, 0.17, "148 unclear paths\nKeep the PR #82 answer", GREY)
    add_arrow(axis, (0.87, 0.65), (0.26, 0.46), "supported branches")
    add_arrow(axis, (0.87, 0.65), (0.50, 0.46))
    add_arrow(axis, (0.87, 0.65), (0.74, 0.46), "fallback")

    add_box(axis, (0.30, 0.04), 0.4, 0.13, "Development result\n170/239 correct server sides • 20 fixes • 13 damages", "#b8d4c8")
    add_arrow(axis, (0.26, 0.29), (0.42, 0.17))
    add_arrow(axis, (0.50, 0.29), (0.50, 0.17))
    add_arrow(axis, (0.74, 0.29), (0.58, 0.17))
    save_figure(figure, "layered_rule_infographic.png")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10})
    metrics = read_json_gz(RESULTS_DIR / "development_metrics.json.gz")
    render_server_attribution()
    render_visible_start_attribution(metrics)
    render_gains_and_losses(metrics)
    render_layered_rule_infographic()


if __name__ == "__main__":
    main()
