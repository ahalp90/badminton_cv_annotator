"""Render the small summary plots used by the contact detector reports."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

TOLERANCES = (5, 10, 15)
BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#6F6F6F"
BLACK = "#202020"


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _percentage(value: float) -> float:
    return 100.0 * value


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=BLACK)


def _label_bars(axis: plt.Axes, bars: Any, *, decimals: int = 1) -> None:
    labels = [f"{bar.get_height():.{decimals}f}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8, color=BLACK)


def plot_contact_coverage(contact_score: dict[str, Any], tree_results: dict[str, Any], output: Path) -> None:
    proposal_results = contact_score["overall"]
    region_results = tree_results["region_ceiling"]["seeded_union"]["operational"]
    eligible_results = tree_results["region_ceiling"]["eligible_intervals"]["operational"]
    series = (
        ("Current final", proposal_results["filtered"], GREY, "-", False),
        ("Raw proposals", proposal_results["raw"], ORANGE, "-", False),
        ("Court-view intervals", eligible_results, SKY, "--", True),
        ("Region v2 with pre-roll", region_results, BLUE, "-", True),
    )
    annotation_offsets = {
        "Current final": -14,
        "Raw proposals": 8,
        "Court-view intervals": -14,
        "Region v2 with pre-roll": 8,
    }

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True, layout="constrained")
    for axis, contact_type, title in zip(
        axes,
        ("non_serve", "serve"),
        ("Non-serve contact coverage", "Serve coverage"),
        strict=True,
    ):
        for label, values, colour, line_style, is_region in series:
            if is_region:
                key = "nonserve_recall" if contact_type == "non_serve" else "serve_recall"
                recalls = [_percentage(values[str(tolerance)][key]) for tolerance in TOLERANCES]
            else:
                recalls = [
                    _percentage(values[str(tolerance)][contact_type]["recall"])
                    for tolerance in TOLERANCES
                ]
            axis.plot(
                TOLERANCES,
                recalls,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                label=label,
                color=colour,
                linestyle=line_style,
            )
            axis.annotate(
                f"{recalls[1]:.1f}",
                (10, recalls[1]),
                xytext=(0, annotation_offsets[label]),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        axis.set_title(title, fontsize=11, weight="bold")
        axis.set_xlabel("Allowed error (base-30 frames)")
        axis.set_xticks(TOLERANCES)
        axis.set_ylim(40, 103)
        _style_axis(axis)
    axes[0].set_ylabel("Contacts covered (%)")
    axes[1].legend(loc="lower right", frameon=False, fontsize=9)
    figure.suptitle("Broader search regions remove the old proposal ceiling", fontsize=13, weight="bold")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def _sum_rally_metric(contact_score: dict[str, Any], metric: str) -> tuple[int, int]:
    correct = 0
    available = 0
    for fixture in contact_score["rally_fits"].values():
        correct += int(fixture[metric]["correct"])
        available += int(fixture[metric]["available"])
    return correct, available


def plot_player_side_accuracy(contact_score: dict[str, Any], output: Path) -> None:
    direct = contact_score["half_attribution"]["filtered"]["10"]["all"]
    rows = (
        (
            "Direct contact side",
            (int(direct["current"]["correct"]), int(direct["current"]["available_predictions"])),
            (int(direct["ankle"]["correct"]), int(direct["ankle"]["available_predictions"])),
        ),
        (
            "Final hitter after\nrally alternation",
            _sum_rally_metric(contact_score, "final_striker_current"),
            _sum_rally_metric(contact_score, "final_striker_geometry"),
        ),
        (
            "Server after\nrally alternation",
            _sum_rally_metric(contact_score, "server_current"),
            _sum_rally_metric(contact_score, "server_geometry"),
        ),
    )
    labels = [label for label, _current, _ankle in rows]
    current_values = [100.0 * correct / available for _label, (correct, available), _ankle in rows]
    ankle_values = [100.0 * correct / available for _label, _current, (correct, available) in rows]

    positions = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.6, 4.5), layout="constrained")
    current_bars = axis.bar(positions - width / 2, current_values, width, label="Current box/net rule", color=BLUE)
    ankle_bars = axis.bar(positions + width / 2, ankle_values, width, label="Ankle-height rule", color=ORANGE)
    _label_bars(axis, current_bars)
    _label_bars(axis, ankle_bars)
    axis.set_title("Changing the side geometry does not fix rally attribution", fontsize=13, weight="bold")
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, loc="upper right")
    _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def plot_tree_trial_summary(tree_results: dict[str, Any], output: Path) -> None:
    models = tree_results["models"]
    configurations = (
        ("HGB\nphysical", models["histogram_boosting"]["physics"]),
        ("HGB\n+ context", models["histogram_boosting"]["physics_context"]),
        ("RF\nphysical", models["random_forest"]["physics"]),
        ("RF\n+ context", models["random_forest"]["physics_context"]),
    )
    metrics = (("Precision", "precision", BLUE), ("Recall", "recall", ORANGE), ("F1", "f1", PURPLE))

    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), layout="constrained")
    positions = np.arange(len(configurations))
    width = 0.24
    for metric_index, (label, key, colour) in enumerate(metrics):
        values = [_percentage(result["metrics"]["10"][key]) for _name, result in configurations]
        bars = axes[0].bar(positions + (metric_index - 1) * width, values, width, label=label, color=colour)
        _label_bars(axes[0], bars)
    axes[0].set_title("Tree comparison at ±10", fontsize=11, weight="bold")
    axes[0].set_ylabel("Event score (%)")
    axes[0].set_xticks(positions, [name for name, _result in configurations])
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, loc="lower left")
    _style_axis(axes[0])

    folds = models["histogram_boosting"]["physics"]["folds"]
    fixtures = [fold["test_fixture"] for fold in folds]
    fold_metrics = (("F1", "f1", PURPLE), ("Non-serve recall", "nonserve_recall", BLUE), ("Serve recall", "serve_recall", ORANGE))
    fixture_positions = np.arange(len(fixtures))
    for metric_index, (label, key, colour) in enumerate(fold_metrics):
        values = [_percentage(fold["metrics"]["10"][key]) for fold in folds]
        bars = axes[1].bar(
            fixture_positions + (metric_index - 1) * width,
            values,
            width,
            label=label,
            color=colour,
        )
        _label_bars(axes[1], bars)
    axes[1].set_title("HGB physical varies by fixture", fontsize=11, weight="bold")
    axes[1].set_xticks(fixture_positions, fixtures)
    axes[1].set_ylim(0, 100)
    axes[1].legend(frameon=False, loc="lower left")
    _style_axis(axes[1])

    figure.suptitle("Histogram boosting is the best tree, but serves remain weak", fontsize=13, weight="bold")
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    contact_score = _read_json_gz(args.data_root / "contact_evidence_score.json.gz")
    tree_results = _read_json_gz(args.data_root / "region_v2" / "tree_contact_results.json.gz")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_contact_coverage(contact_score, tree_results, args.output_dir / "contact_coverage.png")
    plot_player_side_accuracy(contact_score, args.output_dir / "player_side_accuracy.png")
    plot_tree_trial_summary(tree_results, args.output_dir / "tree_trial_summary.png")


if __name__ == "__main__":
    main()
