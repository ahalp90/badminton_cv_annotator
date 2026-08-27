"""Render the issue #104 benchmark figures from detailed results."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


BLUE = "#2878B5"
ORANGE = "#E1812C"
INK = "#252525"
GRID = "#D9D9D9"
GREY = "#A6A6A6"
CORPORA = ("ShuttleSet", "ShuttleSet22")
FEATURES = (
    ("shots_per_rally", "Shots per rally", "shots"),
    ("posture_mad", "Posture MAD", "image-coordinate MAD"),
    ("recovery_distance", "Recovery distance", "normalized court distance"),
    ("movement_inefficiency", "Movement inefficiency", "normalized excess path"),
)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _load(path: Path) -> Mapping[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return _mapping(json.load(handle), str(path))


def _feature_values(video: Mapping[str, object]) -> dict[str, list[float]]:
    values = {name: [] for name, _, _ in FEATURES}
    for raw_rally in _sequence(video["rallies"], "rallies"):
        rally = _mapping(raw_rally, "rally")
        values["shots_per_rally"].append(float(rally["shots_per_rally"]))
        for raw_row in _sequence(rally["posture"], "posture"):
            value = _mapping(raw_row, "posture row")["mad"]
            if value is not None:
                values["posture_mad"].append(float(value))
        recovery = _mapping(rally["recovery"], "recovery")
        for raw_row in _sequence(recovery["observations"], "recovery observations"):
            value = _mapping(raw_row, "recovery row")["mean_distance"]
            if value is not None:
                values["recovery_distance"].append(float(value))
        for raw_row in _sequence(rally["movement_inefficiency"], "movement"):
            row = _mapping(raw_row, "movement row")
            for slot in ("top", "bottom"):
                if row[slot] is not None:
                    values["movement_inefficiency"].append(float(row[slot]))
    return values


def _corpus_values(
    report: Mapping[str, object], *, production: bool
) -> dict[str, dict[str, list[float]]]:
    source = (
        _mapping(report["feature_evaluation"], "feature evaluation")
        if production
        else report
    )
    per_video = _mapping(source["per_video"], "per-video results")
    return {
        str(video_id): _feature_values(_mapping(video, str(video_id)))
        for video_id, video in per_video.items()
    }


def _population(
    report: Mapping[str, object], *, production: bool
) -> Mapping[str, Any]:
    source = (
        _mapping(report["feature_evaluation"], "feature evaluation")
        if production
        else report
    )
    return _mapping(source["population"], "feature population")


def _all_values(
    per_video: Mapping[str, Mapping[str, Sequence[float]]], feature: str
) -> list[float]:
    return [value for video in per_video.values() for value in video[feature]]


def _title(figure: Figure, title: str, subtitle: str) -> None:
    figure.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=14, color=INK)
    figure.text(0.08, 0.925, subtitle, ha="left", fontsize=9.5, color=INK)


def _style_axis(axis: Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK)


def _coverage_figure(
    output: Path,
    production: Mapping[str, object],
    comparison: Mapping[str, object],
) -> None:
    feature_rows = (
        ("Shots per rally", "rallies", "rallies"),
        ("Posture MAD", "posture_eligible", "posture_total"),
        ("Recovery distance", "recovery_eligible", "recovery_total"),
        ("Movement inefficiency", "movement_eligible", "movement_total"),
    )
    populations = (production, comparison)
    y = np.arange(len(feature_rows), dtype=float)
    offsets = (-0.18, 0.18)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for corpus_index, (corpus, population, color, hatch) in enumerate(
        zip(CORPORA, populations, (BLUE, ORANGE), (None, "//"), strict=True)
    ):
        numerators = [int(population[numerator]) for _, numerator, _ in feature_rows]
        denominators = [int(population[denominator]) for _, _, denominator in feature_rows]
        percentages = [100.0 * n / d for n, d in zip(numerators, denominators, strict=True)]
        positions = y + offsets[corpus_index]
        bars = ax.barh(
            positions,
            percentages,
            height=0.31,
            color=color if hatch is None else "white",
            edgecolor=color,
            hatch=hatch,
            linewidth=1.2,
            label=corpus,
        )
        for bar, percentage, numerator, denominator in zip(
            bars, percentages, numerators, denominators, strict=True
        ):
            ax.text(
                min(percentage + 0.7, 100.8),
                bar.get_y() + bar.get_height() / 2,
                f"{percentage:.1f}%  ({numerator:,}/{denominator:,})",
                va="center",
                fontsize=8.7,
                color=INK,
            )
    ax.set_yticks(y, [label for label, _, _ in feature_rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 124)
    ax.set_xlabel("Eligible observations (%)")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        ncol=2,
        frameon=False,
    )
    _style_axis(ax)
    _title(
        fig,
        "Trial feature coverage by corpus",
        "40-video ShuttleSet production intervals and 47-video ShuttleSet22 human intervals",
    )
    fig.text(
        0.08,
        0.035,
        "Coverage measures availability, not feature accuracy. Rally duration is omitted because its end offset is unresolved.",
        fontsize=8.7,
        color=INK,
    )
    fig.subplots_adjust(left=0.23, right=0.96, top=0.84, bottom=0.16)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _truth_benchmark_figure(
    output: Path,
    production: Mapping[str, object],
) -> None:
    aggregate = _mapping(production["aggregate"], "production aggregate")
    rally_total = int(aggregate["n_gt_rallies"])
    rally_rows = (
        ("Covered", int(aggregate["covered"]), BLUE),
        ("Split", int(aggregate["split"]), ORANGE),
        ("Missed", int(aggregate["missed"]), GREY),
    )
    score_rows = (
        (
            "Strict contact precision",
            100.0 * float(aggregate["contact_precision"]),
            f'{int(aggregate["contact_matches"]):,}/{int(aggregate["contact_filtered_total"]):,}',
        ),
        (
            "Strict contact recall",
            100.0 * float(aggregate["contact_recall"]),
            f'{int(aggregate["contact_matches"]):,}/{int(aggregate["contact_gt_total"]):,}',
        ),
        (
            "Strict contact F1",
            100.0 * float(aggregate["contact_f1"]),
            "harmonic mean",
        ),
        (
            "Exact shot count",
            100.0 * float(aggregate["ball_round_primary"]),
            f'{int(aggregate["ball_round_primary_correct"]):,}/{int(aggregate["ball_round_primary_total"]):,}',
        ),
        (
            "Final striker",
            100.0 * float(aggregate["player_primary"]),
            f'{int(aggregate["player_primary_correct"]):,}/{int(aggregate["player_primary_total"]):,}',
        ),
        (
            "Server",
            100.0 * float(aggregate["server_primary"]),
            f'{int(aggregate["server_primary_correct"]):,}/{int(aggregate["server_primary_total"]):,}',
        ),
        (
            "Landing half",
            100.0 * float(aggregate["landing_primary"]),
            f'{int(aggregate["landing_primary_correct"]):,}/{int(aggregate["landing_primary_total"]):,}',
        ),
        (
            "Winner",
            100.0 * float(aggregate["getpoint_primary"]),
            f'{int(aggregate["getpoint_primary_correct"]):,}/{int(aggregate["getpoint_primary_total"]):,}',
        ),
    )

    fig, (rally_axis, score_axis) = plt.subplots(
        1,
        2,
        figsize=(12.2, 6.8),
        gridspec_kw={"width_ratios": (0.75, 1.4)},
    )
    bottom = 0.0
    for label, count, color in rally_rows:
        percentage = 100.0 * count / rally_total
        rally_axis.bar(0, percentage, bottom=bottom, width=0.58, color=color)
        rally_axis.text(
            0,
            bottom + percentage / 2,
            f"{label}\n{percentage:.2f}%\n({count:,})",
            ha="center",
            va="center",
            fontsize=9,
            color="white" if label != "Missed" else INK,
        )
        bottom += percentage
    rally_axis.set_ylim(0, 100)
    rally_axis.set_xlim(-0.6, 0.6)
    rally_axis.set_xticks((0,), (f"All ground-truth rallies\nn={rally_total:,}",))
    rally_axis.set_ylabel("Share of ground-truth rallies (%)")
    rally_axis.set_title("Rally classification", loc="left", fontsize=11, color=INK)
    rally_axis.spines[["top", "right"]].set_visible(False)
    rally_axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    rally_axis.set_axisbelow(True)

    labels = [label for label, _, _ in score_rows]
    values = [value for _, value, _ in score_rows]
    y = np.arange(len(score_rows))
    score_axis.barh(y, values, color=BLUE, height=0.62)
    score_axis.set_yticks(y, labels)
    score_axis.invert_yaxis()
    score_axis.set_xlim(0, 66)
    score_axis.set_xlabel("Score against ShuttleSet ground truth (%)")
    score_axis.set_title("Contact and outcome scores", loc="left", fontsize=11, color=INK)
    score_axis.axhline(2.5, color=INK, linewidth=0.8)
    for yy, (_, value, denominator) in zip(y, score_rows, strict=True):
        score_axis.text(
            value + 0.7,
            yy,
            f"{value:.2f}%  ({denominator})",
            va="center",
            fontsize=8.7,
            color=INK,
        )
    _style_axis(score_axis)
    _title(
        fig,
        "Production outputs against ShuttleSet ground truth",
        "40-video issue #103 replay; metrics that govern the issue #22 feature decisions",
    )
    fig.text(
        0.08,
        0.02,
        "Split means one ground-truth rally's contacts cross multiple predicted spans.\n"
        "Strict contact scores use covered rallies and one-to-one matching within ±5 base-30 frames, scaled to source FPS.\n"
        "Outcome denominators exclude merged mappings and other ineligible annotations; F1 is derived from precision and recall.",
        fontsize=8.7,
        color=INK,
    )
    fig.subplots_adjust(left=0.09, right=0.96, top=0.82, bottom=0.21, wspace=0.45)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_figure(
    output: Path,
    production: Mapping[str, Mapping[str, Sequence[float]]],
    comparison: Mapping[str, Mapping[str, Sequence[float]]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.7))
    for axis, (feature, label, units) in zip(axes.flat, FEATURES, strict=True):
        for y, (corpus, values, color, marker) in enumerate(
            zip(
                CORPORA,
                (_all_values(production, feature), _all_values(comparison, feature)),
                (BLUE, ORANGE),
                ("o", "s"),
                strict=True,
            )
        ):
            p10, median, p90 = np.percentile(values, (10, 50, 90))
            axis.hlines(y, p10, p90, color=color, linewidth=2.2)
            axis.plot(
                median,
                y,
                marker=marker,
                markersize=7,
                markerfacecolor="white" if y else color,
                markeredgecolor=color,
                markeredgewidth=1.5,
            )
            axis.text(
                p90,
                y + 0.19,
                f"p10 {p10:.3g}  median {median:.3g}  p90 {p90:.3g}  n={len(values):,}",
                ha="right",
                fontsize=7.8,
                color=INK,
            )
        axis.set_yticks((0, 1), CORPORA)
        axis.set_title(label, loc="left", fontsize=11, color=INK)
        axis.set_xlabel(units)
        axis.set_ylim(1.55, -0.55)
        _style_axis(axis)
    _title(
        fig,
        "Trial feature distributions",
        "Dots show pooled medians; lines show the 10th to 90th percentile range",
    )
    fig.text(
        0.08,
        0.025,
        "The corpora use different rally intervals. Similar values support portability, not accuracy or interchangeability.",
        fontsize=8.7,
        color=INK,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.84,
        bottom=0.12,
        hspace=0.48,
        wspace=0.34,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _per_video_figure(
    output: Path,
    production: Mapping[str, Mapping[str, Sequence[float]]],
    comparison: Mapping[str, Mapping[str, Sequence[float]]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    for axis, (feature, label, units) in zip(axes.flat, FEATURES, strict=True):
        for y, (values_by_video, color, marker) in enumerate(
            zip((production, comparison), (BLUE, ORANGE), ("o", "s"), strict=True)
        ):
            medians = np.asarray(
                [
                    np.median(video[feature])
                    for video in values_by_video.values()
                    if video[feature]
                ]
            )
            jitter = np.linspace(-0.095, 0.095, len(medians))
            axis.scatter(
                medians,
                y + jitter,
                s=18,
                marker=marker,
                facecolors=color if y == 0 else "white",
                edgecolors=color,
                linewidths=0.8,
                alpha=0.8,
            )
            pooled_video_median = float(np.median(medians))
            axis.plot(
                pooled_video_median,
                y,
                marker="|",
                color=INK,
                markersize=22,
                markeredgewidth=2.0,
            )
            axis.text(
                float(np.max(medians)),
                y + 0.20,
                f"{len(medians)} videos; median {pooled_video_median:.3g}",
                ha="right",
                fontsize=7.8,
                color=INK,
            )
        axis.set_yticks((0, 1), CORPORA)
        axis.set_title(label, loc="left", fontsize=11, color=INK)
        axis.set_xlabel(units)
        axis.set_ylim(1.55, -0.55)
        _style_axis(axis)
    _title(
        fig,
        "Per-video trial feature medians",
        "Each mark is one video; the dark vertical tick is the median across video medians",
    )
    fig.text(
        0.08,
        0.025,
        "The spread makes fixture dependence visible. Pooled leave-one-video-out ranges remain the decision gate.",
        fontsize=8.7,
        color=INK,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.82,
        bottom=0.12,
        hspace=0.72,
        wspace=0.34,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuttleset", type=Path, required=True)
    parser.add_argument("--shuttleset22", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shuttleset = _load(args.shuttleset)
    shuttleset22 = _load(args.shuttleset22)
    production_values = _corpus_values(shuttleset, production=True)
    comparison_values = _corpus_values(shuttleset22, production=False)
    if (len(production_values), len(comparison_values)) != (40, 47):
        raise ValueError("expected 40 ShuttleSet and 47 ShuttleSet22 videos")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _truth_benchmark_figure(
        args.output_dir / "issue_104_production_truth_benchmark.png",
        shuttleset,
    )
    _coverage_figure(
        args.output_dir / "issue_104_feature_coverage.png",
        _population(shuttleset, production=True),
        _population(shuttleset22, production=False),
    )
    _summary_figure(
        args.output_dir / "issue_104_feature_distributions.png",
        production_values,
        comparison_values,
    )
    _per_video_figure(
        args.output_dir / "issue_104_feature_per_video.png",
        production_values,
        comparison_values,
    )


if __name__ == "__main__":
    main()
