"""Write the corrected serve-trajectory report and its supporting plots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

PRIMARY_POPULATION = "primary_239_one_to_one"
COVERED_POPULATION = "covered_249_merge_sensitivity"
ALL_POPULATION = "all_292_end_to_end"

COLOURS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "pink": "#CC79A7",
    "sky": "#56B4E9",
    "purple": "#6A3D9A",
    "grey": "#777777",
    "light_grey": "#D9D9D9",
}


def _metric(metrics: dict[str, object], *keys: str) -> Any:
    """Read one nested metric from the checked result object."""
    value: Any = metrics
    for key in keys:
        value = value[key]
    return value


def _plain_label(label: str) -> str:
    """Return a report label for one nearest-stroke category."""
    return {
        "contact_1": "GT serve",
        "contact_2": "GT first return",
        "later": "Later GT stroke",
        "unmatched": "No GT stroke in window",
        "no_anchor": "No accepted anchor",
    }[label]


def _alignment_table(alignment: dict[str, object]) -> str:
    """Format global alignment counts without hiding ambiguous windows."""
    lines = [
        "| Tolerance | GT serve | GT first return | Later GT stroke | No GT stroke in window | More than one GT stroke in window |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for tolerance in ("5", "10", "30"):
        values = alignment[tolerance]
        labels = values["labels"]
        lines.append(
            f"| ±{tolerance} | {labels.get('contact_1', 0)} | {labels.get('contact_2', 0)} | "
            f"{labels.get('later', 0)} | {labels.get('unmatched', 0)} | {values['multiple']} |"
        )
    return "\n".join(lines)


def _alignment_by_fixture_table(alignment: dict[str, object]) -> str:
    """Format all three primary alignment tolerances by fixture."""
    lines = [
        "| Video | Tolerance | Rallies | GT serve | GT first return | Later GT stroke | No GT stroke in window | Multiple in window |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fixture, fixture_values in alignment.items():
        for tolerance in ("5", "10", "30"):
            values = fixture_values[tolerance]
            labels = values["labels"]
            lines.append(
                f"| {fixture} | ±{tolerance} | {values['n']} | {labels.get('contact_1', 0)} | "
                f"{labels.get('contact_2', 0)} | {labels.get('later', 0)} | "
                f"{labels.get('unmatched', 0)} | {values['multiple']} |"
            )
    return "\n".join(lines)


def _segmentation_table(results: pd.DataFrame) -> str:
    """Format covered, split and missed GT rallies by fixture."""
    lines = [
        "| Video | GT rallies | Covered | Split across spans | Missed by segmentation |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, frame in [("All", results), *results.groupby("fixture", sort=True)]:
        counts = frame["boundary"].value_counts()
        lines.append(
            f"| {label} | {len(frame)} | {counts.get('covered', 0)} | "
            f"{counts.get('split', 0)} | {counts.get('missed', 0)} |"
        )
    return "\n".join(lines)


def _population_table(metrics: dict[str, object]) -> str:
    """Format the three deliberately different rally populations."""
    populations = _metric(metrics, "population_counts")
    rows = (
        (ALL_POPULATION, "All GT rallies", "End-to-end view, including segmentation failures"),
        (COVERED_POPULATION, "Covered rallies", "Sensitivity to the current COVERED definition, including merges"),
        (PRIMARY_POPULATION, "One-to-one rallies", "Analyses that need one predicted rally per GT rally"),
    )
    lines = [
        "| Rally group | All videos | sset_01 | sset_15 | sset_21 | Used for |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for key, label, use in rows:
        values = populations[key]
        by_fixture = values["by_fixture"]
        lines.append(
            f"| {label} | {values['global']} | {by_fixture['sset_01']} | "
            f"{by_fixture['sset_15']} | {by_fixture['sset_21']} | {use} |"
        )
    return "\n".join(lines)


def _unmatched_table(sequence: dict[str, object]) -> str:
    """Format later-contact outcomes globally and by fixture."""
    lines = [
        "| Video | Unmatched anchors | Later contact matches serve | No serve match, but return matches | First match is another GT stroke | No later GT match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, values in [("All", sequence["global"]), *sequence["by_fixture"].items()]:
        lines.append(
            f"| {label} | {values['anchors_unmatched_at_tolerance_10']} | "
            f"{values['later_serve_match']} | {values['no_later_serve_but_first_return_match']} | "
            f"{values['other_later_gt_match']} | {values['no_later_gt_match']} |"
        )
    return "\n".join(lines)


def _path_table(metrics: dict[str, object]) -> str:
    """Format the primary evidence funnel under both source masks."""
    values = _metric(metrics, "path_funnel", PRIMARY_POPULATION)
    lines = [
        "| Track source check | Rallies | Continuous run selected | At least 5 points and close enough to contact | Passes the shared jump check | 0.05-BH incoming calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "recurrence_clean": "Exclude recurrence-flagged points",
        "producer_original": "Also exclude producer-marked inpainted points",
    }
    for variant, label in labels.items():
        row = values["global"][variant]
        lines.append(
            f"| {label} | {row['n']} | {row['selected_paths']} | {row['path_available']} | "
            f"{row['common_path_eligible']} | {row['robust_trend_incoming']} |"
        )
    return "\n".join(lines)


def _path_by_fixture_table(metrics: dict[str, object]) -> str:
    """Format shared-rule evidence availability by fixture."""
    values = _metric(metrics, "path_funnel", PRIMARY_POPULATION, "by_fixture")
    lines = [
        "| Video | One-to-one rallies | Usable paths, recurrence check | Incoming calls | Usable paths, plus producer mask | Incoming calls |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fixture, fixture_values in values.items():
        recurrence = fixture_values["recurrence_clean"]
        producer = fixture_values["producer_original"]
        lines.append(
            f"| {fixture} | {recurrence['n']} | {recurrence['common_path_eligible']} | "
            f"{recurrence['robust_trend_incoming']} | {producer['common_path_eligible']} | "
            f"{producer['robust_trend_incoming']} |"
        )
    return "\n".join(lines)


def _rule_table(rule_rows: pd.DataFrame) -> str:
    """Format the four fixed comparisons on unique ±10 truth."""
    global_rows = rule_rows[rule_rows["scope"].eq("global")]
    labels = {
        ("recurrence_clean", "historical"): "Historical absolute-closure rule; recurrence check",
        ("recurrence_clean", "robust_trend"): "0.05-BH trend rule; recurrence check",
        ("producer_original", "historical"): "Historical rule; recurrence plus producer mask",
        ("producer_original", "robust_trend"): "0.05-BH trend rule; recurrence plus producer mask",
    }
    lines = [
        "| Fixed comparison | Paths eligible for this rule | Correct return calls | False return calls | Returns missed | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        variant, rule = key
        row = global_rows[
            global_rows["path_definition"].eq(variant) & global_rows["rule"].eq(rule)
        ].iloc[0]
        lines.append(
            f"| {label} | {int(row['rule_paths_eligible'])} | {int(row['tp'])} | "
            f"{int(row['fp'])} | {int(row['fn'])} | {row['precision']:.1%} | {row['recall']:.1%} |"
        )
    return "\n".join(lines)


def _trend_inpaint_table(rule_rows: pd.DataFrame) -> str:
    """Format the controlled inpaint comparison for the fixed trend rule."""
    global_rows = rule_rows[
        rule_rows["scope"].eq("global") & rule_rows["rule"].eq("robust_trend")
    ].set_index("path_definition")
    labels = {
        "recurrence_clean": "Exclude recurrence-flagged points",
        "producer_original": "Also exclude producer-marked inpainted points",
    }
    lines = [
        "| Track source check | Labelled paths with usable motion | Correct return calls | False return calls | Returns missed |",
        "|---|---:|---:|---:|---:|",
    ]
    for path_definition, label in labels.items():
        row = global_rows.loc[path_definition]
        lines.append(
            f"| {label} | {int(row['rule_paths_eligible'])}/135 | {int(row['tp'])}/17 | "
            f"{int(row['fp'])}/118 | {int(row['fn'])}/17 |"
        )
    return "\n".join(lines)


def _rule_by_fixture_table(rule_rows: pd.DataFrame) -> str:
    """Format the predeclared 0.05-BH recurrence rule by fixture."""
    rows = rule_rows[
        ~rule_rows["scope"].eq("global")
        & rule_rows["path_definition"].eq("recurrence_clean")
        & rule_rows["rule"].eq("robust_trend")
    ]
    lines = [
        "| Video | Unique ±10 truth | GT returns | Usable paths | Correct return calls | False return calls | Returns missed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['scope']} | {int(row['n_truth'])} | {int(row['gt_first_returns'])} | "
            f"{int(row['rule_paths_eligible'])} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} |"
        )
    return "\n".join(lines)


def _finite_median(series: pd.Series) -> float:
    """Return the median after excluding non-finite diagnostic ratios."""
    values = series.astype(float).to_numpy()
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if len(finite) else math.nan


def _diagnostic_table(diagnostics: pd.DataFrame, group_column: str) -> str:
    """Format continuous recurrence-mask diagnostics by one grouping."""
    eligible = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
    ].copy()
    lines = [
        "| Group | Paths | Median fitted decrease (BH) | Median residual scatter (BH) | Median trend-to-jitter |",
        "|---|---:|---:|---:|---:|",
    ]
    if group_column == "call_correct":
        labels: dict[object, str] = {True: "Correct calls", False: "Incorrect calls"}
    else:
        labels = {"serve": "GT serves", "first_return": "GT first returns"}
    for value, label in labels.items():
        group = eligible[eligible[group_column].eq(value)]
        lines.append(
            f"| {label} | {len(group)} | {_finite_median(group['fitted_decrease_bh']):.3f} | "
            f"{_finite_median(group['residual_rms_bh']):.3f} | "
            f"{_finite_median(group['trend_to_jitter']):.3f} |"
        )
    return "\n".join(lines)


def _path_length_diagnostic_table(diagnostics: pd.DataFrame) -> str:
    """Format diagnostics in coarse path-length groups chosen for description only."""
    eligible = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
    ].copy()
    eligible["length_group"] = pd.cut(
        eligible["path_frames"],
        bins=[4, 5, 9, np.inf],
        labels=["5 points", "6-9 points", "10+ points"],
    )
    lines = [
        "| Observed path length | Paths | Median fitted decrease (BH) | Median residual scatter (BH) |",
        "|---|---:|---:|---:|",
    ]
    for label in ("5 points", "6-9 points", "10+ points"):
        group = eligible[eligible["length_group"].eq(label)]
        lines.append(
            f"| {label} | {len(group)} | {_finite_median(group['fitted_decrease_bh']):.3f} | "
            f"{_finite_median(group['residual_rms_bh']):.3f} |"
        )
    return "\n".join(lines)


def _server_score_table(metrics: dict[str, object], population: str) -> str:
    """Format the server methods needed to answer the investigation question."""
    scores = _metric(metrics, "server_scores", population, "global")
    methods = (
        "old alternating fit",
        "anchor player",
        "historical rule, recurrence mask",
        "0.05-BH trend rule, recurrence mask",
        "0.05-BH trend rule, recurrence plus producer mask",
        "0.05-BH trend evidence only",
        "0.05-BH trend then prepend unknown player",
        "0.05-BH trend then prepend other player",
    )
    labels = {
        "old alternating fit": "Released alternating fit",
        "anchor player": "Assume the earliest contact player served",
        "historical rule, recurrence mask": "Flip player when the historical rule says incoming",
        "0.05-BH trend rule, recurrence mask": (
            "Use earliest-contact player; flip when the 0.05-BH trend says incoming"
        ),
        "0.05-BH trend rule, recurrence plus producer mask": (
            "Same fallback and 0.05-BH flip; also mask producer inpaint"
        ),
        "0.05-BH trend evidence only": "Motion answer only; abstain without usable evidence",
        "0.05-BH trend then prepend unknown player": "Prepend one unknown contact before alternating fit",
        "0.05-BH trend then prepend other player": "Prepend inferred server before alternating fit",
    }
    denominator = int(scores["old alternating fit"]["n"])
    lines = [
        f"| Server method | Correct | Answers made | Overall accuracy (n={denominator}) |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        row = scores[method]
        lines.append(
            f"| {labels[method]} | {row['correct']}/{row['n']} | {row['known']}/{row['n']} | "
            f"{row['accuracy']:.1%} |"
        )
    return "\n".join(lines)


def _server_population_sensitivity_table(metrics: dict[str, object]) -> str:
    """Show the main direct rule under the three non-interchangeable populations."""
    lines = [
        "| Rally group | Released fit | Earliest-contact player | Earliest-contact fallback plus 0.05-BH flip |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        PRIMARY_POPULATION: "239 one-to-one",
        COVERED_POPULATION: "249 covered, including merges",
        ALL_POPULATION: "292 end-to-end, including segmentation failures",
    }
    for population, label in labels.items():
        scores = _metric(metrics, "server_scores", population, "global")
        cells = []
        for method in (
            "old alternating fit",
            "anchor player",
            "0.05-BH trend rule, recurrence mask",
        ):
            row = scores[method]
            cells.append(f"{row['correct']}/{row['n']} ({row['accuracy']:.1%})")
        lines.append(f"| {label} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _server_by_fixture_table(metrics: dict[str, object]) -> str:
    """Format primary server results by fixture."""
    by_fixture = _metric(metrics, "server_scores", PRIMARY_POPULATION, "by_fixture")
    lines = [
        "| Video | Rallies | Released fit | Earliest-contact player | 0.05-BH direct motion rule |",
        "|---|---:|---:|---:|---:|",
    ]
    for fixture, scores in by_fixture.items():
        old = scores["old alternating fit"]
        anchor = scores["anchor player"]
        trend = scores["0.05-BH trend rule, recurrence mask"]
        lines.append(
            f"| {fixture} | {trend['n']} | {old['correct']} | {anchor['correct']} | {trend['correct']} |"
        )
    return "\n".join(lines)


def plot_anchor_alignment(metrics: dict[str, object], plot_dir: Path) -> None:
    """Plot nearest GT stroke categories at all three declared tolerances."""
    alignment = _metric(metrics, "alignment", PRIMARY_POPULATION, "global")
    tolerances = ("5", "10", "30")
    categories = ("contact_1", "contact_2", "later", "unmatched")
    colours = (COLOURS["blue"], COLOURS["orange"], COLOURS["purple"], COLOURS["light_grey"])
    figure, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    bottoms = np.zeros(len(tolerances), dtype=float)
    x_positions = np.arange(len(tolerances))
    for category, colour in zip(categories, colours, strict=True):
        counts = np.array(
            [alignment[tolerance]["labels"].get(category, 0) for tolerance in tolerances],
            dtype=float,
        )
        bars = axis.bar(x_positions, counts, bottom=bottoms, color=colour, label=_plain_label(category))
        for index, (bar, count, bottom) in enumerate(zip(bars, counts, bottoms, strict=True)):
            bar.set_alpha(1.0 if index == 1 else 0.45)
            if index == 1:
                bar.set_edgecolor("#222222")
                bar.set_linewidth(1.2)
            if count >= 9:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + count / 2,
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=10,
                )
        bottoms += counts
    for index, tolerance in enumerate(tolerances):
        multiple = alignment[tolerance]["multiple"]
        axis.text(index, 246, f"{multiple} with multiple\nGT strokes in window", ha="center", va="bottom")
    axis.set(
        xticks=x_positions,
        xticklabels=("±5 strict", "±10 main baseline", "±30 sanity check"),
        ylabel="One-to-one rallies (n=239)",
        ylim=(0, 273),
        title=(
            "Which GT stroke is nearest to the earliest accepted contact?\n"
            "The anchor is an ordinary accepted contact candidate, not a serve detector"
        ),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncols=4)
    axis.get_xticklabels()[1].set_fontweight("bold")
    figure.savefig(plot_dir / "anchor_alignment.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_unmatched_followup(metrics: dict[str, object], plot_dir: Path) -> None:
    """Plot what later accepted contacts reveal after an unmatched anchor."""
    values = _metric(metrics, "unmatched_anchor_sequences", "global")
    outcome_labels = (
        "Any later contact\nmatches serve",
        "Otherwise, first\nreturn matches",
        "Otherwise, another\nGT stroke matches",
        "Otherwise, no\nlater GT match",
    )
    outcome_counts = (
        values["later_serve_match"],
        values["no_later_serve_but_first_return_match"],
        values["other_later_gt_match"],
        values["no_later_gt_match"],
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    bars = axis.bar(
        np.arange(len(outcome_labels)),
        outcome_counts,
        color=(COLOURS["blue"], COLOURS["orange"], COLOURS["purple"], COLOURS["light_grey"]),
    )
    axis.bar_label(bars, padding=3)
    axis.set(
        xticks=np.arange(len(outcome_labels)),
        xticklabels=outcome_labels,
        ylabel="Anchors (n=97)",
        title=(
            "Later accepted contacts recover the serve or first return in 85 of 97 rallies\n"
            "All matches use ±10; categories prioritise serve, then return, then other, then none"
        ),
        ylim=(0, 58),
    )
    axis.tick_params(axis="x", labelrotation=8)
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(plot_dir / "unmatched_anchor_followup.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_motion_evidence_and_inpaint(
    metrics: dict[str, object],
    plot_dir: Path,
) -> None:
    """Show the scarcity of motion evidence under the two fixed source checks."""
    funnel = _metric(metrics, "path_funnel", PRIMARY_POPULATION, "global")
    variants = (
        ("recurrence_clean", "Exclude recurrence-flagged points", COLOURS["blue"]),
        ("producer_original", "Also exclude producer-marked inpaint", COLOURS["orange"]),
    )
    x_positions = np.arange(len(variants))
    usable = np.array([funnel[variant]["common_path_eligible"] for variant, _, _ in variants])
    unavailable = 239 - usable
    figure, axis = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
    bars = axis.bar(
        x_positions,
        usable,
        color=[colour for _, _, colour in variants],
    )
    axis.bar_label(bars, labels=[f"{count} usable" for count in usable], label_type="center", color="white")
    bars = axis.bar(
        x_positions,
        unavailable,
        bottom=usable,
        color=COLOURS["light_grey"],
    )
    axis.bar_label(
        bars,
        labels=[f"{count} without evidence" for count in unavailable],
        label_type="center",
    )
    axis.set(
        xticks=x_positions,
        xticklabels=("Exclude recurrence-\nflagged points", "Also exclude producer-\nmarked inpaint"),
        ylabel="One-to-one rallies (n=239)",
        ylim=(0, 250),
        title=(
            "Usable pre-contact motion is rare and falls from 24 to 14 rallies\n"
            "Removing producer-marked inpaint changes the evidence source, not the 0.05-BH threshold"
        ),
    )
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(plot_dir / "motion_evidence_and_inpaint.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trend_diagnostics(diagnostics: pd.DataFrame, plot_dir: Path) -> None:
    """Plot the predeclared trend measurement and untuned noise diagnostics."""
    eligible = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
    identity_groups = (
        ("serve", "GT serve", COLOURS["blue"]),
        ("first_return", "GT first return", COLOURS["orange"]),
    )
    for identity, label, colour in identity_groups:
        group = eligible[eligible["gt_anchor_identity"].eq(identity)]
        axes[0, 0].scatter(
            group["path_frames"],
            group["fitted_decrease_bh"],
            label=f"{label} (n={len(group)})",
            color=colour,
            s=55,
            alpha=0.85,
        )
        axes[0, 1].scatter(
            group["path_frames"],
            group["residual_rms_bh"],
            label=f"{label} (n={len(group)})",
            color=colour,
            s=55,
            alpha=0.85,
        )
    axes[0, 0].axhline(
        0.05,
        color="#222222",
        linestyle="--",
        label="≥0.05 BH: incoming / call first return",
    )
    axes[0, 0].set(
        xlabel="Observed path points",
        ylabel="Fitted decrease (apparent BH)",
        title="Approach trend against path length",
    )
    axes[0, 1].set(
        xlabel="Observed path points",
        ylabel="Residual scatter (apparent BH)",
        title="Track scatter against path length",
    )
    axes[0, 0].legend(fontsize=9)

    correct_groups = (
        (True, "Correct calls", COLOURS["blue"]),
        (False, "Incorrect calls", COLOURS["pink"]),
    )
    for correct, label, colour in correct_groups:
        group = eligible[eligible["call_correct"].eq(correct)]
        axes[1, 0].scatter(
            group["fitted_decrease_bh"],
            group["residual_rms_bh"],
            label=f"{label} (n={len(group)})",
            color=colour,
            s=60,
            alpha=0.85,
        )
        finite_ratio = group[np.isfinite(group["trend_to_jitter"].astype(float))]
        axes[1, 1].scatter(
            finite_ratio["path_frames"],
            finite_ratio["trend_to_jitter"],
            label=f"{label} (n={len(finite_ratio)})",
            color=colour,
            s=60,
            alpha=0.85,
        )
    axes[1, 0].axvline(0.05, color="#222222", linestyle="--")
    axes[1, 0].set(
        xlabel="Fitted decrease (apparent BH)",
        ylabel="Residual scatter (apparent BH)",
        title="0.05-BH call correctness and track scatter",
    )
    axes[1, 1].axhline(0, color="#777777", linewidth=1)
    axes[1, 1].set(
        xlabel="Observed path points",
        ylabel="Fitted decrease / residual scatter",
        title="Trend-to-jitter is descriptive, not a cutoff",
    )
    axes[1, 0].legend(fontsize=9)
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.suptitle(
        "0.05-BH trend-rule diagnostics for 19 usable paths with unique ±10 serve/return truth"
    )
    figure.savefig(plot_dir / "trend_and_jitter_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_representative_errors(
    results: pd.DataFrame,
    path_points: pd.DataFrame,
    diagnostics: pd.DataFrame,
    plot_dir: Path,
) -> list[str]:
    """Plot every usable-path error made by the fixed recurrence-mask rule."""
    eligible = diagnostics[
        diagnostics["path_definition"].eq("recurrence_clean")
        & diagnostics["common_path_eligible"].astype(bool)
        & ~diagnostics["call_correct"].astype(bool)
    ].copy()
    eligible["error_type"] = np.where(
        eligible["incoming_call"].astype(bool),
        "False return call on a GT serve",
        "Missed GT first return",
    )
    chosen = eligible.sort_values(["error_type", "residual_rms_bh"]).reset_index(drop=True)
    figure, axes = plt.subplots(4, 2, figsize=(12.5, 16), constrained_layout=True)
    filenames: list[str] = []
    result_index = results.set_index(["fixture", "video_id", "set_id", "rally"])
    for axis, (_, row) in zip(axes.flat, chosen.iterrows(), strict=True):
        key = (row["fixture"], int(row["video_id"]), row["set_id"], int(row["rally"]))
        points = path_points[
            path_points["fixture"].eq(row["fixture"])
            & path_points["video_id"].eq(row["video_id"])
            & path_points["set_id"].eq(row["set_id"])
            & path_points["rally"].eq(row["rally"])
            & path_points["path_definition"].eq("recurrence_clean")
        ].sort_values("sample_index")
        distances = points["distance_bh"].to_numpy(dtype=float)
        path_time = np.linspace(0.0, 1.0, len(distances))
        result = result_index.loc[key]
        intercept = float(result["recurrence_clean_robust_intercept_bh"])
        decrease = float(row["fitted_decrease_bh"])
        axis.scatter(path_time, distances, color=COLOURS["blue"], label="Observed distance")
        axis.plot(path_time, intercept - decrease * path_time, color=COLOURS["orange"], label="Robust fitted trend")
        axis.set(
            xlabel="Position through observed path",
            ylabel="Distance to contact player (apparent BH)",
            title=(
                f"{row['error_type']}\n{row['fixture']} {row['set_id']} rally {int(row['rally'])}; "
                f"{len(points)} points"
            ),
        )
        axis.text(
            0.02,
            0.03,
            f"Decrease {decrease:.3f} BH\nResidual {row['residual_rms_bh']:.3f} BH\n"
            f"Trend/jitter {row['trend_to_jitter']:.2f}",
            transform=axis.transAxes,
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": COLOURS["grey"], "alpha": 0.9},
        )
        axis.grid(alpha=0.2)
        filenames.append(f"{row['fixture']} {row['set_id']} rally {int(row['rally'])}")
    axes[0, 0].legend(fontsize=9)
    figure.suptitle("All eight 0.05-BH trend-rule mistakes among 19 usable unique-truth paths")
    figure.savefig(plot_dir / "trend_rule_errors.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    return filenames


def plot_server_attribution(metrics: dict[str, object], plot_dir: Path) -> None:
    """Plot the main server methods on one explicit primary denominator."""
    scores = _metric(metrics, "server_scores", PRIMARY_POPULATION, "global")
    methods = (
        "old alternating fit",
        "anchor player",
        "0.05-BH trend rule, recurrence mask",
        "0.05-BH trend then prepend other player",
    )
    labels = (
        "Released\nalternating fit",
        "Earliest-contact\nplayer",
        "Earliest-contact fallback\n+ motion-backed flips",
        "Inferred-player prepend\n+ alternating refit",
    )
    values = [scores[method]["accuracy"] for method in methods]
    x_positions = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = axis.bar(
        x_positions,
        values,
        color=(COLOURS["grey"], COLOURS["sky"], COLOURS["blue"], COLOURS["orange"]),
    )
    for bar, method in zip(bars, methods, strict=True):
        row = scores[method]
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{row['correct']}/{row['n']} correct\n{row['known']}/{row['n']} answers",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set(
        xticks=x_positions,
        xticklabels=labels,
        ylabel="Correct server over all 239 one-to-one rallies",
        ylim=(0, 0.79),
        title=(
            "The direct motion correction helps; feeding it back into the alternating fit loses the gain\n"
            "The direct method uses the earliest-contact player by default and changes only 15 incoming calls"
        ),
    )
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(plot_dir / "server_attribution.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_report(
    results: pd.DataFrame,
    rule_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metrics: dict[str, object],
    error_cases: list[str],
    report_path: Path,
) -> None:
    """Write a standalone report with the important answer first."""
    primary_alignment = _metric(metrics, "alignment", PRIMARY_POPULATION)
    covered_alignment = _metric(metrics, "alignment", COVERED_POPULATION, "global")
    sequence = _metric(metrics, "unmatched_anchor_sequences")
    primary_scores = _metric(metrics, "server_scores", PRIMARY_POPULATION, "global")
    summary_report = f"""# What the earliest accepted contact tells us about the serve

## Bottom line

All server scores in this paragraph use the same 239 one-to-one rallies. The released alternating fit gets **{primary_scores['old alternating fit']['correct']}** right. Using the player at the earliest accepted contact gets **{primary_scores['anchor player']['correct']}** right. Usable pre-contact motion exists in only **24/239** rallies, so motion can only be a small correction. Using the earliest-contact player by default, then changing the answer when motion clearly approaches that player, reaches **{primary_scores['0.05-BH trend rule, recurrence mask']['correct']}**. The new information helps when used directly. Prepending the inferred missing serve and rerunning the alternating fit falls to **{primary_scores['0.05-BH trend then prepend other player']['correct']}**. The old fit therefore loses most of the direct gain. The practical priority is to improve which accepted contact becomes the anchor and how often a clean motion path exists. More trajectory complexity cannot help rallies that never reach a trustworthy contact or usable path.

## Why anchor selection comes first

At the normal ±10 timing tolerance, **97 of 239** earliest contacts do not match a ShuttleSet stroke. Later accepted contacts recover the serve in 49 of those rallies and the first return in another 36. This means many failures occur before motion classification. An early ordinary candidate often takes the anchor position, or the accepted sequence misses the serve while retaining the return.

## What should we do next?

Improve anchor selection and motion-path availability before adding a richer trajectory classifier. Keep the 0.05-BH rule unchanged while testing it on new videos. This focuses work on the two bottlenecks that limit the current method: the wrong contact can become the anchor, and only 24/239 rallies contain usable motion evidence.

## Extended summary (optional)

This investigation asks whether the shuttle approaches the player at the earliest accepted contact. Incoming motion suggests that this contact is the first return, not the serve. The other player is then the likely server. The useful version of this idea is modest: start with the player measured at the earliest contact and use motion only to correct a small number of calls.

Three rally counts describe different parts of the evaluation. ShuttleSet contains **292 ground-truth rallies**. That number shows end-to-end performance and includes segmentation failures. The current segmentation marks **249 rallies as covered**, but ten of those rallies share five predicted spans. They remain a sensitivity view of the existing `COVERED` definition. The main result uses **239 one-to-one rallies**, where one predicted span maps to one ground-truth rally. This avoids scoring the same contact sequence twice.

These populations are not interchangeable. Using all 292 rallies for trajectory scoring would mix missing predicted spans with motion mistakes. Using all 249 covered rallies would reuse one accepted-contact sequence for both ground-truth rallies inside each merged span. The 239-rally set removes those two problems. The broader populations remain useful later, where they show how segmentation failures and the current merge definition affect end-to-end accuracy.

The earliest accepted contact is an ordinary output of the released contact detector. It is not designed to find serves. The detector starts from shuttle impulses and player proximity, then applies its usual wrist, suppression and exclusion checks. The analysis measures which player is nearest at that accepted frame. It does not use the released alternating server fit to choose that player.

This direct player measurement gives a simple baseline. If the contact is the serve, its player is the server. If the contact is the first return, the other player served. The baseline works only as well as the chosen contact. The timing comparison therefore comes before any motion classification or server score.

The practical timing check allows ±10 base-30fps frames between the accepted contact and an annotated stroke. On the 239 one-to-one rallies, the earliest contact is nearest the serve in 119 cases, the first return in 19, and a later stroke in 4. In **97 rallies**, no annotated stroke lies inside the window. Five windows contain more than one annotated stroke, which the analysis flags separately. Stricter ±5 and broader ±30 views are reported later as supporting checks, but ±10 is the main baseline.

The 97 unmatched earliest contacts show that many failures happen before trajectory classification. A later accepted contact matches the serve in 49 rallies. In another 36, no later contact matches the serve, but one matches the first return. The first pattern is consistent with an early ordinary candidate taking the anchor position. The second is consistent with the serve being absent from the accepted sequence while the return remains detectable. Only nine first match another later stroke, and three have no later match. Better anchor selection is therefore a central finding, not a side issue.

Motion coverage is the next limitation. Only **24 of 239 rallies** have a continuous pre-contact path that passes the shared quality checks. The direct server method does not make 239 trajectory-based decisions. It uses the earliest-contact player as its answer whenever motion is unavailable or does not say incoming. Motion changes the answer only when the fixed 0.05-body-height trend rule finds a clear approach towards that player.

This separation matters when reading the final accuracy. A rally without usable motion is not evidence that the shuttle moved away from the player. It is a rally where the trajectory method cannot answer. The full server method still answers because it falls back to the directly measured contact player.

That limited correction still helps. The released alternating fit gets {primary_scores['old alternating fit']['correct']}/239 rallies right. The earliest-contact player alone gets {primary_scores['anchor player']['correct']}/239. Adding the motion-backed correction reaches **{primary_scores['0.05-BH trend rule, recurrence mask']['correct']}/239**. The gain is not evidence that trajectory solves server attribution generally. It shows that a small amount of usable incoming-motion evidence can correct some otherwise direct contact-player calls.

The prepend experiment gives an important negative result. Supplying an inferred missing serve and rerunning the old alternating fit reaches only {primary_scores['0.05-BH trend then prepend other player']['correct']}/239. The new information is useful when applied directly, but the alternating refit largely throws that improvement away. This result argues against recursive refitting as the next step.

The practical next step is to improve which accepted contact becomes the anchor and how often a clean pre-contact path exists. The fixed 0.05-body-height rule should remain unchanged until it is tested on new videos. More complicated motion classification would add machinery while the larger contact-selection and evidence-availability failures remain unresolved.
"""
    detail = f"""

## What are the 292, 249 and 239 rallies?

The main result uses 239 rallies because each has one predicted span and one contact sequence for one ground-truth rally. The 249-rally and 292-rally views answer broader sensitivity questions; they do not replace that primary comparison.

{_population_table(metrics)}

**Population trail:** 292 ground-truth rallies → 249 covered rallies → 239 one-to-one rallies.

The 249 covered rows use 244 predicted spans. There are 239 one-rally spans and five spans that each cover two ground-truth rallies. The merged rows remain visible in the covered sensitivity results, but the primary analysis never scores their shared contact sequence twice.

The investigation keeps five stages separate:

1. Segmentation maps a ground-truth rally to a predicted span or fails.
2. The earliest accepted contact matches a plausible stroke or does not.
3. A continuous pre-contact path is unavailable, rejected by the quality checks, or usable.
4. A usable path falls above or below the fixed incoming threshold.
5. The resulting server attribution is correct or incorrect.

## Is the first accepted contact actually the serve?

Usually it is closest to the serve, but **97 of 239** earliest contacts do not match any annotated stroke at the main ±10 tolerance. The anchor is therefore useful, but too unreliable to treat as a detected serve.

The offset is `(accepted contact frame - GT stroke frame) × 30 / source fps`. Negative values mean the accepted contact occurs earlier. Each tolerance keeps the nearest stroke identity even when several strokes lie inside the window. The last column reports that ambiguity separately.

{_alignment_table(primary_alignment['global'])}

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

The ±10 bar is the practical baseline. The ±5 strict view and ±30 sanity check show how the result changes with tolerance. The broad ±30 window contains several strokes in 117 rallies, so it is not clean identity truth.

## What happens when the first contact is wrong?

Later accepted contacts recover the serve or first return in **85 of the 97** unmatched rallies. Many unmatched anchors therefore reflect an early candidate taking the anchor position or a missing serve, rather than a failure of the later sequence as a whole.

{_unmatched_table(sequence)}

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Later contacts are checked independently against every annotated stroke at the same ±10 tolerance. A stroke is not consumed after one match. The first later match occurs at accepted-contact rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12. Rank is one-based in the full accepted sequence, so the first later contact has rank 2.

Four first matches have more than one annotated stroke inside ±10. Twenty-seven sequences reuse one stroke ordinal for more than one accepted contact. These flags make the non-consuming check explicit and do not change the outcome categories.

The 55 anchors still unmatched at ±30 are best described as **GT-incompatible candidates under the ±30 sanity criterion**. This wording does not claim a visually verified false contact.

## Can incoming motion help?

Yes, but usable pre-contact motion exists in only **24 of 239** one-to-one rallies. The motion result is a small correction to the earliest-contact fallback, not a stand-alone answer for every rally.

![Usable motion evidence under both TrackNet source checks](outputs/plots/motion_evidence_and_inpaint.png)

The path search looks back at most 30 base-30fps frames within the same court scene and uses the continuous run closest to contact. Both fixed rules require at least five samples, a final sample close to the contact, recurrence guard `NO_FLAG`, finite player-distance and body-height evidence, and no gross single-step jump.

{_path_table(metrics)}

“Continuous run selected” means that at least one source point exists. “At least 5 points and close enough” applies the sample-count and contact-gap checks. “Passes the shared jump check” is the usable-evidence count for the 0.05-BH decision. A rally outside that count has no motion answer.

To judge the motion call itself, the analysis needs an anchor that can be labelled confidently as either the serve or first return. There are **135 such rallies** at ±10: 118 serves and 17 first returns. Nineteen have usable recurrence-checked paths. The fixed 0.05-BH rule correctly identifies 9 returns, makes 4 false return calls on serves, and misses 8 returns. Four of those misses have usable motion below the threshold; four have no usable path.

The 24 usable paths and 15 incoming calls over all 239 rallies are broader availability counts. Restricting to the 135 labelled rallies leaves 19 usable paths and 13 incoming calls. The other five usable paths have an unmatched or later-stroke anchor and cannot enter serve-versus-return scoring.

## Does removing inpainted TrackNet points help?

Removing producer-marked filled or interpolated points eliminates the four false return calls, but it also removes useful evidence. The same fixed 0.05-BH rule then finds 7 returns instead of 9. This is a precision-coverage trade-off, not a retuned comparison.

{_trend_inpaint_table(rule_rows)}

The threshold and every other motion decision remain unchanged between rows. Usable labelled paths fall from 19 to 10. Under the stricter source check, one missed return has usable motion below 0.05 BH and nine have no usable path. Every video loses evidence.

## Does the inferred missing serve improve server identification?

The new information helps when used directly. Feeding it back into the released alternating fit mostly loses the improvement: **163/239 falls to 127/239**.

![Four central server-attribution results](outputs/plots/server_attribution.png)

The direct method starts with the earliest-contact player. It changes that answer only for the 15 rallies where usable motion says the shuttle is incoming. When evidence is unavailable or the path does not say incoming, the earliest-contact player remains the answer.

The full primary table keeps the historical rule, stricter producer mask, evidence-only result and both prepend variants visible:

{_server_score_table(metrics, PRIMARY_POPULATION)}

Accuracy retains all 239 rallies in the denominator. “Answers made” shows whether a method supplied Top or Bottom. The inferred-player prepend answers 217 rallies, while the direct fallback answers all 239.

## Detailed motion methods and diagnostics

The detail below explains the fixed rule comparison and track-noise measurements. It supports the main result, but neither diagnostic changes a call.

### Historical absolute closure versus the 0.05-BH trend

The robust trend takes the median slope between every pair of shuttle-to-player distance samples. Time is normalised from zero to one across the observed path. The fitted decrease is the negative slope. The call is “incoming” only when that decrease reaches 0.05 apparent player body heights.

Both rules first use the shared sample-count, contact-gap, recurrence, finite-evidence and jump checks. The historical rule then adds its 0.25-BH total-movement eligibility floor. The trend rule does not. This is why the historical row has 18 eligible paths rather than 19 under the recurrence check, and 9 rather than 10 after removing producer-marked inpaint. Net closure of 0.25 BH and the 55% approaching-step condition then decide the historical call.

{_rule_table(rule_rows)}

All four rows use the same 135 confidently labelled anchors. “Returns missed” includes both usable paths below threshold and returns without usable evidence. The 0.25 values came from the old analysis, while 55% was selected under old ±5/249 scoring. None is an independently calibrated physical threshold. The 0.05-BH value is an engineering judgement fixed before corrected scoring and was never swept.

{_rule_by_fixture_table(rule_rows)}

### What trend and jitter show

The 0.05-BH fitted decrease alone makes the call. Residual RMS measures scatter around the robust trend. Trend-to-jitter divides fitted decrease by that scatter. Neither diagnostic is an eligibility test or another classifier.

{_diagnostic_table(diagnostics, 'gt_anchor_identity')}

{_diagnostic_table(diagnostics, 'call_correct')}

{_path_length_diagnostic_table(diagnostics)}

The path-length groups are descriptive summaries only. They were not used to make or tune the call.

![Continuous trend and jitter diagnostics](outputs/plots/trend_and_jitter_diagnostics.png)

Serves and first returns have similar median fitted decreases in this small usable set. Correct calls show a much larger median fitted decrease and trend-to-jitter than incorrect calls. Incorrect calls also have slightly more residual scatter. These patterns do not justify another cutoff.

The error plot shows all eight mistakes with usable recurrence-checked paths: four false return calls and four missed returns. The cases are {', '.join(error_cases)}.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Supporting breakdowns (optional)

The tables below retain the per-video and sensitivity evidence without placing it in the main reading path.

### Segmentation by video

{_segmentation_table(results)}

### Contact alignment by video

{_alignment_by_fixture_table(primary_alignment['by_fixture'])}

The merge-sensitive 249-row view is close to the primary result at ±10: {covered_alignment['10']['labels'].get('contact_1', 0)} nearest serves, {covered_alignment['10']['labels'].get('contact_2', 0)} nearest first returns, {covered_alignment['10']['labels'].get('later', 0)} later strokes and {covered_alignment['10']['labels'].get('unmatched', 0)} unmatched anchors. It has {covered_alignment['10']['multiple']} multiple-stroke windows. Similar counts do not make merged rows suitable for one-rally trajectory scoring.

### Motion availability by video

{_path_by_fixture_table(metrics)}

### Server sensitivity and video results

{_server_population_sensitivity_table(metrics)}

The 292-row view includes all 43 segmentation failures. Those failures have no anchor-based answer. The 249-row view includes ten merged ground-truth rows. Neither sensitivity view replaces the 239-rally primary result.

{_server_by_fixture_table(metrics)}

## Limits

- Only 17 first-return anchors have unique ±10 truth. Only 19 unique-truth anchors have usable recurrence-only motion paths.
- Apparent body-height normalisation is image based. It is not a physical court distance, and its meaning can change with player scale and camera geometry.
- A five-point path is allowed. The 0.05-BH engineering threshold is therefore deliberately modest, but it is still uncalibrated.
- TrackNet residual scatter is measured from the observed path. We do not have independent ground truth for TrackNet positional error.
- The ±30 view often contains several GT strokes. It is a sanity check, not clean stroke identity.
- No new manual labels were added. “GT-incompatible” means unmatched to existing GT under the stated tolerance, not visually proven false.
- The three videos are the same videos used in the historical exploration. The corrected thresholds were fixed before scoring, but the reported scores are not external validation.

## Output files

- `outputs/rallies.csv.gz`: one checked row for each of 292 GT rallies.
- `outputs/spans.csv.gz`: all 344 half-open predicted spans.
- `outputs/path_points.csv.gz`: the 1,012 sampled path points used to rebuild motion measurements.
- `outputs/fixed_rules.csv.gz`: the four fixed rule/mask comparisons globally and by video.
- `outputs/trend_diagnostics.csv.gz`: continuous trend and jitter values for the 135 unique ±10 truth anchors under both masks.
- `outputs/metrics.json.gz`: checked population, alignment, funnel and server summaries.
"""
    report_path.write_text(summary_report + detail, encoding="utf-8")


def write_final_outputs(
    results: pd.DataFrame,
    path_points: pd.DataFrame,
    rule_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
    metrics: dict[str, object],
    plot_dir: Path,
    report_path: Path,
) -> None:
    """Replace stale plots and write the corrected standalone report."""
    (plot_dir.parent / "thresholds.csv.gz").unlink(missing_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for path in plot_dir.glob("*.png"):
        path.unlink()
    case_dir = plot_dir / "cases"
    if case_dir.is_dir():
        for path in case_dir.glob("*.png"):
            path.unlink()
        case_dir.rmdir()

    plot_anchor_alignment(metrics, plot_dir)
    plot_unmatched_followup(metrics, plot_dir)
    plot_motion_evidence_and_inpaint(metrics, plot_dir)
    plot_trend_diagnostics(diagnostics, plot_dir)
    error_cases = plot_representative_errors(results, path_points, diagnostics, plot_dir)
    plot_server_attribution(metrics, plot_dir)
    _write_report(results, rule_rows, diagnostics, metrics, error_cases, report_path)
