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
    lines = [
        "| Server method | Correct over all rallies | Answers made | Accuracy |",
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
        for bar, count, bottom in zip(bars, counts, bottoms, strict=True):
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
    rank_counts = values["first_gt_match_rank"]
    rank_labels = ("2nd accepted", "3rd accepted", "4th accepted", "5th or later", "No later match")
    later_rank = sum(count for rank, count in rank_counts.items() if int(rank) >= 5)
    rank_values = (
        rank_counts.get("2", 0),
        rank_counts.get("3", 0),
        rank_counts.get("4", 0),
        later_rank,
        values["no_later_gt_match"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    bars = axes[0].bar(
        np.arange(len(outcome_labels)),
        outcome_counts,
        color=(COLOURS["blue"], COLOURS["orange"], COLOURS["purple"], COLOURS["light_grey"]),
    )
    axes[0].bar_label(bars, padding=3)
    axes[0].set(
        xticks=np.arange(len(outcome_labels)),
        xticklabels=outcome_labels,
        ylabel="Anchors (n=97)",
        title="What did a later accepted contact match?",
        ylim=(0, 58),
    )
    bars = axes[1].bar(
        np.arange(len(rank_labels)),
        rank_values,
        color=(COLOURS["blue"], COLOURS["sky"], COLOURS["purple"], COLOURS["pink"], COLOURS["light_grey"]),
    )
    axes[1].bar_label(bars, padding=3)
    axes[1].set(
        xticks=np.arange(len(rank_labels)),
        xticklabels=rank_labels,
        ylabel="Anchors (n=97)",
        title="Rank of the first later GT-matched contact",
        ylim=(0, 65),
    )
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=12)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "After 97 earliest anchors had no GT match within ±10\n"
        "Later contacts also use ±10; left categories prioritise serve, then return, then other, then none"
    )
    figure.savefig(plot_dir / "unmatched_anchor_followup.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_motion_evidence_and_inpaint(
    results: pd.DataFrame,
    metrics: dict[str, object],
    rule_rows: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Show evidence availability separately from fixed-rule outcomes."""
    funnel = _metric(metrics, "path_funnel", PRIMARY_POPULATION, "global")
    stages = ("selected_paths", "path_available", "common_path_eligible")
    stage_labels = ("Continuous run\nselected", "Enough points and\nnear contact", "Passes shared\nquality checks")
    variants = (
        ("recurrence_clean", "Exclude recurrence-flagged points", COLOURS["blue"]),
        ("producer_original", "Also exclude producer-marked inpaint", COLOURS["orange"]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), constrained_layout=True)
    x_positions = np.arange(len(stages))
    width = 0.36
    for index, (variant, label, colour) in enumerate(variants):
        counts = [funnel[variant][stage] for stage in stages]
        bars = axes[0].bar(x_positions + (index - 0.5) * width, counts, width, label=label, color=colour)
        axes[0].bar_label(bars, padding=3)
    axes[0].set(
        xticks=x_positions,
        xticklabels=stage_labels,
        ylabel="One-to-one rallies (n=239)",
        ylim=(0, 68),
        title="How often is motion evidence available?",
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    global_rules = rule_rows[
        rule_rows["scope"].eq("global") & rule_rows["rule"].eq("robust_trend")
    ].set_index("path_definition")
    x_positions = np.arange(len(variants))
    state_columns = (
        ("No usable evidence", COLOURS["light_grey"]),
        ("Usable; says serve", COLOURS["sky"]),
        ("Usable; says first return", COLOURS["orange"]),
    )
    state_counts: list[tuple[int, int, int]] = []
    for variant, _, _ in variants:
        row = global_rules.loc[variant]
        usable = int(row["rule_paths_eligible"])
        incoming = int(row["incoming_calls"])
        state_counts.append((int(row["n_truth"]) - usable, usable - incoming, incoming))
    bottoms = np.zeros(len(variants), dtype=float)
    for state_index, (state_label, colour) in enumerate(state_columns):
        counts = np.array([values[state_index] for values in state_counts], dtype=float)
        bars = axes[1].bar(x_positions, counts, bottom=bottoms, label=state_label, color=colour)
        for bar, count, bottom in zip(bars, counts, bottoms, strict=True):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bottom + count / 2,
                str(int(count)),
                ha="center",
                va="center",
                fontsize=10,
            )
        bottoms += counts
    for index, (variant, _, _) in enumerate(variants):
        row = global_rules.loc[variant]
        unique_truth = results[
            results["primary_one_to_one"].astype(bool)
            & results["anchor_tolerance_10_label"].isin(["contact_1", "contact_2"])
            & results["anchor_tolerance_10_in_window_count"].eq(1)
        ]
        gt_return = unique_truth["anchor_tolerance_10_label"].eq("contact_2")
        usable = unique_truth[f"{variant}_common_path_eligible"].astype(bool)
        incoming = unique_truth[f"{variant}_robust_trend_incoming"].astype(bool)
        usable_return_misses = int((gt_return & usable & ~incoming).sum())
        no_evidence_return_misses = int((gt_return & ~usable).sum())
        axes[1].text(
            index,
            91,
            f"17 GT returns\n{int(row['tp'])} incoming\n{usable_return_misses} usable below\n"
            f"{no_evidence_return_misses} no evidence\n\n{int(row['incoming_calls'])} incoming calls\n"
            f"{int(row['tp'])} returns + {int(row['fp'])} serves",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": COLOURS["grey"], "alpha": 0.92},
        )
    axes[1].set(
        xticks=x_positions,
        xticklabels=("Exclude recurrence-\nflagged points", "Also exclude producer-\nmarked inpaint"),
        ylabel="Unique ±10 anchors (n=135)",
        ylim=(0, 145),
        title="Evidence states under the same fixed 0.05-BH rule",
    )
    axes[1].legend(loc="upper center", ncols=3, fontsize=8)
    axes[1].grid(axis="y", alpha=0.2)
    figure.suptitle("Masking producer-marked inpaint reduces evidence; the motion threshold does not change")
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
        "historical rule, recurrence mask",
        "0.05-BH trend rule, recurrence mask",
        "0.05-BH trend rule, recurrence plus producer mask",
    )
    labels = (
        "Released\nalternating fit",
        "Earliest-contact\nplayer",
        "Contact player;\nhistorical flip",
        "Contact player; 0.05-BH flip\n24 motion paths",
        "Contact player; 0.05-BH flip\nproducer mask, 14 paths",
    )
    values = [scores[method]["accuracy"] for method in methods]
    x_positions = np.arange(len(methods))
    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    bars = axis.bar(
        x_positions,
        values,
        color=(COLOURS["grey"], COLOURS["sky"], COLOURS["purple"], COLOURS["blue"], COLOURS["orange"]),
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
            "Does incoming-motion evidence improve server attribution?\n"
            "0.05-BH motion is usable on 24/239 with the recurrence mask and 14/239 with the producer mask; "
            "all other rallies use the contact player"
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

## Summary

This investigation asks whether the shuttle clearly approaches the player at the earliest accepted contact. If it does, the contact is probably the first return rather than the serve. That signal can identify the other player as server. The corrected result is useful, but limited by contact timing and scarce trajectory evidence.

Three rally counts answer different questions. ShuttleSet contains **292 GT rallies**. This end-to-end population includes 43 rallies without a covering predicted span. The current scoring calls **249 rallies covered**. Ten belong to five predicted spans that each cover two GT rallies, so 249 remains a sensitivity view of `COVERED`. The main downstream population is **239 one-to-one rallies**. Each has one predicted span mapped to one GT rally. Contact identity, trajectory classification and server attribution use 239 when they require an unambiguous match.

The **earliest accepted contact** is the first ordinary contact-detector candidate that survives the released filters inside a predicted rally. It is not a serve detector. It comes from the usual shuttle impulse and player-proximity process. The wrist check, ordinary suppression and definitive exclusion mask can reject it. Nothing requires serve-like motion. The contact player's court half is measured directly at that frame, without using the alternating server fit.

On the 239 one-to-one rallies, the strict ±5 view gives 87 nearest serves, 15 first returns, 3 later strokes and 134 unmatched anchors. At the main ±10 baseline, the counts are **119 serve, 19 first return, 4 later and 97 unmatched**. Five ±10 windows contain several GT strokes. Keeping nearest identity while flagging ambiguity leaves unique truth for 118 serves and 17 returns. At ±30, the labels are 156 serve, 24 return, 4 later and 55 unmatched. However, 117 windows contain several GT strokes. This is a sanity check, not clean identity truth.

The 97 unmatched ±10 anchors reveal two patterns. In 49 rallies, a later accepted contact matches the serve. This is consistent with an earlier ordinary candidate taking the anchor position. In 36, no later contact matches the serve but one matches the first return. The serve appears missing while the return was detected. Nine first match another GT stroke. Three never match later GT. The first later match has rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12.

Usable motion evidence is rare. The recurrence-only check finds 57 continuous pre-contact runs. Thirty-one have at least five points and end close enough to contact. Only **24/239** pass the shared jump check. The unique ±10 truth set contains 19 usable paths. The fixed 0.05-BH rule makes 13 return calls: 9 correct and 4 false. Of 17 GT returns, 4 have usable paths below 0.05 BH and 4 have no usable path. A negative decision is therefore separate from missing evidence.

The unchanged historical rule adds a 0.25-BH total-movement eligibility floor. Its incoming call then requires 0.25 BH of net closure and 55% of steps towards the player. The 0.25 values came from the old analysis, while 55% was selected under old ±5/249 scoring. The new rule calls incoming from a robust fitted decrease of at least **0.05 apparent body heights**. This engineering judgement was fixed before corrected scoring and never swept. Both correctly identify 9 of 17 GT returns. The new rule adds one false call. The result shows what removing the strong, path-length-dependent 0.25-BH floor changes; it does not show a score advantage.

The inpaint comparison holds both rules fixed. Its only added restriction removes producer-marked filled or interpolated points. For the 0.05-BH rule, usable unique-truth paths fall from 19 to 10. Correct return calls fall from 9 to 7, false calls from 4 to 0, and missed returns rise from 8 to 10. Every video loses evidence.

Residual scatter and trend-to-jitter remain diagnostics, not decision cutoffs. Incorrect usable-path calls have slightly higher median scatter and a much weaker median trend-to-jitter signal. The sample is too small to turn that pattern into another rule.

On 239 one-to-one rallies, the released alternating fit is correct in {primary_scores['old alternating fit']['correct']}/239. Using the earliest contact player gives {primary_scores['anchor player']['correct']}/239. Using that player as the fallback, then flipping to the other player on 0.05-BH incoming evidence, reaches **{primary_scores['0.05-BH trend rule, recurrence mask']['correct']}/239 ({primary_scores['0.05-BH trend rule, recurrence mask']['accuracy']:.1%})**. The producer mask gives {primary_scores['0.05-BH trend rule, recurrence plus producer mask']['correct']}/239. Prepending a hypothetical contact reaches only {primary_scores['0.05-BH trend then prepend unknown player']['correct']}/239 or {primary_scores['0.05-BH trend then prepend other player']['correct']}/239. The useful result is the direct incoming-motion clue, not recursive refitting.

The next step is to improve anchor and path availability before building a richer classifier. Many failures start earlier: an ordinary candidate takes the anchor position, or the accepted sequence lacks the serve. Future work should keep contact and segmentation failures separate from trajectory classification. The unchanged 0.05-BH rule needs testing on new videos before its score can be treated as general performance.
"""
    detail = f"""

## Rally groups and failure stages

{_population_table(metrics)}

{_segmentation_table(results)}

**Main denominator trail:** 292 GT rallies → 249 covered rallies → 239 one-to-one rallies → 135 unique ±10 serve/return anchors → 19 usable unique-truth paths under the recurrence-only check.

The 249 covered rows use 244 predicted spans. There are 239 one-rally spans and five spans that each cover two GT rallies. The merged rows remain visible in the covered sensitivity results, but they are never double-scored in the primary motion comparison.

The investigation keeps these stages separate:

1. Segmentation either maps a GT rally to a predicted span or fails.
2. The earliest accepted contact either matches a plausible GT stroke or does not.
3. A continuous pre-contact path is either unavailable, rejected by the shared quality checks, or usable.
4. A usable path falls above or below the fixed incoming threshold.
5. The resulting server attribution is either correct or incorrect.

## Earliest-contact alignment

The offset is `(accepted contact frame - GT stroke frame) × 30 / source fps`. Negative values mean the accepted contact occurs earlier. Each tolerance uses the nearest GT stroke even when several strokes fall inside the window. The last column reports that ambiguity separately.

{_alignment_table(primary_alignment['global'])}

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

All three tolerances by video are:

{_alignment_by_fixture_table(primary_alignment['by_fixture'])}

The merge-sensitive 249-row view is close to the primary result at ±10: {covered_alignment['10']['labels'].get('contact_1', 0)} nearest serves, {covered_alignment['10']['labels'].get('contact_2', 0)} nearest first returns, {covered_alignment['10']['labels'].get('later', 0)} later strokes and {covered_alignment['10']['labels'].get('unmatched', 0)} unmatched anchors. It has {covered_alignment['10']['multiple']} multiple-stroke windows. This similarity does not make merged rows suitable for one-rally trajectory scoring.

## What follows an unmatched anchor

Later contacts are checked independently against every GT stroke. A GT stroke is not consumed after one match. The rank is one-based in the full accepted-contact sequence, so the first later contact has rank 2.

{_unmatched_table(sequence)}

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Four first matches have more than one GT stroke inside ±10. Twenty-seven sequences reuse a GT ordinal for more than one accepted contact. Those flags make the non-consuming check explicit. They do not change the category order in the table.

The 55 anchors still unmatched at ±30 are best described as **GT-incompatible candidates under the ±30 sanity criterion**. That wording does not claim a visually verified false contact.

## Motion evidence before the contact

The path searches back at most 30 base-30fps frames within the same court scene. It uses the continuous run closest to contact. Both fixed rules require at least five samples, no gap larger than two base-30fps frames before contact, recurrence guard `NO_FLAG`, finite contact-player distance and body-height evidence, and a largest-step to median-step ratio no greater than 4.

{_path_table(metrics)}

{_path_by_fixture_table(metrics)}

![Motion evidence availability and fixed inpaint comparison](outputs/plots/motion_evidence_and_inpaint.png)

“Continuous run selected” only means that at least one usable source point exists. “At least 5 points and close enough” applies the sample-count and contact-gap requirements. “Passes the shared jump check” is the usable-evidence count for the 0.05-BH decision. A rally outside that final count has no usable answer from the motion rule.

The 24 usable paths and 15 incoming calls above cover all 239 one-to-one rallies. Requiring unique ±10 serve/return truth leaves 19 usable paths and 13 incoming calls. The remaining five usable paths have another or unmatched anchor identity, so they cannot enter the serve-versus-return classification score.

## Historical absolute closure versus the 0.05-BH trend

The robust trend takes the median slope between every pair of shuttle-to-player distance samples. Time is normalised from zero to one across the observed path. The fitted decrease is the negative slope. The call is “incoming” only when that decrease reaches 0.05 BH.

Both rules first use the shared five-point, contact-gap, recurrence, finite-evidence and jump checks. The historical rule then adds its 0.25-BH total-movement eligibility floor. The trend rule does not. This is why the historical row has 18 eligible paths rather than 19 under the recurrence mask, and 9 rather than 10 under the producer mask. Net closure and the 55% approaching-step condition decide the historical incoming call after that eligibility check.

{_rule_table(rule_rows)}

All four rows use the same 135 one-to-one anchors with unique ±10 truth: 118 GT serves and 17 GT first returns. “Returns missed” includes both usable paths below the threshold and returns without usable evidence. Under the recurrence-only 0.05-BH rule, four misses have usable negative paths and four have no usable path. Under the producer mask, one missed return has a usable negative path and nine have no usable path. The distinction matters because only the usable negative cases are trajectory decisions.

{_rule_by_fixture_table(rule_rows)}

## What trend and jitter show

The 0.05-BH threshold alone makes the call. Residual RMS measures scatter around the robust trend. Trend-to-jitter divides the fitted decrease by that residual scatter. Neither diagnostic is an eligibility test or another classifier.

{_diagnostic_table(diagnostics, 'gt_anchor_identity')}

{_diagnostic_table(diagnostics, 'call_correct')}

{_path_length_diagnostic_table(diagnostics)}

The path-length groups are descriptive summaries only. They were not used to make or tune the call.

![Continuous trend and jitter diagnostics](outputs/plots/trend_and_jitter_diagnostics.png)

GT serves and first returns have similar median fitted decreases in this small usable set. Correct calls show a much larger median fitted decrease and trend-to-jitter than incorrect calls. Incorrect calls also have slightly more residual scatter. These are descriptive patterns after applying the fixed rule. They do not justify another cutoff.

The error plot shows all eight mistakes with usable recurrence-mask paths: four false return calls and four missed returns. The cases are {', '.join(error_cases)}.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Server attribution

The main server table uses only the 239 one-to-one rallies. Accuracy keeps abstentions in the denominator. “Answers made” shows whether a method supplied Top or Bottom.

{_server_score_table(metrics, PRIMARY_POPULATION)}

![Server attribution on the 239 one-to-one rallies](outputs/plots/server_attribution.png)

The direct 0.05-BH rule uses the anchor player when the path is usable but not incoming. It also uses the anchor player when motion evidence is unavailable. The evidence-only row abstains in the second case. Its {primary_scores['0.05-BH trend evidence only']['known']}/239 answers show the actual evidence coverage.

The same main methods under the two sensitivity populations are:

{_server_population_sensitivity_table(metrics)}

The end-to-end 292-row accuracy includes all 43 segmentation failures. Those failures have no anchor-based server answer. The 249-row result includes ten merged GT rows. Neither sensitivity view replaces the 239-row primary result.

Primary results by video show that no one video supplies the full improvement:

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
    plot_motion_evidence_and_inpaint(results, metrics, rule_rows, plot_dir)
    plot_trend_diagnostics(diagnostics, plot_dir)
    error_cases = plot_representative_errors(results, path_points, diagnostics, plot_dir)
    plot_server_attribution(metrics, plot_dir)
    _write_report(results, rule_rows, diagnostics, metrics, error_cases, report_path)
