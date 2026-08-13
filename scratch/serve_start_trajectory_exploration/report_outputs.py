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
        (ALL_POPULATION, "All ground-truth (GT) rallies", "End-to-end view, including segmentation failures"),
        (COVERED_POPULATION, "Covered rallies", "Check how results change under the current COVERED definition, including merged rallies"),
        (PRIMARY_POPULATION, "One-to-one rallies", "Analyses that need one predicted rally for each GT rally"),
    )
    lines = [
        "| Rally group | All videos | sset_01 | sset_15 | sset_21 | What it is used for |",
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
        "0.05-BH trend then like-for-like refit",
        "0.05-BH trend rule, recurrence plus producer mask",
        "0.05-BH trend evidence only",
        "0.05-BH trend then prepend unknown player",
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
        "0.05-BH trend then like-for-like refit": (
            "Earliest-contact fallback; prepend inferred server and refit on incoming triggers"
        ),
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
        "| Video | Rallies | Released fit | Earliest-contact player | Direct motion correction | Prepend/refit, same fallback |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fixture, scores in by_fixture.items():
        old = scores["old alternating fit"]
        anchor = scores["anchor player"]
        trend = scores["0.05-BH trend rule, recurrence mask"]
        refit = scores["0.05-BH trend then like-for-like refit"]
        lines.append(
            f"| {fixture} | {trend['n']} | {old['correct']} | {anchor['correct']} | {trend['correct']} | "
            f"{refit['correct']} |"
        )
    return "\n".join(lines)


def _prepend_refit_comparison_table(results: pd.DataFrame) -> str:
    """Show how fallback choice and triggered refitting contribute separately."""
    primary = results[results["primary_one_to_one"].astype(bool)]
    triggered = primary["recurrence_clean_robust_trend_incoming"].astype(bool)
    groups = (
        ("No incoming-motion trigger", primary[~triggered]),
        ("Incoming-motion trigger", primary[triggered]),
        ("All primary rallies", primary),
    )
    methods = (
        "anchor_player",
        "motion_rule_server",
        "anchor_fallback_refit_server",
    )
    lines = [
        "| Motion group | Rallies | Earliest-contact baseline | Direct correction | Prepend/refit |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, frame in groups:
        cells = []
        for column in methods:
            correct = int(frame[column].eq(frame["gt_server"]).sum())
            known = int(frame[column].notna().sum())
            cells.append(f"{correct} correct; {known} answers")
        lines.append(f"| {label} | {len(frame)} | {' | '.join(cells)} |")
    return "\n".join(lines)


def _prepend_refit_disagreement_table(results: pd.DataFrame) -> str:
    """List the triggered rallies where direct inference and refitting differ."""
    primary = results[results["primary_one_to_one"].astype(bool)]
    triggered = primary["recurrence_clean_robust_trend_incoming"].astype(bool)
    direct = primary["motion_rule_server"].fillna("Unknown")
    augmented_fit = primary["inferred_player_refit_server"].fillna("Tie")
    disagreements = primary[triggered & direct.ne(augmented_fit)]
    lines = [
        "| Rally | What the augmented fit does | Direct inference | Prepend/refit result | GT server |",
        "|---|---|---|---|---|",
    ]
    for row in disagreements.itertuples(index=False):
        augmented_result = row.inferred_player_refit_server
        if pd.isna(augmented_result):
            behaviour = "Ties; retain earliest-contact fallback"
        else:
            behaviour = f"Later votes override to {augmented_result}"
        lines.append(
            f"| {row.fixture} {row.set_id} rally {int(row.rally)} | {behaviour} | {row.motion_rule_server} | "
            f"{row.anchor_fallback_refit_server} | {row.gt_server} |"
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
    denominator = int(scores["old alternating fit"]["n"])
    incoming_calls = int(
        _metric(
            metrics,
            "path_funnel",
            PRIMARY_POPULATION,
            "global",
            "recurrence_clean",
            "robust_trend_incoming",
        )
    )
    methods = (
        "old alternating fit",
        "anchor player",
        "0.05-BH trend rule, recurrence mask",
        "0.05-BH trend then like-for-like refit",
    )
    labels = (
        "Released\nalternating fit",
        "Earliest-contact\nplayer",
        "Same fallback\n+ direct motion correction",
        "Same fallback\n+ triggered prepend/refit",
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
        ylabel=f"Correct server over all {denominator} one-to-one rallies",
        ylim=(0, 0.79),
        title=(
            "Shared fallback isolates the effect of refitting on motion-triggered rallies\n"
            f"The two right bars differ only on the {incoming_calls} incoming-motion triggers"
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
    summary_report = f"""# What the earliest accepted contact tells us about who served

## Main takeaway

All server results here use the same 239 one-to-one rallies. The released alternating-fit method gets **{primary_scores['old alternating fit']['correct']}** right. If we assume that the player at the earliest accepted contact served, that rises to **{primary_scores['anchor player']['correct']}**.

Only **24/239** rallies have usable shuttle motion before that contact, so motion can only help in a small number of cases. It is a correction to the basic method, not something we can use for every rally. Applying the motion correction directly raises the result to **{primary_scores['0.05-BH trend rule, recurrence mask']['correct']}** correct. The prepend/refit method uses the same earliest-contact fallback and the same 15 motion triggers, but gets **{primary_scores['0.05-BH trend then like-for-like refit']['correct']}** correct. On those 15 triggered rallies, the direct method gets 13 right and prepend/refit gets 9.

Rerunning the full fit after adding the motion-based guess does not help in this sample. The bigger priorities are making sure we start from the right contact and increasing the number of rallies with usable motion paths.

## Why getting the first contact right matters

With the normal ±10 timing tolerance, **97 of 239** earliest accepted contacts do not match any ShuttleSet stroke. Looking at later accepted contacts recovers the serve in 49 of those rallies and the first return in another 36.

Many errors happen before we try to classify the shuttle motion. Sometimes an earlier ordinary contact candidate is accepted first. In other cases, the accepted contact sequence misses the serve but still includes the return.

## What should we try next?

Improve how we choose the starting contact and increase the number of usable motion paths before adding a more complicated trajectory classifier.

Keep the 0.05-BH rule unchanged while testing it on new videos. The two main limits of the current method are: the wrong contact can be chosen as the starting point, and only 24/239 rallies have usable motion evidence.

<!-- TOC START -->
## Contents

- [Main takeaway](#main-takeaway)
- [Why getting the first contact right matters](#why-getting-the-first-contact-right-matters)
- [What should we try next?](#what-should-we-try-next)
- [Why do we use 292, 249 and 239 rallies in different places?](#why-do-we-use-292-249-and-239-rallies-in-different-places)
- [Is the first accepted contact the serve?](#is-the-first-accepted-contact-the-serve)
- [What happens when the first contact does not match?](#what-happens-when-the-first-contact-does-not-match)
- [How does the motion correction work?](#how-does-the-motion-correction-work)
- [How often do we have usable motion?](#how-often-do-we-have-usable-motion)
- [Does excluding producer-marked interpolation help?](#does-excluding-producer-marked-interpolation-help)
- [Does prepend/refit improve the motion-based guess?](#does-prependrefit-improve-the-motion-based-guess)
- [Extra diagnostics and rule comparisons (optional)](#extra-diagnostics-and-rule-comparisons-optional)
  - [Do the diagnostics point to a better rule?](#do-the-diagnostics-point-to-a-better-rule)
  - [How does the older 0.25-BH rule compare?](#how-does-the-older-025-bh-rule-compare)
  - [Which usable paths give the wrong answer?](#which-usable-paths-give-the-wrong-answer)
- [Extra breakdowns (optional)](#extra-breakdowns-optional)
  - [Segmentation by video](#segmentation-by-video)
  - [Contact alignment by video](#contact-alignment-by-video)
  - [Follow-up for unmatched earliest contacts](#follow-up-for-unmatched-earliest-contacts)
  - [Motion availability by video](#motion-availability-by-video)
  - [Fixed motion rules by video](#fixed-motion-rules-by-video)
  - [Server results and broader checks](#server-results-and-broader-checks)
  - [The four triggered rallies where direct and prepend/refit disagree](#the-four-triggered-rallies-where-direct-and-prependrefit-disagree)
- [Note about an earlier exploratory comparison](#note-about-an-earlier-exploratory-comparison)
- [Limits](#limits)
- [Output files](#output-files)
<!-- TOC END -->

"""
    detail = f"""## Why do we use 292, 249 and 239 rallies in different places?

The main comparison uses 239 rallies because each of these has exactly one predicted span and one contact sequence for one ground-truth rally. The 249-rally and 292-rally results show how the findings change when we include broader sets of rallies. They are useful checks, but they do not replace the main 239-rally comparison.

{_population_table(metrics)}

**How the groups narrow down:** 292 ground-truth rallies → 249 covered rallies → 239 one-to-one rallies.

The 249 covered rows come from 244 predicted spans. Of those spans, 239 cover one ground-truth rally each, while five spans each cover two ground-truth rallies. Those merged cases stay in the 249-rally results, but the main analysis does not score the same shared contact sequence twice.

The analysis separates five questions:

1. Did segmentation map the ground-truth rally to a predicted span?
2. Does the earliest accepted contact match a plausible stroke?
3. Is there a usable continuous shuttle path before that contact?
4. If there is a usable path, is its incoming-motion measure above or below the fixed threshold?
5. Does the final guess about who served turn out to be correct?

## Is the first accepted contact the serve?

Often, but not reliably enough to call it a detected serve. The serve is the largest single category, at **119 of 239** rallies, but **97 of 239** earliest contacts do not match any annotated stroke at the main ±10 tolerance.

The earliest accepted contact is the first output accepted by the released contact detector; it is not produced by a dedicated serve detector. The detector begins with shuttle impulses and player proximity, then applies wrist, suppression and exclusion checks. For this analysis, we independently measure which player is nearest at the accepted frame rather than relying on the released alternating fit.

The timing offset is:

`(accepted contact frame - GT stroke frame) × 30 / source fps`

A negative value means the accepted contact happens earlier than the ground-truth stroke. At each tolerance, we keep the nearest stroke as the match even if several strokes fall inside the window. The final column in the results reports those ambiguous cases separately.

![Nearest GT stroke at all three tolerances](outputs/plots/anchor_alignment.png)

The small “multiple” number shows when the timing window contains more than one possible GT stroke. We still use whichever stroke is closest to the accepted contact. This is rare at ±10 (5 rallies), but happens in 117 rallies at ±30, so the wider window is too ambiguous to tell us reliably which stroke the contact belongs to.

We use ±10 as the main tolerance. The stricter ±5 result and the broad ±30 check show how much the answer changes when the tolerance changes.

## What happens when the first contact does not match?

Later accepted contacts recover either the serve or the first return in **85 of the 97** rallies where the earliest contact does not match. This suggests that many bad starting contacts come from an early candidate being accepted first, or from the serve being missed even though later contacts are still present.

![Later-contact outcomes after an unmatched anchor](outputs/plots/unmatched_anchor_followup.png)

Each later accepted contact is checked independently against every annotated stroke using the same ±10 tolerance. A stroke is allowed to match more than one accepted contact. The first later match appears at contact rank 2 in 56 rallies, rank 3 in 17, rank 4 in 9, and rank 5 or later in 12. Ranks count from the start of the full accepted sequence, so the first later contact is rank 2.

Four of these first matches have more than one annotated stroke inside the ±10 window. In 27 sequences, the same stroke number matches more than one accepted contact. These cases are flagged; they do not change the result categories.

There are still 55 earliest contacts with no match even at ±30. We describe them as **GT-incompatible candidates under the ±30 sanity check**. That means they do not match the existing ground truth within ±30; it does not mean we manually inspected them and proved they were false contacts.

## How does the motion correction work?

The motion check asks whether the shuttle is moving towards that player before the earliest accepted contact.

If it is, the contact is more likely to be the first return rather than the serve, which means the other player probably served. If the path does not meet the incoming-motion threshold, we leave the earliest-contact guess unchanged.

To build the motion path, we look back by at most 30 frames on a 30-fps base timeline, staying within the same court scene. We choose the continuous run closest to the contact. A path is usable only if it has at least five samples, ends close enough to the contact, has recurrence guard `NO_FLAG`, has valid player-distance and body-height measurements, and contains no extremely large single-step jump.

For the trend measure, we calculate the slope between every pair of shuttle-to-player distance samples and take the median slope. Time is scaled from zero to one across the path. We then use the negative slope as the fitted decrease in distance. If that decrease is at least **0.05 apparent player body heights (BH)**, we call the path incoming.

The 0.05-BH threshold was chosen in advance as an engineering judgement. It is not a calibrated physical constant.

The direct method changes the basic earliest-contact guess when the shuttle meets the incoming-motion threshold for the contact player. The prepend/refit method instead adds the inferred server to the contact sequence and reruns the full alternating fit. Motion affects few rallies because usable motion evidence is rare.

## How often do we have usable motion?

Under the main recurrence check, only **24 of 239** one-to-one rallies have usable motion before the contact. The stricter check that also excludes producer-marked filled or interpolated points leaves 14. For most rallies, the method keeps the earliest-contact guess.

![Usable motion evidence under both TrackNet source checks](outputs/plots/motion_evidence_and_inpaint.png)

We can judge the motion rule against **135 earliest contacts** where the ±10 ground truth identifies either serve or first return without ambiguity: 118 serves and 17 first returns. Nineteen of those 135 have usable paths under the recurrence check.

The fixed rule marks 13 of those 19 paths as incoming. Nine are genuine first returns, while four are serves and therefore false return calls.

Looking specifically at the 17 ground-truth first returns: 9 are correctly called incoming, 4 have usable paths but stay below the 0.05-BH threshold, and 4 do not have a usable path at all. The distinction matters: "measured but below threshold" is different from "we had no usable motion evidence."

Across all 239 rallies, there are 24 usable paths and 15 incoming calls. Five of those usable paths belong to contacts that are unmatched or match a later stroke, so they cannot be included in the 135-rally serve-versus-return scoring set.

## Does excluding producer-marked interpolation help?

It removes the four false return calls, but it also removes useful evidence. With the same fixed 0.05-BH rule, the number of correctly found returns drops from 9 to 7.

This is a trade-off: fewer false calls, but also fewer rallies where we can make a useful motion-based call. The rule itself has not been retuned.

{_trend_inpaint_table(rule_rows)}

The threshold and all other motion decisions stay the same in both rows. The number of labelled usable paths falls from 19 to 10. With the stricter source check, one missed return has usable motion but stays below 0.05 BH, while nine have no usable path. Every video loses some usable evidence.

## Does prepend/refit improve the motion-based guess?

No, not in this sample. The direct motion method gets **163/239** rallies right. Prepend/refit gets **159/239**, even though both use the same fallback and the same set of motion triggers. The entire four-answer difference comes from the 15 rallies where motion triggers.

![Four central server-attribution results](outputs/plots/server_attribution.png)

When motion does not trigger, both methods choose the player at the earliest accepted contact. When motion does trigger, the direct method chooses the other player as server.

Prepend/refit adds that inferred server to the contact sequence, reruns the alternating fit, and falls back to the earliest-contact player if the fit ties.

{_prepend_refit_comparison_table(results)}

For the 15 triggered rallies, direct inference and prepend/refit agree in 11 cases. In two cases, later contact votes overturn a correct direct guess. In two more, the alternating fit ties.

These four cases show how using the whole contact sequence can sometimes weaken a good local motion clue. Fifteen triggered rallies is too small a sample to know whether this pattern will continue on new videos, but in this sample the extra refitting step does not improve the result.

## Extra diagnostics and rule comparisons (optional)

The sections below contain the diagnostic results, comparisons with the older rule, and individual failure cases.

### Do the diagnostics point to a better rule?

No. There are too few usable paths, and the groups overlap too much. Path length, residual scatter and trend-to-jitter do not show a clear reason to add another cutoff.

The decision itself uses only the 0.05-BH fitted decrease. Residual RMS tells us how much the points scatter around the fitted trend. Trend-to-jitter is the fitted decrease divided by that scatter. These are diagnostics; they do not decide whether a path is usable and they are not separate classifiers.

{_diagnostic_table(diagnostics, 'gt_anchor_identity')}

{_diagnostic_table(diagnostics, 'call_correct')}

{_path_length_diagnostic_table(diagnostics)}

These path-length groups summarise what we observed. They were not used to choose or adjust the rule.

![Continuous trend and jitter diagnostics](outputs/plots/trend_and_jitter_diagnostics.png)

In this small set, serves and first returns have almost the same median fitted decrease. Correct calls have a larger median fitted decrease and a higher trend-to-jitter value than incorrect calls. Incorrect calls also have slightly more residual scatter. These are observations about this sample, not new decision rules.

### How does the older 0.25-BH rule compare?

The older rule requires all three of the following: at least 0.25 BH of total shuttle movement, at least 0.25 BH of net movement towards the player, and at least 55% of steps moving towards the player.

The 0.05-BH trend rule instead checks whether the fitted decrease in shuttle-to-player distance reaches 0.05 BH across the observed path.

Both rules use the same checks for sample count, distance from the final path point to the contact, recurrence flags, valid measurements and large jumps. Because the older rule also requires 0.25 BH of total movement, it leaves 18 eligible paths instead of 19 with the recurrence check, and 9 instead of 10 with the producer mask added.

{_rule_table(rule_rows)}

All four rows use the same 135 earliest contacts with unambiguous ±10 labels. "Returns missed" includes both returns with a usable path that falls below the threshold and returns with no usable motion evidence.

Neither rule was chosen because of the scores in this table. The 0.25-BH values come from the older analysis. The 55% step threshold was chosen using the older ±5/249 scoring setup. The 0.05-BH value was set in advance as an engineering judgement. None of these values has been independently calibrated as a physical threshold.

### Which usable paths give the wrong answer?

There are eight errors among the usable paths: four false return calls on ground-truth serves and four missed ground-truth returns. The traces below show the rally-level evidence for those cases.

The cases are {', '.join(error_cases)}.

![All 0.05-BH false return calls and missed returns with usable paths](outputs/plots/trend_rule_errors.png)

## Extra breakdowns (optional)

The tables below show the per-video results and the broader checks.

### Segmentation by video

{_segmentation_table(results)}

### Contact alignment by video

{_alignment_table(primary_alignment['global'])}

{_alignment_by_fixture_table(primary_alignment['by_fixture'])}

At ±10, the broader 249-row view has {covered_alignment['10']['labels'].get('contact_1', 0)} nearest serves, {covered_alignment['10']['labels'].get('contact_2', 0)} nearest first returns, {covered_alignment['10']['labels'].get('later', 0)} later strokes and {covered_alignment['10']['labels'].get('unmatched', 0)} unmatched earliest contacts. It also has {covered_alignment['10']['multiple']} windows containing more than one stroke.

That similarity does not make the merged rows suitable for trajectory scoring that assumes one predicted rally corresponds to one ground-truth rally.

### Follow-up for unmatched earliest contacts

{_unmatched_table(sequence)}

### Motion availability by video

{_path_table(metrics)}

"Continuous run selected" means there is at least one source point in the selected run. "At least 5 points and close enough" applies the minimum sample count and contact-gap checks. "Passes the shared jump check" is the final count of paths considered usable for the 0.05-BH decision. Rallies outside that count do not get a motion-based answer.

{_path_by_fixture_table(metrics)}

### Fixed motion rules by video

{_rule_by_fixture_table(rule_rows)}

### Server results and broader checks

{_server_score_table(metrics, PRIMARY_POPULATION)}

The accuracy percentage always uses all 239 rallies as the denominator. "Answers made" tells us whether the method supplied Top or Bottom. The direct method and prepend/refit both fall back to the earliest-contact player, so they give an answer for all 239 rallies.

{_server_population_sensitivity_table(metrics)}

The 292-rally view includes all 43 segmentation failures. Those rallies have no contact to use for an earliest-contact answer. The 249-rally view includes ten ground-truth rows that belong to merged spans. These broader results are useful checks, but neither replaces the main 239-rally result.

{_server_by_fixture_table(metrics)}

### The four triggered rallies where direct and prepend/refit disagree

{_prepend_refit_disagreement_table(results)}

## Note about an earlier exploratory comparison

One exploratory calculation combined prepend/refit on triggered rallies with the released alternating-fit method as the fallback on non-triggered rallies. It scored 127/239.

That result is not a like-for-like comparison with the direct method because the fallback is different. Of the 36-correct-answer gap between that calculation and the direct method, 32 answers come from the different fallback and only four come from the refitting on triggered rallies.

The row-level output keeps this calculation so it can still be checked later, but the report's main comparison uses the same earliest-contact fallback for both methods.

## Limits

- Only 17 earliest contacts with unique ±10 ground truth are first returns. Only 19 contacts with unique ground truth also have usable motion paths under the recurrence-only check.
- Body-height normalisation is based on apparent height in the image. It is not a physical distance on the court, and it can change with player scale and camera geometry.
- Paths with five points are allowed. The 0.05-BH threshold is modest for that reason, but it is still not calibrated.
- TrackNet residual scatter is measured from the observed path itself. We do not have separate ground truth for TrackNet position error.
- The ±30 view often contains several possible ground-truth strokes. It is a broad sanity check, not clean evidence of which stroke a contact represents.
- We did not add new manual labels. "GT-incompatible" means that a contact does not match the existing ground truth within the stated tolerance; it does not mean we visually checked it and proved it false.
- The three videos are the same videos used in the earlier exploration. The thresholds reported here were fixed before this scoring, but these results are not an independent external validation.

## Output files

- `outputs/rallies.csv.gz`: one checked row for each of 292 ground-truth rallies.
- `outputs/spans.csv.gz`: all 344 half-open predicted spans.
- `outputs/path_points.csv.gz`: the 1,012 sampled path points used to rebuild the motion measurements.
- `outputs/fixed_rules.csv.gz`: the four fixed rule/mask comparisons, both overall and by video.
- `outputs/trend_diagnostics.csv.gz`: continuous trend and jitter values for the 135 contacts with unique ±10 ground truth, under both masks.
- `outputs/metrics.json.gz`: checked summaries for rally counts, alignment, filtering steps and server results.
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
