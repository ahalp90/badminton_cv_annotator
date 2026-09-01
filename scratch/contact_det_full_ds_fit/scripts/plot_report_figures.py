"""Build the standalone figures used by the contact-detector report pack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")


EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = EXPERIMENT_DIR / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#7A3E9D"
SKY = "#56B4E9"
GREY = "#6B7280"
LIGHT_GREY = "#D1D5DB"
INK = "#17212B"
PAPER = "#FFFFFF"

JsonObject = dict[str, Any]


def load_json(relative_path: str) -> JsonObject:
    """Load one evidence file and require a JSON object at its root."""
    path = EXPERIMENT_DIR / relative_path
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {relative_path}")
    return value


def require_mapping(value: Any, field_name: str) -> JsonObject:
    """Return a required object field with a useful error on bad evidence."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected {field_name} to be a JSON object")
    return value


def require_list(value: Any, field_name: str) -> list[Any]:
    """Return a required list field with a useful error on bad evidence."""
    if not isinstance(value, list):
        raise TypeError(f"Expected {field_name} to be a JSON list")
    return value


def number(value: Any, field_name: str) -> float:
    """Return a required numeric field as a float."""
    if not isinstance(value, int | float):
        raise TypeError(f"Expected {field_name} to be numeric")
    return float(value)


def integer(value: Any, field_name: str) -> int:
    """Return a required integer field."""
    if not isinstance(value, int):
        raise TypeError(f"Expected {field_name} to be an integer")
    return value


def set_style() -> None:
    """Set one readable visual style for the complete figure set."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": LIGHT_GREY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "legend.frameon": False,
            "legend.fontsize": 12,
            "grid.color": LIGHT_GREY,
            "grid.alpha": 0.7,
            "grid.linewidth": 0.8,
        }
    )


def add_footnote(figure: Figure, text: str) -> None:
    """Add a compact evidence note below a figure."""
    figure.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=9.5, color=GREY)


def save_figure(figure: Figure, filename: str) -> None:
    """Save a report figure with stable dimensions and close it."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)


def format_percent(value: float) -> str:
    """Format a proportion as a one-decimal percentage."""
    return f"{value * 100:.1f}%"


def label_vertical_bars(axis: Axes, bars: Any, values: list[float], *, percentages: bool = True) -> None:
    """Put values above a vertical bar series."""
    for bar, value in zip(bars, values, strict=True):
        label = format_percent(value) if percentages else f"{value:,.0f}"
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=11)


def label_horizontal_bars(axis: Axes, bars: Any, values: list[float], *, percentages: bool = True) -> None:
    """Put values at the end of a horizontal bar series."""
    for bar, value in zip(bars, values, strict=True):
        label = format_percent(value) if percentages else f"{value:,.0f}"
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {label}", ha="left", va="center", fontsize=11)


def plot_experiment_route(
    pilot: JsonObject,
    baseline: JsonObject,
    final_setting: JsonObject,
    rally_start_model: JsonObject,
    shuttleset: JsonObject,
) -> None:
    """Show the experiment route and the answer supplied by each stage."""
    pilot_videos = require_list(pilot["video_names"], "pilot video_names")
    if pilot["exact_match"] is not True:
        raise ValueError("The saved pilot feature check no longer has an exact match")
    baseline_runs = require_list(baseline["runs"], "baseline runs")
    validation_videos = require_list(baseline["validation_videos"], "validation videos")
    development = development_metrics(final_setting)
    external = external_metrics(shuttleset)
    choices = require_list(rally_start_model["choices"], "rally-start model choices")
    passing_choices = integer(require_mapping(rally_start_model["checks"], "rally-start checks")["passing_choice_count"], "passes")
    stages = [
        (f"{len(pilot_videos)}-video pilot", "Could the saved features\nrun end to end?", "Yes. The rebuilt features\nmatched exactly"),
        (f"{len(baseline_runs)} model runs\n{len(validation_videos)} validation videos", "Which RF or HGB setup\nworked best?",
         "HGB, with more\nnon-contact examples"),
        ("Failure checks", "Why were complete rallies\nstill wrong?", "First contacts were\nthe main gap"),
        ("Rally-start follow-up", "Could an earlier frame add\nthe missed first contact?",
         f"{passing_choices} of {len(choices)} choices were right\nat least 80% of the time"),
        (f"{len(require_list(final_setting['videos'], 'development videos'))}-video development\nscoring",
         "Did the setup still work\nacross development videos?",
         (f"{format_percent(number(development['precision'], 'precision'))} precision; "
          f"{format_percent(number(development['recall'], 'recall'))} recall")),
        (f"{integer(shuttleset['video_count'], 'external video count')} new test videos", "Did the final model work\non new videos?",
         (f"{format_percent(number(external['precision'], 'precision'))} precision; "
          f"{format_percent(number(external['recall'], 'recall'))} recall")),
    ]
    figure, axis = plt.subplots(figsize=(16, 10))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 10)
    axis.axis("off")
    axis.set_title("The work grew from a 3-video check to a 47-video test with labels kept out", pad=18)

    positions = [(0.8, 6.1), (5.7, 6.1), (10.6, 6.1), (0.8, 1.4), (5.7, 1.4), (10.6, 1.4)]
    for stage_index, ((heading, question, answer), (left, bottom)) in enumerate(zip(stages, positions, strict=True)):
        colour = BLUE if stage_index in {0, 1, 4} else ORANGE if stage_index in {2, 3} else PURPLE
        box = FancyBboxPatch(
            (left, bottom), 4.1, 2.65, boxstyle="round,pad=0.18", facecolor="#F7FAFC", edgecolor=colour, linewidth=2.5
        )
        axis.add_patch(box)
        axis.text(left + 0.25, bottom + 2.22, heading, fontsize=14, weight="bold", color=colour, va="top")
        axis.text(left + 0.25, bottom + 1.48, question, fontsize=11.5, color=INK, va="top", wrap=True)
        axis.text(left + 0.25, bottom + 0.57, answer, fontsize=12, weight="bold", color=INK, va="top", wrap=True)

    arrow_pairs = [((4.95, 7.42), (5.55, 7.42)), ((9.85, 7.42), (10.45, 7.42)), ((12.65, 5.95), (2.85, 4.2)),
                   ((4.95, 2.72), (5.55, 2.72)), ((9.85, 2.72), (10.45, 2.72))]
    for start, end in arrow_pairs:
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, color=GREY, linewidth=1.8))

    add_footnote(
        figure,
        "Sources: records/pilot_feature_check.json; records/baseline_summary.json; "
        "records/missed_contact_summary.json; records/rally_start_model_summary.json; "
        "raw/final_contact_scores/combined_first/final_contact_setting_result.json; "
        "records/shuttleset22_test_summary.json.",
    )
    save_figure(figure, "01_experiment_route.png")


def readable_run_name(run_id: str) -> str:
    """Turn a fixed baseline run identifier into a short human label."""
    names = {
        "hgb_reference_raw_balanced": "HGB · original motion\nbalanced · 12 non-contacts",
        "hgb_reference_common30_balanced": "HGB · 30 fps motion\nbalanced · 12 non-contacts",
        "rf_reference_raw_balanced": "RF · original motion\nbalanced · 12 non-contacts",
        "rf_reference_common30_balanced": "RF · 30 fps motion\nbalanced · 12 non-contacts",
        "hgb_reference_raw_no_weight": "HGB · original motion\nno class weights",
        "rf_reference_raw_no_weight": "RF · original motion\nno class weights",
        "hgb_15_leaves_raw_balanced": "HGB · 15 leaves\nbalanced · 12 non-contacts",
        "hgb_learning_rate_004_raw_balanced": "HGB · learning rate 0.04\nbalanced · 12 non-contacts",
        "hgb_reference_raw_more_negatives": "HGB · original motion\nbalanced · 24 non-contacts",
    }
    if run_id not in names:
        raise KeyError(f"No readable name for baseline run {run_id}")
    return names[run_id]


def plot_nine_run_model_comparison(baseline: JsonObject) -> None:
    """Compare contact scores and whole-section results for the nine fixed runs."""
    runs = require_list(baseline["runs"], "baseline_summary.runs")
    labels: list[str] = []
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    fully_correct_shares: list[float] = []
    fully_correct_counts: list[int] = []
    accepted_counts: list[int] = []
    for raw_run in runs:
        run = require_mapping(raw_run, "baseline run")
        labels.append(readable_run_name(str(run["run_id"])))
        precision.append(number(run["timing_precision"], "timing_precision"))
        recall.append(number(run["timing_recall"], "timing_recall"))
        f1.append(number(run["timing_f1"], "timing_f1"))
        fully_correct_shares.append(number(run["fully_correct_share_of_accepted_sections"], "fully correct share"))
        fully_correct_counts.append(integer(run["fully_correct_video_sections"], "fully correct sections"))
        accepted_counts.append(integer(run["accepted_video_sections"], "accepted sections"))

    row_positions = list(range(len(labels)))
    figure, (score_axis, section_axis) = plt.subplots(1, 2, figsize=(20, 11), gridspec_kw={"width_ratios": [1.35, 1]})
    point_offset = 0.22
    score_axis.scatter(precision, [position + point_offset for position in row_positions], s=105, marker="o", label="Precision", color=BLUE)
    score_axis.scatter(recall, row_positions, s=105, marker="s", label="Recall", color=ORANGE)
    score_axis.scatter(f1, [position - point_offset for position in row_positions], s=115, marker="D", label="F1", color=PURPLE)
    score_axis.set_yticks(row_positions, labels)
    score_axis.invert_yaxis()
    score_axis.set_xlim(0.81, 0.90)
    score_axis.set_xlabel("Score")
    score_axis.set_title("Contact timing")
    score_axis.grid(axis="x")
    score_axis.legend(loc="lower right", ncol=3)

    section_axis.scatter(fully_correct_shares, row_positions, s=125, marker="o", color=BLUE)
    section_axis.set_yticks(row_positions, [""] * len(row_positions))
    section_axis.invert_yaxis()
    section_axis.set_xlim(0.09, 0.18)
    section_axis.set_xlabel("Share fully correct")
    section_axis.set_title("Sections kept by the development scorer")
    section_axis.grid(axis="x")
    for position, share, correct_count, accepted_count in zip(
        row_positions, fully_correct_shares, fully_correct_counts, accepted_counts, strict=True
    ):
        section_axis.annotate(
            f"{format_percent(share)}  ({correct_count}/{accepted_count})",
            (share, position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=10.5,
        )
    chosen_index = labels.index(readable_run_name(str(baseline["chosen_run_id"])))
    for axis in [score_axis, section_axis]:
        axis.axhspan(chosen_index - 0.45, chosen_index + 0.45, color=SKY, alpha=0.13, zorder=0)
    score_axis.text(0.8105, chosen_index, "selected", color=BLUE, weight="bold", va="center", fontsize=11)
    figure.suptitle("All nine runs were close; HGB with more non-contact examples came first", fontsize=20,
                    weight="bold", color=INK)
    add_footnote(
        figure,
        "Source: records/baseline_summary.json. Eight development videos. Contact timing allows 5 frames; the check of "
        "the whole section allows 10. There are 5,696 labelled contacts. RF = random forest; HGB = histogram gradient "
        "boosting.",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_figure(figure, "02_nine_run_model_comparison.png")


def development_metrics(final_setting: JsonObject) -> JsonObject:
    """Return the selected 5-frame metrics for the 40-video held-out scores."""
    return require_mapping(final_setting["selected_metrics_at_5_frames"], "selected_metrics_at_5_frames")


def external_metrics(shuttleset: JsonObject, tolerance: str = "5") -> JsonObject:
    """Return external timing metrics for one frame tolerance."""
    timing = require_mapping(shuttleset["timing"], "shuttleset22_test_summary.timing")
    return require_mapping(timing[tolerance], f"timing.{tolerance}")


def rally_section_metrics(shuttleset: JsonObject) -> JsonObject:
    """Return the post-test recount of detected rally sections."""
    return require_mapping(
        shuttleset["rally_section_recount"],
        "shuttleset22_test_summary.rally_section_recount",
    )


def plot_contact_metrics(baseline: JsonObject, final_setting: JsonObject, shuttleset: JsonObject) -> None:
    """Compare contact metrics at the three main evidence stages."""
    chosen_id = str(baseline["chosen_run_id"])
    runs = [require_mapping(item, "baseline run") for item in require_list(baseline["runs"], "baseline runs")]
    chosen_run = next(run for run in runs if run["run_id"] == chosen_id)
    development = development_metrics(final_setting)
    external = external_metrics(shuttleset)
    stages = ["8-video model choice", "40-video development", "47-video frozen test"]
    precision = [number(chosen_run["timing_precision"], "timing_precision"), number(development["precision"], "precision"),
                 number(external["precision"], "precision")]
    recall = [number(chosen_run["timing_recall"], "timing_recall"), number(development["recall"], "recall"),
              number(external["recall"], "recall")]
    f1 = [number(chosen_run["timing_f1"], "timing_f1"), number(development["f1"], "f1"), number(external["f1"], "f1")]

    figure, axis = plt.subplots(figsize=(14, 8.5))
    group_positions = list(range(len(stages)))
    bar_width = 0.23
    precision_bars = axis.bar([value - bar_width for value in group_positions], precision, bar_width, label="Precision", color=BLUE)
    recall_bars = axis.bar(group_positions, recall, bar_width, label="Recall", color=ORANGE)
    f1_bars = axis.bar([value + bar_width for value in group_positions], f1, bar_width, label="F1", color=PURPLE)
    label_vertical_bars(axis, precision_bars, precision)
    label_vertical_bars(axis, recall_bars, recall)
    label_vertical_bars(axis, f1_bars, f1)
    axis.set_xticks(group_positions, stages)
    axis.set_ylim(0, 1.03)
    axis.set_ylabel("Share of contacts")
    axis.set_title("Recall changed little on new videos; precision fell to 80.6%")
    axis.grid(axis="y")
    axis.legend(loc="lower left", ncol=3)
    add_footnote(
        figure,
        "Sources: records/baseline_summary.json; "
        "raw/final_contact_scores/combined_first/final_contact_setting_result.json; "
        "records/shuttleset22_test_summary.json. A predicted contact may be up to 5 frames from its label on a 30 fps "
        "clock.",
    )
    save_figure(figure, "03_contact_precision_recall_f1.png")


def plot_first_vs_later_recall(baseline: JsonObject, final_setting: JsonObject, shuttleset: JsonObject) -> None:
    """Compare first-contact recall with later-contact recall at each stage."""
    counts = require_mapping(baseline["chosen_run_contact_counts"], "chosen_run_contact_counts")
    development = development_metrics(final_setting)
    external = external_metrics(shuttleset)
    first_recall = [
        integer(counts["first_contacts_matched_at_5_frames"], "first matched") / integer(counts["first_contacts"], "first contacts"),
        number(development["first_contact_recall"], "first_contact_recall"),
        number(external["first_contact_recall"], "first_contact_recall"),
    ]
    later_recall = [
        integer(counts["later_contacts_matched_at_5_frames"], "later matched") / integer(counts["later_contacts"], "later contacts"),
        number(development["other_contact_recall"], "other_contact_recall"),
        number(external["later_contact_recall"], "later_contact_recall"),
    ]
    stages = ["8-video model choice", "40-video development", "47-video frozen test"]
    figure, axis = plt.subplots(figsize=(14, 8.5))
    positions = list(range(len(stages)))
    bar_width = 0.34
    first_bars = axis.bar([value - bar_width / 2 for value in positions], first_recall, bar_width, label="First contact", color=ORANGE)
    later_bars = axis.bar([value + bar_width / 2 for value in positions], later_recall, bar_width, label="Later contacts", color=BLUE)
    label_vertical_bars(axis, first_bars, first_recall)
    label_vertical_bars(axis, later_bars, later_recall)
    axis.set_xticks(positions, stages)
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Recall")
    axis.set_title("The first contact of a rally stayed much harder to find")
    axis.grid(axis="y")
    axis.legend(loc="lower right")
    add_footnote(
        figure,
        "Sources: records/baseline_summary.json; "
        "raw/final_contact_scores/combined_first/final_contact_setting_result.json; "
        "records/shuttleset22_test_summary.json. A predicted contact may be up to 5 frames from its label on a 30 fps "
        "clock.",
    )
    save_figure(figure, "04_first_vs_later_recall.png")


def plot_development_error_mix(baseline: JsonObject) -> None:
    """Show the mutually exclusive failure groups for development rallies."""
    failures = require_mapping(baseline["chosen_run_failed_single_rally_sections"], "failed_single_rally_sections")
    labels = ["Missing contacts only", "Wrong player side only", "Both missing and extra contacts", "Extra contacts only"]
    counts = [
        integer(failures["missing_contacts_and_no_extra_contacts"], "missing only"),
        integer(failures["all_contact_times_correct_but_player_side_wrong"], "side only"),
        integer(failures["both_missing_and_extra_contacts"], "both"),
        integer(failures["extra_contacts_and_no_missing_contacts"], "extra only"),
    ]
    total = integer(failures["failed_sections"], "failed sections")
    figure, axis = plt.subplots(figsize=(14, 8.5))
    bars = axis.barh(labels, counts, color=[ORANGE, PURPLE, GREY, BLUE])
    label_horizontal_bars(axis, bars, [float(value) for value in counts], percentages=False)
    axis.invert_yaxis()
    axis.set_xlim(0, max(counts) * 1.23)
    axis.set_xlabel("Failed single-rally sections")
    axis.set_title("Missing contacts caused most failed development sections")
    axis.grid(axis="x")
    one_short = integer(failures["exactly_one_contact_missing_with_remaining_times_and_sides_correct"], "one short")
    axis.text(0.98, 0.07, f"{one_short} sections were otherwise correct\nand short by exactly one contact", transform=axis.transAxes,
              ha="right", va="bottom", fontsize=13, color=INK, bbox={"boxstyle": "round,pad=0.5", "fc": "#F7FAFC", "ec": LIGHT_GREY})
    add_footnote(
        figure,
        f"Source: records/baseline_summary.json. The four bars include all {total} failed single-rally sections from the "
        "selected 8-video development run. In this whole-rally check, a contact may be up to 10 frames from its label.",
    )
    save_figure(figure, "05_development_error_mix.png")


def plot_external_error_mix(shuttleset: JsonObject) -> None:
    """Show the exclusive whole-section outcome partition from the tracked report."""
    one_rally_sections = integer(
        require_mapping(shuttleset["section_mapping"], "section_mapping")["one_labelled_rally"], "one-rally sections"
    )
    outcome_labels = [
        "Passed old contact + side check",
        "Missing contacts only",
        "Extra contacts only",
        "Missing and extra contacts",
        "Equal contact count, wrong timing",
        "Wrong predicted side",
        "Predicted side unanswered",
    ]
    outcome_counts = [493, 1147, 243, 306, 335, 437, 8]
    if sum(outcome_counts) != one_rally_sections:
        raise ValueError("The tracked whole-section outcomes no longer match the one-rally section count")
    figure, axis = plt.subplots(figsize=(14, 8.5))
    bars = axis.barh(outcome_labels, outcome_counts, color=[BLUE, ORANGE, SKY, GREY, PURPLE, "#9B72CF", LIGHT_GREY])
    label_horizontal_bars(axis, bars, [float(value) for value in outcome_counts], percentages=False)
    axis.invert_yaxis()
    axis.set_xlim(0, max(outcome_counts) * 1.22)
    axis.set_xlabel("One-rally sections")
    axis.set_title("Among one-rally sections, missing contacts caused most failures")
    axis.grid(axis="x")
    add_footnote(
        figure,
        "Source: shuttleset22_test_report.md, checked against records/shuttleset22_test_summary.json. The bars cover all "
        "2,969 sections that match one labelled rally. A contact may be up to 5 frames from its label.",
    )
    save_figure(figure, "06_external_error_mix.png")


def plot_rally_start_followup(candidate: JsonObject, model: JsonObject) -> None:
    """Show candidate coverage and why the trained rally-start repair stopped."""
    targets = require_mapping(candidate["target_first_contacts"], "target_first_contacts")
    tolerance_labels = ["5 frames", "10 frames"]
    covered_counts: list[int] = []
    target_counts: list[int] = []
    for tolerance in ["at_5_frames", "at_10_frames"]:
        values = require_mapping(targets[tolerance], tolerance)
        covered_counts.append(integer(values["covered_contacts"], "covered contacts"))
        target_counts.append(integer(values["target_contacts"], "target contacts"))

    choices = [require_mapping(item, "rally-start model choice") for item in require_list(model["choices"], "choices")]
    figure, (coverage_axis, model_axis) = plt.subplots(1, 2, figsize=(16, 8.5))
    coverage_rates = [covered / target for covered, target in zip(covered_counts, target_counts, strict=True)]
    bars = coverage_axis.bar(tolerance_labels, coverage_rates, color=[SKY, BLUE], width=0.55)
    label_vertical_bars(coverage_axis, bars, coverage_rates)
    coverage_axis.set_ylim(0, 1.05)
    coverage_axis.set_ylabel("Share of 81 missed first contacts found")
    coverage_axis.set_title("The earlier frames included\nmany missed first contacts")
    coverage_axis.grid(axis="y")

    marker_by_model = {"logistic_regression": "o", "shallow_hgb": "s"}
    colour_by_model = {"logistic_regression": ORANGE, "shallow_hgb": BLUE}
    for model_id, marker in marker_by_model.items():
        model_choices = [choice for choice in choices if choice["model_id"] == model_id]
        selected_actions = [integer(choice["selected_actions"], "selected_actions") for choice in model_choices]
        correct_rates = [number(choice["correct_action_rate"], "correct_action_rate") for choice in model_choices]
        label = "Logistic regression" if model_id == "logistic_regression" else "Shallow HGB"
        model_axis.scatter(selected_actions, correct_rates, s=120, marker=marker, color=colour_by_model[model_id], label=label)
        for choice, selected, correct_rate in zip(model_choices, selected_actions, correct_rates, strict=True):
            model_axis.annotate(f"cut-off {number(choice['cutoff'], 'cutoff'):.1f}", (selected, correct_rate), xytext=(7, 7),
                                textcoords="offset points", fontsize=10.5)
    model_axis.axhline(0.80, color=PURPLE, linestyle="--", linewidth=2, label="Required 80% correct")
    model_axis.set_ylim(0, 0.88)
    model_axis.set_xlabel("Earlier contacts added")
    model_axis.set_ylabel("Share of added contacts that were right")
    model_axis.set_title("The model could not choose\nwhich contact to add")
    model_axis.grid()
    model_axis.legend(loc="upper right")
    figure.suptitle("The missed contact was often nearby, but no model chose it safely", fontsize=20,
                    weight="bold", color=INK)
    add_footnote(
        figure,
        "Sources: records/rally_start_candidate_summary.json and records/rally_start_model_summary.json. Candidate "
        "coverage uses the 81 otherwise-correct sections missing a first contact. Each training video was scored by a "
        "model trained on other videos.",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.94))
    save_figure(figure, "07_rally_start_followup.png")


def plot_timing_tolerance(shuttleset: JsonObject) -> None:
    """Show how external contact metrics change with timing tolerance."""
    timing = require_mapping(shuttleset["timing"], "timing")
    tolerances = [1, 2, 5, 10]
    metrics_by_tolerance = [require_mapping(timing[str(tolerance)], f"timing.{tolerance}") for tolerance in tolerances]
    series = {
        "Precision": ([number(item["precision"], "precision") for item in metrics_by_tolerance], BLUE),
        "Recall": ([number(item["recall"], "recall") for item in metrics_by_tolerance], ORANGE),
        "F1": ([number(item["f1"], "f1") for item in metrics_by_tolerance], PURPLE),
    }
    figure, axis = plt.subplots(figsize=(14, 8.5))
    label_offsets = {"Precision": -18, "Recall": 16, "F1": 0}
    for label, (values, colour) in series.items():
        axis.plot(tolerances, values, marker="o", linewidth=3, markersize=9, color=colour, label=label)
        for tolerance, value in zip(tolerances, values, strict=True):
            axis.annotate(
                format_percent(value),
                (tolerance, value),
                xytext=(0, label_offsets[label]),
                textcoords="offset points",
                ha="center",
                fontsize=10.5,
            )
    axis.set_xticks(tolerances)
    axis.set_ylim(0.45, 0.90)
    axis.set_xlabel("Allowed timing error, in frames after scaling to 30 fps")
    axis.set_ylabel("Share of contacts")
    axis.set_title("Scores rose most when the allowed error grew to five frames")
    axis.grid()
    axis.legend(loc="lower right", ncol=3)
    add_footnote(
        figure,
        "Source: records/shuttleset22_test_summary.json. The test has 47 videos. Precision counts 39,994 predictions; "
        "recall counts 38,218 usable labels. Five frames is about 0.17 seconds at 30 fps.",
    )
    save_figure(figure, "08_timing_tolerance.png")


def plot_whole_section_confidence(baseline: JsonObject, shuttleset: JsonObject) -> None:
    """Compare the tracked whole-section retention points without mixing denominators."""
    chosen_id = str(baseline["chosen_run_id"])
    runs = [require_mapping(item, "baseline run") for item in require_list(baseline["runs"], "baseline runs")]
    chosen_run = next(run for run in runs if run["run_id"] == chosen_id)
    development_kept_090 = integer(chosen_run["accepted_video_sections"], "development sections kept")
    development_correct_090 = integer(chosen_run["fully_correct_video_sections"], "development correct sections")
    external_whole = require_mapping(require_mapping(shuttleset["whole_rallies"], "whole_rallies")["5"], "whole_rallies.5")
    external_kept_090 = integer(require_mapping(shuttleset["prediction"], "prediction")["detected_sections"], "detected sections")
    external_correct_090 = integer(external_whole["fully_correct_sections"], "external correct sections")
    labels = [
        "Development · 10-frame\ncut-off 0.90",
        "Development · 10-frame\ncut-off 0.95",
        "New videos · 5-frame\ncut-off 0.90",
        "New videos · 5-frame\ncut-off 0.95",
    ]
    kept_counts = [development_kept_090, 322, external_kept_090, 1754]
    correct_counts = [development_correct_090, 55, external_correct_090, 245]
    accuracy_labels = ["16.26% of kept", "17.08% of kept", "16.60% of 2,969 scored*", "18.23% of 1,344 scored*"]
    colours = [BLUE, PURPLE, BLUE, PURPLE]

    figure, (kept_axis, correct_axis) = plt.subplots(2, 1, figsize=(15, 11), sharex=True)
    positions = list(range(len(labels)))
    kept_bars = kept_axis.bar(positions, kept_counts, color=colours, width=0.62)
    label_vertical_bars(kept_axis, kept_bars, [float(value) for value in kept_counts], percentages=False)
    kept_axis.set_ylabel("Sections kept")
    kept_axis.set_title("Sections left after raising the minimum contact score")
    kept_axis.set_ylim(0, max(kept_counts) * 1.15)
    kept_axis.grid(axis="y")

    correct_bars = correct_axis.bar(positions, correct_counts, color=colours, width=0.62)
    label_vertical_bars(correct_axis, correct_bars, [float(value) for value in correct_counts], percentages=False)
    for position, correct_count, accuracy_label in zip(positions, correct_counts, accuracy_labels, strict=True):
        correct_axis.annotate(
            accuracy_label,
            (position, correct_count),
            xytext=(0, 23),
            textcoords="offset points",
            ha="center",
            fontsize=10.5,
            color=INK,
        )
    correct_axis.set_xticks(positions, labels)
    correct_axis.set_ylabel("Fully correct sections")
    correct_axis.set_xlabel("Videos and minimum contact score")
    correct_axis.set_ylim(0, max(correct_counts) * 1.28)
    correct_axis.grid(axis="y")
    figure.suptitle(
        "Within each dataset, a 0.95 cut-off kept fewer sections and modestly improved correctness",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    add_footnote(
        figure,
        "Sources: records/baseline_summary.json, baseline_report.md, records/shuttleset22_test_summary.json and "
        "shuttleset22_test_report.md.\nOnly the 0.90-to-0.95 change within each dataset is comparable: populations and "
        "scorers differ; development uses 10 frames and new videos use 5.\n*A section is scored when it matches one "
        "labelled rally and has enough human labels to check the player side.",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    save_figure(figure, "09_confidence_vs_yield.png")


def plot_contact_cutoff_tradeoff(final_setting: JsonObject) -> None:
    """Show the development-only trade-off between score cut-off and output."""
    raw_results = require_list(final_setting["setting_results"], "setting_results")
    selected_distance = integer(final_setting["selected_duplicate_distance_at_30_fps"], "selected duplicate distance")
    results: list[JsonObject] = []
    for raw_result in raw_results:
        result = require_mapping(raw_result, "setting result")
        if integer(result["duplicate_distance_at_30_fps"], "duplicate distance") == selected_distance:
            results.append(result)
    results.sort(key=lambda result: number(result["score_cutoff"], "score cutoff"))
    cutoffs = [number(result["score_cutoff"], "score cutoff") for result in results]
    metrics = [require_mapping(result["metrics"], "setting metrics") for result in results]
    precision = [number(item["precision"], "precision") for item in metrics]
    recall = [number(item["recall"], "recall") for item in metrics]
    prediction_counts = [integer(item["prediction_count"], "prediction_count") for item in metrics]

    figure, (quality_axis, yield_axis) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    quality_axis.plot(cutoffs, precision, marker="o", linewidth=3, color=BLUE, label="Precision")
    quality_axis.plot(cutoffs, recall, marker="o", linewidth=3, color=ORANGE, label="Recall")
    quality_axis.set_ylim(0, 1.02)
    quality_axis.set_ylabel("Share of contacts")
    quality_axis.set_title("A higher minimum score gave fewer contacts and higher precision")
    quality_axis.grid()
    quality_axis.legend(loc="upper left")
    yield_axis.plot(cutoffs, prediction_counts, marker="o", linewidth=3, color=PURPLE)
    yield_axis.set_xlabel("Minimum contact score")
    yield_axis.set_ylabel("Predictions kept across 40 videos")
    yield_axis.grid()
    selected_cutoff = number(final_setting["selected_score_cutoff"], "selected score cutoff")
    for axis in [quality_axis, yield_axis]:
        axis.axvline(selected_cutoff, color=GREY, linestyle="--", linewidth=2)
    quality_axis.text(selected_cutoff + 0.012, 0.08, "selected 0.90", color=GREY, fontsize=11)
    add_footnote(
        figure,
        "Source: raw/final_contact_scores/combined_first/final_contact_setting_result.json. Each of the 40 development "
        "videos was scored by a model trained on the other 32. Timing allows 5 frames; nearby predictions join within 6.",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(figure, "10_contact_cutoff_tradeoff.png")


def plot_standalone_gap(shuttleset: JsonObject) -> None:
    """Put the external contact and whole-rally results beside the goal."""
    timing = external_metrics(shuttleset)
    side = require_mapping(require_mapping(shuttleset["player_side"], "player_side")["5"], "player_side.5")
    section_metrics = rally_section_metrics(shuttleset)
    whole_score = require_mapping(
        require_mapping(section_metrics["whole_rally_contact_score"], "whole_rally_contact_score")["5"],
        "whole_rally_contact_score.5",
    )
    labels = [
        "Contact timing\nprecision",
        "Right player\nafter timing match",
        "Passed old check among\none-rally sections",
        "Clean and fully correct\namong all sections",
    ]
    values = [
        number(timing["precision"], "precision"),
        number(side["accuracy_when_both_answered"], "side accuracy"),
        number(whole_score["share_of_one_rally_sections"], "one-rally share"),
        number(whole_score["clean_span_share_of_all_predicted_sections"], "all-section share"),
    ]
    figure, axis = plt.subplots(figsize=(16, 8.5))
    bars = axis.bar(labels, values, color=[BLUE, PURPLE, ORANGE, SKY], width=0.58)
    label_vertical_bars(axis, bars, values)
    axis.axhline(1.0, color=GREY, linestyle="--", linewidth=2)
    axis.text(3.45, 1.005, "near-100% goal", ha="right", va="bottom", color=GREY, fontsize=11)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Share correct")
    axis.set_title("Single contacts are useful; complete rally outputs are still rare")
    axis.grid(axis="y")
    add_footnote(
        figure,
        "Source: records/shuttleset22_test_summary.json. Denominators from left: 39,994 predicted contacts; 32,188 "
        "matched contacts with two side answers; 2,969 one-rally sections; all 3,982 predicted sections. The last bar "
        "also requires one complete rally and no part of another.",
    )
    save_figure(figure, "11_standalone_gap.png")


def plot_rally_section_outcomes(shuttleset: JsonObject) -> None:
    """Show what all predicted sections contained."""
    section_metrics = rally_section_metrics(shuttleset)
    counts = require_mapping(section_metrics["section_counts"], "section_counts")
    labels = ["One whole rally only", "Part of one rally", "No labelled rally", "Parts of several rallies"]
    values = [
        integer(counts["one_complete_rally"], "one complete rally"),
        integer(counts["one_partial_rally"], "one partial rally"),
        integer(counts["no_labelled_rally"], "no labelled rally"),
        integer(counts["several_labelled_rallies"], "several labelled rallies"),
    ]
    figure, axis = plt.subplots(figsize=(14, 8.5))
    bars = axis.barh(labels, values, color=[BLUE, SKY, ORANGE, PURPLE])
    label_horizontal_bars(axis, bars, [float(value) for value in values], percentages=False)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.2)
    axis.set_xlabel("Predicted sections")
    axis.set_title("2,515 of 3,982 sections held one complete rally and no other rally")
    axis.grid(axis="x")
    add_footnote(
        figure,
        "Source: records/shuttleset22_test_summary.json. A rally is complete here when every labelled contact is inside "
        "the section. ShuttleSet22 does not label the true visual start and end of each rally.",
    )
    save_figure(figure, "12_rally_section_outcomes.png")


def plot_rally_section_precision_recall(shuttleset: JsonObject) -> None:
    """Show the one-to-one rally-section precision, recall and F1."""
    section_metrics = rally_section_metrics(shuttleset)
    labels = ["Precision", "Recall", "F1"]
    values = [
        number(section_metrics["precision"], "section precision"),
        number(section_metrics["recall"], "section recall"),
        number(section_metrics["f1"], "section F1"),
    ]
    figure, axis = plt.subplots(figsize=(12, 8.5))
    bars = axis.bar(labels, values, color=[BLUE, ORANGE, PURPLE], width=0.58)
    label_vertical_bars(axis, bars, values)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Share of rally sections")
    axis.set_title("Rally-section precision was 63.2%; recall was 73.5%")
    axis.grid(axis="y")
    add_footnote(
        figure,
        "Source: records/shuttleset22_test_summary.json. Precision is 2,515 clean matches from 3,982 predicted sections. "
        "Recall is the same 2,515 matches from 3,422 usable labelled rallies.",
    )
    save_figure(figure, "13_rally_section_precision_recall.png")


def plot_rally_section_context(shuttleset: JsonObject) -> None:
    """Show the variable context around contacts in clean rally sections."""
    section_metrics = rally_section_metrics(shuttleset)
    frame_rate = number(section_metrics["frame_rate"], "frame rate")
    context = require_mapping(section_metrics["clean_section_context_frames"], "clean_section_context_frames")
    start = require_mapping(context["before_first_labelled_contact"], "before_first_labelled_contact")
    end = require_mapping(context["after_last_labelled_contact"], "after_last_labelled_contact")
    labels = ["Before first contact", "After last contact"]
    medians = [number(start["median"], "start median"), number(end["median"], "end median")]
    p10 = [number(start["p10"], "start p10"), number(end["p10"], "end p10")]
    p90 = [number(start["p90"], "start p90"), number(end["p90"], "end p90")]
    median_seconds = [value / frame_rate for value in medians]
    lower = [(median - low) / frame_rate for median, low in zip(medians, p10, strict=True)]
    upper = [(high - median) / frame_rate for median, high in zip(medians, p90, strict=True)]

    figure, axis = plt.subplots(figsize=(14, 6.5))
    positions = [0, 1]
    axis.errorbar(
        median_seconds, positions, xerr=[lower, upper], fmt="o", markersize=12, linewidth=3, capsize=7, color=BLUE
    )
    for position, value in zip(positions, median_seconds, strict=True):
        axis.annotate(
            f"median {value:.1f} s", (value, position), xytext=(10, 10), textcoords="offset points", fontsize=12,
            weight="bold",
        )
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(p90) / frame_rate * 1.08)
    axis.set_xlabel("Seconds between the section edge and the nearest labelled contact")
    axis.set_title("Space around each rally varied; the end usually had more room")
    axis.grid(axis="x")
    add_footnote(
        figure,
        "Source: records/shuttleset22_test_summary.json. Dots are medians across 2,515 clean one-rally sections. Lines "
        "show the middle 80% (10th to 90th percentile). These are contact labels, not true rally-boundary labels.",
    )
    save_figure(figure, "14_rally_section_context.png")


def main() -> None:
    """Load frozen evidence and rebuild every report figure."""
    set_style()
    baseline = load_json("records/baseline_summary.json")
    final_setting = load_json("raw/final_contact_scores/combined_first/final_contact_setting_result.json")
    missed_contacts = load_json("records/missed_contact_summary.json")
    rally_start_candidate = load_json("records/rally_start_candidate_summary.json")
    rally_start_model = load_json("records/rally_start_model_summary.json")
    shuttleset = load_json("records/shuttleset22_test_summary.json")
    if integer(missed_contacts["otherwise_correct_one_short_sections_at_10_frames"]["section_count"], "one-short sections") != 94:
        raise ValueError("The expected frozen development error count has changed")

    pilot = load_json("records/pilot_feature_check.json")
    plot_experiment_route(pilot, baseline, final_setting, rally_start_model, shuttleset)
    plot_nine_run_model_comparison(baseline)
    plot_contact_metrics(baseline, final_setting, shuttleset)
    plot_first_vs_later_recall(baseline, final_setting, shuttleset)
    plot_development_error_mix(baseline)
    plot_external_error_mix(shuttleset)
    plot_rally_start_followup(rally_start_candidate, rally_start_model)
    plot_timing_tolerance(shuttleset)
    plot_whole_section_confidence(baseline, shuttleset)
    plot_contact_cutoff_tradeoff(final_setting)
    plot_standalone_gap(shuttleset)
    plot_rally_section_outcomes(shuttleset)
    plot_rally_section_precision_recall(shuttleset)
    plot_rally_section_context(shuttleset)

    # records/shuttleset22_test_summary.json confirms that per-video values were independently checked, but does not
    # store them. A per-video plot is therefore omitted: rebuilding it would require an untracked external result file.


if __name__ == "__main__":
    main()
