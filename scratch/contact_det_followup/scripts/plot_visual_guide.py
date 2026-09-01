"""Build the reproducible figures used in the contact follow-up report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch

plt.switch_backend("Agg")


JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP_DIR = REPO_ROOT / "scratch/contact_det_followup"
FIGURE_DIR = FOLLOWUP_DIR / "figures"

INK = "#17212B"
BLUE = "#0072B2"
ORANGE = "#D97706"
PURPLE = "#7A3E9D"
RED = "#C44E52"
GREY = "#667085"
LIGHT_GREY = "#D0D5DD"
PALE_BLUE = "#E7F1F8"
PALE_ORANGE = "#FFF3E0"
PALE_PURPLE = "#F2EAF7"

# These counts are the committed error table in shuttleset22_test_report.md.
BASELINE_ERROR_COUNTS = {
    "Missing contact": 1_147,
    "Wrong player side": 437,
    "Wrong timing": 335,
    "Missing and extra contacts": 306,
    "Extra contact": 243,
    "Player side unanswered": 8,
}
BASELINE_MAPPED_SECTIONS = 2_969
BASELINE_OLD_CORRECT_SECTIONS = 493


@dataclass(frozen=True)
class DataPaths:
    """Files used to rebuild the report figures."""

    side_audit: Path
    side_development: Path
    duplicate_audit: Path
    setting_sweep: Path
    start_best_case: Path
    start_model_development: Path
    start_model_validation: Path
    combined_best_case: Path
    delete_model: Path
    keep_review: Path
    figure_dir: Path


def build_paths() -> DataPaths:
    """Return the committed evidence and output paths."""
    results_dir = FOLLOWUP_DIR / "results"
    return DataPaths(
        side_audit=results_dir / "side_audit.json",
        side_development=results_dir / "side_development.json",
        duplicate_audit=results_dir / "opposite_side_duplicate_audit.json",
        setting_sweep=results_dir / "setting_sweep.json",
        start_best_case=results_dir / "start_best_case.json",
        start_model_development=results_dir / "start_model_development.json",
        start_model_validation=results_dir / "start_model_validation.json",
        combined_best_case=results_dir / "combined_best_case.json",
        delete_model=results_dir / "delete_model_development.json",
        keep_review=results_dir / "keep_review_development.json",
        figure_dir=FIGURE_DIR,
    )


def load_json(path: Path) -> JsonObject:
    """Load a JSON object from a saved result."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def object_value(value: Any, label: str) -> JsonObject:
    """Return an expected JSON object."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected {label} to be an object")
    return value


def list_value(value: Any, label: str) -> list[Any]:
    """Return an expected JSON list."""
    if not isinstance(value, list):
        raise TypeError(f"Expected {label} to be a list")
    return value


def integer(value: Any, label: str) -> int:
    """Return an expected integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {label} to be an integer")
    return value


def number(value: Any, label: str) -> float:
    """Return an expected real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected {label} to be numeric")
    return float(value)


def configure_style() -> None:
    """Set a quiet, print-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": LIGHT_GREY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
        }
    )


def style_axes(axis: Axes, *, grid_axis: str) -> None:
    """Keep the plot frame light and the data easy to scan."""
    axis.grid(axis=grid_axis, color=LIGHT_GREY, alpha=0.65, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LIGHT_GREY)
    axis.spines["bottom"].set_color(LIGHT_GREY)


def add_title(figure: Figure, title: str, subtitle: str) -> None:
    """Add one title and one line of context."""
    figure.suptitle(title, x=0.08, y=0.98, ha="left", fontsize=18, weight="bold", color=INK)
    figure.text(0.08, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color=GREY)


def add_flow_box(
    axis: Axes,
    centre_x: float,
    centre_y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolour: str,
    edgecolour: str,
    fontsize: float = 10.5,
) -> None:
    """Draw one rounded flow box in axis-relative coordinates."""
    box = FancyBboxPatch(
        (centre_x - width / 2, centre_y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolour,
        edgecolor=edgecolour,
        linewidth=1.5,
        transform=axis.transAxes,
    )
    axis.add_patch(box)
    axis.text(
        centre_x,
        centre_y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        transform=axis.transAxes,
    )


def add_flow_arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    colour: str = GREY,
) -> None:
    """Connect two points in axis-relative coordinates."""
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=axis.transAxes,
        textcoords=axis.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": 1.8},
    )


def save_figure(figure: Figure, paths: DataPaths, stem: str, source: str) -> None:
    """Save one figure as PNG and SVG."""
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    figure.text(0.08, 0.025, source, ha="left", va="bottom", fontsize=8, color=GREY)
    figure.savefig(
        paths.figure_dir / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    figure.savefig(
        paths.figure_dir / f"{stem}.svg",
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
    )
    plt.close(figure)


def plot_complete_rallies(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Compare the ShuttleSet22 baseline with the rally-wide side vote."""
    side_audit = data["side_audit"]
    by_tolerance = object_value(
        side_audit["results_by_tolerance_at_30_fps"], "side-audit tolerances"
    )
    main_result = object_value(by_tolerance["5"], "±5 side audit")
    main_vote = object_value(main_result["simple_vote"], "±5 simple vote")
    main_repaired = integer(main_vote["repaired_sections"], "±5 repaired sections")
    main_broken = integer(main_vote["broken_sections"], "±5 broken sections")
    baseline_side_accuracy = number(main_vote["baseline_side_accuracy"], "baseline side accuracy")
    revised_side_accuracy = number(main_vote["revised_side_accuracy"], "revised side accuracy")
    baseline_contact_side_f1 = number(
        main_vote["baseline_contact_and_side_f1"], "baseline contact-and-side F1"
    )
    revised_contact_side_f1 = number(
        main_vote["revised_contact_and_side_f1"], "revised contact-and-side F1"
    )

    labels = ["±5 frames\n(main result)", "±10 frames"]
    baseline_counts: list[int] = []
    revised_counts: list[int] = []
    baseline_rates: list[float] = []
    revised_rates: list[float] = []

    for tolerance in (5, 10):
        if tolerance == 5:
            vote = main_vote
        else:
            result = object_value(by_tolerance[str(tolerance)], f"±{tolerance} side audit")
            vote = object_value(result["simple_vote"], f"±{tolerance} simple vote")
        baseline_counts.append(integer(vote["baseline_strict_fully_correct"], "baseline count"))
        revised_counts.append(integer(vote["revised_strict_fully_correct"], "revised count"))
        baseline_rates.append(number(vote["baseline_full_output_precision"], "baseline precision"))
        revised_rates.append(number(vote["revised_full_output_precision"], "revised precision"))

    figure, axis = plt.subplots(figsize=(10, 7.0))
    add_title(
        figure,
        "Held-out ShuttleSet22: the side rule nearly doubled correct rallies",
        "47 test videos · 3,982 sections · correct timing and player side",
    )
    figure.text(
        0.08,
        0.875,
        f"Main ±5 result: {main_repaired} sections repaired · "
        f"{main_broken} previously correct sections became wrong",
        ha="left",
        va="top",
        fontsize=10.5,
        weight="bold",
        color=BLUE,
    )
    x_positions = [0.0, 1.0]
    width = 0.32
    baseline_bars = axis.bar(
        [position - width / 2 for position in x_positions],
        baseline_counts,
        width,
        label="Baseline",
        color=GREY,
    )
    revised_bars = axis.bar(
        [position + width / 2 for position in x_positions],
        revised_counts,
        width,
        label="Rally-wide side vote",
        color=BLUE,
    )
    axis.set_xticks(x_positions, labels)
    axis.set_ylabel("Fully correct sections")
    axis.set_ylim(0, 1120)
    axis.legend(loc="upper left")
    style_axes(axis, grid_axis="y")

    for bars, counts, rates in (
        (baseline_bars, baseline_counts, baseline_rates),
        (revised_bars, revised_counts, revised_rates),
    ):
        for bar, count, rate in zip(bars, counts, rates, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                count + 22,
                f"{count:,}\n{rate:.2%}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
                color=INK,
            )

    figure.text(
        0.08,
        0.105,
        "The rule is tuned for complete rallies.\n"
        f"Individual-contact trade-off: side accuracy {baseline_side_accuracy:.2%} → {revised_side_accuracy:.2%} · "
        f"contact-and-side F1 {baseline_contact_side_f1:.2%} → {revised_contact_side_f1:.2%}",
        ha="left",
        va="top",
        fontsize=9.3,
        color=GREY,
        linespacing=1.35,
    )
    figure.tight_layout(rect=(0.06, 0.16, 0.98, 0.83))
    save_figure(figure, paths, "01_complete_rallies", "Source: results/side_audit.json")


def plot_baseline_errors(paths: DataPaths) -> None:
    """Show the main error groups among sections mapped to one rally."""
    labels = list(BASELINE_ERROR_COUNTS)
    values = list(BASELINE_ERROR_COUNTS.values())
    colours = [BLUE, PURPLE, GREY, GREY, GREY, GREY]

    figure, axis = plt.subplots(figsize=(10, 6.4))
    add_title(
        figure,
        "Held-out ShuttleSet22: missing contacts caused most failures",
        f"47 test videos · {BASELINE_MAPPED_SECTIONS:,} sections matched to one labelled rally · ±5 frames",
    )
    bars = axis.barh(labels[::-1], values[::-1], color=colours[::-1])
    axis.set_xlabel("Sections")
    axis.set_xlim(0, 1250)
    style_axes(axis, grid_axis="x")
    for bar, value in zip(bars, values[::-1], strict=True):
        axis.text(
            value + 18,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}",
            va="center",
            color=INK,
            fontsize=10.5,
        )
    axis.text(
        0.98,
        0.08,
        f"{BASELINE_OLD_CORRECT_SECTIONS} sections were fully correct under this older one-rally check.",
        transform=axis.transAxes,
        ha="right",
        color=GREY,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.06, 0.07, 0.98, 0.88))
    save_figure(
        figure,
        paths,
        "02_baseline_errors",
        "Source: scratch/contact_det_full_ds_fit/shuttleset22_test_report.md",
    )


def plot_side_rule_explainer(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Show how heuristic side guesses choose one alternating sequence."""
    development = data["side_development"]
    development_choice = object_value(development["selected"], "chosen side rule")
    minimum_lead = integer(development["selected_minimum_vote_gap"], "side-rule lead")

    side_audit = data["side_audit"]
    test_tolerances = object_value(
        side_audit["results_by_tolerance_at_30_fps"], "side-audit tolerances"
    )
    test_at_five = object_value(test_tolerances["5"], "±5 side audit")
    test_vote = object_value(test_at_five["simple_vote"], "±5 side vote")

    figure, axis = plt.subplots(figsize=(13, 6.4))
    axis.axis("off")
    add_title(
        figure,
        "How the whole-rally side rule chooses Top and Bottom",
        "HGB chooses contact frames; a separate heuristic guesses each side; human labels are not used at run time",
    )

    axis.text(
        0.50,
        0.84,
        "Illustrative five-contact rally",
        ha="center",
        va="center",
        fontsize=13,
        weight="bold",
        color=INK,
        transform=axis.transAxes,
    )
    add_flow_box(
        axis,
        0.16,
        0.69,
        0.22,
        0.13,
        "HGB keeps five\ncontact frames",
        facecolour=PALE_BLUE,
        edgecolour=BLUE,
    )
    add_flow_arrow(axis, (0.16, 0.615), (0.16, 0.565))
    add_flow_box(
        axis,
        0.16,
        0.47,
        0.27,
        0.17,
        "Wrist-and-net heuristic\nTop · ? · Bottom · Top · Bottom",
        facecolour="white",
        edgecolour=BLUE,
        fontsize=10,
    )
    axis.text(
        0.16,
        0.34,
        "? gives no vote",
        ha="center",
        va="center",
        fontsize=9.5,
        color=GREY,
        transform=axis.transAxes,
    )

    add_flow_arrow(axis, (0.30, 0.52), (0.36, 0.66))
    add_flow_arrow(axis, (0.30, 0.52), (0.36, 0.43))
    add_flow_box(
        axis,
        0.51,
        0.66,
        0.31,
        0.15,
        "Sequence A\nTop · Bottom · Top · Bottom · Top\n1 agreement",
        facecolour="white",
        edgecolour=GREY,
        fontsize=9.7,
    )
    add_flow_box(
        axis,
        0.51,
        0.43,
        0.31,
        0.15,
        "Sequence B\nBottom · Top · Bottom · Top · Bottom\n3 agreements",
        facecolour=PALE_ORANGE,
        edgecolour=ORANGE,
        fontsize=9.7,
    )

    add_flow_arrow(axis, (0.67, 0.52), (0.73, 0.52), colour=ORANGE)
    add_flow_box(
        axis,
        0.84,
        0.53,
        0.21,
        0.22,
        "Choose sequence B\nlead = 2\n\nOutput alternates\nfor the whole rally",
        facecolour=PALE_ORANGE,
        edgecolour=ORANGE,
        fontsize=10.2,
    )
    axis.text(
        0.84,
        0.36,
        f"Rule acts when the lead is at least {minimum_lead}",
        ha="center",
        va="center",
        fontsize=9.5,
        color=GREY,
        transform=axis.transAxes,
    )

    development_repairs = integer(development_choice["repaired_sections"], "development repairs")
    development_breaks = integer(development_choice["broken_sections"], "development breaks")
    test_repairs = integer(test_vote["repaired_sections"], "test repairs")
    test_breaks = integer(test_vote["broken_sections"], "test breaks")
    add_flow_box(
        axis,
        0.30,
        0.14,
        0.39,
        0.13,
        f"Chosen on 40 ShuttleSet videos\n{development_repairs} wrong sections fixed · "
        f"{development_breaks} correct section damaged",
        facecolour=PALE_BLUE,
        edgecolour=BLUE,
        fontsize=9.8,
    )
    add_flow_box(
        axis,
        0.72,
        0.14,
        0.39,
        0.13,
        f"Held-out ShuttleSet22: 47 videos\n{test_repairs} wrong sections fixed · "
        f"{test_breaks} correct sections damaged",
        facecolour=PALE_PURPLE,
        edgecolour=PURPLE,
        fontsize=9.8,
    )
    save_figure(
        figure,
        paths,
        "06_side_rule_explainer",
        "Sources: results/side_development.json and results/side_audit.json",
    )


def plot_simple_rule_checks(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Show the empty duplicate audit and the cut-off sweep trade-off."""
    duplicate_audit = data["duplicate_audit"]
    development_duplicates = object_value(
        duplicate_audit["development"], "development duplicate audit"
    )
    test_duplicates = object_value(duplicate_audit["frozen_test"], "test duplicate audit")

    setting_sweep = data["setting_sweep"]
    best_setting = object_value(
        setting_sweep["global_descriptive_best"], "best contact-list setting"
    )
    setting_count = integer(setting_sweep["setting_count"], "contact-list setting count")
    repaired = integer(best_setting["repaired_sections"], "setting-sweep repairs")
    broken = integer(best_setting["broken_sections"], "setting-sweep breaks")
    net = integer(best_setting["net_sections"], "setting-sweep net")
    score_cutoff = number(best_setting["score_cutoff"], "best contact cut-off")
    copy_distance = integer(
        best_setting["duplicate_distance_at_30_fps"], "best nearby-copy distance"
    )

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [0.9, 1.2]})
    add_title(
        figure,
        "Neither simple contact-list rule gave a useful gain",
        "Duplicate audit: 40 ShuttleSet + 47 held-out ShuttleSet22 videos · "
        "cut-off sweep: ShuttleSet only · ±5 frames",
    )

    duplicate_axis, cutoff_axis = axes
    duplicate_axis.set_xlim(0, 1)
    duplicate_axis.set_ylim(0, 1)
    duplicate_axis.axis("off")
    duplicate_axis.set_title("Search for close opposite-side duplicates", pad=12)
    duplicate_axis.plot([0.15, 0.85], [0.62, 0.62], color=LIGHT_GREY, linewidth=2)
    duplicate_axis.scatter([0.32], [0.62], s=220, color=BLUE, edgecolor=INK, linewidth=0.8, zorder=3)
    duplicate_axis.scatter([0.68], [0.62], s=220, color=PURPLE, edgecolor=INK, linewidth=0.8, zorder=3)
    duplicate_axis.text(0.32, 0.72, "Top contact", ha="center", va="bottom", color=INK)
    duplicate_axis.text(0.68, 0.72, "Bottom contact", ha="center", va="bottom", color=INK)
    duplicate_axis.annotate(
        "0–2 frames apart",
        xy=(0.68, 0.50),
        xytext=(0.32, 0.50),
        ha="center",
        va="center",
        color=GREY,
        arrowprops={"arrowstyle": "<->", "color": GREY, "linewidth": 1.5},
    )
    duplicate_axis.text(
        0.50,
        0.34,
        "0",
        ha="center",
        va="center",
        fontsize=34,
        weight="bold",
        color=INK,
    )
    duplicate_axis.text(
        0.50,
        0.18,
        "matching pairs\n0 in ShuttleSet · 0 in held-out ShuttleSet22",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GREY,
    )
    if (
        integer(development_duplicates["pair_count"], "development duplicate pairs") != 0
        or integer(test_duplicates["pair_count"], "test duplicate pairs") != 0
    ):
        raise ValueError("The duplicate helper expects both saved audits to be empty")

    change_values = [-broken, repaired]
    change_labels = ["Correct timing sections\ndamaged", "Wrong timing sections\nfixed"]
    bars = cutoff_axis.barh([0, 1], change_values, color=[PURPLE, BLUE], height=0.55)
    cutoff_axis.axvline(0, color=INK, linewidth=0.9)
    cutoff_axis.set_yticks([0, 1], change_labels)
    cutoff_axis.set_xlim(-155, 165)
    cutoff_axis.set_ylim(-0.85, 1.55)
    cutoff_axis.set_xlabel("Sections")
    cutoff_axis.set_title(
        f"Best of {setting_count} settings: lower the HGB cut-off from 0.90 to {score_cutoff:.2f}\n"
        f"Keep the {copy_distance}-frame nearby-copy removal"
    )
    style_axes(cutoff_axis, grid_axis="x")
    for bar, value in zip(bars, change_values, strict=True):
        label_x = value - 7 if value < 0 else value + 7
        horizontal_alignment = "right" if value < 0 else "left"
        cutoff_axis.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            str(abs(value)),
            ha=horizontal_alignment,
            va="center",
            weight="bold",
            color=INK,
        )
    cutoff_axis.text(
        5,
        -0.62,
        f"Net gain: {net} timing-complete sections\n"
        f"Timing only · the {score_cutoff:.2f} setting was not tested on ShuttleSet22",
        ha="center",
        va="center",
        fontsize=10,
        color=GREY,
    )
    figure.tight_layout(rect=(0.05, 0.08, 0.98, 0.84), w_pad=4.0)
    save_figure(
        figure,
        paths,
        "07_simple_rule_checks",
        "Sources: results/opposite_side_duplicate_audit.json and results/setting_sweep.json",
    )


def plot_candidates_and_choosers(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Compare available first-contact repairs with repairs chosen by models."""
    start_best_case = data["start_best_case"]
    start_model = data["start_model_development"]
    start_validation = data["start_model_validation"]

    timing_then_side = object_value(
        start_best_case["timing_then_rally_side"], "first-contact best case"
    )
    chosen = object_value(start_model["chosen"], "chosen first-contact model")
    nested = object_value(start_model["nested_held_out_estimate"], "nested first-contact result")
    validation = object_value(
        object_value(start_validation["by_tolerance_at_30_fps"], "validation tolerances")["5"],
        "±5 first-contact validation",
    )

    first_labels = [
        "Labels checked every add/replace edit\n(shows what could be fixed)",
        "Model chosen and scored on\nthe same results (optimistic)",
        "Model chosen without using\nthe scored group's results",
    ]
    first_values = [
        integer(timing_then_side["repaired_sections"], "first-contact best-case repairs"),
        integer(chosen["repaired_sections"], "pooled first-contact repairs"),
        integer(nested["repaired_sections"], "nested first-contact repairs"),
    ]
    validation_changed = integer(start_validation["number_changed"], "validation sections changed")
    validation_repaired = integer(validation["repaired_sections"], "validation repairs")
    validation_broken = integer(validation["broken_sections"], "validation breaks")

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1.7, 0.8]})
    add_title(
        figure,
        "First-contact edits could make 300 training sections fully correct",
        "Allowed edit: add an earlier saved frame or replace the current first contact · "
        "ShuttleSet only · no ShuttleSet22 · ±5 frames",
    )
    figure.legend(
        handles=[
            Patch(facecolor=ORANGE, label="Labels identify an edit that works"),
            Patch(facecolor=BLUE, label="Model chooses without labels"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.075, 0.89),
        frameon=False,
        ncol=2,
        fontsize=9.5,
    )

    first_axis, validation_axis = axes
    first_positions = range(len(first_values))
    first_bars = first_axis.barh(first_positions, first_values, color=[ORANGE, BLUE, BLUE])
    first_axis.set_title("32 training videos · 2,850 sections")
    first_axis.set_yticks(first_positions, first_labels)
    first_axis.invert_yaxis()
    first_axis.set_xlabel("Wrong sections made fully correct")
    first_axis.set_xlim(0, 340)
    style_axes(first_axis, grid_axis="x")
    for bar, value in zip(first_bars, first_values, strict=True):
        first_axis.text(
            value + 6,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            ha="left",
            va="center",
            weight="bold",
            color=INK,
        )

    validation_axis.axis("off")
    validation_axis.text(
        0.04,
        0.92,
        "8 separate validation videos",
        ha="left",
        va="top",
        fontsize=14,
        weight="bold",
        color=INK,
        transform=validation_axis.transAxes,
    )
    validation_axis.text(
        0.04,
        0.80,
        "Model and cut-off fixed\nbefore these videos",
        ha="left",
        va="top",
        fontsize=10.5,
        color=GREY,
        transform=validation_axis.transAxes,
    )
    validation_metrics = [
        (validation_changed, "sections changed"),
        (validation_repaired, "wrong sections made fully correct"),
        (validation_broken, "correct sections damaged"),
    ]
    for vertical_position, (value, label) in zip(
        (0.58, 0.38, 0.18), validation_metrics, strict=True
    ):
        validation_axis.text(
            0.04,
            vertical_position,
            str(value),
            ha="left",
            va="center",
            fontsize=23,
            weight="bold",
            color=BLUE,
            transform=validation_axis.transAxes,
        )
        validation_axis.text(
            0.25,
            vertical_position,
            label,
            ha="left",
            va="center",
            fontsize=10.5,
            color=INK,
            transform=validation_axis.transAxes,
        )

    figure.tight_layout(rect=(0.05, 0.07, 0.98, 0.80), w_pad=4.0)
    save_figure(
        figure,
        paths,
        "03_candidates_and_choosers",
        "Sources: committed first-contact edit, model-development, and validation results",
    )


def keep_review_points(curve_value: Any, label: str) -> tuple[list[float], list[float], list[int]]:
    """Return non-empty coverage, precision, and accepted counts from a saved curve."""
    rows = list_value(curve_value, label)
    points: list[tuple[float, float, int]] = []
    for index, row_value in enumerate(rows):
        row = object_value(row_value, f"{label}[{index}]")
        if row["precision"] is None:
            continue
        points.append(
            (
                number(row["coverage"], "coverage") * 100,
                number(row["precision"], "precision") * 100,
                integer(row["accepted_count"], "accepted count"),
            )
        )
    points.sort()
    return (
        [point[0] for point in points],
        [point[1] for point in points],
        [point[2] for point in points],
    )


def plot_deletion_model_explainer(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Show how the deletion model acts and how its selected deletions ended."""
    delete_model = data["delete_model"]
    sections = integer(delete_model["sections"], "deletion experiment sections")
    contact_rows = integer(delete_model["delete_rows"], "deletion training rows")
    repairable = object_value(delete_model["ceiling_recoverable"], "deletion repair counts")
    repairable_at_five = integer(repairable["at_5_frames"], "repairable deletion sections")
    best = object_value(delete_model["descriptive_best"], "best deletion-model result")
    changed = integer(best["number_changed"], "model deletions")
    repaired = integer(best["repaired_sections"], "model deletion repairs")
    broken = integer(best["broken_sections"], "model deletion breaks")
    net = integer(best["net_sections"], "model deletion net")
    remained_wrong = changed - repaired - broken

    nested = object_value(delete_model["nested_held_out_estimate"], "held-out deletion estimate")
    group_choices = list_value(nested["choices_by_outer_group"], "held-out deletion choices")
    kept_groups = 0
    for choice_value in group_choices:
        choice = object_value(choice_value, "held-out deletion choice")
        if choice.get("action") == "keep":
            kept_groups += 1

    figure, axes = plt.subplots(1, 2, figsize=(13, 6.7), gridspec_kw={"width_ratios": [1.0, 1.15]})
    add_title(
        figure,
        "The deletion model scored every retained contact, then removed at most one",
        f"32 ShuttleSet training videos · {sections:,} sections · {contact_rows:,} contacts · "
        "no ShuttleSet22 · ±5 frames",
    )

    flow_axis, result_axis = axes
    flow_axis.axis("off")
    flow_axis.text(
        0.50,
        0.91,
        "How the second model works",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        color=INK,
        transform=flow_axis.transAxes,
    )
    add_flow_box(
        flow_axis,
        0.50,
        0.76,
        0.78,
        0.13,
        f"Training: {contact_rows:,} retained contacts\none row for every contact",
        facecolour=PALE_BLUE,
        edgecolour=BLUE,
    )
    add_flow_arrow(flow_axis, (0.50, 0.685), (0.50, 0.635))
    add_flow_box(
        flow_axis,
        0.50,
        0.54,
        0.84,
        0.16,
        "Positive only if deleting this contact\nmakes the whole rally fully correct\n"
        "at most one positive contact per section",
        facecolour=PALE_ORANGE,
        edgecolour=ORANGE,
        fontsize=10,
    )
    add_flow_arrow(flow_axis, (0.50, 0.45), (0.50, 0.40))
    add_flow_box(
        flow_axis,
        0.50,
        0.32,
        0.78,
        0.13,
        "Learn a new deletion score\nfrom 12 contact-and-rally measurements",
        facecolour="white",
        edgecolour=PURPLE,
    )
    add_flow_arrow(flow_axis, (0.50, 0.245), (0.50, 0.20))
    add_flow_box(
        flow_axis,
        0.50,
        0.11,
        0.88,
        0.15,
        "Run time: score every contact → take the highest\n"
        "above cut-off: delete one · below cut-off: keep all",
        facecolour=PALE_PURPLE,
        edgecolour=PURPLE,
        fontsize=9.8,
    )

    result_axis.set_title(
        "What happened when the model and cut-off\nwere chosen on these same videos"
    )
    result_axis.barh([1], [repairable_at_five], color=ORANGE, height=0.48)
    result_axis.text(
        repairable_at_five + 8,
        1,
        str(repairable_at_five),
        ha="left",
        va="center",
        weight="bold",
        color=INK,
    )

    outcome_values = [repaired, broken, remained_wrong]
    outcome_colours = [BLUE, RED, GREY]
    left_edge = 0
    for value, colour in zip(outcome_values, outcome_colours, strict=True):
        result_axis.barh([0], [value], left=left_edge, color=colour, height=0.48)
        result_axis.text(
            left_edge + value / 2,
            0,
            str(value),
            ha="center",
            va="center",
            weight="bold",
            color="white",
            fontsize=9.5,
        )
        left_edge += value
    result_axis.text(
        changed + 8,
        0,
        f"{changed} deletions",
        ha="left",
        va="center",
        weight="bold",
        color=INK,
    )
    result_axis.set_yticks(
        [1, 0],
        [
            "Labels found a repair\nfrom one deletion",
            "Model chose one contact\nto delete",
        ],
    )
    result_axis.set_xlim(0, 550)
    result_axis.set_xlabel("Sections")
    style_axes(result_axis, grid_axis="x")
    result_axis.set_ylim(-1.00, 1.70)
    result_axis.text(0, -0.38, "42 fixed", ha="left", va="center", color=BLUE, weight="bold")
    result_axis.text(105, -0.38, "88 damaged", ha="left", va="center", color=RED, weight="bold")
    result_axis.text(235, -0.38, "367 still wrong", ha="left", va="center", color=GREY, weight="bold")
    result_axis.text(
        275,
        -0.66,
        f"Net result: {abs(net)} fewer fully correct sections.",
        ha="center",
        va="center",
        fontsize=10,
        color=INK,
        weight="bold",
    )
    result_axis.text(
        275,
        -0.84,
        f"Held-out selection kept every contact in {kept_groups} of {len(group_choices)} groups.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=GREY,
    )
    figure.subplots_adjust(left=0.05, right=0.97, top=0.78, bottom=0.13, wspace=0.42)
    save_figure(
        figure,
        paths,
        "08_deletion_model_explainer",
        "Source: results/delete_model_development.json",
    )


def plot_keep_review_curve(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot precision against the share of sections accepted automatically."""
    keep_review = data["keep_review"]
    coverage_5, precision_5, accepted_5 = keep_review_points(
        keep_review["curve_at_5_frames"], "±5 keep/review curve"
    )
    coverage_10, precision_10, _accepted_10 = keep_review_points(
        keep_review["curve_at_10_frames"], "±10 keep/review curve"
    )

    figure, axis = plt.subplots(figsize=(10, 6.4))
    add_title(
        figure,
        "ShuttleSet training data: the rally-level model stayed far below 90% precision",
        "32 videos · 2,850 held-out sections · no ShuttleSet22 · target: 10% coverage",
    )
    axis.fill_between([10, 55], 90, 100, color=PALE_BLUE, alpha=0.9, label="Target region")
    axis.plot(coverage_5, precision_5, marker="o", linewidth=2.2, color=BLUE, label="±5 frames")
    axis.plot(coverage_10, precision_10, marker="s", linewidth=2.2, color=PURPLE, label="±10 frames")
    axis.axvline(10, color=GREY, linestyle="--", linewidth=1)
    axis.axhline(90, color=GREY, linestyle="--", linewidth=1)
    axis.set_xlim(0, 55)
    axis.set_ylim(0, 100)
    axis.set_xlabel("Coverage: sections accepted automatically (%)")
    axis.set_ylabel("Precision: accepted sections that were fully correct (%)")
    axis.legend(loc="lower right")
    style_axes(axis, grid_axis="both")

    for coverage, precision, accepted in zip(coverage_5, precision_5, accepted_5, strict=True):
        if accepted not in {8, 460}:
            continue
        axis.annotate(
            f"{accepted} accepted\n{precision:.1f}% right",
            (coverage, precision),
            xytext=(8, -26 if accepted == 460 else 10),
            textcoords="offset points",
            fontsize=9,
            color=BLUE,
        )

    figure.tight_layout(rect=(0.06, 0.07, 0.98, 0.88))
    save_figure(
        figure,
        paths,
        "04_keep_review_curve",
        "Source: results/keep_review_development.json",
    )


def main() -> None:
    """Load the saved results and rebuild all report figures."""
    configure_style()
    paths = build_paths()
    data = {
        "side_audit": load_json(paths.side_audit),
        "side_development": load_json(paths.side_development),
        "duplicate_audit": load_json(paths.duplicate_audit),
        "setting_sweep": load_json(paths.setting_sweep),
        "start_best_case": load_json(paths.start_best_case),
        "start_model_development": load_json(paths.start_model_development),
        "start_model_validation": load_json(paths.start_model_validation),
        "combined_best_case": load_json(paths.combined_best_case),
        "delete_model": load_json(paths.delete_model),
        "keep_review": load_json(paths.keep_review),
    }
    plot_complete_rallies(data, paths)
    plot_baseline_errors(paths)
    plot_side_rule_explainer(data, paths)
    plot_simple_rule_checks(data, paths)
    plot_candidates_and_choosers(data, paths)
    plot_deletion_model_explainer(data, paths)
    plot_keep_review_curve(data, paths)


if __name__ == "__main__":
    main()
