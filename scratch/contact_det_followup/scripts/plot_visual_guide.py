"""Build the nine-image standalone visual guide for the contact follow-up."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.switch_backend("Agg")


JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP_DIR = REPO_ROOT / "scratch/contact_det_followup"
FIGURE_DIR = FOLLOWUP_DIR / "figures"
TOLERANCES = (5, 10)
FIGURE_SIZE = (16, 10)

PAPER = "#FAFBFC"
INK = "#17212B"
BLUE = "#0072B2"
ORANGE = "#D97706"
PURPLE = "#7A3E9D"
GREY = "#667085"
LIGHT_GREY = "#D0D5DD"
PALE_BLUE = "#E7F1F8"
PALE_ORANGE = "#FFF1DA"
PALE_PURPLE = "#F0E9F5"
PALE_GREY = "#F0F2F5"


@dataclass(frozen=True)
class DataPaths:
    """Input and output paths used by the visual guide."""

    baseline: Path
    side_audit: Path
    setting_sweep: Path
    start_model_development: Path
    start_model_validation: Path
    combined_best_case: Path
    delete_model: Path
    keep_review: Path
    duplicate_audit: Path
    shuttleset_result: Path
    motif: Path
    figure_dir: Path


def build_paths() -> DataPaths:
    """Return paths to the saved evidence records."""
    results_dir = FOLLOWUP_DIR / "results"
    return DataPaths(
        baseline=results_dir / "baseline_recount.json",
        side_audit=results_dir / "side_audit.json",
        setting_sweep=results_dir / "setting_sweep.json",
        start_model_development=results_dir / "start_model_development.json",
        start_model_validation=results_dir / "start_model_validation.json",
        combined_best_case=results_dir / "combined_best_case.json",
        delete_model=results_dir / "delete_model_development.json",
        keep_review=results_dir / "keep_review_development.json",
        duplicate_audit=results_dir / "opposite_side_duplicate_audit.json",
        shuttleset_result=REPO_ROOT
        / "scratch/contact_det_full_ds_fit/raw/shuttleset22-test-result/result.json",
        motif=FIGURE_DIR / "scorecard_court_motif.png",
        figure_dir=FIGURE_DIR,
    )


def load_json(path: Path) -> JsonObject:
    """Load a JSON object from a saved evidence record."""
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def mapping(value: Any, label: str) -> JsonObject:
    """Return a required JSON object."""
    if not isinstance(value, dict):
        raise TypeError(f"Expected {label} to be an object")
    return value


def list_value(value: Any, label: str) -> list[Any]:
    """Return a required JSON list."""
    if not isinstance(value, list):
        raise TypeError(f"Expected {label} to be a list")
    return value


def integer(value: Any, label: str) -> int:
    """Return a required integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Expected {label} to be an integer")
    return value


def real(value: Any, label: str) -> float:
    """Return a required real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"Expected {label} to be numeric")
    return float(value)


def tolerance_record(record: JsonObject, tolerance: int, label: str) -> JsonObject:
    """Return a tolerance-keyed record."""
    records_value = record.get(
        "results_by_tolerance_at_30_fps", record.get("by_tolerance_at_30_fps")
    )
    records = mapping(records_value, f"{label}.tolerance records")
    return mapping(records.get(str(tolerance)), f"{label}[{tolerance}]")


def nested_tolerance_record(
    record: JsonObject, tolerance: int, label: str
) -> JsonObject:
    """Return a record keyed by the plain tolerance field."""
    records = mapping(
        record.get("results_by_tolerance"), f"{label}.results_by_tolerance"
    )
    return mapping(records.get(str(tolerance)), f"{label}[{tolerance}]")


def fraction_text(value: float) -> str:
    """Format a proportion as a percentage."""
    return f"{value * 100:.1f}%"


def configure_style() -> None:
    """Set a shared, readable style for all nine outputs."""
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": LIGHT_GREY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "grid.color": LIGHT_GREY,
            "grid.alpha": 0.6,
            "grid.linewidth": 0.75,
        }
    )


def add_header(figure: Figure, title: str, subtitle: str) -> None:
    """Add a numbered question title and a dataset/tolerance subtitle."""
    figure.text(
        0.045, 0.955, title, ha="left", va="top", fontsize=20, weight="bold", color=INK
    )
    figure.text(0.045, 0.915, subtitle, ha="left", va="top", fontsize=11.5, color=GREY)


def add_footer(figure: Figure, takeaway: str, source: str) -> None:
    """Add the takeaway band and small source/method note."""
    figure.patches.append(
        Rectangle(
            (0.03, 0.062),
            0.94,
            0.065,
            transform=figure.transFigure,
            facecolor=PALE_BLUE,
            edgecolor="none",
            zorder=-1,
        )
    )
    figure.text(
        0.045,
        0.097,
        f"TAKEAWAY  {takeaway}",
        ha="left",
        va="center",
        fontsize=11.5,
        weight="bold",
        color=INK,
    )
    figure.text(0.045, 0.025, source, ha="left", va="bottom", fontsize=8.5, color=GREY)


def save_figure(figure: Figure, paths: DataPaths, stem: str) -> None:
    """Save one figure as PNG and SVG at the requested resolution."""
    paths.figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        paths.figure_dir / f"{stem}.png",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor=PAPER,
    )
    figure.savefig(
        paths.figure_dir / f"{stem}.svg",
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor=PAPER,
    )
    plt.close(figure)


def style_axes(axis: Axes, *, grid_axis: str = "y") -> None:
    """Apply light-grid and spine styling to one axis."""
    axis.grid(axis=grid_axis, color=LIGHT_GREY, alpha=0.6, linewidth=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(LIGHT_GREY)
    axis.spines["bottom"].set_color(LIGHT_GREY)


def draw_metric_card(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    heading: str,
    value: str,
    detail: str,
    colour: str,
    fill: str,
) -> None:
    """Draw one overview card in axis coordinates."""
    card = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        facecolor=fill,
        edgecolor=colour,
        linewidth=2,
    )
    axis.add_patch(card)
    axis.text(
        x + 0.04,
        y + height - 0.1,
        heading,
        va="top",
        fontsize=12.5,
        weight="bold",
        color=colour,
    )
    axis.text(
        x + 0.04,
        y + height * 0.58,
        value,
        va="center",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    axis.text(
        x + 0.04, y + 0.11, detail, va="bottom", fontsize=9.8, color=GREY, wrap=True
    )


def plot_story_scorecard(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot the overview cards for the baseline and follow-up decisions."""
    baseline = data["baseline"]
    side = data["side_audit"]
    setting = data["setting_sweep"]
    validation = data["start_model_validation"]
    delete = data["delete_model"]
    keep = data["keep_review"]
    baseline_sections = integer(
        mapping(baseline["metrics_by_tolerance_at_30_fps"], "baseline metrics")["5"][
            "strict_fully_correct_sections"
        ],
        "baseline sections",
    )
    baseline_sections_10 = integer(
        mapping(baseline["metrics_by_tolerance_at_30_fps"], "baseline metrics")["10"][
            "strict_fully_correct_sections"
        ],
        "baseline sections",
    )
    side_five = mapping(tolerance_record(side, 5, "side")["simple_vote"], "side vote")
    side_ten = mapping(tolerance_record(side, 10, "side")["simple_vote"], "side vote")
    setting_five = mapping(
        tolerance_record(setting, 5, "setting")["global_descriptive_best"],
        "setting best",
    )
    setting_ten = mapping(
        tolerance_record(setting, 10, "setting")["global_descriptive_best"],
        "setting best",
    )
    valid_five = mapping(
        tolerance_record(validation, 5, "validation"), "validation five"
    )
    valid_ten = mapping(
        tolerance_record(validation, 10, "validation"), "validation ten"
    )
    delete_five = mapping(delete["descriptive_best"], "delete best")
    delete_ten = mapping(delete["descriptive_best_at_10_frames"], "delete best ten")
    keep_five = next(
        row
        for row in list_value(keep["curve_at_5_frames"], "keep five curve")
        if real(mapping(row, "keep five row")["threshold"], "threshold") == 0.6
    )
    keep_ten = next(
        row
        for row in list_value(keep["curve_at_10_frames"], "keep ten curve")
        if real(mapping(row, "keep row")["threshold"], "threshold") == 0.6
    )
    figure, axis = plt.subplots(figsize=FIGURE_SIZE)
    add_header(
        figure,
        "01 — Which follow-up choice actually changes the output?",
        "Frozen test for the baseline and side vote; development and held-out checks for later ideas · ±5 primary, ±10 beside it",
    )
    axis.set_xlim(0, 3)
    axis.set_ylim(0, 2.35)
    axis.axis("off")
    cards = [
        (
            "BASELINE",
            f"{baseline_sections:,}  |  {baseline_sections_10:,}",
            "Fully correct sections\n±5  |  ±10",
            BLUE,
            PALE_BLUE,
        ),
        (
            "SIDE VOTE · KEEP",
            f"{integer(side_five['revised_strict_fully_correct'], 'side revised'):,}  |  {integer(side_ten['revised_strict_fully_correct'], 'side revised'):,}",
            "Frozen 47-video test\nNo breaks",
            BLUE,
            PALE_BLUE,
        ),
        (
            "SETTING LEAD · PARK",
            f"+{integer(setting_five['net_sections'], 'setting net')}  |  +{integer(setting_ten['net_sections'], 'setting net ten')}",
            "0.85 / 6 timing only\nSaved decision: stop",
            ORANGE,
            PALE_ORANGE,
        ),
        (
            "HELD-OUT START · STOP",
            f"+{integer(valid_five['net_sections'], 'validation net')}  |  +{integer(valid_ten['net_sections'], 'validation net ten')}",
            "Untouched V · 8 videos\nNo breaks",
            PURPLE,
            PALE_PURPLE,
        ),
        (
            "DELETE · STOP",
            f"{integer(delete_five['net_sections'], 'delete net')}  |  {integer(delete_ten['net_sections'], 'delete net ten')}",
            "Learned delete chooser\nBreaks exceed repairs",
            ORANGE,
            PALE_ORANGE,
        ),
        (
            "KEEP / REVIEW · STOP",
            f"{fraction_text(real(keep_five['precision'], 'keep precision'))}  |  {fraction_text(real(mapping(keep_ten, 'keep ten')['precision'], 'keep ten precision'))}",
            "Precision at threshold 0.60\nTarget: 90% at 10% coverage",
            GREY,
            PALE_GREY,
        ),
    ]
    positions = [
        (0.08, 1.23),
        (1.05, 1.23),
        (2.02, 1.23),
        (0.08, 0.25),
        (1.05, 0.25),
        (2.02, 0.25),
    ]
    for card, (x, y) in zip(cards, positions, strict=True):
        draw_metric_card(axis, x, y, 0.9, 0.84, *card)
    add_footer(
        figure,
        "Keep the whole-rally side vote; park the small leads until better evidence exists.",
        "Sources: baseline_recount.json; side_audit.json; setting_sweep.json; start_model_validation.json; delete_model_development.json; keep_review_development.json.",
    )
    save_figure(figure, paths, "01_story_scorecard")


def plot_baseline_funnel_and_error_map(
    data: dict[str, JsonObject], paths: DataPaths
) -> None:
    """Plot the raw frozen-test contact funnel and one-rally error mix."""
    baseline = data["baseline"]
    result = data["shuttleset_result"]
    rallies = mapping(
        mapping(result["whole_rallies"], "whole rallies")["by_tolerance"],
        "whole-rally tolerances",
    )
    figure, axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [0.95, 1.45]}
    )
    add_header(
        figure,
        "02 — Where does a frozen baseline rally output lose accuracy?",
        "ShuttleSet-22 frozen 47-video test · strict recount plus one-rally errors · ±5 primary, ±10 beside it",
    )
    metrics = mapping(
        mapping(baseline["metrics_by_tolerance_at_30_fps"], "baseline metrics")["5"],
        "±5 baseline",
    )
    rally_five = mapping(rallies["5"], "±5 rallies")
    funnel_labels = [
        "Labelled\ncontacts",
        "Timing\nmatched",
        "Correct\nplayer side",
        "Mapped\nsections",
        "Strictly\ncorrect",
    ]
    funnel_values = [
        integer(metrics["labelled_contacts"], "labelled contacts"),
        integer(metrics["matched_contacts"], "matched contacts"),
        integer(metrics["correct_player_sides"], "correct player sides"),
        integer(rally_five["mapped_sections"], "mapped sections"),
        integer(
            metrics["strict_fully_correct_sections"], "strict fully correct sections"
        ),
    ]
    funnel_bars = axes[0].bar(
        range(len(funnel_labels)),
        funnel_values,
        color=[GREY, BLUE, PURPLE, GREY, BLUE],
        width=0.68,
    )
    axes[0].set_xticks(range(len(funnel_labels)), funnel_labels)
    axes[0].set_ylabel("Contacts, then sections")
    axes[0].set_title("Contact → strict section funnel · ±5", loc="left")
    axes[0].set_ylim(0, max(funnel_values) * 1.15)
    for bar, value in zip(funnel_bars, funnel_values, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    error_labels = [
        "Missing contact",
        "Wrong side",
        "Timing mismatch",
        "Missing + extra",
        "Extra contact",
        "Side unanswered",
    ]
    error_keys = [
        "missing_contacts_only",
        "wrong_predicted_side",
        "timing_mismatch_equal_counts",
        "missing_and_extra_contacts",
        "extra_contacts_only",
        "predicted_side_unanswered",
    ]
    error_values: list[list[int]] = []
    for tolerance in TOLERANCES:
        outcomes = mapping(rallies[str(tolerance)]["outcome_counts"], "outcome counts")
        error_values.append(
            [integer(outcomes[key], f"outcome {key}") for key in error_keys]
        )
    error_positions = list(range(len(error_labels)))
    bar_width = 0.35
    primary_error_bars = axes[1].barh(
        [position - bar_width / 2 for position in error_positions],
        error_values[0],
        bar_width,
        color=BLUE,
        label="±5",
    )
    wider_error_bars = axes[1].barh(
        [position + bar_width / 2 for position in error_positions],
        error_values[1],
        bar_width,
        color=GREY,
        label="±10",
    )
    axes[1].set_yticks(error_positions, error_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("One-rally sections")
    axes[1].set_title("Error categories at both tolerances", loc="left")
    axes[1].set_xlim(0, max(max(values) for values in error_values) * 1.2)
    axes[1].legend(loc="lower right")
    for bars, values in [
        (primary_error_bars, error_values[0]),
        (wider_error_bars, error_values[1]),
    ]:
        for bar, value in zip(bars, values, strict=True):
            axes[1].text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f"  {value:,}",
                va="center",
                fontsize=8.5,
            )
    style_axes(axes[0])
    style_axes(axes[1], grid_axis="x")
    add_footer(
        figure,
        "The dominant section failures are missing contacts and wrong player sides, not isolated score noise.",
        "Sources: baseline_recount.json and the frozen-test result.json. Error categories use the older one-rally outcome counts; the funnel ends on the strict recount.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.28)
    save_figure(figure, paths, "02_baseline_funnel_and_error_map")


def plot_side_vote_gain_and_tradeoff(
    data: dict[str, JsonObject], paths: DataPaths
) -> None:
    """Plot side-vote strict output gains and contact-level trade-offs."""
    side = data["side_audit"]
    figure, axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE, gridspec_kw={"width_ratios": [1.1, 1]}
    )
    add_header(
        figure,
        "03 — Does the whole-rally side vote trade local accuracy for complete outputs?",
        "ShuttleSet-22 frozen 47-video test · fixed label-free vote scored once · ±5 primary, ±10 beside it",
    )
    votes = {
        tolerance: mapping(
            tolerance_record(side, tolerance, "side")["simple_vote"], "simple vote"
        )
        for tolerance in TOLERANCES
    }
    tolerance_positions = [0, 1]
    width = 0.34
    baseline_values = [
        integer(votes[tolerance]["baseline_strict_fully_correct"], "baseline strict")
        for tolerance in TOLERANCES
    ]
    revised_values = [
        integer(votes[tolerance]["revised_strict_fully_correct"], "revised strict")
        for tolerance in TOLERANCES
    ]
    baseline_bars = axes[0].bar(
        [position - width / 2 for position in tolerance_positions],
        baseline_values,
        width,
        color=GREY,
        label="Baseline",
    )
    revised_bars = axes[0].bar(
        [position + width / 2 for position in tolerance_positions],
        revised_values,
        width,
        color=BLUE,
        label="Side vote",
    )
    axes[0].set_xticks(tolerance_positions, ["±5 frames\n(primary)", "±10 frames"])
    axes[0].set_title("Complete section outputs", loc="left")
    axes[0].set_ylabel("Fully correct sections")
    axes[0].set_ylim(0, max(revised_values) * 1.25)
    axes[0].legend(loc="upper left")
    for bars in (baseline_bars, revised_bars):
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=11,
                weight="bold",
            )
    axes[0].text(
        0.98,
        0.96,
        "Repairs +418 / +471 · breaks 0 / 0",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=BLUE,
        fontsize=10,
    )
    style_axes(axes[0])

    metric_labels = [
        "Matched-contact\nside accuracy",
        "Contact + side\nF1",
        "Strict output\nprecision",
    ]
    baseline_fields = [
        "baseline_side_accuracy",
        "baseline_contact_and_side_f1",
        "baseline_full_output_precision",
    ]
    revised_fields = [
        "revised_side_accuracy",
        "revised_contact_and_side_f1",
        "revised_full_output_precision",
    ]
    changes = {
        tolerance: [
            (
                real(votes[tolerance][revised_field], revised_field)
                - real(votes[tolerance][baseline_field], baseline_field)
            )
            * 100
            for baseline_field, revised_field in zip(
                baseline_fields, revised_fields, strict=True
            )
        ]
        for tolerance in TOLERANCES
    }
    metric_positions = list(range(len(metric_labels)))
    bars_five = axes[1].bar(
        [position - width / 2 for position in metric_positions],
        changes[5],
        width,
        color=BLUE,
        label="±5 frames",
    )
    bars_ten = axes[1].bar(
        [position + width / 2 for position in metric_positions],
        changes[10],
        width,
        color=PURPLE,
        label="±10 frames",
    )
    axes[1].axhline(0, color=INK, linewidth=0.8)
    axes[1].set_xticks(metric_positions, metric_labels)
    axes[1].set_ylim(-2.5, 13)
    axes[1].set_ylabel("Change from baseline (percentage points)")
    axes[1].set_title("What changes locally?", loc="left")
    axes[1].legend(loc="upper left")
    for bars in (bars_five, bars_ten):
        for bar in bars:
            value = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.25 if value >= 0 else -0.25),
                f"{value:+.1f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
    style_axes(axes[1])
    add_footer(
        figure,
        "The side vote lifts complete rallies by 418 at ±5 and 471 at ±10, while local side accuracy falls modestly.",
        "Source: side_audit.json. Strict precision and contact + side F1 are scored on the frozen 47-video test; ±10 local metrics are retained in the saved record.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.28)
    save_figure(figure, paths, "03_side_vote_gain_and_tradeoff")


def plot_repairs_breaks_net(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot repairs, breaks and net changes across follow-up populations."""
    side = data["side_audit"]
    setting = data["setting_sweep"]
    start = data["start_model_development"]
    validation = data["start_model_validation"]
    delete = data["delete_model"]
    labels = [
        "Side vote\nfrozen test",
        "0.85 / 6\nall development",
        "First contact\ntraining estimate",
        "First contact\nheld-out check",
        "Delete model\ndevelopment",
    ]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    add_header(
        figure,
        "04 — Which follow-up lines repair more sections than they break?",
        "Net = repairs minus breaks · 47 test videos; development splits 32 fit + 8 held out · ±5 primary",
    )
    for axis, tolerance in zip(axes, TOLERANCES, strict=True):
        side_row = mapping(
            tolerance_record(side, tolerance, "side")["simple_vote"], "side vote"
        )
        setting_row = mapping(
            tolerance_record(setting, tolerance, "setting")["global_descriptive_best"],
            "setting best",
        )
        start_row = mapping(
            start["chosen_at_10_frames"] if tolerance == 10 else start["chosen"],
            "start model chosen",
        )
        valid_row = mapping(
            tolerance_record(validation, tolerance, "validation"), "start validation"
        )
        delete_row = mapping(
            delete["descriptive_best_at_10_frames"]
            if tolerance == 10
            else delete["descriptive_best"],
            "delete best",
        )
        repairs = [
            integer(row[field], field)
            for row, field in [
                (side_row, "repaired_sections"),
                (setting_row, "repaired_sections"),
                (start_row, "repaired_sections"),
                (valid_row, "repaired_sections"),
                (delete_row, "repaired_sections"),
            ]
        ]
        breaks = [
            integer(row[field], field)
            for row, field in [
                (side_row, "broken_sections"),
                (setting_row, "broken_sections"),
                (start_row, "broken_sections"),
                (valid_row, "broken_sections"),
                (delete_row, "broken_sections"),
            ]
        ]
        nets = [
            integer(row[field], field)
            for row, field in [
                (side_row, "net_sections"),
                (setting_row, "net_sections"),
                (start_row, "net_sections"),
                (valid_row, "net_sections"),
                (delete_row, "net_sections"),
            ]
        ]
        x_positions = list(range(len(labels)))
        width = 0.24
        repair_bars = axis.bar(
            [position - width for position in x_positions],
            repairs,
            width,
            color=BLUE,
            label="Repairs",
        )
        break_bars = axis.bar(x_positions, breaks, width, color=ORANGE, label="Breaks")
        net_bars = axis.bar(
            [position + width for position in x_positions],
            nets,
            width,
            color=PURPLE,
            label="Net",
        )
        axis.axhline(0, color=INK, linewidth=0.8)
        axis.set_xticks(x_positions, labels)
        axis.set_title(f"Repairs, breaks and net · ±{tolerance}", loc="left")
        axis.set_ylabel("Sections")
        axis.set_ylim(min(-110, min(nets) - 60), max(repairs) * 1.2)
        axis.legend(loc="upper left", ncol=3)
        for bars in [repair_bars, break_bars, net_bars]:
            for bar in bars:
                value = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (8 if value >= 0 else -15),
                    f"{value:+,}" if value < 0 else f"{value:,}",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8.5,
                )
        style_axes(axis)
        axis.text(
            0.99,
            0.96,
            "PRIMARY" if tolerance == 5 else "WIDER TOLERANCE",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=BLUE if tolerance == 5 else GREY,
            weight="bold",
            fontsize=9,
        )
    add_footer(
        figure,
        "Only the side vote has a large, safe net gain; all other lines remain small leads or fail their safety gate.",
        "Sources: side_audit.json; setting_sweep.json; start_model_development.json; start_model_validation.json; delete_model_development.json. Populations and label access differ by row.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.2)
    save_figure(figure, paths, "04_repairs_breaks_net")


def plot_setting_sweep(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot all 57 timing-only score-cutoff and merge-distance choices."""
    setting = data["setting_sweep"]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    add_header(
        figure,
        "05 — Does any of the 57 contact settings clearly win?",
        "A-D + V development timing-complete sections · each panel shows every cut-off / merge choice · ±5 primary, ±10 beside it",
    )
    for axis, tolerance in zip(axes, TOLERANCES, strict=True):
        setting_records = (
            setting["settings"]
            if tolerance == 5
            else tolerance_record(setting, 10, "setting sweep")["settings"]
        )
        rows = [
            mapping(row, "setting row")
            for row in list_value(setting_records, "setting rows")
        ]
        colours_by_merge = {4: GREY, 5: ORANGE, 6: BLUE}
        for merge_distance in (4, 5, 6):
            merge_rows = sorted(
                (
                    row
                    for row in rows
                    if integer(row["duplicate_distance_at_30_fps"], "merge distance")
                    == merge_distance
                ),
                key=lambda row: real(row["score_cutoff"], "score cutoff"),
            )
            cutoffs = [real(row["score_cutoff"], "score cutoff") for row in merge_rows]
            values = [
                integer(row["timing_complete_sections"], "timing-complete sections")
                for row in merge_rows
            ]
            axis.plot(
                cutoffs,
                values,
                color=colours_by_merge[merge_distance],
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"Merge distance {merge_distance}",
            )
        baseline_row = next(
            row
            for row in rows
            if real(row["score_cutoff"], "cutoff") == 0.9
            and integer(row["duplicate_distance_at_30_fps"], "merge") == 6
        )
        best_row = next(
            row
            for row in rows
            if real(row["score_cutoff"], "cutoff") == 0.85
            and integer(row["duplicate_distance_at_30_fps"], "merge") == 6
        )
        baseline_value = integer(
            baseline_row["timing_complete_sections"],
            "baseline timing-complete sections",
        )
        best_value = integer(
            best_row["timing_complete_sections"], "best timing-complete sections"
        )
        axis.scatter([0.9], [baseline_value], c=GREY, s=85, marker="s", zorder=4)
        axis.scatter(
            [0.85],
            [best_value],
            c=BLUE,
            s=100,
            marker="o",
            edgecolor=INK,
            linewidth=0.8,
            zorder=4,
        )
        axis.annotate(
            f"0.90 / 6 · {baseline_value:,}",
            (0.9, baseline_value),
            xytext=(8, -17),
            textcoords="offset points",
            fontsize=8.5,
            color=GREY,
        )
        axis.annotate(
            f"0.85 / 6 · {best_value:,}",
            (0.85, best_value),
            xytext=(-88, 10),
            textcoords="offset points",
            fontsize=8.5,
            color=BLUE,
            weight="bold",
        )
        axis.set_xticks([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
        axis.set_xlim(0.03, 0.97)
        axis.set_xlabel("Contact-score cut-off")
        axis.set_title(f"Timing-complete sections · ±{tolerance}", loc="left")
        axis.set_ylabel("Sections")
        axis.legend(loc="upper left", fontsize=9)
        style_axes(axis)
    add_footer(
        figure,
        "The 0.85 / 6 point is a timing lead only; most saved alternative frames do not carry player sides. The saved decision remains stop.",
        "Source: setting_sweep.json. All 57 settings are read from the saved settings arrays; this sweep measures timing-complete sections, not complete contact-and-side outputs.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.16)
    save_figure(figure, paths, "05_setting_sweep")


def plot_first_contact_ceiling_vs_achieved(
    data: dict[str, JsonObject], paths: DataPaths
) -> None:
    """Plot the label-guided first-contact ceiling against model results."""
    best_case = data["combined_best_case"]
    development = data["start_model_development"]
    validation = data["start_model_validation"]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    add_header(
        figure,
        "06 — How much of the first-contact ceiling does a label-free model reach?",
        "A-D = 32 development videos · V = 8 videos kept out of training · the ceiling uses labels",
    )
    for axis, tolerance in zip(axes, TOLERANCES, strict=True):
        combined_row = nested_tolerance_record(
            best_case, tolerance, "combined best case"
        )
        start_ceiling = mapping(combined_row["start_only"], "start-only ceiling")
        model_row = mapping(
            development["chosen_at_10_frames"]
            if tolerance == 10
            else development["chosen"],
            "pooled model",
        )
        nested_row = mapping(development["nested_held_out_estimate"], "nested model")
        nested_row = mapping(
            nested_row["at_10_frames"] if tolerance == 10 else nested_row,
            "nested estimate",
        )
        valid_row = mapping(
            tolerance_record(validation, tolerance, "held-out V"), "held-out V"
        )
        labels = [
            "With labels:\nbest allowed edit",
            "Without labels:\n32-video fit",
            "Without labels:\ngroup-held-out check",
            "Without labels:\n8-video V check",
        ]
        values = [
            integer(
                start_ceiling["event_edit_repaired_sections"],
                "start event-edit repairs",
            ),
            integer(model_row["repaired_sections"], "pooled model repairs"),
            integer(nested_row["repaired_sections"], "nested repairs"),
            integer(valid_row["repaired_sections"], "V repairs"),
        ]
        colours = [PURPLE, BLUE, GREY, BLUE]
        hatches = ["///", "", "", ""]
        bars = axis.bar(labels, values, color=colours, hatch=hatches, width=0.58)
        axis.set_title(f"First-contact repairs · ±{tolerance}", loc="left")
        axis.set_ylabel("Fully correct sections repaired")
        axis.set_ylim(0, max(values) * 1.35)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:,}",
                ha="center",
                va="bottom",
                fontsize=10.5,
                weight="bold",
            )
        axis.text(
            0.03,
            0.96,
            "//// label-guided ceiling",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=PURPLE,
            fontsize=9.2,
        )
        axis.text(
            0.03,
            0.91,
            "solid = label-free model",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=BLUE,
            fontsize=9.2,
        )
        style_axes(axis)
    add_footer(
        figure,
        "The cautious model repairs only a small fraction of the label-guided room; V is a different population and stays the decisive check.",
        "Sources: combined_best_case.json; start_model_development.json; start_model_validation.json. A-D uses 32 development videos; V uses eight held-out videos.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.18)
    save_figure(figure, paths, "06_first_contact_ceiling_vs_achieved")


def plot_combined_ceiling_vs_delete_model(
    data: dict[str, JsonObject], paths: DataPaths
) -> None:
    """Plot event-edit ceilings beside the learned delete model."""
    combined = data["combined_best_case"]
    delete = data["delete_model"]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    add_header(
        figure,
        "07 — Can a label-free delete model approach the combined event-edit ceiling?",
        "A-D development sections · label-guided ceilings versus label-free learned-delete outcomes · ±5 primary, ±10 beside it",
    )
    for axis, tolerance in zip(axes, TOLERANCES, strict=True):
        combined_row = nested_tolerance_record(
            combined, tolerance, "combined best case"
        )
        combined_summary = mapping(combined_row["combined"], "combined ceiling")
        delete_ceiling = mapping(combined_row["delete_only"], "delete ceiling")
        delete_model = mapping(
            delete["descriptive_best_at_10_frames"]
            if tolerance == 10
            else delete["descriptive_best"],
            "delete model",
        )
        labels = [
            "Delete-only\nceiling",
            "Combined\nceiling",
            "Model\nrepairs",
            "Model\nbreaks",
            "Model\nnet",
        ]
        values = [
            integer(
                delete_ceiling["event_edit_repaired_sections"], "delete ceiling repairs"
            ),
            integer(
                combined_summary["event_edit_repaired_sections"],
                "combined event-edit repairs",
            ),
            integer(delete_model["repaired_sections"], "delete repairs"),
            integer(delete_model["broken_sections"], "delete breaks"),
            integer(delete_model["net_sections"], "delete net"),
        ]
        colours = [PURPLE, PURPLE, BLUE, ORANGE, BLUE]
        hatches = ["///", "///", "", "", ""]
        bars = axis.bar(labels, values, color=colours, hatch=hatches, width=0.58)
        axis.axhline(0, color=INK, linewidth=0.8)
        axis.set_title(f"Ceiling versus learned delete · ±{tolerance}", loc="left")
        axis.set_ylabel("Sections")
        axis.set_ylim(min(values) - 80, max(values) * 1.18)
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + (10 if value >= 0 else -15),
                f"{value:+,}" if value < 0 else f"{value:,}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9.5,
                weight="bold",
            )
        axis.text(
            0.03,
            0.96,
            "//// label-guided ceiling",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=PURPLE,
            fontsize=9.2,
        )
        axis.text(
            0.03,
            0.91,
            "solid = label-free model",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=BLUE,
            fontsize=9.2,
        )
        style_axes(axis)
    add_footer(
        figure,
        "Labels expose substantial recoverable room, but the learned delete chooser loses more sections than it repairs.",
        "Sources: combined_best_case.json; delete_model_development.json. Ceilings use A-D labels; the learned delete row uses saved label-free predictions scored on A-D.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.18)
    save_figure(figure, paths, "07_combined_ceiling_vs_delete_model")


def plot_keep_review_curve(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot precision against accepted coverage for the keep/review model."""
    keep = data["keep_review"]
    figure, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE, sharey=True)
    add_header(
        figure,
        "08 — Can keep-or-review reach 90% precision at useful coverage?",
        "32-video development set · target: 90% of accepted sections correct while accepting at least 10% of all sections",
    )
    for axis, tolerance in zip(axes, TOLERANCES, strict=True):
        curve = list_value(
            keep["curve_at_5_frames"] if tolerance == 5 else keep["curve_at_10_frames"],
            "keep curve",
        )
        points = [
            mapping(row, "keep curve row")
            for row in curve
            if mapping(row, "keep curve row").get("precision") is not None
        ]
        coverage = [real(row["coverage"], "coverage") for row in points]
        precision = [real(row["precision"], "precision") for row in points]
        thresholds = [real(row["threshold"], "threshold") for row in points]
        axis.plot(
            coverage, precision, color=BLUE, linewidth=2, marker="o", markersize=5
        )
        axis.axvline(0.1, color=ORANGE, linestyle="--", linewidth=1.4)
        axis.axhline(0.9, color=ORANGE, linestyle="--", linewidth=1.4)
        threshold_index = next(
            index for index, threshold in enumerate(thresholds) if threshold == 0.6
        )
        axis.scatter(
            [coverage[threshold_index]],
            [precision[threshold_index]],
            s=95,
            color=PURPLE,
            zorder=4,
        )
        axis.annotate(
            "threshold 0.60",
            (coverage[threshold_index], precision[threshold_index]),
            xytext=(8, -17),
            textcoords="offset points",
            fontsize=9,
            color=PURPLE,
            weight="bold",
        )
        axis.text(
            0.1,
            0.91,
            "90% / 10% target",
            color=ORANGE,
            fontsize=9,
            ha="left",
            va="bottom",
        )
        axis.set_xlim(-0.01, max(coverage) * 1.04)
        axis.set_ylim(0, 1.03)
        axis.set_xlabel("Share of sections accepted (coverage)")
        axis.set_ylabel("Accepted sections correct (precision)")
        axis.set_title(f"Keep / review curve · ±{tolerance}", loc="left")
        axis.xaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
        axis.yaxis.set_major_formatter(lambda value, _position: f"{value * 100:.0f}%")
        style_axes(axis)
    add_footer(
        figure,
        "The best useful slice remains far below the 90% precision target at both tolerances; stop this line.",
        "Source: keep_review_development.json. Null precision points are omitted, as required; the purple point marks threshold 0.60.",
    )
    figure.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.2, wspace=0.18)
    save_figure(figure, paths, "08_keep_review_curve")


def plot_useful_leads(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Plot the final action recommendations as cards with an optional court motif."""
    side = data["side_audit"]
    setting = data["setting_sweep"]
    start = data["start_model_validation"]
    delete = data["delete_model"]
    keep = data["keep_review"]
    duplicate = data["duplicate_audit"]
    keep_five = next(
        row
        for row in list_value(keep["curve_at_5_frames"], "keep five curve")
        if real(mapping(row, "keep five row")["threshold"], "threshold") == 0.6
    )
    figure = plt.figure(figsize=FIGURE_SIZE)
    add_header(
        figure,
        "09 — What should the next release actually use?",
        "Decision guide from the saved follow-up evidence · ±5 remains primary; ±10 is a conditional release metric",
    )
    cards = [
        (
            "USE SIDE VOTE",
            f"+{integer(mapping(tolerance_record(side, 5, 'side')['simple_vote'], 'side vote')['net_sections'], 'side net'):,} strict sections at ±5",
            "Large frozen-test gain with no breaks.",
            BLUE,
            PALE_BLUE,
        ),
        (
            "PARKED ±10 WHAT-IF",
            f"0.85 / 6 → +{integer(mapping(tolerance_record(setting, 10, 'setting')['global_descriptive_best'], 'setting best')['net_sections'], 'setting net ten'):,} at ±10",
            "Timing only. Saved decision: stop. Re-run if ±10 becomes the target.",
            ORANGE,
            PALE_ORANGE,
        ),
        (
            "PARK FIRST-CONTACT LEAD",
            f"+{integer(tolerance_record(start, 5, 'held-out V')['net_sections'], 'V net'):,} on untouched V",
            "Small lead; preserve it for better contact evidence.",
            PURPLE,
            PALE_PURPLE,
        ),
        (
            "STOP DELETE / REVIEW",
            f"Delete {integer(delete['descriptive_best']['net_sections'], 'delete net')} net · review {fraction_text(real(mapping(keep_five, 'keep five')['precision'], 'review precision'))}",
            "Both are 32-video development findings.",
            GREY,
            PALE_GREY,
        ),
        (
            "STOP DUPLICATE CLEANUP",
            f"{integer(mapping(duplicate['development'], 'duplicate development')['pair_count'], 'development duplicate pairs')} dev · {integer(mapping(duplicate['frozen_test'], 'duplicate frozen test')['pair_count'], 'test duplicate pairs')} test pairs",
            "No adjacent opposite-side pairs exist in either saved set.",
            GREY,
            PALE_GREY,
        ),
    ]
    axis = figure.add_axes((0.055, 0.18, 0.67, 0.66))
    axis.set_xlim(0, 2.1)
    axis.set_ylim(0, 2.9)
    axis.axis("off")
    positions = [
        (0.04, 1.95, 0.94),
        (1.08, 1.95, 0.94),
        (0.04, 1.05, 0.94),
        (1.08, 1.05, 0.94),
        (0.48, 0.15, 1.1),
    ]
    for card, (x, y, width) in zip(cards, positions, strict=True):
        draw_metric_card(axis, x, y, width, 0.75, *card)
    if paths.motif.exists():
        motif = mpimg.imread(paths.motif)
        motif_axis = figure.add_axes((0.76, 0.28, 0.2, 0.42))
        motif_axis.imshow(motif)
        motif_axis.set_title("A simple rally motif", fontsize=11, color=GREY, pad=8)
        motif_axis.axis("off")
    else:
        figure.text(
            0.86,
            0.45,
            "Court motif\nnot available",
            ha="center",
            va="center",
            fontsize=11,
            color=GREY,
        )
    add_footer(
        figure,
        "Ship the side vote, keep the small first-contact result as a lead, and stop the other saved-input rules.",
        "Sources: side_audit.json; setting_sweep.json; start_model_validation.json; delete_model_development.json; keep_review_development.json; opposite_side_duplicate_audit.json. Supporting illustration: scorecard_court_motif.png.",
    )
    save_figure(figure, paths, "09_useful_leads")


def main() -> None:
    """Load saved evidence and generate the complete nine-image visual series."""
    configure_style()
    paths = build_paths()
    data = {
        "baseline": load_json(paths.baseline),
        "side_audit": load_json(paths.side_audit),
        "setting_sweep": load_json(paths.setting_sweep),
        "start_model_development": load_json(paths.start_model_development),
        "start_model_validation": load_json(paths.start_model_validation),
        "combined_best_case": load_json(paths.combined_best_case),
        "delete_model": load_json(paths.delete_model),
        "keep_review": load_json(paths.keep_review),
        "duplicate_audit": load_json(paths.duplicate_audit),
        "shuttleset_result": load_json(paths.shuttleset_result),
    }
    plot_story_scorecard(data, paths)
    plot_baseline_funnel_and_error_map(data, paths)
    plot_side_vote_gain_and_tradeoff(data, paths)
    plot_repairs_breaks_net(data, paths)
    plot_setting_sweep(data, paths)
    plot_first_contact_ceiling_vs_achieved(data, paths)
    plot_combined_ceiling_vs_delete_model(data, paths)
    plot_keep_review_curve(data, paths)
    plot_useful_leads(data, paths)


if __name__ == "__main__":
    main()
