"""Build the four plots used in the contact follow-up report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

plt.switch_backend("Agg")


JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
FOLLOWUP_DIR = REPO_ROOT / "scratch/contact_det_followup"
FIGURE_DIR = FOLLOWUP_DIR / "figures"

INK = "#17212B"
BLUE = "#0072B2"
ORANGE = "#D97706"
PURPLE = "#7A3E9D"
GREY = "#667085"
LIGHT_GREY = "#D0D5DD"
PALE_BLUE = "#E7F1F8"

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
    """Compare the frozen-test baseline with the rally-wide side vote."""
    side_audit = data["side_audit"]
    by_tolerance = object_value(
        side_audit["results_by_tolerance_at_30_fps"], "side-audit tolerances"
    )

    labels = ["±5 frames\n(main result)", "±10 frames"]
    baseline_counts: list[int] = []
    revised_counts: list[int] = []
    baseline_rates: list[float] = []
    revised_rates: list[float] = []

    for tolerance in (5, 10):
        result = object_value(by_tolerance[str(tolerance)], f"±{tolerance} side audit")
        vote = object_value(result["simple_vote"], f"±{tolerance} simple vote")
        baseline_counts.append(integer(vote["baseline_strict_fully_correct"], "baseline count"))
        revised_counts.append(integer(vote["revised_strict_fully_correct"], "revised count"))
        baseline_rates.append(number(vote["baseline_full_output_precision"], "baseline precision"))
        revised_rates.append(number(vote["revised_full_output_precision"], "revised precision"))

    figure, axis = plt.subplots(figsize=(10, 6.4))
    add_title(
        figure,
        "One rule gave a large, clean gain",
        "Frozen 47-video test · 3,982 predicted sections · fully correct timing and player side",
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

    figure.tight_layout(rect=(0.06, 0.07, 0.98, 0.88))
    save_figure(figure, paths, "01_complete_rallies", "Source: results/side_audit.json")


def plot_baseline_errors(paths: DataPaths) -> None:
    """Show the main error groups among sections mapped to one rally."""
    labels = list(BASELINE_ERROR_COUNTS)
    values = list(BASELINE_ERROR_COUNTS.values())
    colours = [BLUE, PURPLE, GREY, GREY, GREY, GREY]

    figure, axis = plt.subplots(figsize=(10, 6.4))
    add_title(
        figure,
        "Most failed rallies are missing a contact",
        f"Baseline · {BASELINE_MAPPED_SECTIONS:,} sections mapped to one labelled rally · ±5 frames",
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


def plot_candidates_and_choosers(data: dict[str, JsonObject], paths: DataPaths) -> None:
    """Compare label-guided room with the gains from label-free models."""
    start_best_case = data["start_best_case"]
    start_model = data["start_model_development"]
    start_validation = data["start_model_validation"]
    combined_best_case = data["combined_best_case"]
    delete_model = data["delete_model"]

    timing_then_side = object_value(
        start_best_case["timing_then_rally_side"], "first-contact best case"
    )
    chosen = object_value(start_model["chosen"], "chosen first-contact model")
    nested = object_value(start_model["nested_held_out_estimate"], "nested first-contact result")
    validation = object_value(
        object_value(start_validation["by_tolerance_at_30_fps"], "validation tolerances")["5"],
        "±5 first-contact validation",
    )
    combined = object_value(combined_best_case["combined"], "combined best case")
    learned_delete = object_value(delete_model["descriptive_best"], "learned delete result")

    first_labels = [
        "Best case\n(labels choose)",
        "Pooled model\n(A–D)",
        "Nested hold-out\n(A–D)",
        "Final check\n(V: 8 videos)",
    ]
    first_values = [
        integer(timing_then_side["repaired_sections"], "first-contact best-case repairs"),
        integer(chosen["repaired_sections"], "pooled first-contact repairs"),
        integer(nested["repaired_sections"], "nested first-contact repairs"),
        integer(validation["repaired_sections"], "validation first-contact repairs"),
    ]
    whole_labels = ["Combined best case\n(labels choose)", "Learned delete\n(net effect)"]
    whole_values = [
        integer(combined["repaired_sections"], "combined best-case repairs"),
        integer(learned_delete["net_sections"], "learned delete net effect"),
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [1.45, 1]})
    add_title(
        figure,
        "Better answers existed; the models could not choose them safely",
        "Development results · ±5 frames · orange bars use labels to choose each rally's best edit",
    )

    first_axis, whole_axis = axes
    first_bars = first_axis.bar(
        range(len(first_values)), first_values, color=[ORANGE, BLUE, PURPLE, BLUE]
    )
    first_axis.set_title("First-contact repairs")
    first_axis.set_xticks(range(len(first_labels)), first_labels)
    first_axis.set_ylabel("Fully correct sections gained")
    first_axis.set_ylim(0, 340)
    style_axes(first_axis, grid_axis="y")
    for bar, value in zip(first_bars, first_values, strict=True):
        first_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 8,
            str(value),
            ha="center",
            va="bottom",
            weight="bold",
            color=INK,
        )

    whole_bars = whole_axis.bar(range(len(whole_values)), whole_values, color=[ORANGE, GREY])
    whole_axis.axhline(0, color=INK, linewidth=0.8)
    whole_axis.set_title("Whole-rally edit result")
    whole_axis.set_xticks(range(len(whole_labels)), whole_labels)
    whole_axis.set_ylim(-90, 530)
    style_axes(whole_axis, grid_axis="y")
    for bar, value in zip(whole_bars, whole_values, strict=True):
        vertical_alignment = "bottom" if value >= 0 else "top"
        offset = 12 if value >= 0 else -12
        whole_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            str(value),
            ha="center",
            va=vertical_alignment,
            weight="bold",
            color=INK,
        )
    whole_axis.text(
        1,
        70,
        "Delete model\n42 repairs, 88 breaks",
        ha="center",
        color=GREY,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0.05, 0.07, 0.98, 0.88), w_pad=3.0)
    save_figure(
        figure,
        paths,
        "03_candidates_and_choosers",
        "Sources: results/start_best_case.json; start_model_development.json; "
        "start_model_validation.json; combined_best_case.json; delete_model_development.json",
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
        "Rejecting more rallies did not produce a trustworthy subset",
        "Group-held-out A–D predictions · 2,850 sections · target: 90% precision at 10% coverage",
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
        "start_best_case": load_json(paths.start_best_case),
        "start_model_development": load_json(paths.start_model_development),
        "start_model_validation": load_json(paths.start_model_validation),
        "combined_best_case": load_json(paths.combined_best_case),
        "delete_model": load_json(paths.delete_model),
        "keep_review": load_json(paths.keep_review),
    }
    plot_complete_rallies(data, paths)
    plot_baseline_errors(paths)
    plot_candidates_and_choosers(data, paths)
    plot_keep_review_curve(data, paths)


if __name__ == "__main__":
    main()
