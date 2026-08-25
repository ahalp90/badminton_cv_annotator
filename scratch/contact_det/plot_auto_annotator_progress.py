"""Plot the retained rally-boundary and contact-detector progress evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from annotator.calibration.scoring import (
    RallyBoundary,
    classify_all,
    load_gt_rallies,
    merged_span_indices,
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
GREY = "#6F6F6F"
BLACK = "#202020"
BUFFER_SECONDS = (1, 2, 3, 5)


class RallySummary(NamedTuple):
    """Pooled rally-boundary counts and one-to-one buffer scores."""

    ground_truth_rallies: int
    predicted_spans: int
    covered: int
    split: int
    missed: int
    one_to_one: int
    merged_spans: int
    spurious_spans: int
    fixtures: dict[str, dict[str, int]]
    buffer_scores: dict[str, dict[str, float | int]]


def _read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=BLACK)


def _label_bars(axis: plt.Axes, bars: Any) -> None:
    labels = [f"{bar.get_height():.1f}" for bar in bars]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8, color=BLACK)


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _rally_summary(evidence: Mapping[str, Any], shots_master: pd.DataFrame) -> RallySummary:
    counts = {
        "ground_truth_rallies": 0,
        "predicted_spans": 0,
        "covered": 0,
        "split": 0,
        "missed": 0,
        "one_to_one": 0,
        "merged_spans": 0,
        "spurious_spans": 0,
    }
    edge_padding_seconds: list[tuple[float, float]] = []
    fixture_summaries: dict[str, dict[str, int]] = {}

    for fixture in evidence["fixtures"]:
        fixture_name = str(fixture["fixture"])
        video_id = int(fixture["video_id"])
        fps = float(fixture["fps"])
        spans = [
            (int(span["start_frame"]), int(span["end_frame"]))
            for span in fixture["spans"]
        ]
        rallies = load_gt_rallies(shots_master, video_id)
        classifications = classify_all(spans, rallies)
        gt_frames = np.asarray(
            [frame for rally in rallies for frame in rally.stroke_frames],
            dtype=np.int64,
        )
        rally_ids_by_span: list[set[int]] = []
        fixture_spurious = 0
        for start, end in spans:
            contained_rallies = {
                rally_index
                for rally_index, rally in enumerate(rallies)
                if any(start <= frame < end for frame in rally.stroke_frames)
            }
            rally_ids_by_span.append(contained_rallies)
            fixture_spurious += int(not np.any((start <= gt_frames) & (gt_frames < end)))

        fixture_covered = sum(
            category is RallyBoundary.COVERED for category, _span_id in classifications
        )
        fixture_split = sum(
            category is RallyBoundary.SPLIT for category, _span_id in classifications
        )
        fixture_missed = sum(
            category is RallyBoundary.MISSED for category, _span_id in classifications
        )
        fixture_merged = len(merged_span_indices(spans, rallies))
        fixture_one_to_one = 0
        counts["ground_truth_rallies"] += len(rallies)
        counts["predicted_spans"] += len(spans)
        counts["covered"] += fixture_covered
        counts["split"] += fixture_split
        counts["missed"] += fixture_missed
        counts["merged_spans"] += fixture_merged
        counts["spurious_spans"] += fixture_spurious

        for rally_index, (rally, (category, span_id)) in enumerate(
            zip(rallies, classifications, strict=True)
        ):
            if (
                category is not RallyBoundary.COVERED
                or span_id is None
                or rally_ids_by_span[span_id] != {rally_index}
            ):
                continue
            counts["one_to_one"] += 1
            fixture_one_to_one += 1
            start, end = spans[span_id]
            first, last = rally.extent
            edge_padding_seconds.append(((first - start) / fps, (end - last) / fps))
        fixture_summaries[fixture_name] = {
            "ground_truth_rallies": len(rallies),
            "predicted_spans": len(spans),
            "covered": fixture_covered,
            "split": fixture_split,
            "missed": fixture_missed,
            "one_to_one": fixture_one_to_one,
            "merged_spans": fixture_merged,
            "spurious_spans": fixture_spurious,
        }

    buffer_scores: dict[str, dict[str, float | int]] = {}
    for label, limit in [(str(seconds), float(seconds)) for seconds in BUFFER_SECONDS] + [("any", np.inf)]:
        matched = sum(
            start_padding <= limit and end_padding <= limit
            for start_padding, end_padding in edge_padding_seconds
        )
        precision = matched / counts["predicted_spans"]
        recall = matched / counts["ground_truth_rallies"]
        buffer_scores[label] = {
            "matched": matched,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }

    return RallySummary(
        counts["ground_truth_rallies"],
        counts["predicted_spans"],
        counts["covered"],
        counts["split"],
        counts["missed"],
        counts["one_to_one"],
        counts["merged_spans"],
        counts["spurious_spans"],
        fixture_summaries,
        buffer_scores,
    )


def _plot_rally_segments(summary: RallySummary, output: Path) -> None:
    labels = [str(seconds) for seconds in BUFFER_SECONDS] + ["No limit"]
    keys = [str(seconds) for seconds in BUFFER_SECONDS] + ["any"]
    series = (
        ("Precision", "precision", BLUE),
        ("Recall", "recall", ORANGE),
        ("F1", "f1", PURPLE),
    )

    figure, axis = plt.subplots(figsize=(8.8, 4.8), layout="constrained")
    positions = np.arange(len(labels))
    for name, metric, colour in series:
        values = [100.0 * float(summary.buffer_scores[key][metric]) for key in keys]
        axis.plot(positions, values, marker="o", linewidth=2.2, label=name, color=colour)
    axis.set_title(
        "Current rally spans: one-to-one segment quality by allowed edge buffer\n"
        "292 ShuttleSet rallies in sset_01, sset_15 and sset_21",
        fontsize=12,
        weight="bold",
    )
    axis.set_xlabel("Maximum span padding before the first and after the last contact (seconds)")
    axis.set_ylabel("One-to-one rally segment score (%)")
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 100)
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def _plot_contact_stack(contact_score: Mapping[str, Any], tree_results: Mapping[str, Any], output: Path) -> None:
    current = contact_score["overall"]["filtered"]["10"]
    raw = contact_score["overall"]["raw"]["10"]
    region = tree_results["region_ceiling"]["seeded_union"]["operational"]["10"]
    hgb = tree_results["models"]["histogram_boosting"]["physics"]["metrics"]["10"]

    if (
        int(current["non_serve"]["total"]) != 2836
        or int(current["serve"]["total"]) != 292
        or int(hgb["ground_truth"]) != 3128
    ):
        raise ValueError("retained contact denominators differ from the three-fixture contract")

    recall_rows = (
        (
            "Raw\nproposals",
            float(raw["non_serve"]["recall"]),
            float(raw["serve"]["recall"]),
        ),
        (
            "Final\nheuristics",
            float(current["non_serve"]["recall"]),
            float(current["serve"]["recall"]),
        ),
        (
            "Region v2\nsearch surface",
            float(region["nonserve_recall"]),
            float(region["serve_recall"]),
        ),
        (
            "HGB physical\nevents",
            float(hgb["nonserve_recall"]),
            float(hgb["serve_recall"]),
        ),
    )
    current_precision = float(current["overall"]["physical_precision"])
    current_recall = float(current["overall"]["recall"])
    event_rows = (
        (
            "Final heuristics",
            current_precision,
            current_recall,
            _f1(current_precision, current_recall),
        ),
        (
            "HGB physical",
            float(hgb["precision"]),
            float(hgb["recall"]),
            float(hgb["f1"]),
        ),
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.9), layout="constrained")
    positions = np.arange(len(recall_rows))
    width = 0.34
    nonserve_bars = axes[0].bar(
        positions - width / 2,
        [100.0 * row[1] for row in recall_rows],
        width,
        label="Non-serve",
        color=BLUE,
    )
    serve_bars = axes[0].bar(
        positions + width / 2,
        [100.0 * row[2] for row in recall_rows],
        width,
        label="Serve",
        color=ORANGE,
    )
    _label_bars(axes[0], nonserve_bars)
    _label_bars(axes[0], serve_bars)
    axes[0].set_title("Search coverage and event recall", fontsize=11, weight="bold")
    axes[0].set_ylabel("Contacts covered or matched at ±10 base-30 frames (%)")
    axes[0].set_xticks(positions, [row[0] for row in recall_rows])
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, loc="lower left")
    _style_axis(axes[0])

    event_positions = np.arange(len(event_rows))
    metrics = (("Precision", 1, BLUE), ("Recall", 2, ORANGE), ("F1", 3, PURPLE))
    event_width = 0.24
    for metric_index, (name, value_index, colour) in enumerate(metrics):
        bars = axes[1].bar(
            event_positions + (metric_index - 1) * event_width,
            [100.0 * row[value_index] for row in event_rows],
            event_width,
            label=name,
            color=colour,
        )
        _label_bars(axes[1], bars)
    axes[1].set_title("HGB on Region v2 improves event quality", fontsize=11, weight="bold")
    axes[1].set_ylabel("One-to-one event score at ±10 base-30 frames (%)")
    axes[1].set_xticks(event_positions, [row[0] for row in event_rows])
    axes[1].set_ylim(0, 105)
    axes[1].legend(frameon=False, loc="lower right")
    _style_axis(axes[1])

    figure.suptitle(
        "Contact progress on 3,128 labelled contacts across three ShuttleSet fixtures",
        fontsize=13,
        weight="bold",
    )
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def _summary_payload(summary: RallySummary) -> dict[str, object]:
    return {
        "ground_truth_rallies": summary.ground_truth_rallies,
        "predicted_spans": summary.predicted_spans,
        "covered": summary.covered,
        "split": summary.split,
        "missed": summary.missed,
        "one_to_one": summary.one_to_one,
        "merged_spans": summary.merged_spans,
        "spurious_spans": summary.spurious_spans,
        "fixtures": summary.fixtures,
        "buffer_scores": summary.buffer_scores,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    arguments = parser.parse_args(argv)

    evidence = _read_json_gz(arguments.data_root / "contact_evidence.json.gz")
    contact_score = _read_json_gz(arguments.data_root / "contact_evidence_score.json.gz")
    tree_results = _read_json_gz(arguments.data_root / "region_v2" / "tree_contact_results.json.gz")
    shots_master = pd.read_csv(REPO_ROOT / "training/data/shuttleset/annotations/shots_master.csv")
    summary = _rally_summary(evidence, shots_master)

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    _plot_rally_segments(summary, arguments.output_dir / "auto_annotator_rally_segments.png")
    _plot_contact_stack(contact_score, tree_results, arguments.output_dir / "auto_annotator_contact_stack.png")
    print(json.dumps(_summary_payload(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
