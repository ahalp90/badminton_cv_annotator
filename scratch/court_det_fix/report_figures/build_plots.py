"""Build the two court-detector handover figures from the saved, gzipped measurement files.

Run from the repository root:
    MPLBACKEND=Agg ~/.venvs/badminton-cicd/bin/python \
        scratch/court_det_fix/report_figures/build_plots.py
"""

import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "docs" / "court_detector" / "data"
OUTPUT_DIR = Path(__file__).resolve().parent

# Okabe-Ito colours: blue, amber and grey stay distinct under protan colour vision.
DETECTOR_COLOUR = "#0072B2"
PEOPLE_COLOUR = "#E69F00"
SCENE_COLOUR = "#999999"

EXPECTED_SCENE_COUNT = 9
TOTAL_TOLERANCE_SECONDS = 0.001


def load_gzipped_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as json_file:
        return json.load(json_file)


def save_figure(figure: plt.Figure, stem: str) -> None:
    for suffix in ("png", "svg"):
        output = OUTPUT_DIR / f"{stem}.{suffix}"
        figure.savefig(output, dpi=200, bbox_inches="tight")
        if suffix == "svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(figure)


def build_runtime_plot() -> None:
    backend_files = [
        ("CPU templates", "interval_cpu_timing.json.gz"),
        ("GPU templates", "interval_cuda_timing.json.gz"),
    ]
    arm_labels: list[str] = []
    detector_seconds: list[float] = []
    people_seconds: list[float] = []
    scene_seconds: list[float] = []
    total_seconds: list[float] = []
    for arm_label, file_name in backend_files:
        timing = load_gzipped_json(DATA_DIR / file_name)
        assert timing["scene_count"] == 33, f"{file_name}: expected 33 scenes, got {timing['scene_count']}"
        people = timing["people_setup_seconds"] + timing["people_input_seconds"]
        component_sum = timing["court_detector_seconds"] + people + timing["scene_seconds"]
        assert abs(component_sum - timing["total_seconds"]) < TOTAL_TOLERANCE_SECONDS, (
            f"{file_name}: components sum to {component_sum:.3f} s, saved total is {timing['total_seconds']:.3f} s"
        )
        arm_labels.append(arm_label)
        detector_seconds.append(timing["court_detector_seconds"])
        people_seconds.append(people)
        scene_seconds.append(timing["scene_seconds"])
        total_seconds.append(timing["total_seconds"])

    figure, axes = plt.subplots(figsize=(10, 4), layout="constrained")
    bar_positions = list(range(len(arm_labels)))
    segments = [
        ("Court (incl. line detection)", detector_seconds, DETECTOR_COLOUR, "white"),
        ("People (pose setup + calls)", people_seconds, PEOPLE_COLOUR, "black"),
        ("Scene detection", scene_seconds, SCENE_COLOUR, "black"),
    ]
    left_edges = [0.0] * len(arm_labels)
    for segment_label, segment_seconds, colour, text_colour in segments:
        bars = axes.barh(
            bar_positions, segment_seconds, left=left_edges, height=0.55, color=colour, label=segment_label
        )
        axes.bar_label(bars, labels=[f"{seconds:.1f}" for seconds in segment_seconds], label_type="center",
                       color=text_colour, fontsize=9)
        left_edges = [left + seconds for left, seconds in zip(left_edges, segment_seconds)]

    for position, total in zip(bar_positions, total_seconds):
        axes.text(total + 3, position, f"{total:.1f} s total", va="center", fontsize=10, fontweight="bold")

    axes.set_yticks(bar_positions, arm_labels, fontsize=11)
    axes.invert_yaxis()
    axes.set_xlim(0, max(total_seconds) * 1.22)
    axes.set_xlabel("Whole-run wall time (s)")
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)
    figure.suptitle("Whole-run time by template-scoring backend", x=0.01, ha="left", fontsize=13)
    axes.set_title(
        "Single warm-cache runs, same 300 s source interval (33 scenes), NVIDIA L40, at most 8 CPU cores.\n"
        "Both arms use GPU neural inference. Runs predate three-frame composition and video pooling;"
        " not a final-speed claim.",
        loc="left", fontsize=9, color="#444444",
    )
    save_figure(figure, "runtime")


def build_consistency_plot() -> None:
    scatter = load_gzipped_json(DATA_DIR / "composite_scatter_video040.json.gz")
    paint_choice = scatter["summaries"]["three_originals_present"]["paint_choice"]
    composite = paint_choice["composite_on_same_scenes"]
    assert len(paint_choice["scene_ids"]) == EXPECTED_SCENE_COUNT, paint_choice["scene_ids"]
    assert paint_choice["scenes"] == EXPECTED_SCENE_COUNT and composite["scenes"] == EXPECTED_SCENE_COUNT

    arm_labels = ["Single original frame\n(paint choice)", "Three-frame composite"]
    rms_values = [paint_choice["rms_px"], composite["rms_px"]]

    figure, axes = plt.subplots(figsize=(10, 3.6), layout="constrained")
    bar_positions = list(range(len(arm_labels)))
    bars = axes.barh(bar_positions, rms_values, height=0.55, color=[PEOPLE_COLOUR, DETECTOR_COLOUR])
    axes.bar_label(bars, labels=[f"{rms:.2f} px" for rms in rms_values], padding=6, fontsize=11,
                   fontweight="bold")

    axes.set_yticks(bar_positions, arm_labels, fontsize=11)
    axes.invert_yaxis()
    axes.set_xlim(0, max(rms_values) * 1.25)
    axes.set_xlabel("RMS corner distance from the across-scene median (native 1920 × 1080 px)")
    axes.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Court-corner scatter across nine registered scenes (lower RMS = more consistent)",
                    x=0.01, ha="left", fontsize=13)
    axes.set_title(
        "Video 040, same nine scenes for both bars, after image alignment.\n"
        "Scatter is not accuracy: a consistent but biased court can also have low scatter.",
        loc="left", fontsize=9, color="#444444",
    )
    save_figure(figure, "consistency")


if __name__ == "__main__":
    build_runtime_plot()
    build_consistency_plot()
