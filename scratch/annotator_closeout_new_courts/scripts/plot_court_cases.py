"""Redraw the two earlier court regression cases with the new detector's outlines.

The frames and old outlines come from the earlier evaluation's saved evidence;
the new outlines come from the court release used by the selected model.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

ROOT = Path(__file__).resolve().parents[1]
EARLIER = ROOT.parent / "annotator_wrapup_evaluation"
BLUE = "#0072B2"
ORANGE = "#C65D00"
PURPLE = "#7651A8"
GREEN = "#009E73"
NUMBER_KEY = "Where each corner belongs: 0 top-left · 1 top-right · 2 bottom-right · 3 bottom-left"


def draw_outline(axis: Axes, pixels: np.ndarray, corners: np.ndarray, colours: list[str], title: str) -> None:
    axis.imshow(pixels)
    polygon = np.vstack([corners, corners[0]])
    axis.plot(polygon[:, 0], polygon[:, 1], color=colours[0], linewidth=2.5)
    for number, (position, colour) in enumerate(zip(corners, colours, strict=True)):
        axis.text(*position, str(number), ha="center", va="center", fontsize=15, weight="bold",
                  color="white", bbox={"boxstyle": "circle,pad=0.22", "facecolor": colour,
                                       "edgecolor": "white", "linewidth": 1.4}, zorder=5)
    axis.set_title(title, fontsize=16, loc="left", pad=14)
    axis.set(xlim=(0, 1920), ylim=(1140, 0), xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    axis.set_xticks([0, 640, 1280, 1920])
    axis.set_yticks([0, 360, 720, 1080])
    axis.tick_params(labelsize=11)


def new_scene(video: int, frame: int) -> dict:
    with gzip.open(ROOT / f"raw/run/output/shared/court/test/{video}/court_evidence.json.gz", "rt") as source:
        records = json.load(source)["scene_records"]
    matches = [record for record in records if record["start_frame"] <= frame < record["end_frame"]]
    assert len(matches) == 1 and matches[0]["scene_valid"]
    return matches[0]


def finish(figure: plt.Figure, name: str) -> None:
    figure.subplots_adjust(top=0.77, bottom=0.20, left=0.07, right=0.97, wspace=0.22)
    figure.savefig(ROOT / f"figures/{name}.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run() -> None:
    with gzip.open(EARLIER / "results/visual_geometry.json.gz", "rt") as source:
        geometry = {record["sample_id"]: record for record in json.load(source)}

    scene = new_scene(53, 83084)
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    pixels = plt.imread(EARLIER / "raw/control_sheets/V14_centre.jpg")
    before = np.asarray(geometry["V14"]["scene"]["raw_corners_px"])
    draw_outline(axes[0], pixels, before, [BLUE, ORANGE, BLUE, BLUE], "Old court: OpenCV moved corner 1")
    draw_outline(axes[1], pixels, np.asarray(scene["corners_native_px"]), [GREEN] * 4, "New court detector")
    figure.suptitle("Video 53: the new court follows the lines and the scene is accepted",
                    fontsize=20, weight="bold", y=0.98)
    figure.text(0.07, 0.88, NUMBER_KEY + "\nOrange: old OpenCV replacement · Green: new detector", fontsize=13)
    figure.text(0.07, 0.085, f"Same image: frame 83,084 · Two-player check passes in "
                f"{100 * scene['exactly_two_fraction']:.1f}% of the scene's frames (50% needed).", fontsize=13)
    figure.text(0.07, 0.035, "Labelled hits in this scene matched with the right player: 0 of 12 with the old court, "
                "11 of 12 with the new court.", fontsize=12)
    finish(figure, "video53_court_fixed")

    scene = new_scene(17, 47276)
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2))
    pixels = plt.imread(EARLIER / "raw/control_sheets/V04_centre.jpg")
    before = np.asarray(geometry["V04"]["scene"]["active_corners_native_px"])
    draw_outline(axes[0], pixels, before, [PURPLE] * 4, "Old court: shared outline is too small")
    draw_outline(axes[1], pixels, np.asarray(scene["corners_native_px"]), [GREEN] * 4, "New court detector")
    for axis in axes:
        axis.annotate("Visible far player", (1117, 190), xytext=(80, 140), fontsize=12, color="white",
                      bbox={"facecolor": "#202020", "alpha": 0.9, "pad": 5},
                      arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.5})
    figure.suptitle("Video 17: the new court keeps the far player", fontsize=20, weight="bold", y=0.98)
    figure.text(0.07, 0.88, NUMBER_KEY + "\nPurple: old shared outline used by the player picker · Green: new detector",
                fontsize=13)
    figure.text(0.07, 0.085, "Same image: frame 47,276 · The new run picks both players here and at frame 46,045, "
                "where the old run lost the far player.", fontsize=12)
    figure.text(0.07, 0.035, "The hit at 47,276 now matches with the right player. The hit at 46,045 matches but is "
                "assigned to the near player.", fontsize=12)
    finish(figure, "video17_court_fixed")


if __name__ == "__main__":
    run()
