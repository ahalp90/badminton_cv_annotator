"""Reproduce the recorded CourtKeyNet cross-scene consensus ship check."""

# The standalone script adds the repository's src directory before importing it.
# ruff: noqa: E402, I001

import csv
import gzip
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from courtkeynet.court_corners import CONSENSUS_FLAG_THRESHOLD_PX, consensus_repair


INPUT_DIR = Path(__file__).resolve().parent / "recorded_inputs"
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 720
CORNER_NAMES = ("tl", "tr", "br", "bl")

EVIDENCE = {
    "vid 3 good-scene max dist": (18.1, 1),
    "vid 3 bad-scene min dist": (177.3, 1),
    "vid 21 max dist (control)": (8.2, 1),
    "vid 3 consensus mean vs GT": (4.62, 2),
    "vid 21 consensus mean vs GT": (4.64, 2),
}


def load_ground_truth(video_id: int) -> np.ndarray:
    """Load one ShuttleSet reference quad in TL, TR, BR, BL order."""
    homography_path = REPO_ROOT / "data/shuttleset/set/homography.csv"
    with homography_path.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["id"]) != video_id:
                continue
            return np.array(
                [
                    [float(row["upleft_x"]), float(row["upleft_y"])],
                    [float(row["upright_x"]), float(row["upright_y"])],
                    [float(row["downright_x"]), float(row["downright_y"])],
                    [float(row["downleft_x"]), float(row["downleft_y"])],
                ],
                dtype=np.float64,
            )
    raise ValueError(f"No homography row found for video {video_id}")


def load_recorded_quads(video_id: int) -> tuple[list[int], np.ndarray]:
    """Load fallback scene starts and quads from the recorded compressed CSV."""
    csv_path = INPUT_DIR / f"fb5_{video_id}.csv.gz"
    scene_starts = []
    quads = []
    with gzip.open(csv_path, mode="rt", newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if row["source"] != "fallback":
                continue
            scene_starts.append(int(row["scene_start"]))
            quads.append(
                [
                    [float(row[f"{corner_name}_x"]), float(row[f"{corner_name}_y"])]
                    for corner_name in CORNER_NAMES
                ]
            )
    return scene_starts, np.asarray(quads, dtype=np.float64)


def per_scene_error(quads: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Return mean corner error for each 1280x720 scene quad."""
    return np.linalg.norm(quads - ground_truth, axis=2).mean(axis=1)


def check_evidence(label: str, observed: float) -> None:
    """Raise when a rounded recorded measurement diverges."""
    expected, decimals = EVIDENCE[label]
    rounded = round(observed, decimals)
    if rounded != expected:
        raise AssertionError(f"{label}: observed {rounded}, expected {expected}")
    print(f"{label}: observed {observed:.4f}, rounds to {rounded}")


def run_video(video_id: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the current repair over one recorded video's fallback scene quads."""
    scene_starts, quads = load_recorded_quads(video_id)
    ground_truth = load_ground_truth(video_id)
    repair = consensus_repair(quads)

    flagged_starts = [scene_start for scene_start, flagged in zip(scene_starts, repair.flagged) if flagged]
    before = per_scene_error(quads, ground_truth)
    after = per_scene_error(repair.repaired_quads, ground_truth)
    consensus_mean = float(np.linalg.norm(repair.consensus_quad - ground_truth, axis=1).mean())

    print(f"\nVIDEO {video_id} ({len(quads)} fallback scenes)")
    print(f"flagged ({int(repair.flagged.sum())}): {flagged_starts}")
    print(
        "whole-quad repair: "
        f"mean {before.mean():.4f} -> {after.mean():.4f} refpx; "
        f"p90 {np.percentile(before, 90):.4f} -> {np.percentile(after, 90):.4f} refpx"
    )
    print(f"consensus quad mean vs GT: {consensus_mean:.4f} refpx")
    return repair.distances_px, repair.flagged, consensus_mean


def main() -> None:
    """Check the recorded separation and consensus error."""
    print(f"CONSENSUS_FLAG_THRESHOLD_PX = {CONSENSUS_FLAG_THRESHOLD_PX}")
    distances_3, flagged_3, consensus_mean_3 = run_video(3)
    distances_21, flagged_21, consensus_mean_21 = run_video(21)

    if int(flagged_3.sum()) != 7:
        raise AssertionError(f"Video 3 flagged {int(flagged_3.sum())} scenes; expected 7")
    if int(flagged_21.sum()) != 0:
        raise AssertionError(f"Video 21 flagged {int(flagged_21.sum())} scenes; expected 0")

    print("\nRecorded evidence")
    check_evidence("vid 3 good-scene max dist", float(distances_3[~flagged_3].max()))
    check_evidence("vid 3 bad-scene min dist", float(distances_3[flagged_3].min()))
    check_evidence("vid 21 max dist (control)", float(distances_21.max()))
    check_evidence("vid 3 consensus mean vs GT", consensus_mean_3)
    check_evidence("vid 21 consensus mean vs GT", consensus_mean_21)


if __name__ == "__main__":
    main()
