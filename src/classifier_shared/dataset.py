"""Classifier dataset paths, curation data, and clip-bound helpers.

ShuttleSet upstream annotations live under ``training/data/shuttleset/annotations/``:

  - ``ANNOTATIONS_DIR``      — root of the annotations tree (CSVs + set/)
  - ``SET_INFO_DIR``         — annotations/set/ (per-match folders + match.csv)
  - ``FLAW_RECORDS_PATH``    — annotations/flaw_shot_records.csv
  - ``VIDEO_METADATA_PATH``  — annotations/video_metadata.csv

BST-team split configuration lives alongside this module in
``src/classifier_shared/``. It is a curation decision (small, static,
version-controlled) layered on top of the upstream data, not part
of ShuttleSet itself:

  - ``SPLITS_V2_PATH``       — src/classifier_shared/shuttleset_splits_v2.csv

Curation constants:

  - ``EXCLUDED_VIDEOS``  — set[int]   of fully dropped match IDs
  - ``REMOVED_SHOTS``    — set[tuple] of individually dropped (vid, set, rally, ball_round)
  - ``CLIP_WINDOW``      — BST's default temporal-window strategy name

Both ``EXCLUDED_VIDEOS`` and ``REMOVED_SHOTS`` are derived from
``flaw_shot_records.csv`` at import time. If the file is missing, both
are empty and a warning is emitted — fine for inspecting the module,
not fine for actual pipeline runs.

``SPLITS_BST_BASELINE`` is kept to regenerate the comparison column in
``shots_master.csv``. BRIC metadata generation reads ``SPLITS_V2_PATH``
directly.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# This file lives at <project>/src/classifier_shared/dataset.py.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_DIR = _PROJECT_ROOT / "training" / "data" / "shuttleset" / "annotations"
SET_INFO_DIR = ANNOTATIONS_DIR / "set"
FLAW_RECORDS_PATH = ANNOTATIONS_DIR / "flaw_shot_records.csv"
VIDEO_METADATA_PATH = ANNOTATIONS_DIR / "video_metadata.csv"
HOMOGRAPHY_CSV_PATH = SET_INFO_DIR / "homography.csv"

# BST-team derived split — co-located with this module since it's a
# small static curation artefact, not bulk upstream data.
SPLITS_V2_PATH = Path(__file__).resolve().parent / "shuttleset_splits_v2.csv"


# ---------------------------------------------------------------------------
# BST clip-window default
# ---------------------------------------------------------------------------
# 'between_2_hits_with_max_limits' is BST's default. See compute_clip_bounds:
# [prev_shot, next_shot + 0.25s] clamped to ±1.5s of the target frame.
CLIP_WINDOW = "between_2_hits_with_max_limits"


# ---------------------------------------------------------------------------
# Flaw-record parsing — derives EXCLUDED_VIDEOS + REMOVED_SHOTS from CSV
# ---------------------------------------------------------------------------
def parse_flaw_records(
    csv_path: Path,
) -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Parse flaw_shot_records.csv into (excluded_videos, removed_shots).

    A row's ``measure`` column is ``'removed'`` for an exclusion. If the
    ``stroke_type`` column is ``'whole'``, the entire match is excluded;
    otherwise the specific (set, rally, ball_round) shot is removed.

    :param csv_path: Path to flaw_shot_records.csv.
    :return: (excluded_video_ids, removed_shot_tuples).
    """
    excluded_videos: set[int] = set()
    removed_shots: set[tuple[int, int, int, int]] = set()

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["measure"] != "removed":
                continue
            match_id = int(row["match"])
            if row["stroke_type"] == "whole":
                excluded_videos.add(match_id)
            else:
                removed_shots.add(
                    (
                        match_id,
                        int(row["set"]),
                        int(row["rally"]),
                        int(row["ball_round"]),
                    )
                )
    return excluded_videos, removed_shots


def _load_flaw_records() -> tuple[set[int], set[tuple[int, int, int, int]]]:
    """Load flaw records lazily; warn + return empty sets if file is missing.

    Lets this module be importable for inspection without the CSV present.
    Execution continues with empty curation sets, which can produce incorrect
    pipeline results.
    """
    try:
        return parse_flaw_records(FLAW_RECORDS_PATH)
    except FileNotFoundError:
        warnings.warn(
            f"{FLAW_RECORDS_PATH} not found. EXCLUDED_VIDEOS and "
            f"REMOVED_SHOTS are empty. Fine for module inspection, "
            f"wrong for pipeline runs.",
            stacklevel=2,
        )
        return set(), set()


EXCLUDED_VIDEOS, REMOVED_SHOTS = _load_flaw_records()


# ---------------------------------------------------------------------------
# Train/val/test splits
# ---------------------------------------------------------------------------

# BST's original baseline split.
# Raw — does not strip EXCLUDED_VIDEOS. Apply the filter at the call site if
# needed (e.g. enrichment script does ``if vid in EXCLUDED_VIDEOS: continue``).
# Kept here so the enriched master CSV can populate ``split_bst_baseline`` for
# parity with BST's evaluation. NOT the active split for BRIC training —
# this scheme assigns whole videos to splits without considering player
# overlap, which leaks player identity between train and val/test.
SPLITS_BST_BASELINE: dict[str, list[int]] = {
    "train": list(range(1, 35)),
    "val": list(range(35, 39)) + [41],
    "test": [39, 40, 42, 43, 44],
}


# ---------------------------------------------------------------------------
# Clip-bounds derivation shared by BST-X and BRIC metadata generation.
# ---------------------------------------------------------------------------
def compute_temporal_bounds(
    folder_path: Path,
    shots_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add start_f and end_f columns to shots_df based on adjacent shots.

    For each shot, the start frame is the previous shot's frame in the same
    rally, and the end frame is the next shot's frame. First/last shots in a
    rally get -1 (handled by the clip window as fallback).

    Adapted from gen_my_dataset.py set_between_2_hits_from_pos().

    :param folder_path: Path to the match folder containing set CSVs.
    :param shots_df: DataFrame with 'set', 'rally', 'ball_round', 'frame_num' columns.
    :return: DataFrame with start_f and end_f columns added.
    """
    parts = []
    for set_i, group_idx in shots_df.groupby("set").groups.items():
        df = pd.read_csv(folder_path / f"set{set_i}.csv")
        df = df[["rally", "ball_round", "frame_num"]]

        # Use a shift to find adjacent frames.
        # We look at the previous shot, but if this is the first shot of a rally,
        # there is no 'previous', so we fallback to -1.
        df["start_f"] = df["frame_num"].shift(1)
        df["start_f"] = df["start_f"].where(df.duplicated("rally", keep="first"), -1)
        # Similarly, look at the next shot, but fallback to -1 if it's the last shot of the rally.
        df["end_f"] = df["frame_num"].shift(-1)
        df["end_f"] = df["end_f"].where(df.duplicated("rally", keep="last"), -1)

        merged = pd.merge(
            shots_df.loc[group_idx].reset_index(drop=True),
            df,
            on=["rally", "ball_round", "frame_num"],
        )
        merged = merged[
            [
                "set",
                "rally",
                "ball_round",
                "start_f",
                "frame_num",
                "end_f",
                "roundscore_A",
                "roundscore_B",
                "player",
                "type",
            ]
        ]
        parts.append(merged)

    return pd.concat(parts).reset_index(drop=True)


def compute_clip_bounds(row, clip_window: str, fps: float) -> tuple[int, int]:
    """Compute start and end frame for one clip based on the clip window.

    :param row: A Series (from iterrows) with keys frame_num, start_f, end_f.
    :param clip_window: One of 'middle_in_a_sec', 'between_2_hits',
        'between_2_hits_with_max_limits'.
    :param fps: Video frames per second.
    :return: (start_frame, end_frame) as ints.
    """
    t = int(fps) // 2       # frames in 0.5 sec
    frame_num = int(row["frame_num"])

    if clip_window == "middle_in_a_sec":
        # Fixed 1-second window centred on the shot frame
        return frame_num - t, frame_num + t

    # --- between_2_hits and between_2_hits_with_max_limits ---
    # Use adjacent shot frames if they exist, otherwise fall back to ±0.5 sec
    eps = t // 2  # frames in 0.25 sec (small extension past the next hit)
    start_f = int(row["start_f"]) if row["start_f"] != -1 else (frame_num - t)
    end_f = int(row["end_f"]) + eps if row["end_f"] != -1 else (frame_num + t)

    if clip_window == "between_2_hits_with_max_limits":
        # Clamp so clip never exceeds 1.5 sec each side of the shot
        limit = int(fps) * 3 // 2  # frames in 1.5 sec
        start_f = max(start_f, frame_num - limit)
        end_f = min(end_f, frame_num + limit + eps)

    # Clamp the start to 0: insurance for a shot in the first half-second of a
    # video (unreachable on real match footage, where play starts minutes in).
    return max(0, start_f), end_f
