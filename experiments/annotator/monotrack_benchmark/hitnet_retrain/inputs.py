"""Prepare saved detections using MonoTrack's player-selection rules."""
from __future__ import annotations

import gzip
import json
import lzma
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "monotrack/python/ai-badminton/src"))

from ai_badminton.court import Court  # pyrefly: ignore[missing-import]
from ai_badminton.pose import Pose  # pyrefly: ignore[missing-import]


def read_json(path: Path) -> Any:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    with gzip.open(path, "wt") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read_array(path: Path) -> np.ndarray:
    with lzma.open(path, "rb") as stream:
        return np.load(stream, allow_pickle=False)


def write_array(path: Path, values: np.ndarray) -> None:
    with lzma.open(path, "wb", format=lzma.FORMAT_XZ, preset=9) as stream:
        np.save(stream, values, allow_pickle=False)


def select_players(kps: np.ndarray, kp_scores: np.ndarray, ndet: np.ndarray, court: Court) -> np.ndarray:
    """Return court slots 1/2 in saved detection order, before filling gaps.

    :param kps: Pixel coordinates, one 17-joint pose per frame and detection.
    :param kp_scores: Confidence for each of those joints, not bounding-box scores.
    :param ndet: Number of real detections in each padded frame.
    :param court: MonoTrack Court in its original selection order.
    :return: Coordinates shaped (frames, 2 court halves, 17 joints, 2 axes).
    """
    selected = np.full((len(kps), 2, 17, 2), np.nan)
    for frame, count in enumerate(ndet):
        best_scores: list[float | None] = [None, None]
        for detection in range(int(count)):
            pose = Pose()
            pose.init_from_kparray(kps[frame, detection])
            half = court.in_court(pose.get_base(), slack=[0.05, 0.15])
            if not half:
                continue
            slot = int(half) - 1
            # Pose's text reader sums Python floats in joint order.
            score = sum(float(value) for value in kp_scores[frame, detection])
            if best_scores[slot] is None or score > best_scores[slot]:
                selected[frame, slot] = pose.kp
                best_scores[slot] = score
    return selected


def fill_players(players: np.ndarray) -> np.ndarray:
    """Apply the released pose fill inside one rally, including final zero fill."""
    filled = np.empty_like(players)
    for slot in range(2):
        table = pd.DataFrame(players[:, slot].reshape(len(players), 34))
        table = table.interpolate(method="slinear").bfill().ffill().fillna(0)
        filled[:, slot] = table.to_numpy().reshape(len(players), 17, 2)
    return filled


def shuttle_variants(track: np.ndarray, sidecar: dict[str, Any], resolution: list[int]) -> dict[str, np.ndarray]:
    """Make native-pixel tracks; zero selected inpaint spans before rally cropping."""
    if sidecar["inpaint_status"] != "applied" or sidecar["n_rows"] != len(track):
        raise ValueError("Inpaint sidecar does not describe the supplied track")
    if track.shape != (len(track), 3) or not np.isin(track[:, 2], (0, 1)).all():
        raise ValueError("Shuttle track must contain normalised x, y and binary visibility")
    if np.any(track[track[:, 2] == 0, :2] != 0):
        raise ValueError("Invisible shuttle points must be zero")
    kept = track[:, :2] * np.asarray(resolution, dtype=float)
    zeroed = kept.copy()
    for start, end in sidecar["inpaint_selected"]:
        if not 0 <= start < end <= len(track):
            raise ValueError(f"Inpaint span [{start}, {end}) is outside the video")
        zeroed[start:end] = 0
    return {"kept": kept, "zeroed": zeroed}


def feature_rows(shuttle: np.ndarray, players: np.ndarray, court: Court) -> np.ndarray:
    """Notebook frame layout: shuttle, bottom pose, top pose, training-order corners."""
    from recipe import scale_data

    corners = np.broadcast_to(np.asarray(court.corners).reshape(8), (len(shuttle), 8))
    rows = np.column_stack((shuttle, players.reshape(len(shuttle), 68), corners))
    # Every row occurs in a 12-frame window, so per-axis extrema are identical
    # to scaling the notebook's full window matrix. Missing zeros stay zero.
    return scale_data(rows).astype(np.float32)
