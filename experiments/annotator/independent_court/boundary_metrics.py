"""Diagnostic measurements for clicked court corners and outer boundaries."""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from courtkeynet.court_corners import CORNER_COURT_M, COURT_LENGTH_M, COURT_WIDTH_M

REFERENCE_DIMENSIONS = (1280, 720)
SHORT_CLICK_INSET_M = 0.02

# These are the development frames already present in the frozen stripe replay.
_CASE_IDS: dict[tuple[str, int], str] = {
    ("zYqgJo1L5uM", 54): "am1_window_00_frame_54",
    ("zYqgJo1L5uM", 5352): "am1_window_01_frame_5352",
    ("BkjErIAsZu4", 150): "am2_window_00_frame_150",
    ("BkjErIAsZu4", 28019): "am2_window_01_frame_28019",
    ("C6NrJyBwn6c", 0): "am3_window_00_frame_0",
    ("C6NrJyBwn6c", 10514): "am3_window_01_frame_10514",
    ("C6NrJyBwn6c", 17174): "am3_window_02_frame_17174",
    ("C6NrJyBwn6c", 24515): "am3_window_03_frame_24515",
    ("C1jR4vZmrkI", 0): "am4_window_00_frame_0",
    ("C1jR4vZmrkI", 319): "am4_window_00_frame_319",
    ("C1jR4vZmrkI", 13782): "am4_window_01_frame_13782",
    ("E8WW8DFCnwk_sample", 14): "yellow_short_frame_14",
    ("E8WW8DFCnwk_sample", 90): "yellow_short_frame_90",
    ("E8WW8DFCnwk_sample", 156): "yellow_short_frame_156",
    ("Cb-xs5rPyxI_gameplay", 45): "letterboxed_short_frame_45",
    ("Cb-xs5rPyxI_gameplay", 58): "letterboxed_short_frame_58",
    ("Cb-xs5rPyxI_gameplay", 78): "letterboxed_short_frame_78",
    ("l-I_Di1Ad2Y_h264", 36): "centre_short_frame_36",
    ("l-I_Di1Ad2Y_h264", 64): "centre_short_frame_64",
    ("l-I_Di1Ad2Y_h264", 71): "centre_short_frame_71",
}


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = homogeneous @ homography.T
    denominator = mapped[:, 2]
    if not np.isfinite(mapped).all() or np.any(denominator <= 0):
        raise ValueError(
            "projection has non-positive or non-finite homogeneous denominators"
        )
    projected = mapped[:, :2] / denominator[:, None]
    if not np.isfinite(projected).all():
        raise ValueError("projection is non-finite")
    return projected


def _summary(distances: Iterable[float]) -> tuple[int, float | None, float | None]:
    values = np.asarray(list(distances), dtype=np.float64)
    if not len(values):
        return 0, None, None
    if not np.isfinite(values).all():
        raise ValueError("metric distances must be finite")
    return len(values), float(np.sqrt(np.mean(values**2))), float(np.max(values))


def _inset_corners(click_inset_m: float) -> np.ndarray:
    return np.array(
        [
            [click_inset_m, click_inset_m],
            [COURT_WIDTH_M - click_inset_m, click_inset_m],
            [COURT_WIDTH_M - click_inset_m, COURT_LENGTH_M - click_inset_m],
            [click_inset_m, COURT_LENGTH_M - click_inset_m],
        ],
        dtype=np.float64,
    )


def _boundary_names(point: np.ndarray) -> tuple[str, ...]:
    names = []
    if np.isclose(point[0], 0.0, rtol=0.0, atol=1e-8):
        names.append("left")
    if np.isclose(point[0], COURT_WIDTH_M, rtol=0.0, atol=1e-8):
        names.append("right")
    if np.isclose(point[1], 0.0, rtol=0.0, atol=1e-8):
        names.append("top")
    if np.isclose(point[1], COURT_LENGTH_M, rtol=0.0, atol=1e-8):
        names.append("bottom")
    return tuple(names)


def _segment_distance(point: np.ndarray, segment: np.ndarray) -> float:
    start, end = segment
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("projected boundary segment is non-finite or degenerate")
    fraction = np.clip(np.dot(point - start, direction) / denominator, 0.0, 1.0)
    distance = float(np.linalg.norm(point - (start + fraction * direction)))
    if not np.isfinite(distance):
        raise ValueError("boundary distance is non-finite")
    return distance


def measure(
    corners: np.ndarray,
    reference: dict[str, Any],
    dimensions: tuple[int, int],
    clicked_corners: list[int],
    click_inset_m: float,
) -> dict[str, int | float | None]:
    """Measure clicked corners and finite outer-boundary landmark residuals.

    Distances are Euclidean distances after native pixels are scaled to
    1280x720. A landmark at a court corner contributes once for each adjacent
    boundary, so the boundary count is a side-measurement count.
    """
    corners = np.asarray(corners, dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("corners must contain four finite points")
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, corners.astype(np.float32))
    # OpenCV can return either matrix sign for ill-conditioned corner sets.
    # The finite origin corner fixes a common scale before testing depth.
    homography /= homography[2, 2]
    _project(homography, np.asarray(CORNER_COURT_M, dtype=np.float64))
    width, height = dimensions
    scale = np.array(
        [REFERENCE_DIMENSIONS[0] / width, REFERENCE_DIMENSIONS[1] / height]
    )

    reference_corners = np.asarray(reference["corners_px"], dtype=np.float64)
    inset_corners = _project(homography, _inset_corners(click_inset_m))
    corner_errors = np.linalg.norm(
        (inset_corners[clicked_corners] - reference_corners[clicked_corners]) * scale,
        axis=1,
    )
    clicked_count, clicked_rms, clicked_max = _summary(corner_errors)

    boundary_errors: list[float] = []
    inset_refpx = inset_corners * scale
    segments = {"left": inset_refpx[[0, 3]], "top": inset_refpx[[0, 1]],
                "right": inset_refpx[[1, 2]], "bottom": inset_refpx[[3, 2]]}
    for landmark in reference.get("landmarks", []):
        court_point = np.asarray(landmark["court_m"], dtype=np.float64)
        image_point = np.asarray(landmark["image_px"], dtype=np.float64)
        if not np.isfinite(court_point).all() or not np.isfinite(image_point).all():
            raise ValueError("landmark contains non-finite coordinates")
        for boundary in _boundary_names(court_point):
            boundary_errors.append(_segment_distance(image_point * scale, segments[boundary]))
    boundary_count, boundary_rms, boundary_max = _summary(boundary_errors)
    return {
        "clicked_corner_count": clicked_count,
        "clicked_corner_rms_px": clicked_rms,
        "clicked_corner_max_px": clicked_max,
        "boundary_landmark_count": boundary_count,
        "boundary_landmark_rms_px": boundary_rms,
        "boundary_landmark_max_px": boundary_max,
    }


def _read_rows(path: Path, compressed: bool) -> list[dict[str, str]]:
    opener = gzip.open if compressed else Path.open
    with opener(path, "rt", newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _rows_to_metadata(
    rows: list[dict[str, str]], click_inset_m: float
) -> dict[str, dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (Path(row["video"]).stem, int(row["frame"]))
        grouped.setdefault(key, []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for key, group in grouped.items():
        if key not in _CASE_IDS:
            raise ValueError(f"unexpected corner annotation case {key}")
        indices = [int(row["corner_idx"]) for row in group]
        if len(group) != 4 or set(indices) != set(range(4)):
            raise ValueError(
                f"corner annotation case {key} must contain four corner slots"
            )
        ordered = sorted(group, key=lambda row: int(row["corner_idx"]))
        result[_CASE_IDS[key]] = {
            "indices": [
                int(row["corner_idx"])
                for row in ordered
                if row["visible"] == "1" and row["source"] == "click"
            ],
            "corners_px": [[float(row["x_px"]), float(row["y_px"])] for row in ordered],
            "click_inset_m": click_inset_m,
            "convention": "outside paint edges" if click_inset_m == 0 else "paint centres",
        }
    return result


def load_corner_metadata(repo_data_root: Path) -> dict[str, dict[str, Any]]:
    """Load the frozen 20-case metadata from ``data/amateur_court_corners``."""
    original = _rows_to_metadata(
        _read_rows(repo_data_root / "hand_corners.csv", compressed=False), 0.0
    )
    short = _rows_to_metadata(
        _read_rows(
            repo_data_root / "2026-09-08" / "hand_corners.csv.gz", compressed=True
        ),
        SHORT_CLICK_INSET_M,
    )
    result = {**original, **short}
    if set(result) != set(_CASE_IDS.values()) or len(result) != 20:
        raise ValueError(f"expected 20 frozen corner cases, loaded {len(result)}")
    return result
