"""Geometry and cached-detection checks for the SAM 3 court baseline."""

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.court_detector.benchmark.sam3 import merge_pieces


def test_group_preserves_instances_and_joins_transitively() -> None:
    masks = np.zeros((4, 30, 30), dtype=bool)
    masks[0, 1:4, 1:4] = True
    masks[0, 10:13, 10:13] = True  # Disconnected strokes in one SAM instance.
    masks[1, 12:16, 12:16] = True
    masks[2, 15:19, 15:19] = True
    masks[3, 22:28, 22:28] = True  # Largest individual piece, smaller than the group.
    selected = merge_pieces.largest_group(masks)
    assert selected[2, 2] and selected[11, 11] and selected[18, 18]
    assert not selected[25, 25]
    assert np.count_nonzero(selected) == np.count_nonzero(masks[:3].any(axis=0))


def test_fit_encloses_perspective_mask() -> None:
    vertices = np.array([[30, 10], [70, 10], [95, 85], [5, 85]], dtype=np.int32)
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.polylines(mask, [vertices], True, 1, 1)
    corners = merge_pieces.fit_quadrilateral(mask)
    assert corners is not None and corners.shape == (4, 2)
    assert cv2.isContourConvex(corners)
    assert all(cv2.pointPolygonTest(corners, tuple(map(float, point)), True) >= -1 for point in vertices)
    assert abs(cv2.contourArea(corners) - cv2.contourArea(vertices)) < 100


@pytest.mark.parametrize("kind", ["empty", "line", "triangle"])
def test_insufficient_boundary_returns_no_quad(kind: str) -> None:
    mask = np.zeros((30, 30), dtype=np.uint8)
    if kind == "line":
        mask[10, 5:20] = 1
    elif kind == "triangle":
        cv2.fillPoly(mask, [np.array([[5, 5], [20, 5], [5, 20]], np.int32)], 1)
    assert merge_pieces.fit_quadrilateral(mask) is None


def test_cached_detections_use_fixed_confidence_and_record_no_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame_path), np.zeros((60, 80, 3), dtype=np.uint8))
    masks = np.zeros((2, 60, 80), dtype=bool)
    masks[0, 1:59, 1:79] = True
    masks[1, 20:40, 30:50] = True
    for name, scores in [("court", [0.5, 0.51]), ("empty", [0.4, 0.5])]:
        np.savez_compressed(tmp_path / f"{name}__court_markings.npz", frame_path=str(frame_path),
                            prompt="court markings", masks=masks, scores=scores)
    output = tmp_path / "output"
    monkeypatch.setattr("sys.argv", ["merge_pieces.py", str(tmp_path), "--out-dir", str(output)])
    merge_pieces.main()
    with gzip.open(output / "quadrilaterals.json.gz", "rt") as handle:
        results = json.load(handle)
    court = results["court__court_markings"]
    assert court["retained_pixels"] == 400
    assert court["image_size"] == [80, 60]
    assert all(29 <= x <= 50 and 19 <= y <= 40 for x, y in court["corners"])
    assert results["empty__court_markings"]["corners"] is None
    assert (output / "court__court_markings.jpg").is_file()
