"""Tests for the independent court boundary diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.annotator.independent_court.boundary_metrics import (
    load_corner_metadata,
    measure,
)
from experiments.annotator.independent_court.detector import CORNER_COURT_M

COURT_CORNERS = np.array(
    [[100.0, 100.0], [1120.0, 165.0], [1010.0, 655.0], [165.0, 590.0]]
)


def _project(
    court_points: np.ndarray, corners: np.ndarray = COURT_CORNERS
) -> np.ndarray:
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, corners.astype(np.float32))
    return cv2.perspectiveTransform(
        court_points.reshape(1, -1, 2).astype(np.float32), homography
    )[0]


def _reference(corners: np.ndarray = COURT_CORNERS) -> dict:
    boundary_points = np.array([[0.0, 0.0], [6.1, 6.7], [6.1, 13.4], [3.05, 0.0]])
    projected = _project(CORNER_COURT_M, corners)
    landmarks = [
        {"court_m": point.tolist(), "image_px": image.tolist()}
        for point, image in zip(boundary_points, _project(boundary_points, corners))
    ]
    return {"corners_px": projected.tolist(), "landmarks": landmarks}


def test_diagonal_perspective_direct_clicks_and_boundary_corners_are_exact() -> None:
    metrics = measure(COURT_CORNERS, _reference(), (1280, 720), [0, 1], 0.0)

    assert metrics == {
        "clicked_corner_count": 2,
        "clicked_corner_rms_px": pytest.approx(0.0, abs=1e-4),
        "clicked_corner_max_px": pytest.approx(0.0, abs=1e-4),
        "boundary_landmark_count": 6,
        "boundary_landmark_rms_px": pytest.approx(0.0, abs=1e-4),
        "boundary_landmark_max_px": pytest.approx(0.0, abs=1e-4),
    }


def test_extrapolated_reference_corners_do_not_enter_clicked_corner_metric() -> None:
    reference = _reference()
    reference["corners_px"][2] = [99999.0, -99999.0]
    reference["corners_px"][3] = [-99999.0, 99999.0]

    metrics = measure(COURT_CORNERS, reference, (1280, 720), [0, 1], 0.0)

    assert metrics["clicked_corner_count"] == 2
    assert metrics["clicked_corner_rms_px"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["clicked_corner_max_px"] == pytest.approx(0.0, abs=1e-4)


def test_native_errors_use_per_axis_1280_by_720_scaling() -> None:
    native_corners = COURT_CORNERS * 2
    reference = _reference(native_corners)
    reference["corners_px"][0] = (
        np.asarray(reference["corners_px"][0]) + [2.0, 4.0]
    ).tolist()

    metrics = measure(native_corners, reference, (2560, 1440), [0], 0.0)

    expected = np.sqrt(5.0)
    assert metrics["clicked_corner_rms_px"] == expected
    assert metrics["clicked_corner_max_px"] == expected


def test_boundary_landmark_uses_its_named_side_only() -> None:
    reference = {
        "corners_px": _project(CORNER_COURT_M).tolist(),
        "landmarks": [
            {
                "court_m": [0.0, 6.7],
                "image_px": _project(np.array([[3.05, 0.0]]))[0].tolist(),
            },
        ],
    }

    metrics = measure(COURT_CORNERS, reference, (1280, 720), [], 0.0)

    assert metrics["boundary_landmark_count"] == 1
    assert metrics["boundary_landmark_max_px"] > 100.0


def test_physical_stripe_centres_require_the_020_m_inset() -> None:
    inset = 0.02
    physical_corners = np.array(
        [
            [inset, inset],
            [6.1 - inset, inset],
            [6.1 - inset, 13.4 - inset],
            [inset, 13.4 - inset],
        ]
    )
    nominal_landmarks = np.array([[0.0, 6.7], [3.05, 0.0], [6.1, 6.7], [3.05, 13.4]])
    physical_landmarks = np.array(
        [[inset, 6.7], [3.05, inset], [6.1 - inset, 6.7], [3.05, 13.4 - inset]]
    )
    reference = {
        "corners_px": _project(physical_corners).tolist(),
        "landmarks": [
            {"court_m": nominal.tolist(), "image_px": image.tolist()}
            for nominal, image in zip(nominal_landmarks, _project(physical_landmarks))
        ],
    }

    centred = measure(COURT_CORNERS, reference, (1280, 720), [0, 1, 2, 3], inset)
    outer = measure(COURT_CORNERS, reference, (1280, 720), [0, 1, 2, 3], 0.0)

    assert centred["clicked_corner_rms_px"] == pytest.approx(0.0, abs=1e-4)
    assert centred["boundary_landmark_rms_px"] == pytest.approx(0.0, abs=1e-4)
    assert outer["clicked_corner_rms_px"] > 0.1
    assert outer["boundary_landmark_rms_px"] > 0.1


def test_frozen_metadata_keeps_extrapolated_in_frame_am2_corners() -> None:
    data_root = Path(__file__).resolve().parents[1] / "data" / "amateur_court_corners"
    metadata = load_corner_metadata(data_root)

    am2 = metadata["am2_window_00_frame_150"]
    assert am2["indices"] == [2]
    assert all(
        0 <= coordinate < bound
        for corner in am2["corners_px"][:2]
        for coordinate, bound in zip(corner, (1920, 1080))
    )
    with (data_root / "hand_corners.csv").open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    am2_rows = {
        int(row["corner_idx"]): row
        for row in rows
        if Path(row["video"]).stem == "BkjErIAsZu4" and row["frame"] == "150"
    }
    assert all(am2_rows[index]["source"] == "extrapolated" for index in (0, 1))
    assert metadata["am3_window_00_frame_0"]["indices"] == [0, 1]
    assert metadata["am3_window_00_frame_0"]["click_inset_m"] == 0.0
    assert metadata["yellow_short_frame_14"]["click_inset_m"] == 0.02


def test_empty_categories_report_zero_counts_and_missing_errors() -> None:
    metrics = measure(
        COURT_CORNERS,
        {"corners_px": _project(CORNER_COURT_M).tolist(), "landmarks": []},
        (1280, 720),
        [],
        0.0,
    )

    assert metrics == {
        "clicked_corner_count": 0,
        "clicked_corner_rms_px": None,
        "clicked_corner_max_px": None,
        "boundary_landmark_count": 0,
        "boundary_landmark_rms_px": None,
        "boundary_landmark_max_px": None,
    }
