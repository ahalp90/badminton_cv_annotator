"""Boundary tests for the independent court evaluation runner."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.independent_court import evaluate


def _write_manifest(tmp_path: Path, case: dict) -> Path:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.png"), image)
    manifest = tmp_path / "input.json.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as target:
        json.dump({"cases": [case]}, target)
    return manifest


def _detection(corners: np.ndarray, *, accepted: bool = True) -> evaluate.detector.Detection:
    candidate = evaluate.detector.Candidate(corners, 0.8, (0.75, 0.85), (4, 5))
    return evaluate.detector.Detection(
        (candidate,), accepted, "accepted" if accepted else "ambiguous", 0.1,
        np.empty((0, 4)), (4, 5), 9,
    )


def _run(tmp_path: Path, monkeypatch, case: dict, corners: np.ndarray, *, accepted: bool = True) -> dict:
    manifest = _write_manifest(tmp_path, case)
    monkeypatch.setattr(evaluate.detector, "detect", lambda image, settings: _detection(corners, accepted=accepted))
    output = tmp_path / "result"
    assert evaluate.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    with gzip.open(output / "results.json.gz", "rt", encoding="utf-8") as source:
        return json.load(source)


def test_matching_view_reports_reference_errors_and_portable_output(tmp_path: Path, monkeypatch) -> None:
    reference = np.array([[100, 100], [1100, 100], [1100, 620], [100, 620]], dtype=float)
    predicted = reference + [3, 4]
    court = np.array([[0, 0], [6.1, 0], [3.05, 6.7], [6.1, 13.4]], dtype=float)
    homography = evaluate._homography(predicted)
    assert homography is not None
    projected, _ = evaluate.detector.project(homography[None], court)
    case = {
        "id": "translation/case",
        "image": "frame.png",
        "reference_status": "matching_view",
        "corners_px": reference.tolist(),
        "landmarks": [
            {"court_m": point.tolist(), "image_px": (image + [0, 4]).tolist()}
            for point, image in zip(court, projected[0])
        ],
        "baseline": {"corners_px": reference.tolist(), "accepted": False},
    }
    result = _run(tmp_path, monkeypatch, case, predicted)
    record = result["cases"][0]
    assert record["accepted"] is True
    assert record["reason"] == "accepted"
    assert record["raw_metrics"]["corner_mean_error_px"] == 5.0
    assert record["raw_metrics"]["corner_max_error_px"] == 5.0
    assert record["raw_metrics"]["landmark_rms_px"] == 4.0
    assert record["baseline"]["metrics"]["corner_mean_error_px"] == 0.0
    assert result["manifest"] == "input.json.gz"
    assert str(tmp_path) not in json.dumps(result)
    assert not Path(record["overlay"]).is_absolute()


def test_unverified_view_suppresses_raw_and_baseline_metrics(tmp_path: Path, monkeypatch) -> None:
    corners = np.array([[100, 100], [1100, 100], [1100, 620], [100, 620]], dtype=float)
    case = {
        "id": "unverified",
        "image": "frame.png",
        "reference_status": "view_unverified",
        "corners_px": corners.tolist(),
        "landmarks": [{"court_m": [0, 0], "image_px": [100, 100]}],
        "baseline": {"corners_px": corners.tolist(), "accepted": False},
    }
    result = _run(tmp_path, monkeypatch, case, corners, accepted=False)
    record = result["cases"][0]
    assert record["accepted"] is False
    assert record["reason"] == "ambiguous"
    assert record["raw_metrics"] is None
    assert record["baseline"]["metrics"] is None


def test_degenerate_prediction_marks_landmarks_invalid(tmp_path: Path, monkeypatch) -> None:
    corners = np.full((4, 2), 200.0)
    case = {
        "id": "degenerate",
        "image": "frame.png",
        "reference_status": "matching_view",
        "corners_px": [[100, 100], [1100, 100], [1100, 620], [100, 620]],
        "landmarks": [{"court_m": [0, 0], "image_px": [100, 100]}],
    }
    result = _run(tmp_path, monkeypatch, case, corners)
    metrics = result["cases"][0]["raw_metrics"]
    assert metrics["landmark_rms_px"] is None
    assert metrics["landmark_status"] == "invalid_homography"
