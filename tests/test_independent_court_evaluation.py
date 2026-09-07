"""Boundary tests for the independent court evaluation runner."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.annotator.independent_court import evaluate


def _write_manifest(tmp_path: Path, case: dict) -> Path:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "frame.png"), image)
    manifest = tmp_path / "input.json.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as target:
        json.dump({"cases": [case]}, target)
    return manifest


def _write_multi_manifest(tmp_path: Path, cases: list[dict]) -> Path:
    """Write distinct same-sized images so cache-to-case joins are observable."""
    for index, case in enumerate(cases):
        image = np.full((720, 1280, 3), index * 17, dtype=np.uint8)
        assert cv2.imwrite(str(tmp_path / case["image"]), image)
    manifest = tmp_path / "input.json.gz"
    with gzip.open(manifest, "wt", encoding="utf-8") as target:
        json.dump({"cases": cases}, target)
    return manifest


def _write_line_cache(path: Path, variant: str, cases: list[dict]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump({"variant": variant, "cases": cases}, target)
    return path


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


def test_line_cache_joins_subset_by_id_and_preserves_provenance(
    tmp_path: Path, monkeypatch,
) -> None:
    cases = [
        {"id": "first", "image": "first.png", "reference_status": "unlabelled"},
        {"id": "second", "image": "second.png", "reference_status": "unlabelled"},
    ]
    manifest = _write_multi_manifest(tmp_path, cases)
    first_segments = np.array([[10, 20, 80, 20], [40, 10, 40, 70]], dtype=float)
    second_segments = np.array([[100, 120, 180, 120], [140, 110, 140, 170]], dtype=float)
    cache_records = [
        {
            "id": "second",
            "dimensions": {"width": 1280, "height": 720},
            "image_file_md5": hashlib.md5((tmp_path / "second.png").read_bytes()).hexdigest(),
            "segments_px": second_segments.tolist(),
        },
        {
            "id": "first",
            "dimensions": {"width": 1280, "height": 720},
            "image_file_md5": hashlib.md5((tmp_path / "first.png").read_bytes()).hexdigest(),
            "segments_px": first_segments.tolist(),
        },
        {
            "id": "unrelated-cache-case",
            "dimensions": {"width": 1280, "height": 720},
            "image_file_md5": "unused",
            "segments_px": [],
        },
    ]
    cache = _write_line_cache(tmp_path / "shared-lines.json.gz", "ridge", cache_records)
    seen: dict[int, np.ndarray] = {}

    def detect(image: np.ndarray, settings, *, segments_px: np.ndarray | None = None):
        assert segments_px is not None
        seen[int(image[0, 0, 0])] = segments_px.copy()
        return _detection(np.array([[100, 100], [1100, 100], [1100, 620], [100, 620]], dtype=float))

    monkeypatch.setattr(evaluate.detector, "detect", detect)
    output = tmp_path / "result"
    assert evaluate.main([
        "--manifest", str(manifest), "--output", str(output), "--line-cache", str(cache),
    ]) == 0

    np.testing.assert_array_equal(seen[0], first_segments)
    np.testing.assert_array_equal(seen[17], second_segments)
    with gzip.open(output / "results.json.gz", "rt", encoding="utf-8") as source:
        result = json.load(source)
    assert result["settings"]["extractor"] == "ridge"
    assert result["line_cache"] == {
        "variant": "ridge", "file": cache.name, "file_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(result)


def test_line_cache_missing_case_id_fails(tmp_path: Path) -> None:
    cases = [
        {"id": "first", "image": "frame.png", "reference_status": "unlabelled"},
        {"id": "second", "image": "other.png", "reference_status": "unlabelled"},
    ]
    manifest = _write_multi_manifest(tmp_path, cases)
    cache = _write_line_cache(tmp_path / "lines.json.gz", "ridge", [{
        "id": "first",
        "dimensions": {"width": 1280, "height": 720},
        "image_file_md5": hashlib.md5((tmp_path / "frame.png").read_bytes()).hexdigest(),
        "segments_px": [[10, 10, 20, 10]],
    }])
    with pytest.raises(ValueError, match="missing case IDs"):
        evaluate.main(["--manifest", str(manifest), "--output", str(tmp_path / "result"), "--line-cache", str(cache)])


def test_line_cache_duplicate_case_id_fails(tmp_path: Path) -> None:
    case = {"id": "first", "image": "frame.png", "reference_status": "unlabelled"}
    manifest = _write_manifest(tmp_path, case)
    record = {
        "id": "first",
        "dimensions": {"width": 1280, "height": 720},
        "image_file_md5": hashlib.md5((tmp_path / "frame.png").read_bytes()).hexdigest(),
        "segments_px": [[10, 10, 20, 10]],
    }
    cache = _write_line_cache(tmp_path / "lines.json.gz", "ridge", [record, record])
    with pytest.raises(ValueError, match="unique"):
        evaluate.main(["--manifest", str(manifest), "--output", str(tmp_path / "result"), "--line-cache", str(cache)])


@pytest.mark.parametrize("failure", ["hash", "dimensions"])
def test_line_cache_image_identity_mismatch_fails(tmp_path: Path, failure: str) -> None:
    case = {"id": "first", "image": "frame.png", "reference_status": "unlabelled"}
    manifest = _write_manifest(tmp_path, case)
    dimensions = {"width": 1280, "height": 720} if failure == "hash" else {"width": 960, "height": 540}
    record = {
        "id": "first",
        "dimensions": dimensions,
        "image_file_md5": "wrong" if failure == "hash" else hashlib.md5((tmp_path / "frame.png").read_bytes()).hexdigest(),
        "segments_px": [[10, 10, 20, 10]],
    }
    cache = _write_line_cache(tmp_path / "lines.json.gz", "ridge", [record])
    message = "different image" if failure == "hash" else "dimensions"
    with pytest.raises(ValueError, match=message):
        evaluate.main(["--manifest", str(manifest), "--output", str(tmp_path / "result"), "--line-cache", str(cache)])


def test_line_cache_provenance_identifies_changed_evidence(tmp_path: Path) -> None:
    path = tmp_path / "same-name.json.gz"
    record = {"id": "court", "segments_px": [[0, 0, 10, 10]]}
    _write_line_cache(path, "same-model", [record])
    first, _ = evaluate._load_line_cache(path)
    record["segments_px"] = [[0, 0, 20, 20]]
    _write_line_cache(path, "same-model", [record])
    second, _ = evaluate._load_line_cache(path)
    assert first["file"] == second["file"]
    assert first["variant"] == second["variant"]
    assert first["file_sha256"] != second["file_sha256"]
    assert second["file_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
