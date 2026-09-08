"""Focused tests for the bounded independent-court person exporter."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.annotator.independent_court.export_people import (
    _write_manifests,
    export_window,
)


def _tiny_video(root: Path, frame_count: int = 10) -> Path:
    video_path = root / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (20, 12)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG video writer is unavailable")
    for frame_index in range(frame_count):
        frame = np.full((12, 20, 3), frame_index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


def test_export_keeps_scheduled_indices_and_empty_samples(tmp_path: Path) -> None:
    _tiny_video(tmp_path)
    output = tmp_path / "export"
    detector_calls = 0

    def empty_detector(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal detector_calls
        detector_calls += 1
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    record = export_window(
        {
            "id": "case-a",
            "video": "clip.avi",
            "start_frame": 2,
            "end_frame": 9,
            "anchor_frames": [3, 8],
        },
        tmp_path,
        output,
        empty_detector,
        sample_fps=2.0,
        score_min=0.2,
    )

    assert [sample["frame_index"] for sample in record["samples"]] == [2, 4, 7]
    assert all(sample["bboxes"] == [] and sample["scores"] == [] for sample in record["samples"])
    assert detector_calls == len(record["samples"]) == 3
    assert [anchor["frame_index"] for anchor in record["anchor_images"]] == [3, 8]
    assert record["video"] == "clip.avi"
    assert record["dimensions"] == {"width": 20, "height": 12}
    assert cv2.imread(str(output / record["median_image"])).shape == (12, 20, 3)

    def assert_relative(value: object) -> None:
        if isinstance(value, dict):
            for item in value.values():
                assert_relative(item)
        elif isinstance(value, list):
            for item in value:
                assert_relative(item)
        elif isinstance(value, str):
            assert not Path(value).is_absolute()

    assert_relative(record)


def test_line_manifests_use_relative_images(tmp_path: Path) -> None:
    record = {
        "id": "case-a",
        "median_image": "images/case-a_median.png",
        "anchor_images": [
            {"frame_index": 8, "image": "images/case-a_frame_00000008.png"}
        ],
    }
    _write_manifests(tmp_path, [record])

    with gzip.open(tmp_path / "manifest.json.gz", "rt", encoding="utf-8") as source:
        assert json.load(source) == {
            "windows": [{"id": "case-a", "record": "case-a.json.gz"}]
        }
    with gzip.open(tmp_path / "line_manifest.json.gz", "rt", encoding="utf-8") as source:
        assert json.load(source) == {
            "cases": [
                {
                    "id": "case-a_median",
                    "image": "images/case-a_median.png",
                    "reference_status": "unlabelled",
                },
                {
                    "id": "case-a_frame_8",
                    "image": "images/case-a_frame_00000008.png",
                    "reference_status": "unlabelled",
                },
            ]
        }


def test_invalid_window_interval_fails_before_decode(tmp_path: Path) -> None:
    _tiny_video(tmp_path)

    with pytest.raises(ValueError, match="exceeds"):
        export_window(
            {
                "id": "bad-window",
                "video": "clip.avi",
                "start_frame": 2,
                "end_frame": 11,
                "anchor_frames": [],
            },
            tmp_path,
            tmp_path / "export",
            lambda frame: (np.empty((0, 4)), np.empty((0,))),
            sample_fps=2.0,
            score_min=0.2,
        )
