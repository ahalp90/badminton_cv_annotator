"""Check the benchmark's visible-court metrics and missing-prediction accounting."""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from experiments.court_detector.benchmark.sam3.benchmark_rallies import (
    clipped_iou,
    evaluate_video,
    score_corners,
    summarise,
)


def test_cyclic_corner_matching_and_translation() -> None:
    reference = np.array([[20, 10], [80, 10], [90, 80], [10, 80]], dtype=np.float32)
    for reverse in (reference, reference[::-1]):
        for shift in range(4):
            result = score_corners(np.roll(reverse, shift, axis=0) + [3, 4], reference, [100, 100])
            assert result["corner_error_px"] == pytest.approx(5)


def test_iou_clips_both_polygons_before_scoring() -> None:
    reference = np.array([[-20, 20], [120, 20], [120, 80], [-20, 80]], dtype=np.float32)
    predicted = np.array([[0, 20], [100, 20], [100, 80], [0, 80]], dtype=np.float32)
    assert clipped_iou(predicted, reference, [100, 100]) == pytest.approx(1)
    assert score_corners(predicted, reference, [100, 100])["corner_error_px"] is None
    assert clipped_iou(predicted + [200, 0], reference, [100, 100]) == 0
    assert clipped_iou(predicted + [50, 0], predicted, [100, 100]) == pytest.approx(0.5)


def test_only_visible_true_corners_count_but_predictions_can_be_offscreen() -> None:
    reference = np.array([[20, 20], [80, 20], [120, 120], [-20, 120]], dtype=np.float32)
    predicted = np.array([[20, -10], [80, -10], [400, 300], [-300, 300]], dtype=np.float32)
    result = score_corners(predicted, reference, [100, 100])
    assert result["visible_reference_corners"] == 2
    assert result["corner_error_px"] == pytest.approx(30)


def test_missing_predictions_stay_in_iou_denominator() -> None:
    reference = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)
    records = []
    for corners in (reference, None):
        records.append({"video_id": 8, "corners": None if corners is None else corners.tolist(),
                        **score_corners(corners, reference, [100, 100])})
    summary = summarise(records)
    assert summary["missing_predictions"] == 1
    assert summary["corner_error_px_present"]["count"] == 1
    assert summary["iou_present"]["mean"] == 1
    assert summary["iou_all_rallies_missing_zero"]["mean"] == 0.5


def test_video_midpoint_and_empty_inference_record(tmp_path: Path) -> None:
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 72))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((72, 128, 3), index * 20, np.uint8))
    writer.release()

    class EmptyProcessor:
        def set_image(self, image: object) -> dict:
            # [2, 8) chooses zero-based frame 5, including after real video seeking.
            assert np.asarray(image).mean() == pytest.approx(100, abs=2)
            return {}

        def set_text_prompt(self, state: dict, prompt: str) -> dict:
            assert prompt == "court markings"
            return {"scores": torch.zeros(0), "masks": torch.zeros((0, 1, 72, 128), dtype=torch.bool)}

    item = {"video_id": 1, "source_filename": path.name, "resolution": [128, 72], "frame_count": 10,
            "spans": [{"rally_id": "set1:1", "start_frame": 2, "end_frame": 8}]}
    reference = {"upleft_x": 200, "upleft_y": 200, "upright_x": 800, "upright_y": 200,
                 "downleft_x": 800, "downleft_y": 600, "downright_x": 200, "downright_y": 600}
    records = []
    evaluate_video(item, reference, tmp_path, EmptyProcessor(), records, tmp_path, 1, time.monotonic())
    assert len(records) == 1
    assert records[0]["midpoint_frame"] == 5
    assert records[0]["corners"] is None
    assert records[0]["iou"] == 0
    assert records[0]["visible_reference_corners"] == 4


def test_cropped_near_coincident_edges_keep_their_overlap() -> None:
    # BkjErIAsZu4 frame 150: OpenCV's clipped-polygon intersection lost most
    # of the overlap where both outlines share a segment on the image's left edge.
    predicted = np.array([[1036.4225, 416.54572], [1493.1888, 438.1038],
                          [1159.5358, 892.93146], [-89.520134, 579.4103]])
    reference = np.array([[1047.8158, 422.58163], [1486.8711, 444.2102],
                          [1159.5, 895.0], [-82.16182, 577.76337]])
    for outline in (predicted, predicted[::-1]):
        assert clipped_iou(outline, reference, [1920, 1080]) == pytest.approx(0.9776259, abs=1e-6)
        assert clipped_iou(reference, outline, [1920, 1080]) == pytest.approx(0.9776259, abs=1e-6)


def test_iou_touching_and_coincident_boundaries() -> None:
    reference = np.array([[-10, 10], [50, 10], [50, 70], [-10, 70]], dtype=np.float64)
    touching = np.array([[50, 10], [80, 10], [80, 70], [50, 70]], dtype=np.float64)
    assert clipped_iou(touching, reference, [100, 100]) == 0
    assert clipped_iou(reference[::-1], reference, [100, 100]) == pytest.approx(1)
    nearly_same = reference + [1e-9, 0]
    assert clipped_iou(nearly_same, reference, [100, 100]) == pytest.approx(1)


def test_reference_corner_on_image_edge_does_not_create_a_spurious_edge() -> None:
    reference = np.array([[0, 316.80306872], [1725.89053849, 299.14579011],
                          [1302.50535449, 898.08560575], [-236.34239394, 933.05765906]])
    for outline in (reference, reference[::-1]):
        assert clipped_iou(outline, reference, [1920, 1080]) == pytest.approx(1)


def test_corner_errors_normalise_to_1080_pixel_height() -> None:
    reference = np.array([[200, 100], [800, 100], [900, 600], [100, 600]], dtype=np.float32)
    result = score_corners(reference + [3, 4], reference, [960, 720])
    assert result["corner_error_px"] == pytest.approx(5)
    assert result["corner_error_1080p_px"] == pytest.approx(7.5)
    full_size = score_corners((reference + [3, 4]) * 1.5, reference * 1.5, [1440, 1080])
    assert full_size["corner_error_1080p_px"] == pytest.approx(result["corner_error_1080p_px"])
