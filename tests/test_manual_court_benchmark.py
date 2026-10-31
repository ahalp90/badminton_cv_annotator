"""Check manual-reference averaging and per-frame comparison accounting."""

import csv
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from experiments.court_detector.benchmark.manual_courts import (
    ANNOTATIONS,
    prepare,
    read_json,
    requested_frames,
    score,
)
from experiments.court_detector.benchmark.sam3.benchmark_rallies import write_json


@pytest.mark.parametrize("midpoint_labelled", [False, True])
def test_manual_frames_and_video_mean_remain_separate(tmp_path: Path, midpoint_labelled: bool) -> None:
    video = tmp_path / "fixed.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (128, 72))
    assert writer.isOpened()
    for index in range(10):
        writer.write(np.full((72, 128, 3), index * 20, np.uint8))
    writer.release()
    labels = tmp_path / "labels"
    for name in ANNOTATIONS:
        path = labels / name
        path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "wt") as stream:
            table = csv.writer(stream)
            table.writerow(["video", "frame", "corner_idx", "x_norm", "y_norm"])
            if name == ANNOTATIONS[0]:
                for frame_index, offset in [(1, 0), (5 if midpoint_labelled else 7, 0.05)]:
                    for corner, (x, y) in enumerate([(0.1, 0.2), (0.8, 0.2), (0.8, 0.8), (0.1, 0.8)]):
                        table.writerow([video.name, frame_index, corner, x + offset, y])
    mapping = tmp_path / "paths.json"
    mapping.write_text(json.dumps({"fixed": str(video)}))
    (labels / "additional_examples").mkdir()
    write_json(labels / "additional_examples" / "manifest.json.gz", {"cases": []})
    output = tmp_path / "results"
    prepare(labels, mapping, output, False)
    records = read_json(output / "inputs.json.gz")
    assert len(records) == 3
    midpoint = records[-1]
    assert midpoint["frame_index"] == 5
    assert midpoint["reference_type"] == ("labelled_midpoint" if midpoint_labelled else "video_mean_midpoint")
    assert midpoint["reference_corners"][0] == pytest.approx([19.2 if midpoint_labelled else 16, 14.4])
    predictions = {row["image_key"]: {"corners": row["reference_corners"], "status": "court"} for row in records}
    for source in ("ours", "monotrack", "sam3"):
        write_json(output / f"{source}.json.gz", predictions)
    score(output)
    summary = read_json(output / "comparison_summary.json.gz")
    assert summary["labelled_frame"]["ours"]["frames"] == 2
    assert summary["midpoint"]["ours"]["frames"] == 1
    assert summary["midpoint_reference_counts"]["labelled_midpoint"] == int(midpoint_labelled)
    assert summary["labelled_frame"]["sam3"]["iou_present"]["mean"] == pytest.approx(1)


def test_downloaded_clip_uses_decoded_frame_numbers() -> None:
    path = Path("data/amateur_court_corners/additional_examples/Cb-xs5rPyxI_gameplay.mp4")
    capture = cv2.VideoCapture(str(path))
    try:
        frames = requested_frames(capture, {45, 58, 78}, sequential=True)
    finally:
        capture.release()
    assert set(frames) == {45, 58, 78}
    assert frames[45].shape[:2] == (720, 960)


def test_monotrack_reuses_labels_and_resumes_new_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments.court_detector.benchmark import manual_courts

    write_json(tmp_path / "inputs.json.gz", [{"image_key": key} for key in ("labelled", "finished", "new1", "new2")])
    reused = {"corners": None, "status": "fit_failed"}
    fresh = {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "status": "court"}
    baseline = tmp_path / "baseline.json.gz"
    write_json(baseline, {"labelled": reused, "finished": reused, "unrelated": reused})
    write_json(tmp_path / "monotrack.json.gz", {"finished": fresh})
    calls = []

    def fake_image(binary: Path, path: Path, raw: Path) -> dict:
        calls.append((path.stem, raw.name))
        return fresh

    monkeypatch.setattr(manual_courts, "monotrack_image", fake_image)
    manual_courts.run_monotrack(tmp_path, tmp_path / "binary", baseline, workers=2)
    records = read_json(tmp_path / "monotrack.json.gz")
    assert set(records) == {"labelled", "finished", "new1", "new2"}
    assert records["labelled"] == reused
    assert records["finished"] == fresh
    assert sorted(calls) == [("new1", "new1_landmarks.txt"), ("new2", "new2_landmarks.txt")]
    manual_courts.run_monotrack(tmp_path, tmp_path / "binary", baseline)
    assert len(calls) == 2


def test_completed_models_do_not_reload(tmp_path: Path) -> None:
    from experiments.court_detector.benchmark.manual_courts import run_ours, run_sam

    write_json(tmp_path / "inputs.json.gz", [{"image_key": "done"}])
    for source in ("ours", "sam3"):
        write_json(tmp_path / f"{source}.json.gz", {"done": {"corners": None, "status": "no_court"}})
    run_ours(tmp_path, tmp_path / "missing-source", tmp_path / "missing-weights")
    run_sam(tmp_path)


def test_main_robust_scores_preserve_the_single_image_comparison(tmp_path: Path) -> None:
    reference = [[100, 100], [800, 100], [800, 600], [100, 600]]
    row = {"video_id": "example", "image_key": "example_f1000", "image_size": [960, 720],
           "reference_type": "labelled_frame", "reference_corners": reference}
    write_json(tmp_path / "inputs.json.gz", [row])
    image_only = {row["image_key"]: {"corners": (np.array(reference) + [60, 0]).tolist(), "status": "court"}}
    for source in ("ours", "monotrack", "sam3"):
        write_json(tmp_path / f"{source}.json.gz", image_only)
    robust = {row["image_key"]: {"corners": reference, "status": "court", "source_frame_index": 1000, "court_frames": [1000, 955, 1045],
                               "pose_window": list(range(955, 1046, 3)), "pose_frames": [997, 1000, 1003],
                               "composition": {"court": "composite"}}}
    write_json(tmp_path / "ours_robust.json.gz", robust)
    score(tmp_path, result_prefix="image_only")
    score(tmp_path, ours_source="ours_robust")
    original = read_json(tmp_path / "image_only_frames.json.gz")[0]
    main = read_json(tmp_path / "comparison_frames.json.gz")[0]
    assert original["our_detector_mode"] == "image_only"
    assert original["detectors"]["ours"]["corner_error_1080p_px"] == pytest.approx(90)
    assert main["our_detector_mode"] == "scene_robust"
    assert main["detectors"]["ours"]["corner_error_1080p_px"] == 0
    assert main["detectors"]["ours"]["sampling"]["selected_fit"] == "composite"
    assert read_json(tmp_path / "ours.json.gz") == image_only
