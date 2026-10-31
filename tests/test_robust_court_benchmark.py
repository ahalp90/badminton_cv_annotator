"""Check sparse pose reuse and the existing three-frame detector boundary."""

import lzma
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from court_detector.feet import window_frames
from court_detector.inputs import PersonSample
from experiments.court_detector.benchmark.robust_courts import (
    BenchmarkFrames,
    FilePosePeople,
    ScaledPeople,
    SparsePeople,
    evaluate_anchor,
    run,
)
from experiments.court_detector.benchmark.sam3.benchmark_rallies import write_json


class FakePeople:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        self.calls.extend(frame_indices)
        return [PersonSample(index, np.empty((0, 4)), np.empty((0, 17, 2))) for index in frame_indices]


class FakeFrames:
    fps = 30.0
    size = (128, 72)
    frame_count = 240

    def __init__(self) -> None:
        self.calls: list[int] = []

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        self.calls.extend(frame_indices)
        return [np.full((72, 128, 3), index, np.uint8) for index in frame_indices]


@pytest.mark.parametrize("fps,anchor,expected_endpoints",
                         [(30, 120, [75, 165]), (30, 0, [45, 90]), (59.885, 5, [95, 185])])
def test_anchor_uses_existing_window_and_endpoint_composition(
    tmp_path: Path, fps: float, anchor: int, expected_endpoints: list[int],
) -> None:
    frames = FakeFrames()
    frames.fps = fps
    source = FakePeople()
    people = SparsePeople(source, tmp_path / "poses.json.gz")
    window = window_frames(anchor, frames.fps, 0, 240)
    line_frames = []

    def segments(frame: np.ndarray, frame_index: int) -> np.ndarray:
        assert frame[0, 0, 0] == frame_index
        line_frames.append(frame_index)
        return np.empty((0, 4), np.float32)

    def detect(view: Any, poses: Any, reader: Any, *, endpoint_views: Any) -> Any:
        assert view.frame_index == anchor
        assert view.scene_frames == (0, 240)
        assert reader is frames
        poses.samples(window)
        endpoints = endpoint_views()
        assert [endpoint.frame_index for endpoint in endpoints] == expected_endpoints
        return SimpleNamespace(corners_native_px=np.array([[20., 20.], [80., 20.], [90., 60.], [10., 60.]]),
                               composition={"court": "composite"}, no_court_reason=None, chosen_key="composite")

    tools: Any = SimpleNamespace(lines=SimpleNamespace(segments=segments), detector=SimpleNamespace(detect=detect))
    result = evaluate_anchor(f"label_f{anchor}", anchor, 240, frames, people, tools)
    assert result["status"] == "court"
    assert result["composition"]["court"] == "composite"
    assert result["pose_window"] == window
    assert result["pose_frames"] == window
    assert frames.calls == window
    assert line_frames == [anchor, *expected_endpoints]
    assert result["court_frames"] == [anchor, *expected_endpoints]
    assert len(set(result["court_frames"])) == 3
    assert len(source.calls) == len(set(source.calls)) == 31

    restored = SparsePeople(source, tmp_path / "poses.json.gz")
    restored.samples(window)
    assert len(source.calls) == 31


@pytest.mark.parametrize("fps,count,anchor", [(60000 / 1001, 181, 14), (30000 / 1001, 91, 58), (24, 73, 64)])
def test_short_clip_keeps_committed_sampling_failure(tmp_path: Path, fps: float, count: int, anchor: int) -> None:
    frames = FakeFrames()
    frames.fps = fps
    source = FakePeople()
    people = SparsePeople(source, tmp_path / "poses.json.gz")
    unavailable_tools: Any = None
    result = evaluate_anchor("edge", anchor, count, frames, people, unavailable_tools)
    assert result["status"] == "scene_too_short_for_feet"
    assert result["corners"] is None
    assert result["court_frames"] == []
    assert result["pose_frames"] == []
    assert frames.calls == source.calls == []


def test_saved_poses_keep_source_frame_numbers() -> None:
    source = FakePeople()
    samples = ScaledPeople(source, np.ones(2)).samples([3, 8])
    assert source.calls == [3, 8]
    assert [sample.frame_index for sample in samples] == [3, 8]


def test_complete_checkpoint_skips_model_loading(tmp_path: Path) -> None:
    write_json(tmp_path / "inputs.json.gz", [{"image_key": "done"}])
    write_json(tmp_path / "ours_robust.json.gz", {"done": {"status": "court", "corners": []}})
    run(tmp_path, tmp_path / "missing-map", tmp_path / "missing-source", tmp_path / "missing-weights")


def test_saved_file_poses_scale_and_only_missing_frames_infer(tmp_path: Path) -> None:
    packed = np.concatenate([np.array([[10, 20, 30, 40]]), np.tile([[10, 20]], (1, 17))], axis=1)
    with lzma.open(tmp_path / "frame_000000003.npy.xz", "wb") as stream:
        np.save(stream, packed, allow_pickle=False)
    fallback = FakePeople()
    people = FilePosePeople(tmp_path, fallback, np.array([0.5, 0.25]))
    samples = people.samples([3, 8])
    assert fallback.calls == [8]
    np.testing.assert_allclose(samples[0].boxes_px, [[5, 5, 15, 10]])
    np.testing.assert_allclose(samples[0].keypoints_px, np.tile([[[5, 5]]], (1, 17, 1)))
    assert [sample.frame_index for sample in samples] == [3, 8]


def test_full_source_is_resized_once_per_requested_frame() -> None:
    source: Any = FakeFrames()
    frames = BenchmarkFrames(source, (64, 36))
    first = frames.read([10, 11])
    second = frames.read([11])
    assert frames.source_size == (128, 72)
    assert frames.size == (64, 36)
    assert first[0].shape == (36, 64, 3)
    assert second[0] is first[1]
    assert source.calls == [10, 11]
