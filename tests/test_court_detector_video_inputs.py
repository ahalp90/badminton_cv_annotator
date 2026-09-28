"""Source frame identity and padded pose slots at the video-input boundary."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from court_detector.video_inputs import (
    FRAME_CACHE_SIZE,
    PoseArrays,
    RtmlibPeople,
    VideoFrames,
)


def test_seek_cache_and_request_order_match_sequential_decode(tmp_path: Path) -> None:
    path = tmp_path / 'frames.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 25, (64, 48))
    assert writer.isOpened()
    for index in range(120):
        image = np.full((48, 64, 3), index, dtype=np.uint8)
        image[:, index % 64] = 255 - index
        writer.write(image)
    writer.release()
    capture = cv2.VideoCapture(str(path))
    expected = []
    while True:
        success, image = capture.read()
        if not success:
            break
        expected.append(image)
    capture.release()
    assert len(expected) == 120
    with VideoFrames(path) as frames:
        assert (frames.frame_count, frames.fps, frames.size) == (120, 25, (64, 48))
        for indices in ([98, 96, 97, 96], [0, 2, 7], list(range(80)), [119, 1, 119]):
            actual = frames.read(indices)
            for index, image in zip(indices, actual, strict=True):
                np.testing.assert_array_equal(image, expected[index])
                assert not image.flags.writeable
            assert len(frames.cache) <= FRAME_CACHE_SIZE
        with pytest.raises(IndexError):
            frames.read([-1])
        with pytest.raises(IndexError):
            frames.read([120])
        assert frames.read([]) == []
    assert not frames.capture.isOpened()


def test_pose_counts_exclude_padding_and_keep_source_pixels() -> None:
    boxes = np.full((3, 2, 4), np.nan, dtype=np.float32)
    keypoints = np.full((3, 2, 17, 2), np.nan, dtype=np.float32)
    boxes[1, 0] = [20, 30, 70, 120]
    keypoints[1, 0] = [40, 100]
    poses = PoseArrays(boxes, keypoints, np.array([0, 1, 0], dtype=np.int8))
    samples = poses.samples([2, 1, 1, 0])
    assert [sample.frame_index for sample in samples] == [2, 1, 1, 0]
    assert samples[0].boxes_px.shape == (0, 4)
    assert samples[3].keypoints_px.shape == (0, 17, 2)
    np.testing.assert_array_equal(samples[1].boxes_px, boxes[1, :1])
    np.testing.assert_array_equal(samples[2].keypoints_px, keypoints[1, :1])
    with pytest.raises(IndexError):
        poses.samples([-1])
    with pytest.raises(ValueError, match='counts'):
        PoseArrays(boxes, keypoints, np.array([0, 3, 0]))
    with pytest.raises(ValueError, match='dimensions'):
        PoseArrays(boxes, keypoints[:2], np.array([0, 1, 0]))


def test_live_people_share_one_extractor_but_cache_each_video() -> None:
    class LabelledFrames:
        """Frames whose first two pixel channels hold their video and frame number."""

        def __init__(self, video: int) -> None:
            self.video = video

        def read(self, indices):
            return [np.full((2, 2, 3), (self.video, index, 0), dtype=np.uint8) for index in indices]

    class RecordingExtractor:
        def __init__(self) -> None:
            self.detected: list[tuple[int, int]] = []  # (video, frame) per detect_frame call

        def detect_frame(self, frame):
            video, index = int(frame[0, 0, 0]), int(frame[0, 0, 1])
            self.detected.append((video, index))
            return SimpleNamespace(bboxes=np.array([[video, index, video + 1, index + 1]], dtype=np.float32),
                                   keypoints=np.zeros((1, 17, 2), dtype=np.float32))

    extractor = RecordingExtractor()
    first_video = RtmlibPeople(LabelledFrames(1), extractor)  # type: ignore[arg-type]
    second_video = RtmlibPeople(LabelledFrames(2), extractor)  # type: ignore[arg-type]
    first_samples = first_video.samples([5, 3, 5])
    second_samples = second_video.samples([5])
    first_video.samples([3])
    assert [sample.frame_index for sample in first_samples] == [5, 3, 5]
    # Frame 5 of each video gets its own detection; repeated frames reuse the video's cache.
    assert (first_samples[0].boxes_px[0, 0], second_samples[0].boxes_px[0, 0]) == (1, 2)
    assert extractor.detected == [(1, 3), (1, 5), (2, 5)]
