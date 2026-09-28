"""Read source-video frames and frame-aligned people for the court detector."""

from __future__ import annotations

import lzma
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Self

import cv2
import numpy as np

from .inputs import FrameReader, PersonSample

if TYPE_CHECKING:
    from shared.rtmlib_pose import RtmlibPoseExtractor

FRAME_CACHE_SIZE = 64  # Enough for two 31-frame foot windows; images stay at native size.


class VideoFrames:
    """Seek to requested windows, then decode nearby frames in order.

    Cache only the requested frames. FFmpeg uses one decoding thread. Source
    codecs must pass a seek-versus-sequential identity check before timing a new
    dataset; container frame positions alone do not establish accurate seeking.
    """

    def __init__(self, path: Path) -> None:
        self.capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, (cv2.CAP_PROP_N_THREADS, 1))
        if not self.capture.isOpened():
            raise OSError(f"Cannot open video: {path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(self.fps) or self.fps <= 0 or self.frame_count <= 0 or min(self.size) <= 0:
            self.capture.release()
            raise ValueError(f"Invalid video metadata: {path}")
        self.next_frame = 0
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        """Return native BGR frames in the requested order, including duplicates."""
        if any(index < 0 or index >= self.frame_count for index in frame_indices):
            raise IndexError(f"Frame indices must be in 0..{self.frame_count - 1}")
        requested = {}
        for index in sorted(set(frame_indices)):
            if index in self.cache:
                self.cache.move_to_end(index)
                requested[index] = self.cache[index]
                continue
            if index < self.next_frame or index - self.next_frame > self.fps:
                if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise OSError(f"Video seek to frame {index} failed")
                self.next_frame = index
            while self.next_frame < index:
                if not self.capture.grab():
                    raise OSError(f"Video ended before frame {index}")
                self.next_frame += 1
            success, frame = self.capture.read()
            if not success:
                raise OSError(f"Video could not decode frame {index}")
            self.next_frame += 1
            frame.flags.writeable = False
            requested[index] = frame
            self.cache[index] = frame
            if len(self.cache) > FRAME_CACHE_SIZE:
                self.cache.popitem(last=False)
        return [requested[index] for index in frame_indices]

    def close(self) -> None:
        self.capture.release()
        self.cache.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PoseArrays:
    """Saved RTMLib arrays indexed by source frame, in native source-image pixels.

    Padded slots are excluded with ndet. The arrays must describe the same video
    and coordinate size as FrameReader; this class does not infer a resize/crop.
    """

    def __init__(self, boxes: np.ndarray, keypoints: np.ndarray, counts: np.ndarray) -> None:
        if boxes.ndim != 3 or boxes.shape[2] != 4:
            raise ValueError("Pose boxes must have shape (frames, slots, 4)")
        frames, slots = boxes.shape[:2]
        if keypoints.shape != (frames, slots, 17, 2) or counts.shape != (frames,):
            raise ValueError("Pose boxes, COCO-17 keypoints and counts must share the frame/slot dimensions")
        if not np.issubdtype(counts.dtype, np.integer) or np.any((counts < 0) | (counts > slots)):
            raise ValueError("Pose detection counts must be integers within the available slots")
        self.boxes, self.keypoints, self.counts = boxes, keypoints, counts
        self.frame_count = frames

    @classmethod
    def from_directory(cls, directory: Path) -> PoseArrays:
        arrays = []
        for name in ('bboxes', 'kps', 'ndet'):
            with lzma.open(directory / f'pose_{name}.npy.xz', 'rb') as stream:
                arrays.append(np.load(stream, allow_pickle=False))
        return cls(*arrays)

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        if any(index < 0 or index >= self.frame_count for index in frame_indices):
            raise IndexError(f"Pose frame indices must be in 0..{self.frame_count - 1}")
        return [PersonSample(index, self.boxes[index, :self.counts[index]], self.keypoints[index, :self.counts[index]])
                for index in frame_indices]


class RtmlibPeople:
    """Run the shared person/pose extractor once per requested frame of one video.

    The cache is keyed by this video's frame numbers, so each video needs its own
    RtmlibPeople. The extractor holds the loaded models; pass the same one to every
    video so a batch loads them once.
    """

    def __init__(self, frames: FrameReader, extractor: RtmlibPoseExtractor) -> None:
        self.frames = frames
        self.extractor = extractor
        self.cache: dict[int, PersonSample] = {}

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        missing = sorted(set(frame_indices) - self.cache.keys())
        for index, frame in zip(missing, self.frames.read(missing), strict=True):
            detected = self.extractor.detect_frame(frame)
            self.cache[index] = PersonSample(index, detected.bboxes, detected.keypoints)
        return [self.cache[index] for index in frame_indices]
