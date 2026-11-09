"""Small OpenCV helpers shared by the two manual review tools."""
from __future__ import annotations

import math

import cv2
import numpy as np

from labelling.scene_labels import SceneTruth, VideoMetadata

# OpenCV uses BGR; labels distinguish classes as well as these protan-safe colours.
TRUTH_COLOURS = {
    SceneTruth.LIVE: (178, 114, 0),
    SceneTruth.LIVE_NON_STANDARD: (233, 180, 86),
    SceneTruth.REPLAY: (167, 121, 204),
    SceneTruth.CUTAWAY: (0, 159, 230),
    SceneTruth.OTHER: (160, 160, 160),
}


def video_metadata(capture: cv2.VideoCapture, video_id: str) -> VideoMetadata:
    """Read and validate the metadata used by every output row."""
    if not capture.isOpened():
        raise ValueError("could not open source video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(frame_count_raw) or frame_count_raw <= 0:
        raise ValueError(f"video frame count is invalid: {frame_count_raw!r}")
    frame_count = round(frame_count_raw)
    if not math.isclose(frame_count_raw, frame_count, abs_tol=1e-6):
        raise ValueError(f"video frame count is not integral: {frame_count_raw!r}")
    return VideoMetadata(video_id, fps, frame_count)


def timeline_x(frame: int, width: int, start: int, end: int) -> int:
    fraction = (frame - start) / (end - start)
    return round(np.clip(fraction, 0.0, 1.0) * (width - 1))


def read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    if not ok or image is None:
        raise RuntimeError(f"could not decode frame {frame}")
    return image
