"""Deterministic fake pose extractor for parity experiments.

Duck-types the two ``RtmlibPoseExtractor`` surfaces the extraction paths use
(``detect_frame`` and ``iter_video``) so the same production assembly code
(``raw_extract.extract_raw_frame`` / ``extract_one_clip``) runs unchanged.
The detections tuple is likewise a duck-typed stand-in for
``rtmlib_pose.FrameDetections`` (same four fields) — importing the real one
would drag in rtmlib, which this fake exists to avoid; same pattern as
``tests/test_raw_schema.py``.

Every output value is a pure function of the frame's pixel content (via its
MD5), so sequential and sharded runs agree exactly if and only if they decoded
identical frames in identical order. Detection counts cycle through 0..3,
which exercises empty frames and NaN padding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
from pipeline.config import COCO_N_JOINTS


class FakeFrameDetections(NamedTuple):
    """Field-for-field stand-in for ``rtmlib_pose.FrameDetections``."""

    keypoints: np.ndarray    # (n_people, J, 2) float32
    bboxes: np.ndarray       # (n_people, 4) float32
    bbox_scores: np.ndarray  # (n_people,) float32
    kp_scores: np.ndarray    # (n_people, J) float32


class DeterministicFakeExtractor:
    """Frame-content-derived detections; no models, no onnxruntime."""

    def detect_frame(self, frame_bgr: np.ndarray) -> FakeFrameDetections:
        digest = hashlib.md5(frame_bgr.tobytes()).digest()
        n_people = digest[0] % 4  # 0..3: exercises empty frames and padding
        # 16 digest bytes seeded into a deterministic per-frame value stream.
        seeds = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        values = np.arange(n_people * COCO_N_JOINTS * 2, dtype=np.float32)
        keypoints = (values.reshape(n_people, COCO_N_JOINTS, 2) + seeds[0]) * (1 + seeds[1] / 255)
        bboxes = np.tile(seeds[2:6], (n_people, 1)) + values[:n_people, None] if n_people else np.empty((0, 4), np.float32)
        bbox_scores = (seeds[6:6 + n_people] + 1) / 300 if n_people else np.empty((0,), np.float32)
        joint_seeds = np.resize(seeds, COCO_N_JOINTS)  # cycle the 16 digest bytes to J values
        kp_scores = np.tile(joint_seeds / 255, (n_people, 1))
        return FakeFrameDetections(
            keypoints=keypoints.astype(np.float32),
            bboxes=bboxes.astype(np.float32),
            bbox_scores=bbox_scores.astype(np.float32),
            kp_scores=kp_scores.astype(np.float32),
        )

    def iter_video(self, video_path: Path | str) -> Iterator[FakeFrameDetections]:
        """Sequential whole-video iteration, mirroring the production adapter."""
        cap = cv2.VideoCapture(str(video_path))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.detect_frame(frame)
        finally:
            cap.release()
