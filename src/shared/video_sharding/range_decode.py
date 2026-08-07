"""Decode an exact half-open frame range from a video with cv2.

Two modes:

- ``seek``: ``CAP_PROP_POS_FRAMES`` to ``start``, then read sequentially. This
  is the candidate production mechanism; its frame accuracy on our codecs is
  exactly what gate_decode_identity establishes, so nothing here assumes it.
- ``scan``: decode from frame 0 and discard until ``start``. Slow, but by
  construction identical to a sequential decode; used as the correctness
  control and as the fallback if seek identity fails.

Yielding stops early (without error) if the stream ends before ``end``; the
caller compares yielded count against the requested span and decides loudness.
That keeps this module a pure decoder and puts short-read policy in one place
(the shard worker).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def open_capture(video_path: Path | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open video: {video_path}")
    return cap


def metadata_frame_count(video_path: Path | str) -> int:
    """The container's reported frame count (CAP_PROP_FRAME_COUNT).

    Metadata, not a decode: may disagree with the decodable frame count on a
    damaged file. The last shard worker's end-of-range probe catches an
    undercount; a short read catches an overcount.
    """
    cap = open_capture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def iter_frame_range(
    video_path: Path | str,
    start: int,
    end: int,
    mode: str = "seek",
) -> Iterator[np.ndarray]:
    """Yield BGR frames for logical frames ``[start, end)`` of ``video_path``."""
    if not 0 <= start < end:
        raise ValueError(f"bad range [{start}, {end})")
    if mode not in ("seek", "scan"):
        raise ValueError(f"mode must be 'seek' or 'scan', got {mode!r}")

    cap = open_capture(video_path)
    try:
        if start > 0:
            if mode == "seek":
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, start):
                    raise RuntimeError(f"seek to frame {start} failed: {video_path}")
            else:
                for _ in range(start):
                    ok, _ = cap.read()
                    if not ok:
                        return
        for _ in range(end - start):
            ok, frame = cap.read()
            if not ok:
                return
            yield frame
    finally:
        cap.release()


def md5_frame(frame_bgr: np.ndarray) -> str:
    """MD5 of the decoded frame bytes. MD5 (not equality) because identity is
    compared across hosts; hex digests travel, 6 MB frames do not."""
    return hashlib.md5(frame_bgr.tobytes()).hexdigest()


def md5_file(path: Path | str, chunk_bytes: int = 1 << 22) -> str:
    """Streaming MD5 of a file; identifies the source video in shard manifests."""
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
