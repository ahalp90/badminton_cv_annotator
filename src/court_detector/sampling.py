"""Reuse grey-image samples while measuring and scoring one view."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import cv2
import numpy as np

# cv2.remap rejects maps with 32,767 (SHRT_MAX) or more points along one side.
REMAP_MAX_POINTS = 32_766


def remap_points(image: np.ndarray, xy: np.ndarray, border_mode: int) -> np.ndarray:
    """Sample an image at many points by bilinear interpolation, in pieces cv2.remap accepts.

    A request under the cap is a single cv2.remap call, so its samples match an unsplit call exactly.

    :param image: The image to sample, with one channel or several.
    :param xy: float32 sample positions in pixels, one row per point, shape (points, 2).
    :param border_mode: The cv2 border mode for positions outside the image.
    :return: Samples shaped (1, points) or (1, points, channels), as one cv2.remap call returns them.
    """
    if len(xy) <= REMAP_MAX_POINTS:
        return cv2.remap(image, xy[:, 0], xy[:, 1], cv2.INTER_LINEAR, borderMode=border_mode)
    pieces = np.split(xy, range(REMAP_MAX_POINTS, len(xy), REMAP_MAX_POINTS))
    # Every piece is under the cap, so each takes the single-call branch above.
    return np.concatenate([remap_points(image, piece, border_mode) for piece in pieces], axis=1)


@contextmanager
def prepared_measurements(verifier: ModuleType) -> Iterator[dict[str, int]]:
    """Reuse greyscale while a worker measures one immutable view at a time.

    :param verifier: The measurements module used by this worker.
    :return: Counts of image conversions and sampling calls for equality checks.
    """
    original_sample = verifier.grayscale_sample
    original_junctions = verifier.raw_junctions
    cached_image = None
    grey = None
    counts = {"greyscale_conversions": 0, "sampling_calls": 0}

    def sample(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        nonlocal cached_image, grey
        if image is not cached_image:
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cached_image = image
            counts["greyscale_conversions"] += 1
        counts["sampling_calls"] += 1
        maps = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        values = remap_points(grey, maps, cv2.BORDER_REPLICATE)
        return values.reshape(np.asarray(points).shape[:-1])

    verifier.grayscale_sample = sample
    verifier.raw_junctions = lambda *_args: {"status": "omitted_unused_diagnostic"}
    try:
        yield counts
    finally:
        verifier.grayscale_sample = original_sample
        verifier.raw_junctions = original_junctions
