"""Reuse grey-image samples while measuring and scoring one view."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType

import cv2
import numpy as np


@contextmanager
def prepared_measurements(verifier: ModuleType) -> Iterator[dict[str, int]]:
    """Reuse greyscale while a worker measures one immutable view at a time.

    :param verifier: The frozen W5 verifier module used by this worker.
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
        values = cv2.remap(grey, maps[:, 0], maps[:, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return values.reshape(np.asarray(points).shape[:-1])

    verifier.grayscale_sample = sample
    verifier.raw_junctions = lambda *_args: {"status": "omitted_unused_diagnostic"}
    try:
        yield counts
    finally:
        verifier.grayscale_sample = original_sample
        verifier.raw_junctions = original_junctions
