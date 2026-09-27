"""Where the court detector's line fragments come from: saved extracts or a live DeepLSD model.

Both sources return (fragments, 4) finite float64 x1, y1, x2, y2 in the frame's
native pixels, the layout `ViewInputs.segments_px` expects. Only `DeepLSDLines`
needs torch, the DeepLSD checkout and the research exporter. It imports them
when it is built, so this module and `SavedLines` need none of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt


class LineSource(Protocol):
    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """(fragments, 4) finite float64 x1, y1, x2, y2 in the native pixels of this BGR frame."""
        ...


def _checked_segments(segments: npt.ArrayLike) -> np.ndarray:
    """Saved fragments as a (fragments, 4) float64 array; an empty extract becomes zero rows."""
    array = np.asarray(segments, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError(f"expected finite (fragments, 4) line segments, got shape {array.shape}")
    return array


def _working_grey(frame: np.ndarray, max_dimension: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Shrink a BGR frame to at most `max_dimension` on its longest side, then make it greyscale.

    This repeats the saved export's resize with a configurable size. Rounding
    each side to whole pixels makes the x and y scale factors differ slightly.

    :param frame: (height, width, 3) BGR uint8 frame at native size.
    :param max_dimension: longest side of the working image; smaller frames keep their size.
    :return: (working height, working width) uint8 greyscale image, and its (width, height).
    """
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise TypeError(f"expected a BGR uint8 frame, got {frame.dtype} with shape {frame.shape}")
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(width, height))
    working_width, working_height = round(width * scale), round(height * scale)
    if scale < 1.0:
        frame = cv2.resize(frame, (working_width, working_height), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (working_width, working_height)


class SavedLines:
    """Fragments already extracted for known video frames, such as a saved DeepLSD export."""

    def __init__(self, segments_by_frame: Mapping[int, npt.ArrayLike]) -> None:
        """:param segments_by_frame: native-pixel x1, y1, x2, y2 fragments keyed by video frame index."""
        self._segments_by_frame = {frame_index: _checked_segments(segments)
                                   for frame_index, segments in segments_by_frame.items()}

    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """The saved fragments for this frame. A frame without a saved extract raises KeyError."""
        return self._segments_by_frame[frame_index]


class DeepLSDLines:
    """Fragments from a DeepLSD model, extracted the way the saved line inputs were made.

    The exporter's helpers load the network once and compute its distance and
    angle fields. `detect_afm_lines` then uses the exporter's settings.
    Gradient validation defaults on, matching the saved amateur example.
    Set `grad_nfa=False` to request the less selective hard variant. Loading puts
    the DeepLSD checkout first on `sys.path`, because DeepLSD is imported from
    that checkout rather than from an installed package.
    """

    def __init__(
        self,
        source: Path,
        weights: Path,
        *,
        device: str = "cuda",
        max_dimension: int = 960,
        grad_nfa: bool = True,
    ) -> None:
        """Load the network once for every later frame.

        :param source: DeepLSD source checkout.
        :param weights: checkpoint file holding the network's `model` state.
        :param device: torch device; a CUDA request fails when CUDA is unavailable.
        :param max_dimension: longest side of the image the network sees.
        :param grad_nfa: DeepLSD's gradient-based line validation; on by default, off for the hard variant.
        """
        # A missing checkout would let the import fall through to any installed deeplsd.
        if not source.is_dir():
            raise FileNotFoundError(f"DeepLSD source checkout does not exist: {source}")
        # The core package must not import the research tree or torch, so both load
        # here. The exporter module itself imports only the standard library.
        import torch

        from experiments.annotator.independent_court import export_lines

        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} was requested but CUDA is unavailable")
        self._network = export_lines._load_deeplsd(source, weights, self._device)
        self._max_dimension = max_dimension
        self._grad_nfa = grad_nfa

    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """(fragments, 4) x1, y1, x2, y2 in native pixels. `frame_index` is unused."""
        from experiments.annotator.independent_court import export_lines

        grey, (working_width, working_height) = _working_grey(frame, self._max_dimension)
        distance_field, angle_field = export_lines._deeplsd_fields(
            self._network, grey, (working_width, working_height), self._device,
        )
        lines = self._network.detect_afm_lines(
            grey, distance_field, angle_field, **export_lines.DEEPLSD_LINE_PARAMS, grad_nfa=self._grad_nfa,
        )
        working_segments = export_lines._segment_array(lines)  # (fragments, 4) in working pixels
        height, width = frame.shape[:2]
        x_factor, y_factor = width / working_width, height / working_height
        return working_segments * np.array([x_factor, y_factor, x_factor, y_factor])
