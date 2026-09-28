"""Where the court detector's line fragments come from: saved extracts or a live DeepLSD model.

Both sources return (fragments, 4) finite float32 x1, y1, x2, y2 in the frame's
native pixels, the layout `ViewInputs.segments_px` expects. Only `DeepLSDLines`
needs torch and the DeepLSD checkout. It imports them when it is built, so this
module and `SavedLines` need neither. The research line exporter uses the DeepLSD
helpers here too, so saved extracts and live lines come from one implementation.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

# DeepLSD's detect_afm_lines settings for every saved extract. grad_nfa is chosen per call.
DEEPLSD_LINE_PARAMS = {
    "filtering": "normal",
    "merge": False,
    "grad_thresh": 3,
}


class LineSource(Protocol):
    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """(fragments, 4) finite float32 x1, y1, x2, y2 in the native pixels of this BGR frame."""
        ...


def _checked_segments(segments: npt.ArrayLike) -> np.ndarray:
    """Saved fragments as a (fragments, 4) float32 array; an empty extract becomes zero rows."""
    array = np.asarray(segments, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
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


def add_source_path(source: Path) -> None:
    """Put a model's source checkout first on `sys.path`, ahead of any installed copy."""
    source_text = str(source.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != source_text]
    sys.path.insert(0, source_text)


def load_deeplsd(source: Path, weights: Path, device: torch.device) -> Any:
    """Load DeepLSD from its source checkout with line detection left to `detect_afm_lines`.

    :param source: DeepLSD source checkout; it goes first on `sys.path`.
    :param weights: checkpoint file holding the network's `model` state.
    :return: the network on `device`, in evaluation mode.
    """
    import torch

    add_source_path(source)
    DeepLSD = import_module("deeplsd.models.deeplsd_inference").DeepLSD

    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    network = DeepLSD({"detect_lines": False})
    network.load_state_dict(checkpoint["model"], strict=True)
    return network.to(device).eval()


def deeplsd_fields(network: Any, grey: np.ndarray, working_size: tuple[int, int],
                   device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """DeepLSD's distance and line-angle fields for one working greyscale image.

    :param grey: (working height, working width) uint8 greyscale image.
    :param working_size: (width, height) of `grey`; each field must match it.
    :return: distance field and angle field, each (working height, working width).
    """
    import torch

    input_tensor = torch.from_numpy(grey).to(device=device, dtype=torch.float32) / 255.0
    with torch.inference_mode():
        prediction = network({"image": input_tensor[None, None]})
    expected_shape = (working_size[1], working_size[0])
    fields = []
    for name in ("df", "line_level"):
        field = prediction[name]
        if field.ndim != 3 or field.shape[0] != 1 or tuple(field.shape[1:]) != expected_shape:
            raise ValueError(f"{name} has unexpected shape {tuple(field.shape)}")
        array = field[0].detach().cpu().numpy()
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        fields.append(array)
    return fields[0], fields[1]


def segment_array(segments: Any) -> np.ndarray:
    """A line detector's fragments as a (fragments, 4) float64 x1, y1, x2, y2 array.

    DeepLSD returns (fragments, 2, 2) endpoints; those are flattened.
    """
    array = np.asarray(segments, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if array.ndim == 3 and array.shape[1:] == (2, 2):
        array = array.reshape(-1, 4)
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError(f"line detector returned unexpected shape {array.shape}")
    return array


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

    The network loads once. Each frame's distance and angle fields then go to
    `detect_afm_lines` with the saved extracts' settings.
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
        # Only this adapter needs torch, so it loads here.
        import torch

        self._device = torch.device(device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"device {device!r} was requested but CUDA is unavailable")
        self._network = load_deeplsd(source, weights, self._device)
        self._max_dimension = max_dimension
        self._grad_nfa = grad_nfa

    def segments(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """(fragments, 4) float32 x1, y1, x2, y2 in native pixels. `frame_index` is unused."""
        grey, (working_width, working_height) = _working_grey(frame, self._max_dimension)
        distance_field, angle_field = deeplsd_fields(self._network, grey, (working_width, working_height),
                                                     self._device)
        lines = self._network.detect_afm_lines(
            grey, distance_field, angle_field, **DEEPLSD_LINE_PARAMS, grad_nfa=self._grad_nfa,
        )
        working_segments = segment_array(lines)  # (fragments, 4) in working pixels
        height, width = frame.shape[:2]
        x_factor, y_factor = width / working_width, height / working_height
        # Scale in float64 and round once, so exact working corners still map to exact native ones.
        return (working_segments * np.array([x_factor, y_factor, x_factor, y_factor])).astype(np.float32)
