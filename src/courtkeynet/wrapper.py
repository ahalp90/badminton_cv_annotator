"""Frame in, four court corners out, with confidence signals.

Wraps the vendored CourtKeyNet model (see PROVENANCE.md) with the input
preprocessing it expects and an inverse mapping back to original-frame pixels.
Each detection carries per-corner peak and entropy signals plus a validity gate
that *flags* a suspect detection rather than silently dropping it: consumers
(e.g. the per-scene homography step) decide what to do with a flagged frame.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from ._vendor.models import CourtKeyNet
from .constants import DEFAULT_CORNER_MIN_PEAK_CONF

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "courtkeynet_finetuned.safetensors"
CONFIG_PATH = Path(__file__).parent / "_vendor" / "configs" / "courtkeynet.yaml"

# Gate thresholds measured on 10 ShuttleSet broadcast videos (400 court-view
# frames vs the recorded homographies, plus a 63-frame hand-checked non-court
# sample). Peak confidence varies per video while localisation stays accurate,
# so the floor sits low: 0.02 keeps 88% of court frames (100% on 7 of 10 vids)
# and still rejected the entire non-court sample in pad mode. One video's court
# frames run below even this floor and fail closed, which is the designed
# behaviour (ShuttleSet's recorded homography is the backstop). Amateur footage
# is unmeasured; re-measure before trusting these there. The peak floor's default
# now lives in constants.py as DEFAULT_CORNER_MIN_PEAK_CONF; the two below are the
# other gate thresholds from that same measurement.
DEFAULT_MAX_ENTROPY = 0.8
DEFAULT_AREA_BOUNDS = (0.01, 0.95)

MODEL_INPUT_SIZE = 640


@dataclass(frozen=True)
class CornerDetection:
    """One frame's court-corner detection with its confidence signals and flags."""

    corners_px: np.ndarray  # (4, 2) float32, original-frame pixel xy, order TL TR BR BL
    peak: np.ndarray  # (4,) softmax max prob per corner heatmap
    entropy: np.ndarray  # (4,) per-corner heatmap entropy, normalised to [0, 1] by log(H*W)
    flags: tuple[str, ...]  # empty tuple == clean detection

    @property
    def passed(self) -> bool:
        """:return: True when the detection cleared the validity gate (no flags)."""
        return not self.flags


class InverseGeometry(NamedTuple):
    """Exact per-frame resize geometry, enough to invert model coords to original pixels.

    scale_x/scale_y are the EFFECTIVE per-axis scales (resized extent / original
    extent), not the theoretical fit scale: cv2.resize works on the rounded integer
    extents, so inverting through the theoretical scale drifts up to ~0.5/scale px
    at the far edge. Squash is the pad-free case (pads zero, per-axis scales differ).
    """

    orig_w: int
    orig_h: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int


def _letterbox_geometry(w: int, h: int, size: int = MODEL_INPUT_SIZE) -> tuple[InverseGeometry, int, int]:
    """Aspect-preserving letterbox geometry: scale to fit, then centre-pad to a square.

    :param w: original frame width in pixels
    :param h: original frame height in pixels
    :param size: target square side length
    :return: (inverse geometry, new_w, new_h) with new_w/new_h the resized content extent
    """
    fit = min(size / w, size / h)
    new_w = round(w * fit)
    new_h = round(h * fit)
    # Centre pad; an odd remainder gives the right/bottom edge the extra pixel.
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    return InverseGeometry(w, h, new_w / w, new_h / h, pad_x, pad_y), new_w, new_h


def _squash_geometry(w: int, h: int, size: int = MODEL_INPUT_SIZE) -> InverseGeometry:
    """Squash geometry: each axis maps onto the full square independently, no pad.

    :param w: original frame width in pixels
    :param h: original frame height in pixels
    :param size: target square side length
    :return: the inverse geometry
    """
    return InverseGeometry(w, h, size / w, size / h, 0, 0)


def _invert_corners(corners_norm: np.ndarray, geom: InverseGeometry, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Map normalised model-space corners back to original-frame pixels.

    One formula covers both modes: normalised -> padded-square pixels, strip the
    pad, divide out the effective per-axis scale (squash just has zero pad).

    :param corners_norm: (4, 2) normalised [0, 1] xy in the size x size model space
    :param geom: the frame's forward-resize geometry
    :param size: padded square side length
    :return: (4, 2) float32 original-frame pixel xy
    """
    px = corners_norm * size - np.array([geom.pad_x, geom.pad_y], dtype=np.float64)
    return (px / np.array([geom.scale_x, geom.scale_y], dtype=np.float64)).astype(np.float32)


def _peak_entropy(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-corner peak probability and normalised entropy from heatmap logits.

    Softmax runs over each corner's flattened heatmap. Peak is the max prob;
    entropy is normalised by log(H*W) so a uniform heatmap scores ~1.

    :param heatmaps: (K, H, W) raw heatmap logits for one frame
    :return: (peak (K,), entropy (K,)) both float32
    """
    num_kpt, height, width = heatmaps.shape
    # float64 for a numerically stable softmax on large logit spikes.
    flat = heatmaps.reshape(num_kpt, -1).astype(np.float64)
    flat = flat - flat.max(axis=1, keepdims=True)
    exp = np.exp(flat)
    probs = exp / exp.sum(axis=1, keepdims=True)
    peak = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1) / np.log(height * width)
    return peak.astype(np.float32), entropy.astype(np.float32)


def _is_convex(corners_norm: np.ndarray) -> bool:
    """:return: True when the TL->TR->BR->BL quad is convex (consistent turn direction).

    :param corners_norm: (4, 2) normalised xy in TL, TR, BR, BL order
    """
    edges = np.roll(corners_norm, -1, axis=0) - corners_norm  # (4, 2) edge vectors around the quad
    next_edges = np.roll(edges, -1, axis=0)  # (4, 2)
    crosses = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]  # (4,)
    return bool((crosses >= 0).all() or (crosses <= 0).all())


def _shoelace_area(corners_norm: np.ndarray) -> float:
    """:return: quad area in normalised units via the shoelace formula.

    :param corners_norm: (4, 2) normalised xy in TL, TR, BR, BL order
    """
    x = corners_norm[:, 0]
    y = corners_norm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _quadrants_ok(corners: np.ndarray) -> bool:
    """:return: True when each corner sits in its expected quadrant about the centroid.

    y grows downward, so TL is the top-left quadrant (x<cx and y<cy), and so on
    clockwise. A swapped or mislabelled corner lands in the wrong quadrant.

    Deliberately stricter than pure vertex order: real broadcast and handheld
    courts project to roughly axis-aligned trapezoids, so a diamond-like quad
    (which this rejects) is treated as suspect by design; flagging fails closed.

    :param corners: (4, 2) xy in TL, TR, BR, BL order (any consistent pixel space)
    """
    cx, cy = corners.mean(axis=0)
    x = corners[:, 0]
    y = corners[:, 1]
    tl_ok = x[0] < cx and y[0] < cy
    tr_ok = x[1] > cx and y[1] < cy
    br_ok = x[2] > cx and y[2] > cy
    bl_ok = x[3] < cx and y[3] > cy
    return bool(tl_ok and tr_ok and br_ok and bl_ok)


def _geometry_flags(
    corners: np.ndarray,
    frame_wh: tuple[float, float],
    area_bounds: tuple[float, float] = DEFAULT_AREA_BOUNDS,
) -> tuple[str, ...]:
    """Shape-only validity flags for a corner quad.

    The area check runs as a fraction of the ORIGINAL frame, not of the padded
    model square: in pad mode the letterbox shrinks normalised areas by the
    content fraction, which would make a fixed threshold mean different things
    per aspect ratio and per resize mode (and skew a pad-vs-squash comparison).
    Convexity and the quadrant check are affine-invariant, so the space of the
    corners does not matter for them.

    :param corners: (4, 2) xy in TL, TR, BR, BL order
    :param frame_wh: (width, height) of the frame the corners live in
    :param area_bounds: (min, max) allowed quad area as a fraction of the frame
    :return: any of ('non_convex', 'bad_area', 'bad_corner_order'); empty when clean
    """
    flags: list[str] = []
    if not _is_convex(corners):
        flags.append("non_convex")
    lo, hi = area_bounds
    area_frac = _shoelace_area(corners) / (frame_wh[0] * frame_wh[1])
    if area_frac < lo or area_frac > hi:
        flags.append("bad_area")
    if not _quadrants_ok(corners):
        flags.append("bad_corner_order")
    return tuple(flags)


def _gate_flags(
    corners: np.ndarray,
    frame_wh: tuple[float, float],
    peak: np.ndarray,
    entropy: np.ndarray,
    corner_min_peak_conf: float,
    max_entropy: float,
    area_bounds: tuple[float, float],
) -> tuple[str, ...]:
    """Full validity gate: confidence signals plus geometry.

    :param corners: (4, 2) xy in TL, TR, BR, BL order
    :param frame_wh: (width, height) of the frame the corners live in
    :param peak: (4,) per-corner softmax max prob
    :param entropy: (4,) per-corner normalised entropy
    :param corner_min_peak_conf: any corner below this flags 'low_peak'
    :param max_entropy: any corner above this flags 'high_entropy'
    :param area_bounds: (min, max) allowed area as a fraction of the frame
    :return: all triggered flags; empty tuple == clean detection
    """
    flags: list[str] = []
    if (peak < corner_min_peak_conf).any():
        flags.append("low_peak")
    if (entropy > max_entropy).any():
        flags.append("high_entropy")
    flags.extend(_geometry_flags(corners, frame_wh, area_bounds))
    return tuple(flags)


class CourtKeyNetDetector:
    """Loads finetuned CourtKeyNet and turns frames into gated corner detections."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        device: str = "cpu",
        resize_mode: str = "pad",
        corner_min_peak_conf: float = DEFAULT_CORNER_MIN_PEAK_CONF,
        max_entropy: float = DEFAULT_MAX_ENTROPY,
        area_bounds: tuple[float, float] = DEFAULT_AREA_BOUNDS,
    ) -> None:
        """Build the model, load weights strict, and set the eval device.

        :param weights_path: safetensors finetuned weights
        :param device: torch device string, e.g. 'cpu' or 'cuda'
        :param resize_mode: 'pad' (aspect-preserving letterbox) or 'squash' (upstream's)
        :param corner_min_peak_conf: gate threshold; any corner peak below flags 'low_peak'
        :param max_entropy: gate threshold; any corner entropy above flags 'high_entropy'
        :param area_bounds: (min, max) allowed normalised quad area
        """
        if resize_mode not in {"pad", "squash"}:
            raise ValueError(f"resize_mode must be 'pad' or 'squash', got {resize_mode!r}")
        self.device = device
        self.resize_mode = resize_mode
        self.corner_min_peak_conf = corner_min_peak_conf
        self.max_entropy = max_entropy
        self.area_bounds = area_bounds
        self.size = MODEL_INPUT_SIZE

        with open(CONFIG_PATH) as config_file:
            config = yaml.safe_load(config_file)
        model = CourtKeyNet(config)
        state = load_file(str(weights_path))
        model.load_state_dict(state, strict=True)
        model.eval()
        self.model = model.to(device)

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, InverseGeometry]:
        """Resize one BGR frame to the model input and record its inverse geometry.

        Both modes finish BGR->RGB, float32/255, HWC->CHW. The returned geometry
        carries the effective per-axis scales and pads, so the corner mapping
        back to original pixels is exact.

        :param frame_bgr: (H, W, 3) uint8 BGR frame
        :return: (chw (3, 640, 640) float32, the frame's inverse geometry)
        """
        height, width = frame_bgr.shape[:2]
        if self.resize_mode == "squash":
            resized = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            geom = _squash_geometry(width, height, self.size)
        else:  # 'pad', validated in __init__
            geom, new_w, new_h = _letterbox_geometry(width, height, self.size)
            small = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            resized = np.zeros((self.size, self.size, 3), dtype=frame_bgr.dtype)
            resized[geom.pad_y : geom.pad_y + new_h, geom.pad_x : geom.pad_x + new_w] = small

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        return chw, geom

    def detect_batch(self, frames_bgr: Sequence[np.ndarray]) -> list[CornerDetection]:
        """Detect court corners across frames that may have different sizes.

        Each frame is preprocessed independently, stacked, and run through one
        forward pass; corners then map back to each frame's own pixel space.

        :param frames_bgr: sequence of (H, W, 3) uint8 BGR frames, sizes may differ
        :return: one gated CornerDetection per input frame, in order
        """
        chw_stack = []
        geometries = []
        for frame_bgr in frames_bgr:
            chw, geom = self._preprocess(frame_bgr)
            chw_stack.append(chw)
            geometries.append(geom)

        batch = torch.from_numpy(np.stack(chw_stack)).to(self.device)  # (B, 3, 640, 640)
        with torch.inference_mode():
            output = self.model(batch)
        heatmaps = output["heatmaps"].cpu().numpy()  # (B, 4, 320, 320) logits
        corners_norm = output["kpts_refined"].cpu().numpy()  # (B, 4, 2) normalised xy

        detections = []
        for frame_idx, geom in enumerate(geometries):
            corners_px = _invert_corners(corners_norm[frame_idx], geom, self.size)
            peak, entropy = _peak_entropy(heatmaps[frame_idx])
            flags = _gate_flags(
                corners_px, (geom.orig_w, geom.orig_h), peak, entropy,
                self.corner_min_peak_conf, self.max_entropy, self.area_bounds,
            )
            detections.append(CornerDetection(corners_px=corners_px, peak=peak, entropy=entropy, flags=flags))
        return detections


def ckn_scene_corners(detections: Sequence[CornerDetection]) -> np.ndarray | None:
    """Per-corner median over the detections that passed the gate.

    Per-scene protocol: the caller samples ~10 frames per PySceneDetect scene and
    gates each. With no passers the scene fails closed (no homography for it), so
    this returns None rather than averaging in suspect corners.

    :param detections: gated detections sampled from one scene
    :return: (4, 2) float32 median corner pixels, or None if none passed
    """
    passed = [detection.corners_px for detection in detections if detection.passed]
    if not passed:
        return None
    median = np.median(np.stack(passed), axis=0).astype(np.float32)  # (4, 2) TL TR BR BL
    # Per-axis medians of individually-valid quads can still synthesise a broken
    # one (e.g. mid-scene camera pan), so re-check shape before handing it out.
    # Area is not re-checked: each median coordinate stays inside the passers'
    # own coordinate envelope, so area cannot leave their ballpark.
    if not (_is_convex(median) and _quadrants_ok(median)):
        return None
    return median
