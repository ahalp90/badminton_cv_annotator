"""Tests for the CourtKeyNet wrapper (src/courtkeynet/wrapper.py).

Split in two: model-free tests exercise the pure preprocessing/gate helpers with
no model load, and a small model-backed block loads the detector once per module
and keeps to four forward passes total (CPU ~0.7 s each).
"""

import numpy as np
import pytest

from courtkeynet.wrapper import (
    CornerDetection,
    CourtKeyNetDetector,
    _gate_flags,
    _geometry_flags,
    _invert_corners,
    _letterbox_geometry,
    _peak_entropy,
    _squash_geometry,
    ckn_scene_corners,
)


# --- Model-free: coordinate round-trips ------------------------------------

@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280)])
def test_letterbox_round_trip(width: int, height: int) -> None:
    """Original pixels -> the true forward letterbox -> inverse recovers the pixels."""
    geom, _new_w, _new_h = _letterbox_geometry(width, height)
    corners_px = np.array(  # a plausible court inset from the frame edges
        [[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100]], dtype=np.float32
    )
    # Forward map exactly as the resize applies it: effective per-axis scales, then pad.
    small = corners_px * np.array([geom.scale_x, geom.scale_y])
    corners_norm = (small + np.array([geom.pad_x, geom.pad_y])) / 640
    recovered = _invert_corners(corners_norm, geom)
    assert np.allclose(recovered, corners_px, atol=1e-3)


def test_letterbox_inverse_uses_effective_scale() -> None:
    """Rounded resize extents shift the effective scale; the inverse must track it.

    4001x3000 makes round(h * fit) land a whole step off h * fit, the case where
    inverting through the theoretical fit scale drifts ~0.75 px at the far edge.
    """
    geom, new_w, new_h = _letterbox_geometry(4001, 3000)
    corners_px = np.array([[0, 0], [4001, 0], [4001, 3000], [0, 3000]], dtype=np.float32)
    small = corners_px * np.array([geom.scale_x, geom.scale_y])
    corners_norm = (small + np.array([geom.pad_x, geom.pad_y])) / 640
    recovered = _invert_corners(corners_norm, geom)
    assert np.allclose(recovered, corners_px, atol=1e-3)


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280)])
def test_squash_round_trip(width: int, height: int) -> None:
    """Original pixels -> squash normalised -> inverse recovers the pixels."""
    corners_px = np.array(
        [[100, 100], [width - 100, 100], [width - 100, height - 100], [100, height - 100]], dtype=np.float32
    )
    corners_norm = corners_px / np.array([width, height])
    recovered = _invert_corners(corners_norm, _squash_geometry(width, height))
    assert np.allclose(recovered, corners_px, atol=1e-3)


# --- Model-free: geometry gate ---------------------------------------------

UNIT_FRAME = (1.0, 1.0)  # quads below live in [0, 1], so the frame is the unit square


def test_geometry_gate_plausible_quad_clean() -> None:
    """A broadcast-shaped court quad clears the geometry checks."""
    quad = np.array([[0.2, 0.35], [0.8, 0.35], [0.95, 0.9], [0.05, 0.9]], dtype=np.float32)
    assert _geometry_flags(quad, UNIT_FRAME) == ()


def test_geometry_gate_concave_flags_non_convex() -> None:
    """A dented quad trips 'non_convex' (membership; it may also fail corner order)."""
    quad = np.array([[0.1, 0.1], [0.9, 0.1], [0.5, 0.35], [0.9, 0.9]], dtype=np.float32)
    assert "non_convex" in _geometry_flags(quad, UNIT_FRAME)


def test_geometry_gate_tiny_flags_bad_area() -> None:
    """A pinhole quad falls below the area floor."""
    quad = np.array([[0.49, 0.49], [0.51, 0.49], [0.51, 0.51], [0.49, 0.51]], dtype=np.float32)
    assert "bad_area" in _geometry_flags(quad, UNIT_FRAME)


def test_geometry_gate_swapped_corners_flags_order() -> None:
    """Swapping TL and TR puts corners in the wrong quadrants (and likely non-convex)."""
    quad = np.array([[0.8, 0.35], [0.2, 0.35], [0.95, 0.9], [0.05, 0.9]], dtype=np.float32)
    assert "bad_corner_order" in _geometry_flags(quad, UNIT_FRAME)


def test_geometry_gate_area_is_frame_fraction() -> None:
    """The area bound reads a fraction of the ORIGINAL frame, not of the model square.

    The same court quad in 1920x1080 pixels occupies the same frame fraction as
    its normalised form, so the verdict must match the unit-square case; a fixed
    bound on padded-normalised coords would have flagged wide frames differently.
    """
    quad_norm = np.array([[0.2, 0.35], [0.8, 0.35], [0.95, 0.9], [0.05, 0.9]], dtype=np.float32)
    quad_px = quad_norm * np.array([1920, 1080], dtype=np.float32)
    assert _geometry_flags(quad_px, (1920, 1080)) == ()


def test_gate_flags_combiner() -> None:
    """The full gate composes confidence and geometry flags, empty when all clean."""
    quad = np.array([[0.2, 0.35], [0.8, 0.35], [0.95, 0.9], [0.05, 0.9]], dtype=np.float32)
    good_peak = np.full(4, 0.5, dtype=np.float32)
    good_entropy = np.full(4, 0.3, dtype=np.float32)
    clean = _gate_flags(quad, UNIT_FRAME, good_peak, good_entropy, 0.1, 0.8, (0.01, 0.95))
    assert clean == ()
    weak_peak = np.array([0.5, 0.5, 0.02, 0.5], dtype=np.float32)  # one corner under the floor
    flagged = _gate_flags(quad, UNIT_FRAME, weak_peak, good_entropy, 0.1, 0.8, (0.01, 0.95))
    assert flagged == ("low_peak",)


# --- Model-free: peak / entropy --------------------------------------------

def test_peak_entropy_sharp_spike() -> None:
    """A single hot cell gives peak ~1 and entropy ~0."""
    heatmaps = np.zeros((4, 32, 32), dtype=np.float32)
    heatmaps[:, 5, 7] = 50.0
    peak, entropy = _peak_entropy(heatmaps)
    assert peak.shape == (4,) and entropy.shape == (4,)
    assert np.all(peak > 0.99)
    assert np.all(entropy < 0.01)


def test_peak_entropy_uniform() -> None:
    """All-zero logits softmax to uniform: peak ~1/(H*W) and entropy ~1."""
    height = width = 32
    heatmaps = np.zeros((4, height, width), dtype=np.float32)
    peak, entropy = _peak_entropy(heatmaps)
    assert np.allclose(peak, 1 / (height * width), atol=1e-6)
    assert np.allclose(entropy, 1.0, atol=1e-6)


# --- Model-free: ckn_scene_corners ---------------------------------------------

def _clean_detection(corners_px: np.ndarray) -> CornerDetection:
    """Build a passing detection (empty flags) with dummy confidence signals."""
    return CornerDetection(
        corners_px=corners_px.astype(np.float32),
        peak=np.full(4, 0.5, dtype=np.float32),
        entropy=np.full(4, 0.2, dtype=np.float32),
        flags=(),
    )


def test_ckn_scene_corners_median_ignores_outlier() -> None:
    """Two frames agree, one has an outlier corner: per-corner median wins."""
    base = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
    outlier = base.copy()
    outlier[0] = [900, 900]
    detections = [_clean_detection(base), _clean_detection(base), _clean_detection(outlier)]
    result = ckn_scene_corners(detections)
    assert result is not None
    assert np.allclose(result, base)


def test_ckn_scene_corners_all_flagged_returns_none() -> None:
    """No passers means fail closed: no corners for the scene."""
    base = np.array([[100, 100], [500, 100], [500, 400], [100, 400]], dtype=np.float32)
    flagged = CornerDetection(
        corners_px=base, peak=np.full(4, 0.05, dtype=np.float32), entropy=np.full(4, 0.9, dtype=np.float32),
        flags=("low_peak",),
    )
    assert ckn_scene_corners([flagged, flagged, flagged]) is None


def test_ckn_scene_corners_empty_returns_none() -> None:
    """No detections at all also fails closed."""
    assert ckn_scene_corners([]) is None


def test_ckn_scene_corners_broken_median_returns_none() -> None:
    """A synthesised median that is not court-shaped fails closed too.

    Per-axis medians of individually-passing quads can land on a broken shape
    (e.g. mid-scene camera pan); hand-build passing detections whose median IS
    the broken quad and check the re-validation catches it.
    """
    concave = np.array([[100, 100], [900, 100], [500, 350], [900, 900]], dtype=np.float32)
    detections = [_clean_detection(concave) for _ in range(3)]
    assert ckn_scene_corners(detections) is None


# --- Model-backed: one detector load, four forwards total ------------------

@pytest.fixture(scope="module")
def detector() -> CourtKeyNetDetector:
    """Load the finetuned detector once for the whole module (pad mode, CPU)."""
    return CourtKeyNetDetector(device="cpu", resize_mode="pad")


@pytest.fixture(scope="module")
def frame_landscape() -> np.ndarray:
    """A fixed random 720x1280 BGR frame (H, W, 3)."""
    return np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def frame_small() -> np.ndarray:
    """A fixed random 480x640 BGR frame, a different size for the batch check."""
    return np.random.default_rng(1).integers(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def detect_landscape(detector: CourtKeyNetDetector, frame_landscape: np.ndarray) -> CornerDetection:
    """Single-item batch on the landscape frame (forward pass 1)."""
    return detector.detect_batch([frame_landscape])[0]


@pytest.fixture(scope="module")
def detect_small(detector: CourtKeyNetDetector, frame_small: np.ndarray) -> CornerDetection:
    """Single-item batch on the small frame (forward pass 2)."""
    return detector.detect_batch([frame_small])[0]


def test_param_count_and_strict_load(detector: CourtKeyNetDetector) -> None:
    """Fixture load covers strict weight load; pin the exact parameter count."""
    assert sum(p.numel() for p in detector.model.parameters()) == 1238856


def test_detect_shapes_and_bounds(detect_landscape: CornerDetection) -> None:
    """Shapes, finiteness, loose pixel bounds, and a tuple flags field."""
    detection = detect_landscape
    assert detection.corners_px.shape == (4, 2)
    assert detection.peak.shape == (4,) and detection.entropy.shape == (4,)
    assert np.all(np.isfinite(detection.corners_px))
    assert np.all(np.isfinite(detection.peak)) and np.all(np.isfinite(detection.entropy))
    # Loose sanity: a clamped [0, 1] model output in pad mode can invert to at
    # most pad/scale pixels beyond each frame edge (the letterbox pad region), so
    # bound to that reachable range rather than a fixed margin. Random input still
    # flags, so we do NOT assert passed.
    geom, _new_w, _new_h = _letterbox_geometry(1280, 720)
    margin_x = geom.pad_x / geom.scale_x + 1
    margin_y = geom.pad_y / geom.scale_y + 1
    assert np.all(detection.corners_px[:, 0] >= -margin_x) and np.all(detection.corners_px[:, 0] <= 1280 + margin_x)
    assert np.all(detection.corners_px[:, 1] >= -margin_y) and np.all(detection.corners_px[:, 1] <= 720 + margin_y)
    assert isinstance(detection.flags, tuple)


def test_batch_equals_sequential(
    detector: CourtKeyNetDetector,
    frame_landscape: np.ndarray,
    frame_small: np.ndarray,
    detect_landscape: CornerDetection,
    detect_small: CornerDetection,
) -> None:
    """Batching two different-size frames matches two single-item batches."""
    batch = detector.detect_batch([frame_landscape, frame_small])  # forward pass 3 (B=2)
    assert len(batch) == 2
    assert np.allclose(batch[0].corners_px, detect_landscape.corners_px, atol=1e-4)
    assert np.allclose(batch[1].corners_px, detect_small.corners_px, atol=1e-4)


def test_determinism(
    detector: CourtKeyNetDetector, frame_landscape: np.ndarray, detect_landscape: CornerDetection
) -> None:
    """The same frame twice gives identical corner pixels."""
    again = detector.detect_batch([frame_landscape])[0]  # forward pass 4
    assert np.array_equal(again.corners_px, detect_landscape.corners_px)
