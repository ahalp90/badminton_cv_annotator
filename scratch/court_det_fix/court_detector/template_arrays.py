"""Score line-template hypotheses with NumPy on the CPU or CuPy on a GPU.

Each scoring function takes the array module first, as `xp`. NumPy and CuPy then run
the same expressions, so both devices share one scoring definition. CPU results are
the reference. CuPy's rounding can differ in the last bits, which occasionally
moves a line sample to a neighbouring pixel and changes a support mean. Rectangle
setup, admission and the W5 gates stay on the CPU in line_templates.py.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

TEMPLATE_DEVICES = ("cpu", "cuda")
SUPPORT_DISTANCE = 4.0
SAMPLES_PER_LINE = 24


class ViewArrays(NamedTuple):
    """One view's fixed scoring inputs, copied to the scoring device once."""

    templates: Any  # (templates, 3, 3) float64 court-template transforms
    distance_map: Any  # (height, width) float32 distance from each working pixel to the nearest fragment
    corner_points: Any  # (4, 3) homogeneous court corners in metres
    segment_points: Any  # (24, 3) homogeneous painted-line ends in metres, two per line
    sample_steps: Any  # (SAMPLES_PER_LINE,) evenly spaced fractions along each clipped line
    focals: Any  # (200,) focal lengths the camera check tries, in working pixels
    image_size: Any  # (2,) working width and height, for per-axis bounds
    size: tuple[int, int]  # the same width and height as Python integers


class ScoredHypotheses(NamedTuple):
    """Host arrays for scored hypotheses; every field after `valid` has one row per valid hypothesis."""

    valid: np.ndarray  # (hypotheses,) bool
    corners: np.ndarray  # (valid, 4, 2) float64 court corners in working pixels
    means: np.ndarray  # (valid, 2) float32 mean line support, lengthwise then cross-court
    visibility: np.ndarray  # (valid, 2) int16 visible pieces, lengthwise then cross-court
    camera_errors: np.ndarray  # (valid,) float64 vector camera errors, before the scalar recheck


def array_module(device: str) -> ModuleType:
    """NumPy for "cpu", or CuPy for "cuda". A CUDA request never falls back to the CPU."""
    if device == "cpu":
        return np
    if device != "cuda":
        raise ValueError(f"template device must be one of {TEMPLATE_DEVICES}, not {device!r}")
    try:
        import cupy  # pyrefly: ignore[missing-import]
    except ImportError as error:
        raise RuntimeError(f"template device 'cuda' needs CuPy, which could not be imported: {error}") from error
    # Without a usable driver, CuPy raises its own CUDA error here.
    if cupy.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("template device 'cuda' needs a CUDA GPU, but CuPy found none")
    return cupy


def homogeneous(points: np.ndarray) -> np.ndarray:
    """(..., 2) court points in metres, extended as geometry.project does and flattened to (points, 3)."""
    return np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1).reshape(-1, 3)


def camera_focals(image_width: int) -> np.ndarray:
    """The focal lengths the camera check tries, in working pixels."""
    return np.geomspace(0.4 * image_width, 4.0 * image_width, 200)


def place_view(xp: ModuleType, distance_map: np.ndarray, size: tuple[int, int], court_model: ModuleType) -> ViewArrays:
    """Copy one view's fixed inputs to the scoring device.

    NumPy builds the sample steps and focal lengths, so every device scores with the same values.
    """
    width, _ = size
    return ViewArrays(
        templates=xp.asarray(court_model.TEMPLATE_TRANSFORMS),
        distance_map=xp.asarray(distance_map),
        corner_points=xp.asarray(homogeneous(court_model.CORNER_COURT_M)),
        segment_points=xp.asarray(homogeneous(court_model.SEGMENTS_M)),
        sample_steps=xp.asarray(np.linspace(0, 1, SAMPLES_PER_LINE)),
        focals=xp.asarray(camera_focals(width)),
        image_size=xp.asarray(np.asarray(size)),
        size=size,
    )


def project(xp: ModuleType, homographies: Any, points: Any) -> tuple[Any, Any]:
    """geometry.project, with the homogeneous points prepared once per view.

    :return: (hypotheses, points, 2) image points and (hypotheses, points) homogeneous denominators.
    """
    mapped = xp.einsum("...ij,pj->...pi", homographies, points)
    denominator = mapped[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = mapped[..., :2] / denominator[..., None]
    return pixels, denominator


def visible_samples(xp: ModuleType, endpoints: Any, view: ViewArrays) -> tuple[Any, Any]:
    """geometry._visible_samples: clip each projected line to the image, then sample the rest evenly.

    :param endpoints: (hypotheses, lines, 2, 2) projected line ends in working pixels.
    :return: (hypotheses, lines, SAMPLES_PER_LINE, 2) samples, and whether each line counts as visible.
    """
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = xp.zeros(starts.shape[:2])
    upper = xp.ones(starts.shape[:2])
    visible = xp.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(view.size):
        stationary = xp.abs(vectors[..., axis]) < 1e-8
        visible &= ~stationary | ((starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1))
        divisor = xp.where(stationary, 1, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        lower = xp.maximum(lower, xp.where(stationary, -xp.inf, xp.minimum(first, last)))
        upper = xp.minimum(upper, xp.where(stationary, xp.inf, xp.maximum(first, last)))
    visible &= upper > lower
    # np.linalg.norm adds x*x + y*y in this order, so NumPy's bits match geometry's.
    length = xp.sqrt(vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1])
    visible &= (upper - lower) * length >= 12
    fractions = lower[..., None] + (upper - lower)[..., None] * view.sample_steps
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    return samples, visible


def geometry_and_support(xp: ModuleType, homographies: Any, view: ViewArrays) -> tuple[Any, Any, Any, Any, Any]:
    """Check each hypothesis's court shape, then score its line support in two directions.

    A valid court has finite corners in front of the camera, is convex, and spans at least
    15% of the image each way. A projected line piece is visible when at least 12 of its
    pixels remain in the image. The first six pieces are the lengthwise x-family: sidelines
    plus the split centre line. The second six are the cross-court y-family.

    :param homographies: (hypotheses, 3, 3) court metres to working pixels.
    :return: the valid mask, one per hypothesis. Then, for valid hypotheses only: their
        homographies, (4, 2) image corners, float32 mean support per direction, and int16
        visible pieces per direction.
    """
    corners, denominators = project(xp, homographies, view.corner_points)
    in_front = xp.isfinite(corners).all(axis=(1, 2)) & xp.all(denominators > 1e-6, axis=1)
    edges = xp.roll(corners, -1, axis=1) - corners
    turns = edges[..., 0] * xp.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * xp.roll(edges[..., 0], -1, axis=1)
    convex = xp.all(turns > 0, axis=1)
    visible_lower = xp.maximum(corners.min(axis=1), 0)
    visible_upper = xp.minimum(corners.max(axis=1), view.image_size - 1)
    wide_enough = xp.all((visible_upper - visible_lower) / view.image_size >= 0.15, axis=1)
    valid = in_front & convex & wide_enough

    valid_homographies = homographies[valid]
    endpoints, _ = project(xp, valid_homographies, view.segment_points)
    samples, visible = visible_samples(xp, endpoints.reshape(-1, 12, 2, 2), view)
    samples = xp.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    width, height = view.size
    # Clip before truncating to pixel indices: CuPy wraps out-of-range indices silently.
    pixel_x = xp.clip(samples[..., 0], 0, width - 1).astype(int)
    pixel_y = xp.clip(samples[..., 1], 0, height - 1).astype(int)
    # NumPy's mean sums then divides. Spelling that out keeps CuPy to the same rounding.
    support = (view.distance_map[pixel_y, pixel_x] <= SUPPORT_DISTANCE).sum(axis=-1) / SAMPLES_PER_LINE
    support *= visible
    lengthwise_visible = visible[:, :6].sum(axis=1)
    cross_court_visible = visible[:, 6:].sum(axis=1)
    means = xp.stack(
        [
            support[:, :6].sum(axis=1) / xp.maximum(lengthwise_visible, 1),
            support[:, 6:].sum(axis=1) / xp.maximum(cross_court_visible, 1),
        ],
        axis=1,
    ).astype(xp.float32)
    visibility = xp.stack([lengthwise_visible, cross_court_visible], axis=1).astype(xp.int16)
    return valid, valid_homographies, corners[valid], means, visibility


def camera_errors(xp: ModuleType, homographies: Any, focals: Any, size: tuple[int, int]) -> Any:
    """Vectorise the frozen camera diagnostic over its 200 focal lengths.

    Each court direction's x, y and w parts stay separate (courts, focal lengths) arrays, because
    numpy is several times slower on trailing axes of length 3 and 2. The sums run left to right,
    as numpy's length-3 reductions do, so errors are bit-identical to the stacked form.
    """
    # The stacked form kept float32 input in float32; this form would promote it to float64.
    if homographies.dtype != np.float64:
        raise TypeError(f"camera errors need float64 homographies, got {homographies.dtype}")
    image_width, image_height = size
    # Column 0 of a homography images the court's width direction, column 1 its length direction.
    # The image-plane parts move to the principal point and scale by focal length; w does neither.
    width_x = (homographies[:, 0, 0] - image_width / 2.0 * homographies[:, 2, 0])[:, None] / focals
    width_y = (homographies[:, 1, 0] - image_height / 2.0 * homographies[:, 2, 0])[:, None] / focals
    length_x = (homographies[:, 0, 1] - image_width / 2.0 * homographies[:, 2, 1])[:, None] / focals
    length_y = (homographies[:, 1, 1] - image_height / 2.0 * homographies[:, 2, 1])[:, None] / focals
    width_w, length_w = homographies[:, 2, 0, None], homographies[:, 2, 1, None]
    width_norm = xp.sqrt(xp.square(width_x) + xp.square(width_y) + xp.square(width_w))
    length_norm = xp.sqrt(xp.square(length_x) + xp.square(length_y) + xp.square(length_w))
    with np.errstate(divide="ignore", invalid="ignore"):
        dot = width_x * length_x + width_y * length_y + width_w * length_w
        cosine = dot / (width_norm * length_norm)
        ratio = xp.log(width_norm / length_norm)
        errors = xp.hypot(cosine, ratio)
    usable = (xp.isfinite(errors) & xp.isfinite(width_norm) & xp.isfinite(length_norm)
              & (width_norm > 0) & (length_norm > 0))
    return xp.where(usable, errors, xp.inf).min(axis=1)


def score_rectangles(xp: ModuleType, rectangles: Any, view: ViewArrays) -> ScoredHypotheses:
    """Score every court template inside each rectangle, then copy the compact results to the CPU.

    Hypotheses run rectangle by rectangle, with the templates varying fastest.

    :param rectangles: (rectangles, 3, 3) float64 unit square to working pixels, on the scoring device.
    """
    homographies = (rectangles[:, None] @ view.templates).reshape(-1, 3, 3)
    valid, valid_homographies, corners, means, visibility = geometry_and_support(xp, homographies, view)
    errors = camera_errors(xp, valid_homographies, view.focals, view.size)
    scored = (valid, corners, means, visibility, errors)
    if xp is np:
        return ScoredHypotheses(*scored)
    return ScoredHypotheses(*(xp.asnumpy(array) for array in scored))
