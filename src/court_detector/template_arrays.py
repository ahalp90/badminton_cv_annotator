"""Score line-template hypotheses with NumPy on the CPU or CuPy on a GPU.

Each scoring function takes the array module, NumPy or CuPy, first as `array_module`.
Both then run the same expressions, so both devices share one scoring definition. CPU
results are the reference. CuPy's rounding can differ in the last bits, which
occasionally moves a line sample to a neighbouring pixel and changes a support mean.
Rectangle setup, admission and the candidate gates stay on the CPU in line_templates.py.
Scoring runs in float32. Its largest arrays hold one sample per line per hypothesis;
float64 would double their memory, and consumer GPUs run float64 many times slower.
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

    templates: Any  # (templates, 3, 3) float32 court-template transforms
    distance_map: Any  # (height, width) float32 distance from each working pixel to the nearest fragment
    corner_points: Any  # (4, 3) float32 homogeneous court corners in metres
    segment_points: Any  # (24, 3) float32 homogeneous painted-line ends in metres, two per line
    sample_steps: Any  # (SAMPLES_PER_LINE,) float32 evenly spaced fractions along each clipped line
    focals: Any  # (200,) float32 focal lengths the camera check tries, in working pixels
    image_size: Any  # (2,) float32 working width and height, for per-axis bounds
    size: tuple[int, int]  # the same width and height as Python integers


class ScoredHypotheses(NamedTuple):
    """Host arrays for scored hypotheses; every field after `valid` has one row per valid hypothesis."""

    valid: np.ndarray  # (hypotheses,) bool
    corners: np.ndarray  # (valid, 4, 2) float32 court corners in working pixels
    means: np.ndarray  # (valid, 2) float32 mean line support, lengthwise then cross-court
    visibility: np.ndarray  # (valid, 2) int16 visible pieces, lengthwise then cross-court
    camera_errors: np.ndarray  # (valid,) float32 vector camera errors, before the scalar recheck


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
    return np.concatenate((points, np.ones((*points.shape[:-1], 1), dtype=points.dtype)), axis=-1).reshape(-1, 3)


def camera_focals(image_width: int) -> np.ndarray:
    """The float32 focal lengths the camera check tries, in working pixels."""
    return np.geomspace(0.4 * image_width, 4.0 * image_width, 200, dtype=np.float32)


def place_view(
    array_module: ModuleType, distance_map: np.ndarray, size: tuple[int, int], court_model: ModuleType,
) -> ViewArrays:
    """Copy one view's fixed inputs to the scoring device as float32.

    NumPy builds the sample steps and focal lengths, so every device scores with the same values.
    """
    width, _ = size
    float32 = array_module.float32
    return ViewArrays(
        templates=array_module.asarray(court_model.TEMPLATE_TRANSFORMS, dtype=float32),
        distance_map=array_module.asarray(distance_map, dtype=float32),
        corner_points=array_module.asarray(homogeneous(court_model.CORNER_COURT_M), dtype=float32),
        segment_points=array_module.asarray(homogeneous(court_model.SEGMENTS_M), dtype=float32),
        sample_steps=array_module.asarray(np.linspace(0, 1, SAMPLES_PER_LINE, dtype=np.float32)),
        focals=array_module.asarray(camera_focals(width)),
        image_size=array_module.asarray(size, dtype=float32),
        size=size,
    )


def project(array_module: ModuleType, homographies: Any, points: Any) -> tuple[Any, Any]:
    """geometry.project, with the homogeneous points prepared once per view.

    :return: (hypotheses, points, 2) image points and (hypotheses, points) homogeneous denominators.
    """
    mapped = points @ homographies.swapaxes(-1, -2)
    denominator = mapped[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = mapped[..., :2] / denominator[..., None]
    return pixels, denominator


def visible_samples(array_module: ModuleType, endpoints: Any, view: ViewArrays) -> tuple[Any, Any]:
    """geometry._visible_samples: clip each projected line to the image, then sample the rest evenly.

    :param endpoints: (hypotheses, lines, 2, 2) projected line ends in working pixels.
    :return: (hypotheses, lines, SAMPLES_PER_LINE, 2) samples, and whether each line counts as visible.
    """
    starts = endpoints[:, :, 0]
    vectors = endpoints[:, :, 1] - starts
    lower = array_module.zeros(starts.shape[:2], dtype=starts.dtype)
    upper = array_module.ones(starts.shape[:2], dtype=starts.dtype)
    # A line that stays still along an axis must start inside the image on that axis.
    stationary_axes_inside = array_module.ones(starts.shape[:2], dtype=bool)
    for axis, limit in enumerate(view.size):
        stationary = array_module.abs(vectors[..., axis]) < 1e-8
        starts_inside = (starts[..., axis] >= 0) & (starts[..., axis] <= limit - 1)
        stationary_axes_inside = stationary_axes_inside & (~stationary | starts_inside)
        divisor = array_module.where(stationary, 1, vectors[..., axis])
        first = -starts[..., axis] / divisor
        last = (limit - 1 - starts[..., axis]) / divisor
        axis_lower = array_module.where(stationary, -array_module.inf, array_module.minimum(first, last))
        axis_upper = array_module.where(stationary, array_module.inf, array_module.maximum(first, last))
        lower = array_module.maximum(lower, axis_lower)
        upper = array_module.minimum(upper, axis_upper)
    span_remains = upper > lower
    # np.linalg.norm adds x*x + y*y in this order, so NumPy's bits match geometry's.
    length = array_module.sqrt(vectors[..., 0] * vectors[..., 0] + vectors[..., 1] * vectors[..., 1])
    long_enough = (upper - lower) * length >= 12
    visible = stationary_axes_inside & span_remains & long_enough
    fractions = lower[..., None] + (upper - lower)[..., None] * view.sample_steps
    samples = starts[..., None, :] + fractions[..., None] * vectors[..., None, :]
    return samples, visible


def geometry_and_support(
    array_module: ModuleType, homographies: Any, view: ViewArrays,
) -> tuple[Any, Any, Any, Any, Any]:
    """Check each hypothesis's court shape, then score its line support in two directions.

    A valid court has finite corners in front of the camera, is convex, and spans at least
    15% of the image each way. A projected line piece is visible when at least 12 of its
    pixels remain in the image. The first six pieces are the lengthwise x-family: sidelines
    plus the split centre line. The second six are the cross-court y-family.

    :param homographies: (hypotheses, 3, 3) float32 court metres to working pixels.
    :return: the valid mask, one per hypothesis. Then, for valid hypotheses only: their
        homographies, (4, 2) image corners, float32 mean support per direction, and int16
        visible pieces per direction.
    """
    corners, denominators = project(array_module, homographies, view.corner_points)
    in_front = array_module.isfinite(corners).all(axis=(1, 2)) & array_module.all(denominators > 1e-6, axis=1)
    # Corners near the horizon can overflow float32 below. The depth test already rejects
    # those courts, so their NaN turns need no warning.
    with np.errstate(over="ignore", invalid="ignore"):
        edges = array_module.roll(corners, -1, axis=1) - corners
        # Each edge crossed with the next: positive at every corner of a convex court.
        next_edge_x = array_module.roll(edges[..., 0], -1, axis=1)
        next_edge_y = array_module.roll(edges[..., 1], -1, axis=1)
        turns = edges[..., 0] * next_edge_y - edges[..., 1] * next_edge_x
    convex = array_module.all(turns > 0, axis=1)
    visible_lower = array_module.maximum(corners.min(axis=1), 0)
    visible_upper = array_module.minimum(corners.max(axis=1), view.image_size - 1)
    wide_enough = array_module.all((visible_upper - visible_lower) / view.image_size >= 0.15, axis=1)
    valid = in_front & convex & wide_enough

    valid_homographies = homographies[valid]
    endpoints, _ = project(array_module, valid_homographies, view.segment_points)
    samples, visible = visible_samples(array_module, endpoints.reshape(-1, 12, 2, 2), view)
    samples = array_module.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    width, height = view.size
    # Clip before truncating to pixel indices: CuPy wraps out-of-range indices silently.
    # int32 covers the working image without the extra storage of int64 indices.
    pixel_x = array_module.clip(samples[..., 0], 0, width - 1).astype(array_module.int32)
    pixel_y = array_module.clip(samples[..., 1], 0, height - 1).astype(array_module.int32)
    # Sum integer sample counts before dividing so equal support keeps an exact tie.
    supported_samples = view.distance_map[pixel_y, pixel_x] <= SUPPORT_DISTANCE
    support_counts = supported_samples.sum(axis=-1, dtype=array_module.int16)
    support_counts *= visible
    direction_counts = array_module.stack(
        [support_counts[:, :6].sum(axis=1, dtype=array_module.int16),
         support_counts[:, 6:].sum(axis=1, dtype=array_module.int16)], axis=1,
    )
    visibility = array_module.stack(
        [visible[:, :6].sum(axis=1, dtype=array_module.int16),
         visible[:, 6:].sum(axis=1, dtype=array_module.int16)], axis=1,
    )
    means = direction_counts.astype(array_module.float32) / (
        array_module.maximum(visibility, 1) * SAMPLES_PER_LINE)
    return valid, valid_homographies, corners[valid], means, visibility


def camera_errors(array_module: ModuleType, homographies: Any, focals: Any, size: tuple[int, int]) -> Any:
    """Vectorise the frozen camera diagnostic over its 200 focal lengths.

    Each court direction's x, y and w parts stay separate (courts, focal lengths) arrays, because
    numpy is several times slower on trailing axes of length 3 and 2. The sums run left to right,
    as numpy's length-3 reductions do, so errors are bit-identical to the stacked form.
    """
    # A float64 input would promote every (courts, focal lengths) array to float64.
    if homographies.dtype != np.float32 or focals.dtype != np.float32:
        raise TypeError(f"camera errors need float32 homographies and focal lengths, "
                        f"got {homographies.dtype} and {focals.dtype}")
    image_width, image_height = size
    principal_x, principal_y = image_width / 2.0, image_height / 2.0
    # Column 0 of a homography images the court's width direction, column 1 its length direction.
    width_column_x, width_column_y, width_column_w = (
        homographies[:, 0, 0], homographies[:, 1, 0], homographies[:, 2, 0],
    )
    length_column_x, length_column_y, length_column_w = (
        homographies[:, 0, 1], homographies[:, 1, 1], homographies[:, 2, 1],
    )
    # The image-plane parts move to the principal point and scale by focal length; w does neither.
    width_x = (width_column_x - principal_x * width_column_w)[:, None] / focals
    width_y = (width_column_y - principal_y * width_column_w)[:, None] / focals
    length_x = (length_column_x - principal_x * length_column_w)[:, None] / focals
    length_y = (length_column_y - principal_y * length_column_w)[:, None] / focals
    width_w, length_w = width_column_w[:, None], length_column_w[:, None]
    width_norm = array_module.sqrt(
        array_module.square(width_x) + array_module.square(width_y) + array_module.square(width_w)
    )
    length_norm = array_module.sqrt(
        array_module.square(length_x) + array_module.square(length_y) + array_module.square(length_w)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        dot = width_x * length_x + width_y * length_y + width_w * length_w
        cosine = dot / (width_norm * length_norm)
        ratio = array_module.log(width_norm / length_norm)
        errors = array_module.hypot(cosine, ratio)
    all_finite = array_module.isfinite(errors) & array_module.isfinite(width_norm) & array_module.isfinite(length_norm)
    usable = all_finite & (width_norm > 0) & (length_norm > 0)
    return array_module.where(usable, errors, array_module.inf).min(axis=1)


def score_rectangles(array_module: ModuleType, rectangles: Any, view: ViewArrays) -> ScoredHypotheses:
    """Score every court template inside each rectangle, then copy the compact results to the CPU.

    Hypotheses run rectangle by rectangle, with the templates varying fastest.

    :param rectangles: (rectangles, 3, 3) float32 unit square to working pixels, on the scoring device.
    """
    homographies = (rectangles[:, None] @ view.templates).reshape(-1, 3, 3)
    valid, valid_homographies, corners, means, visibility = geometry_and_support(array_module, homographies, view)
    errors = camera_errors(array_module, valid_homographies, view.focals, view.size)
    scored = (valid, corners, means, visibility, errors)
    if array_module is np:
        return ScoredHypotheses(*scored)
    return ScoredHypotheses(*(array_module.asnumpy(array) for array in scored))
