"""Measure line support and keep the best distinct court shapes."""


from __future__ import annotations

import math

import numba
import numpy as np

from . import geometry as detector
from . import line_observations as assignment

FULL_SAMPLES = 64  # samples per marking for the full line-support score
MIN_VISIBLE_PX = 12.  # shortest clipped interval that counts, as in geometry._visible_fractions
# Map 0 scores the six intervals at constant court x, map 1 the six at constant court y.
INTERVAL_MAPS = np.repeat(np.arange(2), 6)
# Marking m owns MARKING_INTERVAL_IDS[MARKING_BOUNDS[m]:MARKING_BOUNDS[m + 1]]; the split
# centre line is one marking over two intervals.
MARKING_INTERVAL_IDS = np.asarray([interval for intervals in assignment.MARKING_INTERVALS for interval in intervals])
MARKING_BOUNDS = np.cumsum([0, *(len(intervals) for intervals in assignment.MARKING_INTERVALS)])


def continuous_support(
    homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int], samples: int = FULL_SAMPLES,
) -> np.ndarray:
    """Score finite visible intervals smoothly while counting the split centre once.

    :param homographies: (courts, 3, 3) float32 court metres to working pixels.
    :param maps: (2, height, width) float32 distance maps, one per direction.
    :param samples: evenly spaced samples along each marking's visible interval. Fewer
        give a cheaper, coarser score.
    :return: (courts,) float32 scores.
    """
    width, height = size
    return _score_courts(
        homographies, maps, width, height, detector.SEGMENTS_M, INTERVAL_MAPS, MARKING_INTERVAL_IDS, MARKING_BOUNDS,
        np.linspace(0, 1, samples), np.float32(assignment.DISTANCE_SIGMA_PX), MIN_VISIBLE_PX,
    )


# The compiled functions below run serially: the search already spreads direction pairs over
# worker processes. cache=True saves the machine code beside this file, so each spawned worker
# loads it rather than recompiling. Numba freezes globals at compile time and its cache misses
# changes to them, so project constants arrive as arguments. error_model="numpy" gives inf and
# NaN on division by zero, as NumPy does, rather than raising.
@numba.njit(cache=True, error_model="numpy")
def _score_courts(
    homographies: np.ndarray, maps: np.ndarray, width: int, height: int, segments_m: np.ndarray,
    interval_maps: np.ndarray, marking_interval_ids: np.ndarray, marking_bounds: np.ndarray,
    sample_fractions: np.ndarray, sigma_px: np.float32, min_visible_px: float,
) -> np.ndarray:
    """Each court's mean over visible markings of the marking's mean Gaussian line response.

    Projection and clipping use float64, which costs little at 24 endpoints per court. Every
    response and mean stays float32. Numba reads Python float literals as float64, so the
    float32 constants are spelled out.
    """
    last_column, last_row = width - 1., height - 1.
    neg_half = np.float32(-.5)
    sample_count = np.float32(len(sample_fractions))
    scores = np.empty(len(homographies), dtype=np.float32)
    for court in range(len(homographies)):
        homography = homographies[court]
        support_total = np.float32(0.)
        visible_markings = 0
        for marking in range(len(marking_bounds) - 1):
            marking_total = np.float32(0.)
            visible_intervals = 0
            for interval in marking_interval_ids[marking_bounds[marking]:marking_bounds[marking + 1]]:
                start_x, start_y = _project(homography, segments_m[interval, 0, 0], segments_m[interval, 0, 1])
                end_x, end_y = _project(homography, segments_m[interval, 1, 0], segments_m[interval, 1, 1])
                vector_x, vector_y = end_x - start_x, end_y - start_y
                lower, upper, visible = _visible_fractions(
                    start_x, start_y, vector_x, vector_y, last_column, last_row, min_visible_px)
                if not visible:
                    continue
                distance_map = maps[interval_maps[interval]]
                response_total = np.float32(0.)
                for fraction in sample_fractions:
                    position = lower + (upper - lower) * fraction
                    pixel_x = int(min(max(start_x + position * vector_x, 0.), last_column))
                    pixel_y = int(min(max(start_y + position * vector_y, 0.), last_row))
                    scaled = distance_map[pixel_y, pixel_x] / sigma_px
                    response_total += math.exp(neg_half * scaled * scaled)
                marking_total += response_total / sample_count
                visible_intervals += 1
            if visible_intervals > 0:
                support_total += marking_total / np.float32(visible_intervals)
                visible_markings += 1
        scores[court] = support_total / np.float32(max(visible_markings, 1))
    return scores


@numba.njit(cache=True, error_model="numpy")
def _project(homography: np.ndarray, court_x: float, court_y: float) -> tuple[float, float]:
    """One court point in working pixels, computed in float64."""
    court_x, court_y = float(court_x), float(court_y)
    depth = float(homography[2, 0]) * court_x + float(homography[2, 1]) * court_y + float(homography[2, 2])
    pixel_x = float(homography[0, 0]) * court_x + float(homography[0, 1]) * court_y + float(homography[0, 2])
    pixel_y = float(homography[1, 0]) * court_x + float(homography[1, 1]) * court_y + float(homography[1, 2])
    return pixel_x / depth, pixel_y / depth


@numba.njit(cache=True, error_model="numpy")
def _visible_fractions(
    start_x: float, start_y: float, vector_x: float, vector_y: float,
    last_column: float, last_row: float, min_visible_px: float,
) -> tuple[float, float, bool]:
    """geometry._visible_fractions for one projected interval.

    :return: first and last in-image fractions of its length, and whether it counts as visible.
    """
    # Numba skips bounds checks, so a NaN sample position could read outside the maps. NumPy's
    # NaN arithmetic already hid intervals with non-finite endpoints; hide them here explicitly.
    if not (math.isfinite(start_x) and math.isfinite(start_y)
            and math.isfinite(vector_x) and math.isfinite(vector_y)):
        return 0., 0., False
    lower, upper, visible = 0., 1., True
    for start, vector, last_pixel in ((start_x, vector_x, last_column), (start_y, vector_y, last_row)):
        if abs(vector) < 1e-8:
            visible = visible and 0. <= start <= last_pixel
        else:
            first = -start / vector
            last = (last_pixel - start) / vector
            lower = max(lower, min(first, last))
            upper = min(upper, max(first, last))
    clipped_length = (upper - lower) * math.hypot(vector_x, vector_y)
    return lower, upper, visible and upper > lower and clipped_length >= min_visible_px


def geometry(homographies: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Use the detector's original positive-depth, convexity and visible-span conditions."""
    corners, denominator = detector.project(homographies, detector.CORNER_COURT_M)
    # Corners near the horizon can overflow float32 here; the depth test rejects those courts.
    with np.errstate(over="ignore", invalid="ignore"):
        edges = np.roll(corners, -1, axis=1) - corners
        turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    image_size = np.asarray(size, dtype=corners.dtype)
    span = np.minimum(corners.max(axis=1), image_size - 1) - np.maximum(corners.min(axis=1), 0)
    valid = (np.isfinite(corners).all(axis=(1, 2)) & np.all(denominator > 1e-6, axis=1)
             & np.all(turns > 0, axis=1)
             & np.all(span / image_size >= detector.DEFAULT_SETTINGS.min_visible_span_fraction, axis=1))
    return valid, corners


def retain(candidates: list[detector.Candidate], settings: detector.Settings) -> list[detector.Candidate]:
    """Vectorise distances while preserving the existing greedy retention order."""
    retained = []
    corners = np.empty((settings.keep_candidates, 4, 2))
    for candidate in sorted(candidates, key=lambda item: -item.score):
        separation = np.linalg.norm(corners[:len(retained)] - candidate.corners_px, axis=2).max(axis=1)
        if np.any(separation <= settings.distinct_corner_distance):
            continue
        corners[len(retained)] = candidate.corners_px
        retained.append(candidate)
        if len(retained) == settings.keep_candidates:
            break
    return retained
