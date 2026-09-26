"""Measure line support and keep the best distinct court shapes."""


from __future__ import annotations

import numpy as np

from . import geometry as detector
from . import line_observations as assignment


def continuous_support(homographies: np.ndarray, maps: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Score finite visible intervals smoothly while counting the split centre once."""
    projected, _ = detector.project(homographies, detector.SEGMENTS_M)
    endpoints = projected.reshape(-1, 12, 2, 2)
    lower, upper, visible = detector._visible_fractions(endpoints, size)
    fractions = lower[..., None] + (upper - lower)[..., None] * np.linspace(0, 1, 64)
    # Same arithmetic as detector._visible_samples, with x and y as separate arrays: numpy is
    # much slower on a trailing axis of length 2. The scores are bit-identical.
    start_x, start_y = endpoints[:, :, 0, 0, None], endpoints[:, :, 0, 1, None]
    end_x, end_y = endpoints[:, :, 1, 0, None], endpoints[:, :, 1, 1, None]
    pixel_x = np.clip(start_x + fractions * (end_x - start_x), 0, size[0] - 1).astype(int)
    pixel_y = np.clip(start_y + fractions * (end_y - start_y), 0, size[1] - 1).astype(int)
    family = np.repeat([0, 1], 6)[None, :, None]
    distance = maps[family, pixel_y, pixel_x]
    response = np.exp(-.5 * np.square(distance / assignment.DISTANCE_SIGMA_PX)).mean(axis=2)
    response *= visible
    per_marking, marking_visible = [], []
    for intervals in assignment.MARKING_INTERVALS:
        count = visible[:, intervals].sum(axis=1)
        per_marking.append(response[:, intervals].sum(axis=1) / np.maximum(count, 1))
        marking_visible.append(count > 0)
    return np.sum(per_marking, axis=0) / np.maximum(np.sum(marking_visible, axis=0), 1)


def geometry(homographies: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Use the detector's original positive-depth, convexity and visible-span conditions."""
    corners, denominator = detector.project(homographies, detector.CORNER_COURT_M)
    edges = np.roll(corners, -1, axis=1) - corners
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    span = np.minimum(corners.max(axis=1), np.asarray(size) - 1) - np.maximum(corners.min(axis=1), 0)
    valid = (np.isfinite(corners).all(axis=(1, 2)) & np.all(denominator > 1e-6, axis=1)
             & np.all(turns > 0, axis=1)
             & np.all(span / size >= detector.DEFAULT_SETTINGS.min_visible_span_fraction, axis=1))
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
