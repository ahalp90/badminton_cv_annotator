"""Isolate fragment access and merged-line access with original raster distances."""

from __future__ import annotations

import numpy as np

from . import detector
from .fixed_floor_evidence import (
    FAMILY_INDEX,
    INTERVAL_NAMES,
    MATCH_ANGLE_DEG,
    FloorObservations,
    _line_matches,
    _summary,
    prepare,
)

__all__ = ["MATCH_ANGLE_DEG", "measure", "prepare"]

# The first four arms cross two independent access choices. The fifth removes
# fragment direction filtering at the same all-fragment merged pool.
ARMS = {
    "family_family": ("family", "family"),
    "projected_family": ("projected", "family"),
    "family_all": ("family", "all"),
    "projected_all": ("projected", "all"),
    "all_all": ("all", "all"),
}


def measure(
    homography: np.ndarray, observations: FloorObservations,
    size: tuple[int, int], settings: detector.Settings,
) -> dict:
    """Score a fixed court using the detector's raster and line-count conventions.

    :param homography: Court metres to working-image pixels; kept unchanged.
    :return: Five arms with separate point support and merged-line identities.
    """
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    endpoints = projected.reshape(12, 2, 2)
    samples, visible = detector._visible_samples(endpoints[None], size, settings.samples_per_line)
    samples = np.nan_to_num(samples[0], nan=0, posinf=0, neginf=0)
    visible = visible[0]
    pixels = np.clip(samples, 0, np.asarray(size) - 1).astype(int)
    vectors = endpoints[:, 1] - endpoints[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    compatible = np.abs(directions @ observations.directions.T) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))
    segments = observations.segments.reshape(-1, 4)
    family_distances = observations.maps[FAMILY_INDEX[:, None], pixels[..., 1], pixels[..., 0]]
    projected_distances = np.empty(samples.shape[:2])
    # Each pair supplies the existing helper's two maps without duplicating a map.
    for first in range(0, 12, 2):
        pair = (segments[compatible[first]], segments[compatible[first + 1]])
        maps = detector._distance_maps(pair, size)
        pair_pixels = pixels[first:first + 2]
        projected_distances[first:first + 2] = maps[np.arange(2)[:, None], pair_pixels[..., 1], pair_pixels[..., 0]]
    all_map = detector._distance_maps((segments, np.empty((0, 4))), size)[0]
    distances = {"family": family_distances, "projected": projected_distances,
                 "all": all_map[pixels[..., 1], pixels[..., 0]]}
    arms = {}
    for name, (fragment_access, line_access) in ARMS.items():
        raster_distances = distances[fragment_access]
        covered = (raster_distances <= settings.support_distance) & visible[:, None]
        support = covered.mean(axis=1)
        pools = observations.family_lines if line_access == "family" else (observations.all_lines, observations.all_lines)
        matches, residuals, counts = _line_matches(samples, support, directions, pools, settings, directional=False)
        counted = (support >= settings.min_family_support) & (residuals <= settings.support_distance * 2)
        arms[name] = {
            **_summary(support, visible, counts, settings),
            "interval_support": support.tolist(), "covered_samples": covered.tolist(),
            "sample_raster_distances_px": raster_distances.tolist(),
            "nearest_line_ids": matches.tolist(), "counted_line_ids": np.where(counted, matches, -1).tolist(),
            "nearest_line_residual_px": np.where(np.isfinite(residuals), residuals, -1).tolist(),
            "fragment_access": fragment_access, "line_pool": line_access,
        }
    original = detector._score(homography[None], observations.maps, settings, observations.family_lines)
    geometry_pass = bool(len(original[1]))
    if geometry_pass:
        baseline = arms["family_family"]
        if (not np.array_equal(baseline["family_means"], original[2][0])
                or not np.array_equal(baseline["distinct_line_counts"], original[3][0])
                or baseline["floor_score"] != original[1][0]):
            raise ValueError("raster baseline differs from detector._score")
    return {
        "geometry_pass": geometry_pass, "original_score_control_exact": geometry_pass,
        "interval_names": INTERVAL_NAMES, "visible": visible.tolist(),
        "samples_working_px": samples.tolist(), "sample_pixels": pixels.tolist(),
        "projected_intervals_working_px": endpoints.tolist(),
        "projected_angles_deg": np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])).tolist(),
        "projected_fragment_ids": [np.flatnonzero(mask).tolist() for mask in compatible], "arms": arms,
    }
