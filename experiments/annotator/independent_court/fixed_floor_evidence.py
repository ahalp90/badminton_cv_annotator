"""Compare floor evidence access while holding court geometry and gate settings fixed.

Measurements use the detector's twelve painted intervals, clipping, sample count,
family means and distinct-line rule. Only observation access and the explicitly
named raster/finite distance control vary. No labels, search or refitting enter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import detector
from .assignment import MATCH_ANGLE_DEG, distances_to_segments

INTERVAL_NAMES = (
    "left_doubles", "left_singles", "far_centre", "near_centre",
    "right_singles", "right_doubles", "far_baseline", "far_long_service",
    "far_short_service", "near_short_service", "near_long_service", "near_baseline",
)
FAMILY_INDEX = np.repeat([0, 1], 6)


@dataclass(frozen=True)
class FloorObservations:
    segments: np.ndarray  # Raw cache order, in working pixels: (fragments, 2, xy).
    directions: np.ndarray
    family_ids: tuple[np.ndarray, np.ndarray]
    maps: np.ndarray
    family_lines: tuple[np.ndarray, np.ndarray]
    all_lines: np.ndarray


def prepare(segments: np.ndarray, size: tuple[int, int], settings: detector.Settings) -> FloorObservations:
    """Prepare original and unrestricted evidence pools with unchanged merging.

    :param segments: Raw XYXY fragments in the detector's working coordinates.
    :return: Original family pools and one shared all-fragment merged pool.
    """
    segments = np.asarray(segments, dtype=float).reshape(-1, 4)
    families = detector._wide_line_families(segments) if settings.wide_families else detector._line_families(segments)
    # Recover raw IDs by exact row membership, including duplicate detector rows.
    family_ids = tuple(np.flatnonzero((segments[:, None] == family[None]).all(axis=2).any(axis=1))
                       for family in families)
    endpoints = segments.reshape(-1, 2, 2)
    vectors = endpoints[:, 1] - endpoints[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    family_lines = tuple(detector._merge_lines(family, settings) for family in families)
    return FloorObservations(endpoints, directions, family_ids, detector._distance_maps(families, size),
                             family_lines, detector._merge_lines(segments, settings))


def _line_matches(
    samples: np.ndarray, support: np.ndarray, directions: np.ndarray,
    pools: tuple[np.ndarray, np.ndarray], settings: detector.Settings, directional: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the original endpoint residual, nearest-line choice and distinct count."""
    matches = np.full(12, -1, dtype=int)
    residuals = np.full(12, np.inf)
    usable = np.zeros(12, dtype=bool)
    counts = np.zeros(2, dtype=int)
    for family, lines in enumerate(pools):
        intervals = np.flatnonzero(FAMILY_INDEX == family)
        if not len(lines):
            continue
        endpoints = samples[intervals][:, [0, -1]]
        distances = np.abs(np.einsum("ied,ld->iel", endpoints, lines[:, :2]) + lines[:, 2])
        maximum = distances.max(axis=1)
        if directional:
            line_directions = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
            parallel = np.abs(directions[intervals] @ line_directions.T) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))
            maximum = np.where(parallel, maximum, np.inf)
        nearest = maximum.argmin(axis=1)
        best = maximum[np.arange(len(intervals)), nearest]
        matches[intervals] = np.where(np.isfinite(best), nearest, -1)
        residuals[intervals] = best
        usable[intervals] = ((support[intervals] >= settings.min_family_support)
                             & (best <= settings.support_distance * 2))
        counts[family] = len(np.unique(matches[intervals][usable[intervals]]))
    return matches, residuals, counts


def _summary(
    support: np.ndarray, visible: np.ndarray, counts: np.ndarray, settings: detector.Settings,
) -> dict:
    means = np.array([support[:6].sum() / max(visible[:6].sum(), 1),
                      support[6:].sum() / max(visible[6:].sum(), 1)])
    reasons = []
    for family, name in enumerate(("lengthwise", "crosscourt")):
        if means[family] < settings.min_family_support:
            reasons.append(f"{name}_mean")
        if counts[family] < settings.min_supported_lines:
            reasons.append(f"{name}_distinct_lines")
    return {"family_means": means.tolist(), "distinct_line_counts": counts.tolist(),
            "floor_score": -1.0 if reasons else float(means.mean()), "rejection_reasons": reasons}


def measure(
    homography: np.ndarray, observations: FloorObservations,
    size: tuple[int, int], settings: detector.Settings,
) -> dict:
    """Measure one fixed court, retaining raw fragment and merged-line identities.

    :param homography: Court metres to working-image pixels; geometry is never changed.
    :return: Four evidence arms and the separate original geometry-gate outcome.
    """
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    endpoints = projected.reshape(12, 2, 2)
    samples, visible = detector._visible_samples(endpoints[None], size, settings.samples_per_line)
    samples = np.nan_to_num(samples[0], nan=0, posinf=0, neginf=0)
    visible = visible[0]
    vectors = endpoints[:, 1] - endpoints[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    compatibility = np.abs(directions @ observations.directions.T) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))
    family_access = np.zeros_like(compatibility)
    for interval, family in enumerate(FAMILY_INDEX):
        family_access[interval, observations.family_ids[family]] = True
    distances = distances_to_segments(samples.reshape(-1, 2), observations.segments)
    distances = distances.reshape(12, settings.samples_per_line, -1)
    arms = {}
    for name in ("raster_family", "finite_family", "finite_projected", "finite_all"):
        directional = name == "finite_projected"
        original_family = name in ("raster_family", "finite_family")
        allowed = family_access if original_family else compatibility
        if name == "finite_all":
            allowed = np.ones_like(compatibility)
        masked = np.where(allowed[:, None], distances, np.inf)
        nearest = np.full(samples.shape[:2], -1, dtype=int)
        nearest_distance = np.full(samples.shape[:2], np.inf)
        if len(observations.segments):
            nearest = masked.argmin(axis=2)
            nearest_distance = np.take_along_axis(masked, nearest[..., None], axis=2)[..., 0]
            nearest = np.where(np.isfinite(nearest_distance), nearest, -1)
        covered = nearest_distance <= settings.support_distance
        if name == "raster_family":
            pixels = np.clip(samples, 0, np.asarray(size) - 1).astype(int)
            raster_distances = observations.maps[FAMILY_INDEX[:, None], pixels[..., 1], pixels[..., 0]]
            covered = raster_distances <= settings.support_distance
        support = covered.mean(axis=1) * visible
        pools = observations.family_lines if original_family else (observations.all_lines, observations.all_lines)
        matches, residuals, counts = _line_matches(samples, support, directions, pools, settings, directional)
        counted = (support >= settings.min_family_support) & (residuals <= settings.support_distance * 2)
        arm = {
            **_summary(support, visible, counts, settings),
            "interval_support": support.tolist(), "compatible_fragment_counts": allowed.sum(axis=1).tolist(),
            "covered_samples": (covered & visible[:, None]).tolist(),
            "nearest_line_ids": matches.tolist(), "counted_line_ids": np.where(counted, matches, -1).tolist(),
            "nearest_line_residual_px": np.where(np.isfinite(residuals), residuals, -1).tolist(),
            "line_pool": "original_families" if original_family else "all_fragments",
        }
        if name != "raster_family":
            # Raster distances do not identify the raw fragment that supplied a pixel.
            arm["nearest_fragment_ids"] = nearest.tolist()
            finite_distance = np.where(np.isfinite(nearest_distance), nearest_distance, -1)
            arm["nearest_fragment_distances_px"] = finite_distance.tolist()
        arms[name] = arm
    # The original detector may never reach camera evaluation for these courts.
    # This fixed-geometry probe records floor evidence independently of that order.
    original = detector._score(homography[None], observations.maps, settings, observations.family_lines)
    geometry_pass = bool(len(original[1]))
    if geometry_pass:
        actual = arms["raster_family"]
        if (not np.array_equal(actual["family_means"], original[2][0])
                or not np.array_equal(actual["distinct_line_counts"], original[3][0])
                or actual["floor_score"] != original[1][0]):
            raise ValueError("reconstructed original evidence differs from detector._score")
    return {"geometry_pass": geometry_pass, "original_score_control_exact": geometry_pass,
            "interval_names": INTERVAL_NAMES, "visible": visible.tolist(),
            "samples_working_px": samples.tolist(), "projected_intervals_working_px": endpoints.tolist(),
            "projected_angles_deg": np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])).tolist(), "arms": arms}
