"""Label-free direction estimation and pruning of existing rectangle identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations

import cv2
import numpy as np

from experiments.annotator.independent_court import assignment, detector


@dataclass(frozen=True)
class Settings:
    angle_deg: float = 1.5
    direction_lines: int = 128
    pencils: int = 16
    overlap: float = 0.8
    rectangles: int = 16_384
    candidate_batch: int = 256
    pencil_selection: str = "ranked"


def angular_residuals(lines: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Line/ray angles in degrees at each line's foot from the coordinate origin.

    :return: One residual per homogeneous point and observed line. Undefined
        rays receive 90 degrees, including a finite VP at the line's foot.
    """
    normals = lines[:, :2]
    feet = -lines[:, 2, None] * normals / np.sum(normals**2, axis=1)[:, None]
    directions = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
    rays = points[:, None, :2] - points[:, None, 2:] * feet[None]
    dot = np.abs(np.einsum("pli,li->pl", rays, directions))
    cross = np.abs(rays[..., 0] * directions[:, 1] - rays[..., 1] * directions[:, 0])
    angles = np.degrees(np.arctan2(cross, dot))
    return np.where(np.linalg.norm(rays, axis=2) > 1e-12, angles, 90.0)


def normalisation(size: tuple[int, int]) -> np.ndarray:
    """Map isotropic, centred coordinates back to working pixels."""
    width, height = size
    diagonal = np.hypot(width, height)
    return np.array([[diagonal, 0, width / 2], [0, diagonal, height / 2], [0, 0, 1]])


def retain_pencils(
    masks: np.ndarray, counts: np.ndarray, supports: np.ndarray, candidate_ids: np.ndarray, settings: Settings,
) -> tuple[list[int], np.ndarray]:
    """Retain competing pencils by raw support or additional observation coverage."""
    if settings.pencil_selection not in ("ranked", "coverage"):
        raise ValueError(f"Unknown pencil selection: {settings.pencil_selection}")
    eligible = counts >= 2
    status = np.where(eligible, "capped", "below_two_lines").astype("U16")
    covered = np.zeros(masks.shape[1], dtype=bool)
    retained: list[int] = []
    for _ in range(settings.pencils):
        indices = np.flatnonzero(eligible)
        if not len(indices):
            break
        novelty = (
            (masks[indices] & ~covered).sum(axis=1) if settings.pencil_selection == "coverage" else counts[indices]
        )
        order = np.lexsort((candidate_ids[indices], -supports[indices], -counts[indices], -novelty))
        index = int(indices[order[0]])
        retained.append(index)
        covered |= masks[index]
        union = np.count_nonzero(masks[indices] | masks[index], axis=1)
        intersection = np.count_nonzero(masks[indices] & masks[index], axis=1)
        redundant = indices[intersection / union > settings.overlap]
        eligible[redundant] = False
        status[redundant] = "redundant"
        status[index] = "retained"
    return retained, status


def estimate(segments: np.ndarray, size: tuple[int, int], settings: Settings) -> tuple[np.ndarray, dict]:
    """Estimate competing homogeneous VPs from fragments without court labels."""
    observations = assignment.prepare_observations(segments, size)
    merge_settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=300)
    all_lines = detector._merge_lines(observations.segments.reshape(-1, 4), merge_settings)
    lines = all_lines[:settings.direction_lines]
    transform = normalisation(size)
    normalised_lines = lines @ transform
    pairs = np.asarray(list(combinations(range(len(lines)), 2)), dtype=int).reshape(-1, 2)
    intersections = np.cross(normalised_lines[pairs[:, 0]], normalised_lines[pairs[:, 1]])
    infinity = np.column_stack((normalised_lines[:, 1], -normalised_lines[:, 0], np.zeros(len(lines))))
    candidates = np.concatenate((intersections, infinity))
    norms = np.linalg.norm(candidates, axis=1)
    nondegenerate = norms > 1e-12
    candidate_ids = np.flatnonzero(nondegenerate)
    candidates = candidates[nondegenerate] / norms[nondegenerate, None]
    masks, counts, supports = [], [], []
    for offset in range(0, len(candidates), settings.candidate_batch):
        angles = angular_residuals(normalised_lines, candidates[offset:offset + settings.candidate_batch])
        batch_masks = angles <= settings.angle_deg
        masks.extend(batch_masks)
        counts.extend(batch_masks.sum(axis=1))
        supports.extend(np.maximum(0, 1 - angles / settings.angle_deg).sum(axis=1))
    masks = np.asarray(masks, dtype=bool).reshape(-1, len(lines)) if len(lines) else np.empty((0, 0), dtype=bool)
    counts = np.asarray(counts, dtype=int)
    supports = np.asarray(supports)
    retained, status = retain_pencils(masks, counts, supports, candidate_ids, settings)
    selected = candidates[retained]
    native = selected @ transform.T
    provenance = []
    for line in lines:
        distances = np.abs(observations.segments @ line[:2] + line[2]).max(axis=1)
        angles = np.abs(observations.directions @ line[:2])
        compatible = (distances <= merge_settings.merge_distance) & (
            angles <= np.sin(np.deg2rad(merge_settings.merge_angle_deg))
        )
        provenance.append(observations.fragment_ids[compatible].tolist())
    details = {
        "raw_fragments": len(segments), "visible_fragments": len(observations.segments),
        "visible_raw_ids": observations.fragment_ids.tolist(),
        "preparation_groups": len(observations.groups),
        "merge_input_cap": 300, "merge_input_excluded": max(0, len(observations.segments) - 300),
        "merged_direction_count": len(all_lines), "direction_cap_excluded": len(all_lines) - len(lines),
        "direction_lines": lines.tolist(), "compatible_raw_ids": provenance,
        "provenance_note": "Geometric compatibility after fitting, not exact merge membership",
        "pair_candidates": len(pairs), "infinity_candidates": len(infinity),
        "degenerate_candidates": int((~nondegenerate).sum()),
        "candidate_ids": candidate_ids.tolist(), "support_counts": counts.tolist(),
        "candidate_status": status.tolist(), "retained_candidate_ids": candidate_ids[retained].tolist(),
        "retained_support_masks": masks[retained].tolist(), "points_working": native.tolist(),
        "normalised_to_working": transform.tolist(),
    }
    return native, details


def rectangle_population(
    families: tuple[np.ndarray, np.ndarray], size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce original centre ordering and convexity; retain original seed IDs."""
    x_lines, y_lines = families
    width, height = size
    x_order = np.argsort(-(x_lines[:, 1] * height / 2 + x_lines[:, 2]) / x_lines[:, 0])
    y_order = np.argsort(-(y_lines[:, 0] * width / 2 + y_lines[:, 2]) / y_lines[:, 1])
    intersections = np.cross(x_lines[x_order, None], y_lines[None, y_order])
    with np.errstate(divide="ignore", invalid="ignore"):
        points = intersections[..., :2] / intersections[..., 2:]
    x_pairs = np.asarray(list(combinations(range(len(x_lines)), 2)), dtype=int).reshape(-1, 2)
    y_pairs = np.asarray(list(combinations(range(len(y_lines)), 2)), dtype=int).reshape(-1, 2)
    quads = points[x_pairs[:, None, [0, 1, 1, 0]], y_pairs[None, :, [0, 0, 1, 1]]].reshape(-1, 4, 2)
    ranks = np.concatenate((np.repeat(x_order[x_pairs], len(y_pairs), axis=0),
                            np.tile(y_order[y_pairs], (len(x_pairs), 1))), axis=1)
    edges = np.roll(quads, -1, axis=1) - quads
    turns = edges[..., 0] * np.roll(edges[..., 1], -1, axis=1) - edges[..., 1] * np.roll(edges[..., 0], -1, axis=1)
    convex = np.isfinite(quads).all(axis=(1, 2)) & np.all(turns > 0, axis=1)
    ids = np.flatnonzero(convex)
    return quads[convex], ranks[convex], ids


def select(
    families: tuple[np.ndarray, np.ndarray], points: np.ndarray, size: tuple[int, int],
    detector_settings: detector.Settings, settings: Settings,
) -> tuple[np.ndarray, dict]:
    """Allocate a bounded rectangle union across every ordered pencil pair."""
    if min(map(len, families)) < 2:
        return np.empty((0, 3, 3)), {"reason": "insufficient_lines", "retained_pair_product_ids": []}
    quads, ranks, ids = rectangle_population(families, size)
    transform = normalisation(size)
    normalised_points = points @ np.linalg.inv(transform).T
    masks = [angular_residuals(lines @ transform, normalised_points) <= settings.angle_deg for lines in families]
    pools, pair_records = [], []
    for x_index in range(len(points)):
        for y_index in range(len(points)):
            if x_index == y_index:
                continue
            x_mask, y_mask = masks[0][x_index], masks[1][y_index]
            retained = x_mask[ranks[:, :2]].all(axis=1) & y_mask[ranks[:, 2:]].all(axis=1)
            pool = np.flatnonzero(retained)
            generator = np.random.default_rng(detector_settings.seed)
            pools.append(generator.permutation(pool))
            pair_records.append({"pencils": [x_index, y_index],
                                 "family_ranks": [np.flatnonzero(x_mask).tolist(), np.flatnonzero(y_mask).tolist()],
                                 "convex_rectangles": len(pool)})
    union = np.unique(np.concatenate(pools)) if pools else np.empty(0, dtype=int)
    fallback = len(union) == 0
    if fallback:
        generator = np.random.default_rng(detector_settings.seed)
        chosen = (generator.choice(len(quads), detector_settings.max_rectangles, replace=False)
                  if len(quads) > detector_settings.max_rectangles else np.arange(len(quads)))
    else:
        selected: list[int] = []
        seen: set[int] = set()
        cursors = np.zeros(len(pools), dtype=int)
        while len(selected) < min(settings.rectangles, len(union)):
            for pool_index, pool in enumerate(pools):
                cursor = cursors[pool_index]
                while cursor < len(pool) and int(pool[cursor]) in seen:
                    cursor += 1
                if cursor < len(pool):
                    index = int(pool[cursor])
                    selected.append(index)
                    seen.add(index)
                    cursor += 1
                cursors[pool_index] = cursor
                if len(selected) == settings.rectangles:
                    break
        chosen = np.asarray(selected, dtype=int)
    rectangles, area_ids = [], []
    for index in chosen:
        quad = quads[index].astype(np.float32)
        if cv2.contourArea(quad) >= 100:
            rectangles.append(cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad))
            area_ids.append(int(ids[index]))
    details = {
        "mode": "vp_pruned", "family_line_counts": list(map(len, families)),
        "available_convex_quads": len(quads), "vp_pairs": pair_records,
        "union_convex_quads": len(union), "union_pair_product_ids": ids[union].tolist(),
        "selected_quads": len(chosen), "selected_pair_product_ids": ids[chosen].tolist(),
        "selected_line_ranks": ranks[chosen].tolist(), "retained_pair_product_ids": area_ids,
        "area_retained_rectangles": len(rectangles), "fallback": fallback,
        "budget_excluded": (len(quads) if fallback else len(union)) - len(chosen),
    }
    return np.asarray(rectangles).reshape(-1, 3, 3), details
