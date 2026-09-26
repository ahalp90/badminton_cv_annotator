"""Merged-line membership and anchor-dependent angular residuals (E0 and E1 geometry)."""

from __future__ import annotations

import cv2
import numpy as np

from experiments.annotator.independent_court import detector

MERGE_INPUT_CAP = 300  # detector._merge_lines merges only the 300 longest fragments.
UNDEFINED_RAY_NORM = 1e-12  # vp_pruning.angular_residuals gives shorter rays 90 degrees.
UNDEFINED_ANGLE_DEG = 90.0


def merge_lines_with_membership(segments: np.ndarray, settings: detector.Settings) -> tuple[np.ndarray, list[dict]]:
    """Private copy of detector._merge_lines that also returns each line's member fragments.

    Every computation and ordering of the original is preserved; the only additions are
    the bookkeeping lists. Membership is the actual group assignment, so it differs from
    the post-fit geometric compatibility the estimator saves as ``compatible_raw_ids``.

    :param segments: (fragments, 4) working-pixel XYXY rows in observation order.
    :param settings: detector settings; only the merge angle, distance and line cap matter.
    :return: (lines, 3) merged coefficients in covered-length order, and one sidecar entry
        per returned line listing the observation rows merged into it in insertion order.
    """
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    groups: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    members: list[list[int]] = []
    for index in np.argsort(-lengths, kind='stable')[:MERGE_INPUT_CAP]:
        points = segments[index].reshape(2, 2)
        direction = (points[1] - points[0]) / lengths[index]
        target = None
        for group_index, line in enumerate(coefficients):
            parallel = abs(np.dot(direction, line[:2])) < np.sin(np.deg2rad(settings.merge_angle_deg))
            close = np.max(np.abs(points @ line[:2] + line[2])) <= settings.merge_distance
            if parallel and close:
                target = group_index
                break
        if target is None:
            groups.append(points)
            coefficients.append(np.zeros(3))
            members.append([])
            target = len(groups) - 1
        else:
            groups[target] = np.concatenate((groups[target], points))
        members[target].append(int(index))
        dx, dy, centre_x, centre_y = cv2.fitLine(groups[target].astype(np.float32), cv2.DIST_L2, 0, .01, .01).ravel()
        coefficients[target] = np.array([-dy, dx, dy * centre_x - dx * centre_y], dtype=np.float64)
    extents = [detector._covered_length(points, line) for points, line in zip(groups, coefficients, strict=True)]
    order = np.argsort(-np.asarray(extents), kind='stable')[:settings.max_family_lines]
    lines = np.asarray(coefficients, dtype=np.float64).reshape(-1, 3)[order]
    sidecar = []
    for line_id, group_index in enumerate(order):
        sidecar.append({
            'line_id': line_id,
            'merge_group_index': int(group_index),
            'covered_length_px': float(extents[group_index]),
            'member_observation_ids': members[group_index],
        })
    return lines, sidecar


def longest_member(members: list[int], lengths: np.ndarray) -> int:
    """Longest clipped contributing fragment; equal lengths break by canonical observation index."""
    return min(members, key=lambda index: (-lengths[index], index))


def line_feet(lines: np.ndarray) -> np.ndarray:
    """Foot of the perpendicular from the coordinate origin: the original agreement anchor."""
    normals = lines[:, :2]
    return -lines[:, 2, None] * normals / np.sum(normals ** 2, axis=1)[:, None]


def project_onto_lines(points: np.ndarray, lines: np.ndarray) -> np.ndarray:
    """Perpendicular projection of one point per line onto that line, in the same frame.

    :param points: (lines, 2) one Cartesian point per line.
    :param lines: (lines, 3) homogeneous coefficients ``(a, b, c)``.
    :return: (lines, 2) ``m - n (n.m + c) / (n.n)`` with ``n = (a, b)``.
    """
    normals = lines[:, :2]
    offsets = np.einsum('li,li->l', normals, points) + lines[:, 2]
    return points - normals * (offsets / np.einsum('li,li->l', normals, normals))[:, None]


def to_normalised(points_working: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map Cartesian working points through the inverse of ``normalised_to_working``."""
    homogeneous = np.column_stack((points_working, np.ones(len(points_working))))
    normalised = np.linalg.solve(transform, homogeneous.T).T
    return normalised[:, :2] / normalised[:, 2:]


def to_working(points_normalised: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Map Cartesian normalised points to working pixels."""
    homogeneous = np.column_stack((points_normalised, np.ones(len(points_normalised))))
    working = homogeneous @ transform.T
    return working[:, :2] / working[:, 2:]


def angular_residuals_at(lines: np.ndarray, points: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """vp_pruning.angular_residuals with an explicit anchor per line.

    With ``anchors = line_feet(lines)`` this reproduces the original function exactly.
    Infinite candidates have a zero homogeneous weight, so their rays ignore the anchor.

    :param lines: (lines, 3) homogeneous line coefficients.
    :param points: (candidates, 3) homogeneous candidate points.
    :param anchors: (lines, 2) Cartesian anchor per line, in the same frame as ``lines``.
    :return: (candidates, lines) angles in degrees; undefined rays receive 90.
    """
    directions = np.stack((lines[:, 1], -lines[:, 0]), axis=1)
    rays = points[:, None, :2] - points[:, None, 2:] * anchors[None]
    dot = np.abs(np.einsum('pli,li->pl', rays, directions))
    cross = np.abs(rays[..., 0] * directions[:, 1] - rays[..., 1] * directions[:, 0])
    angles = np.degrees(np.arctan2(cross, dot))
    return np.where(np.linalg.norm(rays, axis=2) > UNDEFINED_RAY_NORM, angles, UNDEFINED_ANGLE_DEG)


def residual_matrix(lines: np.ndarray, points: np.ndarray, anchors: np.ndarray, batch: int) -> np.ndarray:
    """Evaluate ``angular_residuals_at`` in the estimator's candidate batches."""
    blocks = []
    for offset in range(0, len(points), batch):
        blocks.append(angular_residuals_at(lines, points[offset:offset + batch], anchors))
    return np.concatenate(blocks) if blocks else np.empty((0, len(lines)))
