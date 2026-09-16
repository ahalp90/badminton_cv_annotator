"""Partial matching of observed line groups to finite, named court markings.

This frozen-geometry experiment uses working-image pixels throughout. It has
no access to reference labels, player observations, search or refinement.
Groups describe collinear detector responses, not verified physical stripes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import SEGMENTS_M, _visible_samples, project

MARKINGS = (
    "left_doubles", "left_singles", "centre", "right_singles", "right_doubles",
    "far_baseline", "far_long_service", "far_short_service",
    "near_short_service", "near_long_service", "near_baseline",
)
MARKING_INTERVALS = ((0,), (1,), (2, 3), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,))
GROUP_ANGLE_DEG = 3.0
GROUP_DISTANCE_PX = 3.0
MATCH_ANGLE_DEG = 5.0
DISTANCE_SIGMA_PX = 2.0
SUPPORT_DISTANCE_PX = 5.0
MARKING_SAMPLES = 64
FRAGMENT_SAMPLES = 16
EXPLAINED_FRACTION = 0.5
ABSENT_FRACTION = 0.2
EXPLAINED_LENGTH_PX = 40.0
COMPARISON_LENGTH_PX = 100.0


@dataclass(frozen=True)
class Observations:
    segments: np.ndarray  # (visible fragments, 2 endpoints, xy)
    fragment_ids: np.ndarray
    groups: tuple[np.ndarray, ...]  # Member positions in segments; raw IDs stay separate.
    group_lengths: np.ndarray  # Union length per observed group.
    directions: np.ndarray
    lengths: np.ndarray
    samples: np.ndarray  # (visible fragments, samples along fragment, xy)


@dataclass(frozen=True)
class Support:
    forward: np.ndarray  # (11 marking identities, observed groups)
    reverse: np.ndarray
    forward_binary: np.ndarray
    reverse_binary: np.ndarray
    visible: np.ndarray  # One flag per marking identity, including the split centre.


def union_length(intervals: np.ndarray) -> float:
    """Measure covered length without counting overlapping fragments twice."""
    ordered = sorted((float(min(pair)), float(max(pair))) for pair in intervals)
    total = 0.0
    previous_end = -np.inf
    for start, end in ordered:
        total += max(0.0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def prepare_observations(
    segments: np.ndarray, size: tuple[int, int], fragment_ids: np.ndarray | None = None,
) -> Observations:
    """Clip and group raw fragments once, independently of all candidate courts.

    :param fragment_ids: Stable source IDs; defaults to positions in the raw cache.
    :return: Canonically ordered observations retaining raw fragment provenance.
    """
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2).copy()
    if fragment_ids is None:
        fragment_ids = np.arange(len(segments))
    swap = (segments[:, 0, 0] > segments[:, 1, 0]) | (
        (segments[:, 0, 0] == segments[:, 1, 0]) & (segments[:, 0, 1] > segments[:, 1, 1])
    )
    segments[swap] = segments[swap, ::-1]
    samples, visible = _visible_samples(segments[None], size, FRAGMENT_SAMPLES)
    samples = samples[0, visible[0]]
    fragment_ids = fragment_ids[visible[0]]
    segments = samples[:, [0, -1]]
    vectors = segments[:, 1] - segments[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    order = sorted(range(len(segments)),
                   key=lambda index: (-lengths[index], *segments[index].ravel(), fragment_ids[index]))
    segments, samples = segments[order], samples[order]
    lengths, fragment_ids = lengths[order], fragment_ids[order]
    directions = (segments[:, 1] - segments[:, 0]) / lengths[:, None]
    normals = np.stack((-directions[:, 1], directions[:, 0]), axis=1)
    groups: list[list[int]] = []
    angle_limit = np.cos(np.deg2rad(GROUP_ANGLE_DEG))
    for index in range(len(segments)):
        for members in groups:
            leader = members[0]
            parallel = abs(directions[index] @ directions[leader]) >= angle_limit
            distance = np.abs((segments[index] - segments[leader, 0]) @ normals[leader]).max()
            if parallel and distance <= GROUP_DISTANCE_PX:
                members.append(index)
                break
        else:
            groups.append([index])
    group_lengths = []
    for members in groups:
        leader = members[0]
        intervals = (segments[members] - segments[leader, 0]) @ directions[leader]
        group_lengths.append(union_length(intervals))
    return Observations(segments, fragment_ids, tuple(np.asarray(members) for members in groups),
                        np.asarray(group_lengths), directions, lengths, samples)


def distances_to_segments(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """Return finite-segment distances: (points, segments)."""
    vectors = segments[:, 1] - segments[:, 0]
    delta = points[:, None] - segments[None, :, 0]
    fraction = np.einsum("psd,sd->ps", delta, vectors) / np.square(vectors).sum(axis=1)
    nearest = segments[None, :, 0] + np.clip(fraction, 0, 1)[..., None] * vectors[None]
    return np.linalg.norm(points[:, None] - nearest, axis=-1)


def measure_support(homography: np.ndarray, observations: Observations, size: tuple[int, int]) -> Support:
    """Measure both directions of support before choosing any assignment.

    The centre's two finite intervals share an identity. Its unpainted middle
    contributes neither forward samples nor reverse support.
    """
    projected, _ = project(homography[None], SEGMENTS_M)
    projected = projected.reshape(12, 2, 2)
    samples, interval_visible = _visible_samples(projected[None], size, MARKING_SAMPLES)
    samples, interval_visible = samples[0], interval_visible[0]
    vectors = projected[:, 1] - projected[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    compatible = np.abs(directions @ observations.directions.T) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))
    shape = (len(MARKINGS), len(observations.groups))
    forward, reverse = np.zeros(shape), np.zeros(shape)
    forward_binary, reverse_binary = np.zeros(shape), np.zeros(shape)
    visible = np.zeros(len(MARKINGS), dtype=bool)
    for marking, intervals in enumerate(MARKING_INTERVALS):
        active = [index for index in intervals if interval_visible[index]]
        visible[marking] = bool(active)
        if not active or not observations.groups:
            continue
        forward_distances = []
        reverse_distances = []
        for interval in active:
            distances = distances_to_segments(samples[interval], observations.segments)
            forward_distances.append(np.where(compatible[interval][None], distances, np.inf))
            distances = distances_to_segments(observations.samples.reshape(-1, 2), projected[[interval]])
            distances = distances.reshape(len(observations.segments), FRAGMENT_SAMPLES)
            reverse_distances.append(np.where(compatible[interval][:, None], distances, np.inf))
        forward_distance = np.concatenate(forward_distances)
        reverse_distance = np.min(reverse_distances, axis=0)
        reverse_gaussian = np.exp(-0.5 * np.square(reverse_distance / DISTANCE_SIGMA_PX)).mean(axis=1)
        reverse_covered = (reverse_distance <= SUPPORT_DISTANCE_PX).mean(axis=1)
        for group, members in enumerate(observations.groups):
            nearest = forward_distance[:, members].min(axis=1)
            forward[marking, group] = np.exp(-0.5 * np.square(nearest / DISTANCE_SIGMA_PX)).mean()
            forward_binary[marking, group] = (nearest <= SUPPORT_DISTANCE_PX).mean()
            weights = observations.lengths[members]
            reverse[marking, group] = np.average(reverse_gaussian[members], weights=weights)
            reverse_binary[marking, group] = np.average(reverse_covered[members], weights=weights)
    return Support(forward, reverse, forward_binary, reverse_binary, visible)


def choose_assignment(support: Support, lengths: np.ndarray) -> dict:
    """Maximise the mean of forward and reverse coverage with optional unmatched nodes."""
    marking_count, group_count = support.forward.shape
    visible_count = max(int(support.visible.sum()), 1)
    weights = lengths / lengths.sum() if len(lengths) else lengths
    benefit = 0.5 * (support.forward / visible_count + support.reverse * weights[None])
    with_unmatched = np.concatenate((benefit, np.zeros((marking_count, marking_count))), axis=1)
    rows, columns = linear_sum_assignment(with_unmatched, maximize=True)
    pairs = [(int(row), int(column)) for row, column in zip(rows, columns)
             if column < group_count and benefit[row, column] > 0]
    forward = sum(support.forward[row, column] for row, column in pairs) / visible_count
    reverse = sum(support.reverse[row, column] * weights[column] for row, column in pairs)
    return {"score": float((forward + reverse) / 2), "forward": float(forward), "reverse": float(reverse),
            "pairs": pairs}


def independent_support(support: Support, lengths: np.ndarray) -> dict:
    """Use the same measurements while allowing every node to reuse its best counterpart."""
    if not len(lengths):
        return {"score": 0.0, "forward": 0.0, "reverse": 0.0}
    forward = support.forward.max(axis=1).sum() / max(int(support.visible.sum()), 1)
    reverse = np.average(support.reverse.max(axis=0), weights=lengths)
    return {"score": float((forward + reverse) / 2), "forward": float(forward), "reverse": float(reverse)}


def ledger_summary(support: Support, lengths: np.ndarray) -> dict:
    """Build a raw-line differential ledger on the shared whole-image observations."""
    if len(lengths):
        forward = support.forward_binary.max(axis=1)
        explained = support.reverse_binary.max(axis=0) >= EXPLAINED_FRACTION
    else:
        forward = np.zeros(len(MARKINGS))
        explained = np.zeros(0, dtype=bool)
    absent = support.visible & (forward < ABSENT_FRACTION)
    return {"explained": explained.tolist(), "absent": np.flatnonzero(absent).tolist(),
            "explained_count": int((explained & (lengths >= EXPLAINED_LENGTH_PX)).sum()),
            "forward": independent_support(support, lengths)["forward"]}


def ledger_order(entries: list[dict], lengths: np.ndarray) -> tuple[list[str], dict]:
    """Rank by pairwise wins, retaining all alternatives and reporting tied leaders."""
    wins = np.zeros(len(entries), dtype=int)
    long_groups = lengths >= COMPARISON_LENGTH_PX
    edges = np.zeros((len(entries), len(entries)), dtype=bool)
    for first in range(len(entries)):
        for second in range(first + 1, len(entries)):
            first_row, second_row = entries[first]["ledger"], entries[second]["ledger"]
            first_explained = np.asarray(first_row["explained"], dtype=bool)
            second_explained = np.asarray(second_row["explained"], dtype=bool)
            difference = int((first_explained & ~second_explained & long_groups).sum())
            difference -= int((second_explained & ~first_explained & long_groups).sum())
            rules = (difference, len(second_row["absent"]) - len(first_row["absent"]),
                     first_row["explained_count"] - second_row["explained_count"],
                     first_row["forward"] - second_row["forward"])
            decision = next((int(np.sign(rule)) for rule in rules if rule != 0), 0)
            wins[first] += decision
            wins[second] -= decision
            edges[first, second], edges[second, first] = decision > 0, decision < 0
    order = sorted(range(len(entries)), key=lambda index: (-wins[index], entries[index]["id"]))
    # Every directed triangle is a comparison cycle; matrix counting visits it three times.
    integer_edges = edges.astype(np.int64)
    triangles = int(np.trace(integer_edges @ integer_edges @ integer_edges) // 3)
    tied = [entries[index]["id"] for index in order if wins[index] == wins[order[0]]] if order else []
    return [entries[index]["id"] for index in order], {"tied_leaders": tied, "three_cycles": triangles}
