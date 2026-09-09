"""Test fragment membership and paint-edge positions at fixed court geometry.

Each raw fragment can take one marking identity and one position hypothesis.
Many fragments may explain the same marking. Geometry and labels never change
inside this module; paired-edge evidence is a diagnostic, not an acceptance rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import paint_geometry
from .assignment import (
    DISTANCE_SIGMA_PX,
    MARKING_INTERVALS,
    MARKING_SAMPLES,
    MARKINGS,
    MATCH_ANGLE_DEG,
    Observations,
    distances_to_segments,
)
from .detector import SEGMENTS_M, _visible_samples, project
from .paint_geometry import POSITION_OFFSETS_M, positioned_segments

STRIPE_WIDTH_M = paint_geometry.STRIPE_WIDTH_M
POSITION_NAMES = ("centre", "negative_edge", "positive_edge")
RESOLVABLE_WIDTH_PX = 2 * DISTANCE_SIGMA_PX


@dataclass(frozen=True)
class StripeEvidence:
    forward: tuple[np.ndarray, ...]  # Per marking: (3 positions, visible samples, raw fragments).
    reverse: np.ndarray  # (11 markings, 3 positions, raw fragments).
    resolvable: tuple[np.ndarray, ...]  # Per marking: both edges visible and sufficiently separated.
    visible: np.ndarray


def fragment_weights(observations: Observations) -> np.ndarray:
    """Distribute each observed group's union length over its raw fragments."""
    weights = np.zeros(len(observations.segments))
    for group, members in enumerate(observations.groups):
        lengths = observations.lengths[members]
        weights[members] = observations.group_lengths[group] * lengths / lengths.sum()
    return weights / weights.sum() if len(weights) else weights


def interval_evidence(
    homography: np.ndarray,
    inverse: np.ndarray,
    interval: int,
    centre_samples: np.ndarray,
    observations: Observations,
    size: tuple[int, int],
    centres: np.ndarray = SEGMENTS_M,
    boundary_tolerance_px: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure all positions at matching court locations on one finite interval."""
    normal_m = np.array([1.0, 0.0]) if interval < 6 else np.array([0.0, 1.0])
    offsets = POSITION_OFFSETS_M[:, None] * normal_m
    shifted_segments = positioned_segments(centres, np.full(3, interval), np.arange(3))
    projected, _ = project(homography[None], shifted_segments)
    projected = projected.reshape(3, 2, 2)
    vectors = projected[:, 1] - projected[:, 0]
    directions = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    compatible = np.abs(directions @ observations.directions.T) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))

    court_samples, _ = project(inverse[None], centre_samples)
    shifted_samples, _ = project(homography[None], court_samples[0][None] + offsets[:, None])
    shifted_samples = shifted_samples.reshape(3, len(centre_samples), 2)
    inside = ((shifted_samples >= -boundary_tolerance_px)
              & (shifted_samples <= np.asarray(size) - 1 + boundary_tolerance_px)).all(axis=2)
    forward = []
    for position in range(len(POSITION_OFFSETS_M)):
        distances = distances_to_segments(shifted_samples[position], observations.segments)
        response = np.exp(-0.5 * np.square(distances / DISTANCE_SIGMA_PX))
        forward.append(np.where(inside[position, :, None] & compatible[position, None], response, 0.0))

    distances = distances_to_segments(observations.samples.reshape(-1, 2), projected)
    distances = distances.reshape(len(observations.segments), observations.samples.shape[1], 3)
    distances = np.where(compatible.T[:, None], distances, np.inf)
    width = np.linalg.norm(shifted_samples[1] - shifted_samples[2], axis=1)
    resolvable = inside[1] & inside[2] & (width >= RESOLVABLE_WIDTH_PX)
    return np.asarray(forward), distances, resolvable


def measure(
    homography: np.ndarray, observations: Observations, size: tuple[int, int], centres: np.ndarray = SEGMENTS_M,
    boundary_tolerance_px: float = 0.0,
) -> StripeEvidence:
    """Cache nominal-centre and paint-edge evidence for the complete comparison."""
    projected, _ = project(homography[None], centres)
    samples, interval_visible = _visible_samples(projected.reshape(1, 12, 2, 2), size, MARKING_SAMPLES)
    inverse = np.linalg.inv(homography)
    forward = []
    reverse = np.zeros((len(MARKINGS), len(POSITION_OFFSETS_M), len(observations.segments)))
    resolvable = []
    visible = np.zeros(len(MARKINGS), dtype=bool)
    for marking, intervals in enumerate(MARKING_INTERVALS):
        forward_parts, reverse_parts, resolved_parts = [], [], []
        for interval in intervals:
            if not interval_visible[0, interval]:
                continue
            found, distances, resolved = interval_evidence(
                homography, inverse, interval, samples[0, interval], observations, size, centres, boundary_tolerance_px,
            )
            forward_parts.append(found)
            reverse_parts.append(distances)
            resolved_parts.append(resolved)
        visible[marking] = bool(forward_parts)
        if forward_parts:
            forward.append(np.concatenate(forward_parts, axis=1))
            nearest = np.minimum.reduce(reverse_parts)
            reverse[marking] = np.exp(-0.5 * np.square(nearest / DISTANCE_SIGMA_PX)).mean(axis=1).T
            resolvable.append(np.concatenate(resolved_parts))
        else:
            forward.append(np.empty((3, 0, len(observations.segments))))
            resolvable.append(np.zeros(0, dtype=bool))
    return StripeEvidence(tuple(forward), reverse, tuple(resolvable), visible)


def resolve_fragments(reverse: np.ndarray) -> dict[str, np.ndarray]:
    """Choose whole-fragment marking/position pairs by maximum mean reverse support.

    This solves the separable reverse objective. It does not jointly optimise
    forward coverage, whose value depends on the other assigned fragments.
    """
    marking_count, position_count, fragment_count = reverse.shape
    flattened = reverse.reshape(marking_count * position_count, fragment_count)
    selected = flattened.argmax(axis=0)
    strength = flattened[selected, np.arange(fragment_count)]
    marking, position = selected // position_count, selected % position_count
    return describe_assignment(reverse, np.where(strength > 0, marking, -1), np.where(strength > 0, position, -1))


def describe_assignment(reverse: np.ndarray, marking: np.ndarray, position: np.ndarray) -> dict[str, np.ndarray]:
    """Measure a supplied identity assignment without changing it when support vanishes."""
    marking_count, _, fragment_count = reverse.shape
    strength = reverse[np.maximum(marking, 0), np.maximum(position, 0), np.arange(fragment_count)]
    strength = np.where(marking >= 0, strength, 0.0)
    by_marking = reverse.max(axis=1)
    alternatives = np.where(np.arange(marking_count)[:, None] == marking[None], -1.0, by_marking)
    alternative_marking = alternatives.argmax(axis=0)
    alternative_strength = alternatives[alternative_marking, np.arange(fragment_count)]
    return {"marking": marking, "position": position, "strength": strength,
            "alternative_marking": np.where(alternative_strength > 0, alternative_marking, -1),
            "alternative_strength": np.maximum(alternative_strength, 0)}


def score_model(
    evidence: StripeEvidence, weights: np.ndarray, position_count: int, fixed_assignment: dict | None = None,
) -> dict:
    """Compare evidence reuse and exclusive fragment identities for one position model."""
    responses = evidence.reverse[:, :position_count]
    if fixed_assignment is None:
        resolved = resolve_fragments(responses)
    else:
        resolved = describe_assignment(responses, np.asarray(fixed_assignment["marking"]),
                                       np.asarray(fixed_assignment["position"]))
    independent_per_marking = np.zeros(len(MARKINGS))
    exclusive_per_marking = np.zeros(len(MARKINGS))
    paired_per_marking: list[float | None] = []
    paired_samples = []
    for marking, full_forward in enumerate(evidence.forward):
        response = full_forward[:position_count]
        allowed = ((resolved["marking"][None] == marking)
                   & (resolved["position"][None] == np.arange(position_count)[:, None]))
        exclusive = np.where(allowed[:, None], response, 0.0)
        if evidence.visible[marking]:
            independent_per_marking[marking] = response.max(axis=(0, 2), initial=0).mean()
            exclusive_per_marking[marking] = exclusive.max(axis=(0, 2), initial=0).mean()
        selected_samples = evidence.resolvable[marking]
        paired_samples.append(int(selected_samples.sum()) if position_count == 3 else 0)
        if position_count == 3 and selected_samples.any():
            negative = exclusive[1].max(axis=1, initial=0)
            positive = exclusive[2].max(axis=1, initial=0)
            paired_per_marking.append(float(np.minimum(negative, positive)[selected_samples].mean()))
        else:
            paired_per_marking.append(None)
    visible_count = max(int(evidence.visible.sum()), 1)
    reverse = float(resolved["strength"] @ weights)
    independent = float(independent_per_marking.sum() / visible_count)
    exclusive = float(exclusive_per_marking.sum() / visible_count)
    return {"independent": {"forward": independent, "reverse": reverse, "score": (independent + reverse) / 2},
            "exclusive": {"forward": exclusive, "reverse": reverse, "score": (exclusive + reverse) / 2},
            "independent_per_marking": independent_per_marking.tolist(),
            "exclusive_per_marking": exclusive_per_marking.tolist(),
            "assignments": {key: values.tolist() for key, values in resolved.items()},
            "paired_per_marking": paired_per_marking, "paired_samples": paired_samples}


def compare(evidence: StripeEvidence, weights: np.ndarray) -> dict:
    return {"centre": score_model(evidence, weights, 1), "stripe": score_model(evidence, weights, 3),
            "visible_markings": np.flatnonzero(evidence.visible).tolist()}
