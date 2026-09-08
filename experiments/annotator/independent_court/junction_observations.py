"""Inspect centre-line terminations and crossings on frozen court geometries.

Missing line responses are uncertain evidence of missing paint. This diagnostic
changes neither ranking nor acceptance.
"""

from __future__ import annotations

import numpy as np

from .assignment import (
    DISTANCE_SIGMA_PX,
    MATCH_ANGLE_DEG,
    Observations,
    distances_to_segments,
)
from .detector import SEGMENTS_M, project
from .stripe_observations import POSITION_OFFSETS_M, STRIPE_WIDTH_M

ARM_START_M = STRIPE_WIDTH_M
ARM_END_M = ARM_START_M + 0.20
ARM_SAMPLES = 16
MIN_VISIBLE_SAMPLES = 8
MIN_SPAN_PX = 4 * DISTANCE_SIGMA_PX
PRESENT_SUPPORT = 0.55
ABSENT_SUPPORT = 0.20
SITE_NAMES = ("far_baseline", "far_long_service", "far_short_service",
              "near_short_service", "near_long_service", "near_baseline")
ARM_DIRECTIONS = {"left": (-1.0, 0.0), "right": (1.0, 0.0),
                  "far": (0.0, -1.0), "near": (0.0, 1.0)}


def arm_support(
    homography: np.ndarray,
    junction_m: np.ndarray,
    direction_m: np.ndarray,
    observations: Observations,
    boxes: np.ndarray,
    size: tuple[int, int],
) -> float | None:
    """Measure a visible ray while retaining uncertain stripe positions."""
    normal_m = np.array([-direction_m[1], direction_m[0]])
    along = np.linspace(ARM_START_M, ARM_END_M, ARM_SAMPLES)
    centre = junction_m + along[:, None] * direction_m
    queries_m = centre[None] + POSITION_OFFSETS_M[:, None, None] * normal_m
    queries, _ = project(homography[None], queries_m)
    queries = queries.reshape(3, ARM_SAMPLES, 2)
    supports = []
    for samples in queries:
        in_frame = ((samples >= 0) & (samples <= np.asarray(size) - 1)).all(axis=1)
        inside_box = ((samples[:, None] >= boxes[None, :, :2])
                      & (samples[:, None] <= boxes[None, :, 2:])).all(axis=2).any(axis=1)
        usable = in_frame & ~inside_box
        if usable.sum() < MIN_VISIBLE_SAMPLES:
            continue
        visible_samples = samples[usable]
        vector = samples[-1] - samples[0]
        span = np.linalg.norm(visible_samples[-1] - visible_samples[0])
        if span < MIN_SPAN_PX:
            continue
        direction = vector / np.linalg.norm(vector)
        compatible = np.abs(observations.directions @ direction) >= np.cos(np.deg2rad(MATCH_ANGLE_DEG))
        distances = distances_to_segments(visible_samples, observations.segments)
        response = np.exp(-0.5 * np.square(distances / DISTANCE_SIGMA_PX))
        response = np.where(compatible[None], response, 0.0)
        supports.append(float(response.max(axis=1, initial=0).mean()))
    return max(supports) if supports else None


def expected_vertical_arms(junction_y: float) -> dict[str, bool]:
    """Read painted intervals from the metric template, not detector endpoints."""
    expected = {}
    for name, sign in (("far", -1), ("near", 1)):
        query_y = junction_y + sign * (ARM_START_M + ARM_END_M) / 2
        painted = False
        for segment in SEGMENTS_M[[2, 3]]:
            painted |= bool(segment[:, 1].min() < query_y < segment[:, 1].max())
        expected[name] = painted
    return expected


def classify_site(arms: dict[str, float | None], expected: dict[str, bool]) -> dict:
    """Classify supported, observable sites; intermediate evidence stays unresolved."""
    numeric = {name: value for name, value in arms.items() if value is not None}
    usable = len(numeric) == len(arms)
    if usable:
        usable = (numeric["left"] >= PRESENT_SUPPORT and numeric["right"] >= PRESENT_SUPPORT
                  and max(numeric["far"], numeric["near"]) >= PRESENT_SUPPORT)
    disagreements, agreements = [], []
    if usable:
        for name, painted in expected.items():
            support = numeric[name]
            if support >= PRESENT_SUPPORT:
                (agreements if painted else disagreements).append(name)
            elif support <= ABSENT_SUPPORT:
                (disagreements if painted else agreements).append(name)
    return {"usable": bool(usable), "agreements": agreements, "disagreements": disagreements}


def measure(
    homography: np.ndarray, observations: Observations, boxes: np.ndarray, size: tuple[int, int],
) -> dict:
    """Inspect all six named centre-line junctions without changing the candidate."""
    sites = []
    for name, transverse in zip(SITE_NAMES, SEGMENTS_M[6:]):
        junction_m = np.array([SEGMENTS_M[2, 0, 0], transverse[0, 1]])
        arms = {}
        for arm, direction in ARM_DIRECTIONS.items():
            arms[arm] = arm_support(homography, junction_m, np.asarray(direction), observations, boxes, size)
        expected = expected_vertical_arms(float(junction_m[1]))
        sites.append({"marking": name, "arms": arms, "expected": expected, **classify_site(arms, expected)})
    return {"sites": sites, "usable_sites": sum(site["usable"] for site in sites),
            "disagreements": sum(len(site["disagreements"]) for site in sites)}
