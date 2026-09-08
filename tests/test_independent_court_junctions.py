"""Distinguish measured junction arms from raw detector endpoints and missing views."""

import numpy as np

from experiments.annotator.independent_court import junction_observations as junctions
from experiments.annotator.independent_court.assignment import prepare_observations
from experiments.annotator.independent_court.detector import SEGMENTS_M, project

SIZE = (500, 400)
HOMOGRAPHY = np.array([[60.0, 0, 50], [0, 125.0, -1350], [0, 0, 1]])
NO_BOXES = np.empty((0, 4))


def test_template_distinguishes_terminations_and_crossings() -> None:
    assert junctions.expected_vertical_arms(0) == {"far": False, "near": True}
    assert junctions.expected_vertical_arms(0.76) == {"far": True, "near": True}
    assert junctions.expected_vertical_arms(4.72) == {"far": True, "near": False}
    assert junctions.expected_vertical_arms(8.68) == {"far": False, "near": True}
    assert junctions.expected_vertical_arms(12.64) == {"far": True, "near": True}
    assert junctions.expected_vertical_arms(13.4) == {"far": True, "near": False}


def test_ideal_baseline_and_long_service_have_different_arm_patterns() -> None:
    observed = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 4)
    observations = prepare_observations(observed, SIZE)
    result = junctions.measure(HOMOGRAPHY, observations, NO_BOXES, SIZE)
    long_service, baseline = result["sites"][-2:]
    assert long_service["usable"] and baseline["usable"]
    assert long_service["arms"]["far"] > 0.99 and long_service["arms"]["near"] > 0.99
    assert baseline["arms"]["far"] > 0.99 and baseline["arms"]["near"] < 0.01
    assert baseline["disagreements"] == []
    wrong_identity = junctions.classify_site(baseline["arms"], {"far": True, "near": True})
    assert wrong_identity["disagreements"] == ["near"]


def test_occluded_outside_arm_is_unknown_not_absent() -> None:
    observed = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 4)
    observations = prepare_observations(observed, SIZE)
    boxes = np.array([[220, 327, 245, 360]])
    baseline = junctions.measure(HOMOGRAPHY, observations, boxes, SIZE)["sites"][-1]
    assert baseline["arms"]["near"] is None
    assert not baseline["usable"]
    assert baseline["disagreements"] == []


def test_clipped_outside_arm_is_unknown_not_absent() -> None:
    size = (500, 334)
    observed = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 4)
    observations = prepare_observations(observed, size)
    baseline = junctions.measure(HOMOGRAPHY, observations, NO_BOXES, size)["sites"][-1]
    assert baseline["arms"]["near"] is None
    assert not baseline["usable"]


def test_painted_continuation_contradicts_a_baseline_termination() -> None:
    continuation = np.array([[[3.05, 13.40], [3.05, 14.00]]])
    template = np.concatenate((SEGMENTS_M, continuation))
    observed = project(HOMOGRAPHY[None], template)[0].reshape(-1, 4)
    observations = prepare_observations(observed, SIZE)
    baseline = junctions.measure(HOMOGRAPHY, observations, NO_BOXES, SIZE)["sites"][-1]
    assert baseline["usable"]
    assert baseline["disagreements"] == ["near"]


def test_weak_or_intermediate_evidence_stays_unresolved() -> None:
    expected = {"far": True, "near": True}
    weak = junctions.classify_site({"left": 0.1, "right": 0.9, "far": 0.9, "near": 0.0}, expected)
    assert not weak["usable"] and weak["disagreements"] == []
    intermediate = junctions.classify_site({"left": 0.9, "right": 0.9, "far": 0.9, "near": 0.4}, expected)
    assert intermediate["usable"] and intermediate["disagreements"] == []


def test_empty_observations_do_not_establish_a_junction() -> None:
    observations = prepare_observations(np.empty((0, 4)), SIZE)
    result = junctions.measure(HOMOGRAPHY, observations, NO_BOXES, SIZE)
    assert result["usable_sites"] == 0
    assert result["disagreements"] == 0
