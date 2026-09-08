"""Exercise paint-edge measurement and exclusive whole-fragment assignments."""

import gzip
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.annotator.independent_court import run_stripes
from experiments.annotator.independent_court import stripe_observations as stripes
from experiments.annotator.independent_court.assignment import prepare_observations
from experiments.annotator.independent_court.detector import SEGMENTS_M, project
from experiments.annotator.independent_court.run_assignment import (
    frozen_entries,
    read_replay,
)

SIZE = (400, 400)
HOMOGRAPHY = np.array([[40.0, 0, 50], [0, 125.0, -1350], [0, 0, 1]])


def painted_edges() -> np.ndarray:
    edges = []
    for interval, segment in enumerate(SEGMENTS_M):
        normal = np.array([1.0, 0]) if interval < 6 else np.array([0, 1.0])
        for offset in (-0.02, 0.02):
            edges.append(segment + offset * normal)
    return project(HOMOGRAPHY[None], np.asarray(edges))[0].reshape(-1, 4)


def test_thick_stripes_gain_position_support_with_unchanged_geometry() -> None:
    observations = prepare_observations(painted_edges(), SIZE)
    measured = stripes.measure(HOMOGRAPHY, observations, SIZE)
    result = stripes.compare(measured, stripes.fragment_weights(observations))
    assert result["stripe"]["exclusive"]["score"] > 0.999
    assert result["stripe"]["exclusive"]["score"] > result["centre"]["exclusive"]["score"]
    assert result["stripe"]["paired_per_marking"][10] > 0.999


def test_one_edge_cannot_supply_both_sides_of_a_stripe() -> None:
    negative_edge = SEGMENTS_M[11] + [0, -0.02]
    observed = project(HOMOGRAPHY[None], negative_edge)[0].reshape(-1, 4)
    observations = prepare_observations(observed, SIZE)
    measured = stripes.measure(HOMOGRAPHY, observations, SIZE)
    result = stripes.compare(measured, stripes.fragment_weights(observations))["stripe"]
    assert result["assignments"]["marking"] == [10]
    assert result["assignments"]["position"] == [1]
    assert result["paired_per_marking"][10] == 0
    assert result["exclusive_per_marking"][10] > 0.999


def test_complementary_fragments_support_one_marking_without_requiring_two_edges() -> None:
    pieces = np.array([[[0, 13.38], [3.05, 13.38]], [[3.05, 13.42], [6.10, 13.42]]])
    observed = project(HOMOGRAPHY[None], pieces)[0].reshape(-1, 4)
    observations = prepare_observations(observed, SIZE)
    measured = stripes.measure(HOMOGRAPHY, observations, SIZE)
    result = stripes.compare(measured, stripes.fragment_weights(observations))["stripe"]
    assert result["assignments"]["marking"] == [10, 10]
    assert sorted(result["assignments"]["position"]) == [1, 2]
    assert result["exclusive_per_marking"][10] > 0.999
    assert result["paired_per_marking"][10] < 0.03


def test_fragment_identity_is_constant_and_alternative_is_distinct() -> None:
    reverse = np.array([[[0.9], [0.8]], [[0.7], [0.6]]])
    result = stripes.resolve_fragments(reverse)
    assert result["marking"].tolist() == [0]
    assert result["position"].tolist() == [0]
    assert result["alternative_marking"].tolist() == [1]
    assert result["alternative_strength"].tolist() == [0.7]


def test_centre_gap_has_no_finite_stripe_support() -> None:
    homography = np.array([[40.0, 0, 50], [0, 40.0, 40], [0, 0, 1]])
    gap = np.array([[3.05, 5.5], [3.05, 7.8]])
    observations = prepare_observations(project(homography[None], gap)[0], (400, 620))
    measured = stripes.measure(homography, observations, (400, 620))
    assert measured.reverse[2].max() < 1e-30


def test_unresolved_edges_remain_an_unavailable_diagnostic() -> None:
    homography = np.array([[40.0, 0, 50], [0, 40.0, 40], [0, 0, 1]])
    observed = project(homography[None], SEGMENTS_M)[0].reshape(-1, 4)
    observations = prepare_observations(observed, (400, 620))
    measured = stripes.measure(homography, observations, (400, 620))
    result = stripes.compare(measured, stripes.fragment_weights(observations))["stripe"]
    assert result["paired_per_marking"] == [None] * 11
    assert result["exclusive"]["score"] > 0.999


def test_empty_observations_and_pool() -> None:
    observations = prepare_observations(np.empty((0, 4)), SIZE)
    measured = stripes.measure(HOMOGRAPHY, observations, SIZE)
    result = stripes.compare(measured, stripes.fragment_weights(observations))
    assert result["stripe"]["exclusive"]["score"] == 0
    assert all(order == [] for order in run_stripes.rank([]).values())


def test_fragment_weights_and_scores_ignore_order_and_endpoint_direction() -> None:
    observed = painted_edges().reshape(-1, 2, 2)
    ids = np.arange(len(observed))
    permutation = np.random.default_rng(92).permutation(len(observed))
    first = prepare_observations(observed, SIZE, ids)
    second = prepare_observations(observed[permutation, ::-1], SIZE, ids[permutation])
    first_weights, second_weights = stripes.fragment_weights(first), stripes.fragment_weights(second)
    np.testing.assert_array_equal(first_weights, second_weights)
    assert first_weights.sum() == pytest.approx(1)
    first_score = stripes.compare(stripes.measure(HOMOGRAPHY, first, SIZE), first_weights)
    second_score = stripes.compare(stripes.measure(HOMOGRAPHY, second, SIZE), second_weights)
    assert first_score == second_score


def test_saved_baseline_boundary_and_candidate_order() -> None:
    root = Path(__file__).resolve().parents[1] / "experiments/annotator/independent_court/recorded/player_guided"
    packed, saved = read_replay(root / "marking_refit_replay.zip")
    baseline = json.loads(gzip.decompress((root / "assignment_results.json.gz").read_bytes()))
    case = packed["cases"][0]
    entries = [entry for entry in frozen_entries(saved["records"][0]) if entry["eligible"]][:2]
    first = run_stripes.run_case(case, deepcopy(entries), baseline["records"][0])
    second = run_stripes.run_case(case, deepcopy(entries[::-1]), baseline["records"][0])
    assert first["orders"] == second["orders"]
    for old, new in zip(entries, first["entries"]):
        assert old["corners_px"] == new["corners_px"]
        for model in ("centre", "stripe"):
            scores = new["stripe_evidence"][model]
            assert scores["exclusive"]["score"] <= scores["independent"]["score"] + 1e-12
    changed = deepcopy(entries)
    changed[0]["corners_px"][0][0] += 1
    with pytest.raises(ValueError, match="Baseline geometry or eligibility differs"):
        run_stripes.run_case(case, changed, baseline["records"][0])
