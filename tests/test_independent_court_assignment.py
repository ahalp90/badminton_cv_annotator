"""Check assignment semantics and frozen-replay boundaries without model inference."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from experiments.annotator.independent_court import assignment, run_assignment
from experiments.annotator.independent_court.detector import SEGMENTS_M, project

SIZE = (400, 620)
HOMOGRAPHY = np.array([[40.0, 0, 50], [0, 40.0, 40], [0, 0, 1]])


def test_exact_finite_court_has_eleven_marking_identities() -> None:
    segments = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 4)
    observations = assignment.prepare_observations(segments, SIZE)
    support = assignment.measure_support(HOMOGRAPHY, observations, SIZE)
    result = assignment.choose_assignment(support, observations.group_lengths)
    assert len(observations.groups) == 11
    assert len(result["pairs"]) == 11
    assert result["score"] == pytest.approx(1.0)
    centre_group = dict(result["pairs"])[2]
    assert sorted(observations.fragment_ids[observations.groups[centre_group]]) == [2, 3]


def test_centre_gap_does_not_supply_painted_support() -> None:
    gap = np.array([[3.05, 5.5], [3.05, 7.8]])
    segments = project(HOMOGRAPHY[None], gap)[0].reshape(1, 4)
    observations = assignment.prepare_observations(segments, SIZE)
    support = assignment.measure_support(HOMOGRAPHY, observations, SIZE)
    assert support.reverse_binary[2, 0] == 0
    assert support.reverse[2, 0] < 1e-30


def test_assignment_solves_global_competition_instead_of_greedy_pairs() -> None:
    matrix = np.array([[0.9, 0.8], [0.85, 0.1]])
    support = assignment.Support(matrix, matrix, matrix, matrix, np.ones(2, dtype=bool))
    result = assignment.choose_assignment(support, np.ones(2))
    assert result["pairs"] == [(0, 1), (1, 0)]
    assert result["score"] == pytest.approx(0.825)


def test_one_observation_cannot_support_two_marking_identities() -> None:
    matrix = np.ones((2, 1))
    support = assignment.Support(matrix, matrix, matrix, matrix, np.ones(2, dtype=bool))
    matched = assignment.choose_assignment(support, np.ones(1))
    independent = assignment.independent_support(support, np.ones(1))
    assert len(matched["pairs"]) == 1
    assert matched["forward"] == 0.5
    assert independent["forward"] == 1.0
    assert matched["score"] < independent["score"]


def test_observation_outside_candidate_stays_in_comparison() -> None:
    segments = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 4)
    original = assignment.prepare_observations(segments, SIZE)
    with_background = assignment.prepare_observations(np.vstack((segments, [380, 40, 380, 580])), SIZE)
    clean_support = assignment.measure_support(HOMOGRAPHY, original, SIZE)
    background_support = assignment.measure_support(HOMOGRAPHY, with_background, SIZE)
    clean = assignment.choose_assignment(clean_support, original.group_lengths)
    background = assignment.choose_assignment(background_support, with_background.group_lengths)
    assert len(with_background.groups) == len(original.groups) + 1
    assert background["reverse"] < clean["reverse"]


def test_grouping_and_assignment_ignore_endpoint_and_input_order() -> None:
    segments = project(HOMOGRAPHY[None], SEGMENTS_M)[0].reshape(-1, 2, 2)
    ids = np.arange(len(segments))
    permutation = np.random.default_rng(19).permutation(len(segments))
    original = assignment.prepare_observations(segments, SIZE, ids)
    reordered = assignment.prepare_observations(segments[permutation, ::-1], SIZE, ids[permutation])
    np.testing.assert_array_equal(original.segments, reordered.segments)
    np.testing.assert_array_equal(original.fragment_ids, reordered.fragment_ids)
    np.testing.assert_array_equal(original.group_lengths, reordered.group_lengths)
    first = assignment.measure_support(HOMOGRAPHY, original, SIZE)
    second = assignment.measure_support(HOMOGRAPHY, reordered, SIZE)
    first_assignment = assignment.choose_assignment(first, original.group_lengths)
    assert first_assignment == assignment.choose_assignment(second, reordered.group_lengths)


def test_duplicate_fragment_does_not_double_group_length() -> None:
    segments = np.array([[50, 50, 150, 50], [50, 50, 150, 50], [100, 50, 200, 50]])
    observations = assignment.prepare_observations(segments, SIZE)
    np.testing.assert_array_equal(observations.group_lengths, [150])


def test_empty_observations_and_candidate_pool() -> None:
    observations = assignment.prepare_observations(np.empty((0, 4)), SIZE)
    support = assignment.measure_support(HOMOGRAPHY, observations, SIZE)
    assert assignment.choose_assignment(support, observations.group_lengths)["score"] == 0
    assert assignment.independent_support(support, observations.group_lengths)["score"] == 0
    orders, diagnostics = run_assignment.rank_entries([], observations.group_lengths)
    assert all(order == [] for order in orders.values())
    assert diagnostics == {"tied_leaders": [], "three_cycles": 0}


def test_offscreen_markings_remain_unobserved() -> None:
    shifted = HOMOGRAPHY.copy()
    shifted[1, 2] = -200
    observations = assignment.prepare_observations(np.empty((0, 4)), SIZE)
    support = assignment.measure_support(shifted, observations, SIZE)
    assert not support.visible[5]  # Far baseline above the image.
    assert support.visible[10]
    assert 5 not in assignment.ledger_summary(support, observations.group_lengths)["absent"]


def test_replay_strips_labels_preserves_geometry_and_ignores_candidate_order() -> None:
    root = Path(__file__).resolve().parents[1]
    replay = root / "experiments/annotator/independent_court/recorded/player_guided/marking_refit_replay.zip"
    inputs, saved = run_assignment.read_replay(replay)
    case = inputs["cases"][0]
    source = saved["records"][0]
    entries = [entry for entry in run_assignment.frozen_entries(source) if entry["eligible"]][:3]
    before = deepcopy(entries)
    assert entries
    assert all("metrics" not in entry for entry in entries)
    poisoned = deepcopy(source)
    for entry in poisoned["entries"]:
        entry["metrics"] = {"corner_max_error_px": -999}
    assert run_assignment.frozen_entries(poisoned) == run_assignment.frozen_entries(source)
    first = run_assignment.run_case(case, deepcopy(entries))
    second = run_assignment.run_case(case, deepcopy(entries[::-1]))
    assert first["orders"] == second["orders"]
    for previous, current in zip(before, first["entries"]):
        assert previous["corners_px"] == current["corners_px"]
        assert current["assignment"]["score"] <= current["independent"]["score"] + 1e-12


def test_reference_metrics_do_not_change_ranking() -> None:
    orders = {scheme: ["0000:original"] for scheme in run_assignment.SCHEMES}
    corners = project(HOMOGRAPHY[None], run_assignment.CORNER_COURT_M)[0][0].tolist()
    record = {"id": "test", "dimensions": {"width": 400, "height": 620},
              "entries": [{"id": "0000:original", "corners_px": corners}], "orders": orders}
    before = deepcopy(orders)
    run_assignment.attach_metrics([record], {"test": {"corners_px": corners, "landmarks": []}})
    assert record["orders"] == before
    assert record["entries"][0]["metrics"]["corner_max_error_px"] == 0
