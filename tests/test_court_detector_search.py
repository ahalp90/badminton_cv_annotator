"""Synthetic checks for direction search, axis matching and camera pruning."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from court_detector import (
    candidate_pool,
    directions,
    generation,
    geometry,
    line_matching,
    line_observations,
    players,
    proposals,
)

COURT_WIDTH_M, COURT_LENGTH_M = players.COURT_SIZE_M


@pytest.mark.parametrize(("limit", "expected"), [(0, []), (1, [3]), (3, [3, 1, 2]), (8, [3, 1, 2, 4])])
def test_rectangle_round_robin_preserves_order_when_pools_run_out(limit: int, expected: list[int]) -> None:
    pools = [np.array(pool, dtype=int) for pool in ([], [3, 1, 2], [3], [], [1, 4, 2], [])]
    np.testing.assert_array_equal(directions.round_robin_union(pools, limit), expected)


def axis_matches(parameters: np.ndarray) -> line_matching.AxisMatches:
    """Axis hypotheses that are all retained; only parameters and retained matter to the player test."""
    count = len(parameters)
    every = np.arange(count)
    return line_matching.AxisMatches(parameters, np.zeros(count), np.full((count, 1), -1),
                                       np.zeros((count, 4), dtype=int), np.zeros(count, dtype=int),
                                       every, every, {})


def axis_parameters(rng: np.random.Generator, count: int, extent: float) -> np.ndarray:
    """Scale/offset pairs near the true axis, half of them reversed so canonicalise must relabel."""
    scale = rng.uniform(.8, 1.2, count) * np.where(np.arange(count) % 2, -1., 1.)
    shift = np.where(scale < 0, extent, 0.) + rng.uniform(-1.5, 1.5, count)
    return np.column_stack((scale, shift))


def test_joint_player_fractions_match_players_on_every_combined_court() -> None:
    rng = np.random.default_rng(7)
    court_corners = np.array([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
                             dtype=np.float32)
    image_corners = np.array([[380, 150], [580, 150], [760, 470], [200, 470]], dtype=np.float32)
    basis = cv2.getPerspectiveTransform(court_corners, image_corners).astype(float)
    horizontal = axis_matches(axis_parameters(rng, 40, COURT_WIDTH_M))
    vertical = axis_matches(axis_parameters(rng, 40, COURT_LENGTH_M))

    # Feet spread over and around the court, as (sampled frames, foot samples, image xy), with missing slots.
    feet_m = rng.uniform((-2., -3.), (COURT_WIDTH_M + 2., COURT_LENGTH_M + 3.), size=(12, 6, 2))
    feet_px = cv2.perspectiveTransform(feet_m.reshape(1, -1, 2), basis).reshape(feet_m.shape)
    feet_px[rng.random(feet_px.shape[:2]) < .2] = np.nan

    transforms, _ = proposals.combine(basis, horizontal, vertical)
    transforms, rotated = proposals.canonicalise(transforms)
    expected_one, expected_two = players.player_fractions(transforms, feet_px)
    one, two = line_matching.joint_player_fractions(basis, horizontal, vertical, feet_px)

    assert rotated.any() and not rotated.all()
    usable = (expected_one == 1) & (expected_two >= .5)
    assert usable.any() and not usable.all()
    np.testing.assert_array_equal(one, expected_one)
    np.testing.assert_array_equal(two, expected_two)
    # The three tiers partition the courts: the final choice's player rule, then a player on
    # the court in at least half the samples, then the rest.
    tiers = line_matching.support_tiers(one, two)
    np.testing.assert_array_equal(tiers == 0, usable)
    np.testing.assert_array_equal(tiers == 1, ~usable & (expected_one >= .5))
    # At the thresholds: (one player, both halves) fractions and their tier.
    edge_one, edge_two = np.array([1., 1., .5, .49, 0.]), np.array([.5, .49, .5, .49, 0.])
    assert line_matching.support_tiers(edge_one, edge_two).tolist() == [0, 1, 1, 2, 2]


@pytest.mark.parametrize("feet_px", [np.empty((0, 0, 2)), np.full((3, 2, 2), np.nan), np.empty((3, 0, 2))])
def test_courts_without_player_observations_get_zero_fractions_and_the_weakest_tier(feet_px: np.ndarray) -> None:
    """No sampled frames, as without people inputs; frames where nobody stands; frames without player slots."""
    rng = np.random.default_rng(7)
    horizontal = axis_matches(axis_parameters(rng, 4, COURT_WIDTH_M))
    vertical = axis_matches(axis_parameters(rng, 5, COURT_LENGTH_M))
    one, two = line_matching.joint_player_fractions(np.eye(3), horizontal, vertical, feet_px)
    np.testing.assert_array_equal(one, np.zeros(20))
    np.testing.assert_array_equal(two, np.zeros(20))
    assert line_matching.support_tiers(one, two).tolist() == [geometry.NO_PLAYER_SUPPORT] * 20


def test_retention_orders_exactly_tied_scores_by_player_tier() -> None:
    """Both candidate caps keep the higher score first, then the stronger tier, then the earlier court."""
    base = np.array([[300., 150.], [660., 150.], [800., 450.], [160., 450.]])
    # (x shift in working px, score, player tier). The last court lies within the 2 px
    # retention distance of the first and ties its score with a stronger tier.
    courts = [(0., .9, 2), (40., .9, 1), (80., .9, 0), (120., .9, 0), (160., .95, 2), (200., .8, 0), (1., .9, 1)]
    candidates = [geometry.Candidate(base + [shift, 0.], score, (0., 0.), (0, 0), tier)
                  for shift, score, tier in courts]
    positions = {id(candidate): position for position, candidate in enumerate(candidates)}
    for limit, expected in [(4, [4, 2, 3, 1]), (7, [4, 2, 3, 1, 6, 5])]:
        retained = generation.select(candidate_pool, candidates, limit)
        assert [positions[id(candidate)] for candidate in retained] == expected


def test_axis_scores_match_the_einsum_form_exactly() -> None:
    """score_axes' per-endpoint distances must reproduce the einsum form bit for bit."""
    rng = np.random.default_rng(20260924)
    for trial in range(40):
        axis = trial % 2
        coordinates = geometry.X_COORDS if axis == 0 else geometry.Y_COORDS
        basis = np.array([[300., 5., 480.], [10., -200., 270.], [.01, .2, 1.]])
        basis[:2] += rng.normal(0, 20, (2, 3))
        parameters = np.column_stack((rng.uniform(.1, 2, 64) * rng.choice([-1, 1], 64), rng.uniform(-3, 3, 64)))
        endpoints = rng.uniform(-100, 1100, (int(rng.integers(3, 129)), 2, 2))

        inverse = np.linalg.inv(basis)
        predicted = parameters[:, :1] * coordinates + parameters[:, 1:]
        lines = inverse[axis][None, None] - predicted[..., None] * inverse[2]
        norms = np.linalg.norm(lines[..., :2], axis=2)
        distances = np.abs(np.einsum('hmd,ged->hmge', lines[..., :2], endpoints) + lines[..., 2, None, None])
        distances = distances.max(axis=3) / np.maximum(norms[..., None], 1e-15)
        nearest = distances.argmin(axis=2)
        residual = np.take_along_axis(distances, nearest[..., None], axis=2)[..., 0]
        response = np.exp(-.5 * np.square(residual / line_observations.DISTANCE_SIGMA_PX))
        same_group = nearest[:, :, None] == nearest[:, None, :]
        stronger = response[:, None, :] > response[:, :, None]
        order = np.arange(len(coordinates))
        tied_before = (response[:, None, :] == response[:, :, None]) & (order[None, None, :] < order[None, :, None])
        exclusive = ~np.any(same_group & (stronger | tied_before), axis=2)
        supported = exclusive & (residual <= line_observations.SUPPORT_DISTANCE_PX)
        expected = ((response * exclusive).mean(axis=1), np.where(supported, nearest, -1), supported.sum(axis=1))

        actual = line_matching.score_axes(parameters, coordinates, endpoints, basis, axis)
        # Bytes rather than values, so signed zeros and NaN bit patterns must match too.
        for actual_part, expected_part in zip(actual, expected, strict=True):
            assert (actual_part.dtype, actual_part.shape, actual_part.tobytes()) == (
                expected_part.dtype, expected_part.shape, expected_part.tobytes())


SIZE = (960, 540)
CAMERA_MATRIX = np.array([[800., 0., 480.], [0., 800., 270.], [0., 0., 1.]])
# Court on the floor (z = 0): x across the 6.1 m width, y along the 13.4 m length.
COURT_CORNERS_M = np.array([[0., 0., 0.], [6.1, 0., 0.], [6.1, 13.4, 0.], [0., 13.4, 0.]])
CAMERA_POSITION_M = np.array([3.05, -6., 5.])


def camera_rows(pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Camera x (right), y (down) and z (forward) axes in floor coordinates, one per row.

    The camera stands behind the near baseline facing along the court, pitched down, then
    rolled about its forward axis.
    """
    pitch, roll = np.radians(pitch_deg), np.radians(roll_deg)
    right = np.array([1., 0., 0.])
    forward = np.array([0., np.cos(pitch), -np.sin(pitch)])
    down = np.cross(forward, right)
    return np.array([np.cos(roll) * right + np.sin(roll) * down,
                     -np.sin(roll) * right + np.cos(roll) * down, forward])


def view(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The court's two vanishing points (homogeneous px) and its corners (1, 4, 2) in px."""
    vanishing_points = (CAMERA_MATRIX @ rows @ np.eye(3)[:, :2]).T
    projected = (CAMERA_MATRIX @ rows @ (COURT_CORNERS_M - CAMERA_POSITION_M).T).T
    return vanishing_points, (projected[:, :2] / projected[:, 2:])[None]


def test_upright_and_mildly_rolled_cameras_pass_both_tests() -> None:
    for roll in (0., 10., -30.):
        points, corners = view(camera_rows(30., roll))
        tilt = candidate_pool.horizon_tilt_deg(points, SIZE)
        assert tilt is not None and abs(tilt - abs(roll)) < 1e-6
        assert proposals.below_horizon(points, corners, SIZE).tolist() == [True]


def test_a_camera_on_its_side_has_a_steep_horizon() -> None:
    points, _ = view(camera_rows(30., 90.))
    tilt = candidate_pool.horizon_tilt_deg(points, SIZE)
    assert tilt is not None and abs(tilt - 90.) < 1e-6


def test_an_upside_down_camera_has_a_level_horizon_but_the_court_above_it() -> None:
    points, corners = view(camera_rows(30., 180.))
    tilt = candidate_pool.horizon_tilt_deg(points, SIZE)
    assert tilt is not None and tilt < 1e-6
    assert proposals.below_horizon(points, corners, SIZE).tolist() == [False]


def test_cameras_looking_straight_or_nearly_straight_down_pass() -> None:
    overhead = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    for rows in (overhead, camera_rows(89.9, 170.)):
        points, corners = view(rows)
        assert proposals.horizon(points, SIZE) is None
        assert candidate_pool.horizon_tilt_deg(points, SIZE) is None
        assert proposals.below_horizon(points, corners, SIZE).tolist() == [True]


def estimator(count: int) -> dict:
    return {
        "points_working": [[float(index), 1., 1.] for index in range(count)],
        "direction_lines": [[1., 0., 0.], [0., 1., 0.]],
        "retained_support_masks": [[True, True] for _ in range(count)],
        "normalised_to_working": np.eye(3).tolist(),
    }


def test_full_direction_budget_preserves_ids_without_svd(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_svd(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("The full direction budget must bypass SVD")

    monkeypatch.setattr(np.linalg, "svd", forbidden_svd)
    screen = generation.screen_groups(estimator(16), 16)
    assert screen["ranked_original_ids"] == list(range(16))
    assert screen["selected_original_ids"] == list(range(16))
    assert screen["diagnostics"] == []


def test_empty_estimator_has_no_direction_groups() -> None:
    points, record = directions.estimate(
        np.empty((0, 4)), (640, 360), directions.Settings(pencil_selection="coverage")
    )
    assert points.shape == (0, 3)
    assert record["direction_lines"] == []
    assert record["retained_support_masks"] == []
    for budget in (12, 16):
        screen = generation.screen_groups(record, budget)
        assert screen["ranked_original_ids"] == []
        assert screen["selected_original_ids"] == []


def test_invalid_direction_records_fail_before_search() -> None:
    with pytest.raises(ValueError, match="at most 16"):
        generation.screen_groups(estimator(17), 16)
    with pytest.raises(ValueError, match="12 or 16"):
        generation.screen_groups(estimator(2), 8)
    record = estimator(2)
    record["normalised_to_working"] = np.eye(2).tolist()
    with pytest.raises(ValueError, match="transform or support-mask shape"):
        generation.screen_groups(record, 16)
    record = estimator(2)
    record["points_working"][0][0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        generation.screen_groups(record, 16)


@pytest.mark.parametrize("native_size", [(1920, 1080), (1366, 768)])
@pytest.mark.parametrize("points", [
    np.array([[500., 100., 1.], [100., -200., 1.]]),
    np.array([[1., 0., 0.], [0., 1., 0.]]),
])
def test_generation_checks_camera_bound_in_native_coordinates(
    monkeypatch: pytest.MonkeyPatch, native_size: tuple[int, int], points: np.ndarray,
) -> None:
    size = (960, 540)
    calls = []

    def record_bound(pair: np.ndarray, actual_size: tuple[int, int]) -> float:
        calls.append((pair.copy(), actual_size))
        return 1.

    def forbidden_matcher(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Camera-rejected directions must bypass axis matching")

    monkeypatch.setattr(candidate_pool, "prepare", lambda _source: (np.empty((0, 4)), (), size))
    monkeypatch.setattr(candidate_pool, "evaluate_pool", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(candidate_pool, "camera_direction_bound", record_bound)
    monkeypatch.setattr(candidate_pool, "propose_role", forbidden_matcher)
    monkeypatch.setattr(line_observations, "prepare_observations",
                        lambda *_args: SimpleNamespace(fragment_ids=np.array([], dtype=int), groups=[]))
    source = {"id": "synthetic", "dimensions": {"width": native_size[0], "height": native_size[1]},
              "all_feet_px": [[[0., 0.], [1., 1.]]]}
    record = estimator(2)
    record["points_working"] = points.tolist()
    saved = {"working_size": list(size), "settings": {"pencil_selection": "coverage"}, "estimator": record}
    result = generation.generate(source, saved, players, candidate_pool, 16)
    scale = np.append(np.asarray(native_size) / size, 1.)
    assert len(calls) == 2
    for (actual_points, actual_size), order in zip(calls, ([0, 1], [1, 0]), strict=True):
        np.testing.assert_array_equal(actual_points, points[order] * scale)
        assert actual_size == native_size
    assert [pair["status"] for pair in result["pairs"]] == ["camera_direction_bound"] * 2
    assert result["camera_bound_coordinate_space"] == "native"
    assert result["entries"] == []
