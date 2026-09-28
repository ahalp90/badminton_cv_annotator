"""An optional cheap line-support score picks which courts each direction pair fully scores.

Real courts come from the synthetic court in test_court_detector_parallel_search. Its
first direction pair builds 140 usable courts. This module also doubles as the helpers
module that generation.generate calls, with a fake propose_role, so spawned workers can
import it by name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from court_detector import (
    candidate_pool,
    directions,
    generation,
    geometry,
    proposals,
    search,
)
from court_detector import line_observations as assignment
from court_detector.candidate_geometry import continuous_support
from tests import test_court_detector_parallel_search as parallel_search

HELPERS = sys.modules[__name__]
# generation.generate reads these from its helpers module: the detector's own retention,
# and the other module's fakes for everything but propose_role.
Settings = candidate_pool.Settings
KEEP_COURTS = candidate_pool.KEEP_COURTS
select_pool = candidate_pool.select_pool
retain = candidate_pool.retain
detector = candidate_pool.detector
CAMERA_ERROR_LIMIT = candidate_pool.CAMERA_ERROR_LIMIT
CAMERA_ROUNDING_MARGIN = candidate_pool.CAMERA_ROUNDING_MARGIN
prepare = parallel_search.prepare
camera_direction_bound = parallel_search.camera_direction_bound
evaluate_pool = parallel_search.evaluate_pool

LIMIT = 41  # splits a group of the real pair's courts with tied cheap scores (checked below)
# The fake pair's four usable courts, 40 px apart so retention keeps them all.
FAKE_COURTS = parallel_search.COURT_A + np.array([[0., 0.], [40., 0.], [80., 0.], [120., 0.]])[:, None, :]
FAKE_SCORES = np.array([.6, .9, .5, .8])
FAKE_CHEAP_SURVIVORS = np.array([1, 3])  # the usable positions the fake keeps under any limit
FAKE_SETTINGS = {"keep_per_pair": 4, "keep_global": 4, "max_matched_pairs": 2}


def propose_role(_points: np.ndarray, _observations: object, _feet: np.ndarray, _size: tuple[int, int],
                 _settings: object, upright_only: bool, full_score_limit: int | None = None) -> proposals.RoleProposals:
    """Four usable courts; under any limit, only those at usable positions 1 and 3.

    Each court's axis IDs and homography encode its usable position, so tests can check
    that its details stayed with it.
    """
    kept = np.arange(len(FAKE_COURTS)) if full_score_limit is None else FAKE_CHEAP_SURVIVORS
    candidates = []
    for court, score in zip(FAKE_COURTS[kept], FAKE_SCORES[kept], strict=True):
        candidates.append(detector.Candidate(court, float(score), (0., 0.), (0, 0)))
    record = {"full_score_limit": full_score_limit, "upright_only": upright_only, "worker_pid": os.getpid(),
              "geometry_players": len(FAKE_COURTS)}
    return proposals.RoleProposals(
        record, None, None, candidates, np.column_stack((kept, kept)), kept % 2 == 1, FAKE_SCORES[kept],
        np.eye(3) * (kept + 1.)[:, None, None], usable_positions=None if full_score_limit is None else kept,
        cheap_ranks=None if full_score_limit is None else np.array([2, 1]),
    )


def scores(proposed: proposals.RoleProposals) -> np.ndarray:
    """The candidates' scores, back in the float32 that continuous_support computed them in."""
    return np.asarray([candidate.score for candidate in proposed.candidates], dtype=np.float32)


def comparable_role(proposed: proposals.RoleProposals) -> tuple:
    """Everything propose_role returns except the matched axes, with arrays as dtype, shape and bytes."""
    corners = np.asarray([candidate.corners_px for candidate in proposed.candidates])
    arrays = [corners, scores(proposed), proposed.axis_ids, proposed.rotated, proposed.axis_scores,
              proposed.homographies, proposed.combined_corners, proposed.valid, proposed.usable, proposed.player_any,
              proposed.player_both_halves]
    return proposed.record, [(array.dtype, array.shape, array.tobytes()) for array in arrays], proposed.usable_positions


def assert_aligned(limited: proposals.RoleProposals, exhaustive: proposals.RoleProposals) -> None:
    """Each kept court carries the exhaustive run's corners, full score and details for its usable position."""
    positions = limited.usable_positions
    assert len(limited.candidates) == len(positions) == limited.record["fully_scored"]
    for candidate_position, usable_position in enumerate(positions):
        original = exhaustive.candidates[usable_position]
        assert limited.usable_position(candidate_position) == usable_position
        np.testing.assert_array_equal(limited.candidates[candidate_position].corners_px, original.corners_px)
        # Exact: a court's full score does not depend on which courts share its batch.
        assert limited.candidates[candidate_position].score == original.score
        assert limited.detail(candidate_position) == exhaustive.detail(usable_position)
    # The pregate copy still describes every combined court.
    for name in ("combined_corners", "valid", "usable", "player_any", "player_both_halves"):
        np.testing.assert_array_equal(getattr(limited, name), getattr(exhaustive, name))


@pytest.fixture(scope="module")
def real_pair() -> tuple:
    """propose_role's inputs for the synthetic court's first direction pair."""
    source = parallel_search.synthetic_court_source()
    _, estimator = directions.estimate(np.asarray(source["segments_px"]), parallel_search.SIZE,
                                       directions.Settings(**search.DIRECTION_SETTINGS))
    segments, _, size = candidate_pool.prepare(source)
    observations = assignment.prepare_observations(segments, size)
    feet = np.asarray(source["all_feet_px"], dtype=float)
    points = np.asarray(estimator["points_working"])[[0, 1]]
    return points, observations, feet, size, candidate_pool.Settings(keep_axes=64)


@pytest.fixture(scope="module")
def exhaustive(real_pair: tuple) -> proposals.RoleProposals:
    return proposals.propose_role(*real_pair)


def test_default_fully_scores_every_usable_court_in_order(
    real_pair: tuple, exhaustive: proposals.RoleProposals,
) -> None:
    _, observations, _, size, _ = real_pair
    assert "axes" not in exhaustive.record
    assert exhaustive.usable_positions is None and "fully_scored" not in exhaustive.record
    assert exhaustive.record["geometry_players"] == len(exhaustive.candidates) == exhaustive.usable.sum() > LIMIT
    corners = np.asarray([candidate.corners_px for candidate in exhaustive.candidates], dtype=np.float32)
    np.testing.assert_array_equal(corners, exhaustive.combined_corners[exhaustive.usable])
    maps = proposals.pair_line_maps(observations, exhaustive.axes, size)
    assert scores(exhaustive).tobytes() == continuous_support(exhaustive.homographies, maps, size, 64).tobytes()


@pytest.mark.parametrize("above_count", [0, 1, 10 ** 6])
def test_a_limit_at_or_above_the_usable_count_changes_nothing(
    real_pair: tuple, exhaustive: proposals.RoleProposals, above_count: int,
) -> None:
    limit = len(exhaustive.candidates) + above_count
    limited = proposals.propose_role(*real_pair, full_score_limit=limit)
    assert comparable_role(limited) == comparable_role(exhaustive)


def test_best_positions_keeps_the_earliest_tied_scores_in_original_order() -> None:
    tied = np.array([.5, .9, .7, .9, .9, .95])
    assert proposals.best_positions(tied, 1)[0].tolist() == [5]
    assert proposals.best_positions(tied, 3)[0].tolist() == [1, 3, 5]
    assert proposals.best_positions(tied, 4)[0].tolist() == [1, 3, 4, 5]
    assert proposals.best_positions(np.zeros(5), 2)[0].tolist() == [0, 1]
    assert proposals.best_positions(np.array([.9, .1, .95]), 2)[0].tolist() == [0, 2]


def test_a_limit_keeps_the_best_cheap_scores_with_their_usable_positions_and_details(
    real_pair: tuple, exhaustive: proposals.RoleProposals,
) -> None:
    _, observations, _, size, _ = real_pair
    limited = proposals.propose_role(*real_pair, full_score_limit=LIMIT)
    maps = proposals.pair_line_maps(observations, exhaustive.axes, size)
    cheap = continuous_support(exhaustive.homographies, maps, size, 16)  # the agreed trial's samples per marking
    ranking = sorted(range(len(cheap)), key=lambda position: (-cheap[position], position))
    kept_tie, dropped_tie = ranking[LIMIT - 1], ranking[LIMIT]
    assert cheap[kept_tie] == cheap[dropped_tie]
    positions = limited.usable_positions.tolist()
    assert kept_tie in positions and dropped_tie not in positions
    assert positions == sorted(ranking[:LIMIT]) != list(range(LIMIT))
    assert limited.record == {**exhaustive.record, "fully_scored": LIMIT}
    assert limited.cheap_ranks.tolist() == [ranking.index(position) + 1 for position in positions]
    assert_aligned(limited, exhaustive)


def test_a_limited_search_builds_the_pair_maps_once(real_pair: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    distance_maps = geometry._distance_maps

    def counted_distance_maps(*args: object) -> np.ndarray:
        calls.append(args)
        return distance_maps(*args)

    monkeypatch.setattr(geometry, "_distance_maps", counted_distance_maps)
    proposals.propose_role(*real_pair)
    assert len(calls) == 1
    proposals.propose_role(*real_pair, full_score_limit=LIMIT)
    assert len(calls) == 2


@pytest.mark.parametrize(("limit", "ranking", "message"), [
    (0, "finite", "must be positive"), (-1, "finite", "must be positive"), (5, "axis", "combined_ranking='finite'"),
])
def test_propose_role_refuses_a_limit_it_cannot_apply(
    real_pair: tuple, limit: int, ranking: str, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        proposals.propose_role(*real_pair, combined_ranking=ranking, full_score_limit=limit)


@pytest.fixture(scope="module")
def limited_runs() -> dict[int, dict]:
    """The fake search under a limit, with one worker and with two."""
    source, saved = parallel_search.fake_inputs()
    runs = {}
    for workers in (1, 2):
        runs[workers] = generation.generate(source, saved, None, HELPERS, 16, workers=workers,
                                            full_score_limit=2, **FAKE_SETTINGS)
    return runs


@pytest.mark.parametrize("workers", [1, 2])
def test_pair_search_forwards_the_limit_and_keeps_usable_candidate_ids(
    limited_runs: dict[int, dict], workers: int,
) -> None:
    result = limited_runs[workers]
    assert result["full_score_limit"] == 2
    matched = [pair for pair in result["pairs"] if pair["status"] == "matched"]
    assert len(matched) == 2
    for pair in matched:
        assert pair["role"]["full_score_limit"] == 2
        assert (pair["role"]["worker_pid"] == os.getpid()) == (workers == 1)
        assert pair["raw_parent_count"] == len(FAKE_COURTS)
        assert pair["role"]["retained_cheap_ranks"] == [2, 1]
        # Best first: usable position 1 scores .9 and position 3 scores .8.
        assert [court["candidate_id"] for court in pair["shortlist"]] == [f"{pair['pair_id']}:1", f"{pair['pair_id']}:3"]
        for court in pair["shortlist"]:
            usable_position = int(court["candidate_id"].split(":")[1])
            assert court["axis_ids"] == [usable_position, usable_position]
            np.testing.assert_array_equal(court["homography_working"], np.eye(3) * (usable_position + 1))
            np.testing.assert_array_equal(court["corners_px"], FAKE_COURTS[usable_position])


def test_two_workers_give_the_serial_record_under_a_limit(limited_runs: dict[int, dict]) -> None:
    assert parallel_search.comparable(limited_runs[2]) == parallel_search.comparable(limited_runs[1])


def test_without_a_limit_the_record_and_helpers_see_none() -> None:
    source, saved = parallel_search.fake_inputs()
    result = generation.generate(source, saved, None, HELPERS, 16, **FAKE_SETTINGS)
    assert "full_score_limit" not in result
    for pair in result["pairs"]:
        if pair["status"] == "matched":
            assert pair["role"]["full_score_limit"] is None
            assert [court["candidate_id"].split(":")[1] for court in pair["shortlist"]] == ["1", "3", "0", "2"]


def test_pool_capture_is_refused_under_a_limit(tmp_path: Path) -> None:
    source, saved = parallel_search.fake_inputs()
    pool_path = tmp_path / "pool.pkl"
    with pytest.raises(ValueError, match="pool capture"):
        generation.generate(source, saved, None, HELPERS, 16, pool_path, full_score_limit=2,
                            **FAKE_SETTINGS)
    assert not pool_path.exists()


@pytest.mark.parametrize("limit", [0, -3])
def test_generate_refuses_a_limit_below_one(limit: int) -> None:
    source, saved = parallel_search.fake_inputs()
    with pytest.raises(ValueError, match="full_score_limit must be positive"):
        generation.generate(source, saved, None, HELPERS, 16, full_score_limit=limit, **FAKE_SETTINGS)
