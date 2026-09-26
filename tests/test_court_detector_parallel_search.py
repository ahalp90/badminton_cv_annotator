"""Direction search in worker processes gives the serial search's record.

This module doubles as the helpers module that generation.generate calls. Spawned workers
import helpers by module name, so fakes must live at module scope in an importable module.
Only propose_role, the camera and horizon tests and the pool input and output are fake;
retention is the detector's own.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scratch.court_det_fix.court_detector import (
    candidate_pool,
    directions,
    generation,
    geometry,
    players,
    proposals,
    search,
)

HELPERS = sys.modules[__name__]
Settings = candidate_pool.Settings
KEEP_COURTS = candidate_pool.KEEP_COURTS
select_pool = candidate_pool.select_pool
retain = candidate_pool.retain
detector = candidate_pool.detector
CAMERA_ERROR_LIMIT = candidate_pool.CAMERA_ERROR_LIMIT
CAMERA_ROUNDING_MARGIN = candidate_pool.CAMERA_ROUNDING_MARGIN

# The fixture sets this in the test process only. A worker that inherited this process's
# memory, as a forked one would, would see it set.
PATCHED_IN_PARENT = False
SIZE = (960, 540)
# Every fake pair proposes these shapes, in working px. Their corners differ by 40 px, so
# retention keeps each apart.
COURT_A = np.array([[300., 150.], [660., 150.], [800., 450.], [160., 450.]])
COURT_B = COURT_A + [40., 0.]
COURT_C = COURT_A - [40., 0.]
COURT_D = COURT_A + [0., 40.]
BEST_PENCILS = (2, 1)  # the one pair that also proposes court D, the best-scored court
FAILING_MARKER = -1.  # a direction's y coordinate that makes every pair using it fail
# Pairs in permutations order for four directions: pencils (first, second) -> pair ID.
PAIR_IDS = {(0, 1): 0, (0, 2): 1, (0, 3): 2, (1, 0): 3, (1, 2): 4, (1, 3): 5,
            (2, 0): 6, (2, 1): 7, (2, 3): 8, (3, 0): 9, (3, 1): 10, (3, 2): 11}
# Direction 1 fails the camera test as the first pencil; direction 3 fails the horizon test as the second.
EXPECTED_STATUSES = ["matched", "matched", "horizon_tilt", "camera_direction_bound", "camera_direction_bound",
                     "camera_direction_bound", "matched", "matched", "horizon_tilt", "matched",
                     "smoke_pair_limit", "smoke_pair_limit"]
FAKE_SETTINGS = {"keep_per_pair": 2, "keep_global": 3, "max_matched_pairs": 5, "max_horizon_tilt_deg": 45.,
                 "legacy_evidence": False}


def prepare(_source: dict) -> tuple[np.ndarray, tuple, tuple[int, int]]:
    return np.empty((0, 4)), (), SIZE


def pencils_of(points: np.ndarray) -> tuple[int, int]:
    """The fake directions' x coordinates are their direction IDs."""
    return int(points[0, 0]), int(points[1, 0])


def camera_direction_bound(points: np.ndarray, _native_size: tuple[int, int]) -> float:
    first, _ = pencils_of(points)
    return 1. if first == 1 else 0.


def horizon_tilt_deg(points: np.ndarray, _size: tuple[int, int]) -> float | None:
    _, second = pencils_of(points)
    if second == 0:
        return None
    return 90. if second == 3 else 0.


def propose_role(points: np.ndarray, _observations: object, _feet: np.ndarray, _size: tuple[int, int],
                 _settings: object, upright_only: bool) -> proposals.RoleProposals:
    """Courts with exactly tied scores, and duplicates within and across pairs."""
    pencils = pencils_of(points)
    if (points[:, 1] == FAILING_MARKER).any():
        raise RuntimeError(f"search failed for pencils {pencils}")
    # Under the 2 px retention distance, so each pair's A is a duplicate of every other pair's A.
    jitter = .1 * (4 * pencils[0] + pencils[1])
    corners = [COURT_C, COURT_A + jitter, COURT_A + jitter + .5, COURT_B + jitter]
    scores = [.5, .9, .9, .9]
    if pencils == BEST_PENCILS:
        corners.append(COURT_D)
        scores.append(.95)
    count = len(corners)
    candidates = []
    for court, score in zip(corners, scores, strict=True):
        candidates.append(detector.Candidate(court, score, (0., 0.), (0, 0)))
    record = {"combined": count + 1, "pencils": list(pencils), "upright_only": upright_only,
              "worker_pid": os.getpid(), "patched_in_parent": PATCHED_IN_PARENT}
    positions = np.arange(count)
    combined = count + 1
    return proposals.RoleProposals(
        record, None, None, candidates, np.column_stack((positions, np.full(count, 4 * pencils[0] + pencils[1]))),
        positions % 2 == 1, np.asarray(scores), np.eye(3) * (positions + 1.)[:, None, None],
        np.asarray([*corners, COURT_C + 5.], dtype=np.float32), np.arange(combined) < count,
        np.arange(combined) % 2 == 0, np.linspace(0., 1., combined, dtype=np.float32),
        np.full(combined, .5, dtype=np.float32),
    )


def evaluate_pool(_source: dict, shortlist: list[dict], *_args: object, legacy_evidence: bool = True) -> list[dict]:
    assert not legacy_evidence
    return [dict(details) for details in shortlist]


def write_pool(path: Path, records: list[tuple]) -> None:
    path.write_bytes(pickle.dumps(records))


def fake_inputs(failing_direction: int | None = None) -> tuple[dict, dict]:
    """A source and a saved direction record with four directions."""
    points = [[float(direction), 1., 1.] for direction in range(4)]
    if failing_direction is not None:
        points[failing_direction][1] = FAILING_MARKER
    estimator = {"points_working": points, "direction_lines": [[1., 0., 0.], [0., 1., 0.]],
                 "retained_support_masks": [[True, True]] * 4, "normalised_to_working": np.eye(3).tolist()}
    source = {"id": "fake", "dimensions": {"width": SIZE[0], "height": SIZE[1]}, "all_feet_px": [[None, None]]}
    saved = {"working_size": list(SIZE), "settings": {"pencil_selection": "coverage"}, "estimator": estimator}
    return source, saved


def comparable(result: dict) -> str:
    """The record without its timings or the searching process's identity, as JSON text."""
    process_keys = {"elapsed_s", "cpu_s", "worker_pid", "patched_in_parent"}

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in process_keys}
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    return json.dumps(strip(result), default=lambda value: value.tolist())


@pytest.fixture(scope="module")
def fake_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[int, tuple[dict, list]]:
    """The fake search with one worker and with two, each with its captured pool."""
    runs = {}
    source, saved = fake_inputs()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(HELPERS, "PATCHED_IN_PARENT", True)
        for workers in (1, 2):
            pool_path = tmp_path_factory.mktemp(f"workers_{workers}") / "pool.pkl"
            result = generation.generate(source, saved, None, Path("."), HELPERS, 16, pool_path, workers=workers,
                                         **FAKE_SETTINGS)
            runs[workers] = (result, pickle.loads(pool_path.read_bytes()))
    return runs


def test_two_workers_give_the_serial_record_and_pool(fake_runs: dict[int, tuple[dict, list]]) -> None:
    (serial, serial_pool), (parallel, parallel_pool) = fake_runs[1], fake_runs[2]
    assert comparable(parallel) == comparable(serial)
    assert len(parallel_pool) == len(serial_pool)
    for parallel_record, serial_record in zip(parallel_pool, serial_pool, strict=True):
        assert parallel_record[:2] == serial_record[:2]
        parallel_arrays = [*parallel_record[2:4], *parallel_record[4]]
        serial_arrays = [*serial_record[2:4], *serial_record[4]]
        for parallel_array, serial_array in zip(parallel_arrays, serial_arrays, strict=True):
            assert (parallel_array.dtype, parallel_array.shape, parallel_array.tobytes()) == (
                serial_array.dtype, serial_array.shape, serial_array.tobytes())


@pytest.mark.parametrize("workers", [1, 2])
def test_tied_courts_keep_the_earliest_pairs_court_and_id(
    fake_runs: dict[int, tuple[dict, list]], workers: int,
) -> None:
    result, _ = fake_runs[workers]
    best_pair = PAIR_IDS[BEST_PENCILS]
    # D beats everything; A and B tie with every other pair's copies, so pair 0's are kept.
    assert [entry["candidate_id"] for entry in result["entries"]] == [f"{best_pair}:4", "0:1", "0:3"]
    np.testing.assert_array_equal(result["entries"][1]["corners_px"], COURT_A + .1)
    np.testing.assert_array_equal(result["entries"][2]["corners_px"], COURT_B + .1)
    assert result["entries"][1]["axis_ids"] == [1, 1]
    assert result["global_cap_reached"] and result["pooled_candidates"] == 10
    shortlists = {pair["pair_id"]: [court["candidate_id"] for court in pair["shortlist"]]
                  for pair in result["pairs"] if pair["status"] == "matched"}
    assert shortlists == {0: ["0:1", "0:3"], 1: ["1:1", "1:3"], 6: ["6:1", "6:3"],
                          best_pair: [f"{best_pair}:4", f"{best_pair}:1"], 9: ["9:1", "9:3"]}


@pytest.mark.parametrize("workers", [1, 2])
def test_skipped_and_capped_pairs_keep_pair_order(fake_runs: dict[int, tuple[dict, list]], workers: int) -> None:
    result, _ = fake_runs[workers]
    assert [pair["pair_id"] for pair in result["pairs"]] == list(range(12))
    assert [pair["status"] for pair in result["pairs"]] == EXPECTED_STATUSES
    for pair in result["pairs"]:
        if pair["status"] == "matched":
            assert pair["role"]["pencils"] == pair["pencils"] and pair["role"]["upright_only"]
            assert pair["per_pair_cap_reached"] and pair["raw_parent_count"] in (4, 5)


def test_pool_capture_keeps_each_pairs_arrays(fake_runs: dict[int, tuple[dict, list]]) -> None:
    _, pool = fake_runs[2]
    best_pair = PAIR_IDS[BEST_PENCILS]
    assert [record[0] for record in pool] == [0, 1, 6, best_pair, 9]
    for pair_id, count, corners, retained_index, pregate in pool:
        assert corners.dtype == np.float32 and corners.shape == (count, 4, 2)
        assert retained_index.dtype == np.int32
        assert retained_index.tolist() == ([4, 1] if pair_id == best_pair else [1, 3])
        combined_corners, valid, usable, player_any, player_both_halves = pregate
        assert combined_corners.shape == (count + 1, 4, 2) and combined_corners.dtype == np.float32
        assert valid.shape == usable.shape == player_any.shape == player_both_halves.shape == (count + 1,)


def test_pairs_are_searched_in_spawned_workers(fake_runs: dict[int, tuple[dict, list]]) -> None:
    for workers, (result, _) in fake_runs.items():
        roles = [pair["role"] for pair in result["pairs"] if pair["status"] == "matched"]
        in_this_process = [role["worker_pid"] == os.getpid() for role in roles]
        saw_patch = [role["patched_in_parent"] for role in roles]
        if workers == 1:
            assert all(in_this_process) and all(saw_patch)
        else:
            assert not any(in_this_process) and not any(saw_patch)


@pytest.mark.parametrize("workers", [1, 2])
def test_a_failing_pair_raises_in_the_caller(workers: int) -> None:
    source, saved = fake_inputs(failing_direction=2)
    with pytest.raises(RuntimeError, match=r"search failed for pencils \(0, 2\)"):
        generation.generate(source, saved, None, Path("."), HELPERS, 16, workers=workers, **FAKE_SETTINGS)


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_must_be_positive(workers: int) -> None:
    source, saved = fake_inputs()
    with pytest.raises(ValueError, match="workers must be positive"):
        generation.generate(source, saved, None, Path("."), HELPERS, 16, workers=workers, **FAKE_SETTINGS)


def synthetic_court_source() -> dict:
    """A court's painted lines and two players' feet, seen by an upright camera behind the near baseline."""
    camera_matrix = np.array([[800., 0., 480.], [0., 800., 270.], [0., 0., 1.]])
    camera_position_m = np.array([3.05, -6., 5.])
    pitch = np.radians(30.)
    right = np.array([1., 0., 0.])
    forward = np.array([0., np.cos(pitch), -np.sin(pitch)])
    rows = np.array([right, np.cross(forward, right), forward])

    def project(floor_m: np.ndarray) -> np.ndarray:
        points = np.column_stack((floor_m.reshape(-1, 2), np.zeros(floor_m.size // 2)))
        projected = (camera_matrix @ rows @ (points - camera_position_m).T).T
        return (projected[:, :2] / projected[:, 2:]).reshape(floor_m.shape)

    # (sampled frames, player slots, floor xy): one player in each half.
    feet_m = np.array([[[2.5, 3.], [3.5, 10.]], [[1.5, 2.], [4.5, 11.]], [[3., 4.], [2., 9.]]])
    return {"id": "synthetic", "dimensions": {"width": SIZE[0], "height": SIZE[1]},
            "segments_px": project(geometry.SEGMENTS_M).reshape(-1, 4).tolist(), "bbox_px": [],
            "all_feet_px": project(feet_m).tolist()}


def test_real_search_in_two_workers_gives_the_serial_record() -> None:
    source = synthetic_court_source()
    _, estimator = directions.estimate(np.asarray(source["segments_px"]), SIZE,
                                       directions.Settings(**search.DIRECTION_SETTINGS))
    saved = {"working_size": list(SIZE), "settings": search.DIRECTION_SETTINGS, "estimator": estimator}
    results = []
    for workers in (1, 2):
        results.append(generation.generate(
            source, saved, players, Path("."), candidate_pool, 16, keep_axes=64, keep_per_pair=8, keep_global=8,
            max_matched_pairs=3, legacy_evidence=False, max_horizon_tilt_deg=45., workers=workers,
        ))
    serial, parallel = results
    assert [pair["status"] for pair in serial["pairs"]].count("matched") == 3
    assert serial["entries"]
    assert comparable(parallel) == comparable(serial)

