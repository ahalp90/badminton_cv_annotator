"""Measuring and refitting parents in spawned worker processes gives the serial scoring.

The parents are real: the committed am3 frame-0 baseline generation record's first five
entries (three refit to a valid child, two have a rank-deficient refit), one entry made
hard-invalid, a compatible duplicate and a conflicting duplicate that shares a homography.
The gxBQ frame-0 record's first four entries give a second view for one pool to score next.

This module doubles as the probe that one test runs inside a worker. Spawned workers import
it by module name, so the probe lives at module scope.
"""

from __future__ import annotations

import copy
import gzip
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
import pytest

from court_detector import generation, measurements, scoring
from court_detector.detect import (
    LiveModules,
    freeze_arrays,
    load_live_modules,
)
from scratch.court_det_fix.court_detector import frozen_cases

COURT_ROOT = Path(__file__).resolve().parents[1] / "scratch/court_det_fix"
CASE_ID = "am3_window_00_frame_0"
SECOND_CASE_ID = "gxBQ_window_00_frame_0"
SECOND_VIEW_ENTRIES = 4
VALID_CHILD_ENTRIES = (0, 1, 2)
RANK_DEFICIENT_ENTRIES = (3, 4)
HARD_INVALID_ENTRY = 5
PLACEHOLDER_JUNCTIONS = {"status": "omitted_unused_diagnostic"}
NO_SAMPLING = {"greyscale_conversions": 0, "sampling_calls": 0}


class RealView(NamedTuple):
    live: LiveModules
    context: Any  # measurements.ViewContext, frozen
    all_line_entries: list[dict]
    painted_line_entries: list[dict]
    entry_index: dict[str, int]  # candidate ID -> its index in the baseline record's entries


class SecondView(NamedTuple):
    context: Any  # measurements.ViewContext, frozen
    all_line_entries: list[dict]
    serial: scoring.ScoredPopulations
    serial_cache: dict


class Run(NamedTuple):
    scored: scoring.ScoredPopulations
    cache: dict  # the caller's measurement cache afterwards
    counts: dict[str, int]  # the caller's sampling counts
    messages: list[str]  # progress lines


def baseline_view(case_id: str, entry_count: int) -> tuple[Any, list[dict]]:
    """A frozen view's context and its baseline generation record's first entries."""
    context = frozen_cases.prepare_view(COURT_ROOT, case_id)
    freeze_arrays(context)
    with gzip.open(COURT_ROOT / f"frozen_views/baseline_generation/{case_id}.json.gz", "rt") as stream:
        entries = json.load(stream)["entries"][:entry_count]
    return context, entries


@pytest.fixture(scope="module")
def view() -> RealView:
    live = load_live_modules()
    context, entries = baseline_view(CASE_ID, HARD_INVALID_ENTRY + 1)
    all_line_entries = entries[:HARD_INVALID_ENTRY]
    hard_invalid = copy.deepcopy(entries[HARD_INVALID_ENTRY])
    hard_invalid["gates"]["geometry_valid"] = False
    all_line_entries.append(hard_invalid)
    # Court 0 under another pair ID: a separate parent measuring the same homography.
    conflicting = {**entries[0], "pair_id": entries[0]["pair_id"] + 1000}
    compatible = copy.deepcopy(entries[1])  # merges into the second all-lines parent
    entry_index = {entry["candidate_id"]: index for index, entry in enumerate(entries)}
    return RealView(live, context, all_line_entries, [conflicting, compatible], entry_index)


@pytest.fixture(scope="module")
def runs(view: RealView) -> dict[int, Run]:
    results = {}
    for workers in (1, 2):
        cache: dict = {}
        messages: list[str] = []
        with view.live.prepared_measurements(measurements) as counts:
            callers_sampler = (measurements.grayscale_sample, measurements.raw_junctions)
            scored = scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [],
                                               view.live.runtime, cache, messages.append, workers=workers)
            # The workers patch their own modules, never this process's.
            assert (measurements.grayscale_sample, measurements.raw_junctions) == callers_sampler
        results[workers] = Run(scored, cache, dict(counts), messages)
    return results


@pytest.fixture(scope="module")
def second_view(view: RealView) -> SecondView:
    context, entries = baseline_view(SECOND_CASE_ID, SECOND_VIEW_ENTRIES)
    cache: dict = {}
    with view.live.prepared_measurements(measurements):
        serial = scoring.score_populations(context, entries, [], [], view.live.runtime, cache, print)
    return SecondView(context, entries, serial, cache)


def assert_same(serial: Any, parallel: Any, path: str = "") -> None:
    """Equal types and values, with NaN equal to NaN; arrays also keep their dtype and shape."""
    assert type(parallel) is type(serial), path
    if isinstance(serial, np.ndarray):
        assert (parallel.dtype, parallel.shape) == (serial.dtype, serial.shape), path
        np.testing.assert_array_equal(parallel, serial, err_msg=path)
    elif isinstance(serial, dict):
        assert list(parallel) == list(serial), path
        for key in serial:
            assert_same(serial[key], parallel[key], f"{path}.{key}")
    elif isinstance(serial, (list, tuple)):
        assert len(parallel) == len(serial), path
        for index, (serial_item, parallel_item) in enumerate(zip(serial, parallel, strict=True)):
            assert_same(serial_item, parallel_item, f"{path}[{index}]")
    elif isinstance(serial, float) and math.isnan(serial):
        assert math.isnan(parallel), path
    else:
        assert parallel == serial, path


def assert_same_scoring(serial: scoring.ScoredPopulations, parallel: scoring.ScoredPopulations) -> None:
    for field in scoring.ScoredPopulations._fields:
        assert_same(getattr(serial, field), getattr(parallel, field), field)


def assert_same_cache(serial_cache: dict, parallel_cache: dict) -> None:
    """The same measurements. Workers add each parent's and its child's in turn, so the key order differs."""
    assert set(parallel_cache) == set(serial_cache)
    for key, measured in serial_cache.items():
        assert_same(measured, parallel_cache[key], repr(key[:8]))


def test_two_workers_give_every_serial_field(runs: dict[int, Run]) -> None:
    assert_same_scoring(runs[1].scored, runs[2].scored)


def test_the_fixture_covers_children_rejections_and_duplicates(view: RealView, runs: dict[int, Run]) -> None:
    scored = runs[2].scored
    assert len(scored.parents) == 7 and scored.identity_resolution["duplicate_group_count"] == 1
    assert scored.identity_resolution["conflicting_geometry_groups"] == [[
        scored.parents[0]["origin_key"], scored.parents[-1]["origin_key"],
    ]]
    statuses = {row["origin_key"]: row["status"] for row in scored.fit_rows}
    for parent in scored.parents:
        entry_index = view.entry_index[parent["candidate_id"]]
        if entry_index in VALID_CHILD_ENTRIES:
            child_key = f"{parent['origin_key']}/child"
            assert statuses[parent["origin_key"]] == "valid_child"
            assert parent["refit"] == {"status": "valid_child", "child_origin_key": child_key}
        elif entry_index in RANK_DEFICIENT_ENTRIES:
            assert statuses[parent["origin_key"]] == "rank_deficient" and "refit" not in parent
        else:
            assert entry_index == HARD_INVALID_ENTRY and not parent["hard_valid"] and "evidence" not in parent
            assert statuses[parent["origin_key"]] == "parent_invalid" and "refit" not in parent
    # Entries 0 to 2, and the conflicting copy of entry 0.
    assert len(scored.children) == 4


@pytest.mark.parametrize("workers", [1, 2])
def test_arrays_are_the_candidates_own(runs: dict[int, Run], workers: int) -> None:
    scored = runs[workers].scored
    expected_keys = []
    for candidate in scored.c_candidates:
        for name, values in candidate["_arrays"].items():
            expected_keys.append(f"{candidate['origin_key']}::{name}")
            assert scored.arrays[expected_keys[-1]] is values
    assert list(scored.arrays) == expected_keys
    assert not scored.line_maps.flags.writeable


@pytest.mark.parametrize("workers", [1, 2])
def test_measurements_use_the_samplers_placeholder_junctions(runs: dict[int, Run], workers: int) -> None:
    for candidate in runs[workers].scored.c_candidates:
        assert candidate["evidence"]["junctions"] == PLACEHOLDER_JUNCTIONS


def test_workers_measure_in_their_own_processes(runs: dict[int, Run]) -> None:
    assert runs[1].counts["greyscale_conversions"] == 1 and runs[1].counts["sampling_calls"] > 0
    assert runs[2].counts == NO_SAMPLING
    assert runs[2].messages[1] == "measuring and refitting each parent in one of 2 worker processes"


def test_the_callers_cache_gains_every_measurement(runs: dict[int, Run]) -> None:
    serial_cache, parallel_cache = runs[1].cache, runs[2].cache
    # Two parents share one homography, and so do their children.
    assert len(serial_cache) == len(runs[1].scored.c_candidates) - 2
    assert_same_cache(serial_cache, parallel_cache)


def test_parallel_scoring_refuses_other_runtimes_and_callers_outside_the_sampler(view: RealView) -> None:
    with pytest.raises(ValueError, match="measure inside sampling.prepared_measurements"):
        scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [],
                                  view.live.runtime, {}, print, workers=2)
    other_runtime = {**view.live.runtime, "zone": object()}
    with view.live.prepared_measurements(measurements), pytest.raises(ValueError, match="accept only that runtime"):
        scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [], other_runtime,
                                  {}, print, workers=2)


@pytest.mark.parametrize("workers", [0, -1])
def test_workers_must_be_positive(view: RealView, workers: int) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [],
                                  view.live.runtime, {}, print, workers=workers)


def test_a_workers_exception_reaches_the_caller(view: RealView) -> None:
    broken = {key: value for key, value in view.all_line_entries[1].items() if key != "corners_px"}
    with view.live.prepared_measurements(measurements), pytest.raises(KeyError, match="corners_px") as raised:
        scoring.score_populations(view.context, [view.all_line_entries[0], broken], [], [], view.live.runtime, {},
                                  print, workers=2)
    assert type(raised.value.__cause__).__name__ == '_RemoteTraceback'


def test_one_pool_scores_consecutive_views_with_each_views_own_inputs(
    view: RealView, runs: dict[int, Run], second_view: SecondView,
) -> None:
    """A worker that kept the first view's image or cache would change the second view's scores."""
    first_cache: dict = {}
    second_cache: dict = {}
    with view.live.prepared_measurements(measurements), generation.worker_pool(2) as pool:
        first = scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [],
                                          view.live.runtime, first_cache, print, workers=2, pool=pool)
        second = scoring.score_populations(second_view.context, second_view.all_line_entries, [], [],
                                           view.live.runtime, second_cache, print, workers=2, pool=pool)
    assert_same_scoring(runs[1].scored, first)
    assert_same_scoring(second_view.serial, second)
    assert_same_cache(runs[1].cache, first_cache)
    assert_same_cache(second_view.serial_cache, second_cache)


def test_a_failed_view_leaves_the_pool_usable_and_removes_its_pickle(
    view: RealView, runs: dict[int, Run], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    broken = {key: value for key, value in view.all_line_entries[1].items() if key != "corners_px"}
    waited = []
    original_wait = scoring.wait

    def wait_before_removing_input(futures: list) -> Any:
        assert len(list(tmp_path.glob('court-scoring-*/view-*.pickle'))) == 1
        result = original_wait(futures)
        assert all(future.done() for future in futures)
        waited.append(True)
        return result

    monkeypatch.setattr(scoring, "wait", wait_before_removing_input)
    with view.live.prepared_measurements(measurements), generation.worker_pool(2) as pool:
        with pytest.raises(KeyError, match="corners_px"):
            entries = [broken, view.all_line_entries[0], *view.all_line_entries[2:]]
            scoring.score_populations(view.context, entries, [], [], view.live.runtime,
                                      {}, print, workers=2, pool=pool)
        assert waited == [True]
        assert list(tmp_path.iterdir()) == []
        after = scoring.score_populations(view.context, view.all_line_entries, view.painted_line_entries, [],
                                          view.live.runtime, {}, print, workers=2, pool=pool)
    assert_same_scoring(runs[1].scored, after)


def test_worker_startup_failure_with_a_large_view_reaches_the_caller() -> None:
    # Spawn cannot re-import a stdin script. A large initargs payload used to hang
    # while the parent wrote to the pipe of the worker that had already died.
    script = '''
import numpy as np
from court_detector import scoring
from court_detector.detect import load_live_modules
live = load_live_modules()
with live.prepared_measurements(live.verifier):
    scoring.measure_and_refit_in_workers(
        np.zeros(1_000_000), [{}], live.runtime, {}, np.zeros((2, 1, 1)), print, 1,
    )
'''
    environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path)}
    process = subprocess.Popen([sys.executable, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True, env=environment)
    try:
        _, errors = process.communicate(script, timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        pytest.fail('worker start-up failure hung instead of raising')
    assert process.returncode != 0
    assert 'BrokenProcessPool' in errors, errors


def worker_settings() -> tuple[int, bool, bool]:
    """In a worker after a scoring task: OpenCV's thread count and whether its view's arrays are writeable."""
    worker_view = scoring.worker_view
    assert worker_view is not None
    return cv2.getNumThreads(), worker_view.context.frame.flags.writeable, worker_view.line_maps.flags.writeable


def test_a_worker_uses_one_opencv_thread_and_a_read_only_view(view: RealView, tmp_path: Path) -> None:
    line_maps = scoring.view_line_maps(view.context)
    view_path = tmp_path / 'view.pickle'
    with view_path.open('wb') as stream:
        pickle.dump((view.context, line_maps, {}), stream)
    parent_identities, _ = scoring.canonicalise_populations(view.all_line_entries, [], [])
    # One worker runs both tasks, so the probe sees the view that the scoring task read.
    with generation.worker_pool(1) as executor:
        executor.submit(scoring.score_parent_in_worker, view_path, parent_identities[0]).result()
        assert executor.submit(worker_settings).result() == (1, False, False)
