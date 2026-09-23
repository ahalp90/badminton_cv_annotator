"""Focused checks for live SVD family screening and population cache identity."""

from __future__ import annotations

import gzip
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "w5_holistic"))
sys.path.insert(0, str(Path(__file__).parent))

import automatic_generation
import generation
import run_cases
from generation import _cached, screen_matches


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def estimator(count: int) -> dict:
    return {"points_working": [[float(index), 1, 1] for index in range(count)],
            "direction_lines": [[1, 0, 0], [0, 1, 0]],
            "retained_support_masks": [[True, True] for _ in range(count)],
            "normalised_to_working": np.eye(3).tolist()}


def test_nine_saved_ranks_and_residuals_match_review() -> None:
    replay = read(ROOT / "evidence/webui_followup3_20260922/review_20260923/replay/results.json.gz")
    for case in replay["cases"]:
        baseline = read(ROOT / "frozen_views/baseline_directions" / f"{case['case_id']}.json.gz")
        screen = automatic_generation.screen_groups(baseline["estimator"], 12)
        assert screen["ranked_original_ids"] == case["rank_order"]
        assert screen["selected_original_ids"] == case["rank_order"][:12]
        for actual, expected in zip(screen["diagnostics"], case["groups"], strict=True):
            assert actual["line_count"] == expected["line_count"]
            assert actual["algebraic_rms"] == pytest.approx(expected["algebraic_rms"], abs=1e-12)


def test_full_budget_empty_small_ties_and_two_line_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    record = estimator(2)
    original_svd = np.linalg.svd
    screen = automatic_generation.screen_groups(record, 12)
    assert screen["ranked_original_ids"] == [0, 1]
    assert screen["selected_original_ids"] == [0, 1]
    assert screen["diagnostics"][0]["algebraic_rms"] == pytest.approx(0)
    assert automatic_generation.screen_groups(estimator(0), 12)["selected_original_ids"] == []

    def forbidden_svd(*args: object, **kwargs: object) -> None:
        raise AssertionError("16 must bypass SVD")

    monkeypatch.setattr(np.linalg, "svd", forbidden_svd)
    assert automatic_generation.screen_groups(record, 16)["selected_original_ids"] == [0, 1]
    monkeypatch.setattr(np.linalg, "svd", original_svd)
    with pytest.raises(ValueError, match="at most 16"):
        automatic_generation.screen_groups(estimator(17), 12)
    with pytest.raises(ValueError, match="12 or 16"):
        automatic_generation.screen_groups(record, 8)


def test_actual_empty_estimator_record(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "frozen_helpers_20260914/vp_pruning"))
    vp_pruning = importlib.import_module("vp_pruning")
    _, record = vp_pruning.estimate(
        np.empty((0, 4)), (640, 360), vp_pruning.Settings(pencil_selection="coverage")
    )
    assert record["direction_lines"] == []
    assert record["retained_support_masks"] == []
    assert record["points_working"] == []
    for budget in (12, 16):
        screen = automatic_generation.screen_groups(record, budget)
        assert screen["ranked_original_ids"] == []
        assert screen["selected_original_ids"] == []


def test_run_case_imports_generation_after_runtime_in_fresh_process() -> None:
    script = """
import sys
from pathlib import Path

sys.path.insert(0, 'scratch/court_det_fix/wider_evaluation')
import run_cases

original_load_runtime = run_cases.load_runtime

def stop_after_runtime(root, control_pack=None):
    run_w5, verifier, runtime = original_load_runtime(root, control_pack)
    def prepare_view(*_args):
        raise RuntimeError('prepare-view sentinel')
    verifier.prepare_view = prepare_view
    return run_w5, verifier, runtime

run_cases.load_runtime = stop_after_runtime
try:
    run_cases.run_case(Path('scratch/court_det_fix'), Path('/tmp/svd-entry-check'),
                       'shuttleset_03_scene_0019')
except RuntimeError as error:
    assert str(error) == 'prepare-view sentinel'
else:
    raise AssertionError('prepare_view was not reached')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT.parents[1],
        env={**os.environ, "PYTHONPATH": "src:."}, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr


@dataclass
class Settings:
    keep_axes: int


def test_pair_ids_and_original_points_survive_screen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    points = [[float(index), 1, 1] for index in range(13)]
    record = estimator(13)
    record["points_working"] = points
    calls = []
    observations = SimpleNamespace(fragment_ids=np.array([], dtype=int), groups=[])
    monkeypatch.setattr(automatic_generation.assignment, "prepare_observations",
                        lambda *_args: observations)
    helpers = ModuleType("fake_automatic")
    helpers.prepare = lambda _source: (np.empty((0, 4)), (), (100, 100))
    helpers.Settings = Settings
    helpers.CAMERA_ERROR_LIMIT = .1
    helpers.CAMERA_ROUNDING_MARGIN = 1e-6
    helpers.KEEP_COURTS = 256
    helpers.camera_direction_bound = lambda *_args: 0.

    def propose(pair_points: np.ndarray, *_args: object) -> SimpleNamespace:
        calls.append(pair_points.copy())
        empty = np.empty((0,), dtype=float)
        candidate = SimpleNamespace(corners_px=np.zeros((4, 2)), score=1.)
        return SimpleNamespace(candidates=[candidate], details=[{}], combined_corners=np.empty((0, 4, 2)),
                               valid=empty, usable=empty, player_any=empty, player_both_halves=empty,
                               record={"combined": 0})

    helpers.propose_role = propose
    helpers.select_pool = lambda candidates: candidates
    helpers.evaluate_pool = lambda *_args: []
    helpers.winner_ids = lambda _entries: (None, None)
    helpers.write_pool = lambda *_args: None
    source = {"id": "test", "dimensions": {"width": 100, "height": 100},
              "all_feet_px": [[[0, 0], [0, 0]]]}
    saved = {"working_size": [100, 100], "settings": {"pencil_selection": "coverage"}, "estimator": record}
    result = automatic_generation.generate(source, saved, None, tmp_path, helpers)
    selected = set(result["direction_screen"]["selected_original_ids"])
    assert len(result["pairs"]) == 13 * 12
    assert len(calls) == 12 * 11
    for pair in result["pairs"]:
        first, second = pair["pencils"]
        assert pair["pair_id"] == first * 12 + second - (second > first)
        if {first, second} <= selected:
            assert pair["status"] == "matched"
            assert [entry["candidate_id"] for entry in pair["shortlist"]] == [f"{pair['pair_id']}:0"]
        else:
            assert pair["status"] == "skipped_svd_mask"
    assert all(any(np.array_equal(pair, np.asarray(points)[[first, second]])
                   for first in selected for second in selected if first != second) for pair in calls)
    assert result["estimator"] == record


def test_cache_budget_and_method_identity(tmp_path: Path) -> None:
    fake_w5 = ModuleType("fake_w5")
    fake_w5.validate_generation_record = lambda *_args, **_kwargs: None
    path = tmp_path / "population.json.gz"
    base = {"case_id": "case", "entries": [], "pairs": [], "line_winner_id": None,
            "paint_winner_id": None}
    for screen, accepted in ((None, 16), ({"method": automatic_generation.SCREEN_METHOD, "budget": 12}, 12)):
        record = {**base}
        if screen is not None:
            record["direction_screen"] = screen
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump(record, stream)
        assert _cached(path, "case", fake_w5, "G0", accepted) == record
        with pytest.raises(ValueError, match="direction screen"):
            _cached(path, "case", fake_w5, "G1", 28 - accepted)
        assert screen_matches(record, accepted)
    assert not screen_matches({**base, "direction_screen": {"method": "other", "budget": 12}}, 12)


def test_both_populations_receive_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    context = SimpleNamespace(case_id="case", source={"id": "case", "dimensions": {"width": 2, "height": 2}},
                              provenance=None, size=(2, 2))
    run_w5 = ModuleType("fake_w5")
    run_w5.validate_generation_record = lambda *_args, **_kwargs: None
    helpers = ModuleType("fake_automatic")
    helpers.frame_path = lambda *_args: tmp_path / "frame.png"
    verifier = {"read_json_gz": read, "frame_path": lambda *_args: tmp_path / "frame.png"}
    runtime = {"verifier": verifier, "run_w5": run_w5, "run_automatic": helpers,
               "vp_pruning": ModuleType("fake_vp_pruning"), "zone": None}
    monkeypatch.setattr(generation, "_new_direction", lambda *_args: {"settings": {}, "estimator": {}})
    monkeypatch.setattr(generation, "_filter_module", lambda *_args: SimpleNamespace(
        paint_masks=lambda *_args: {"paint": np.zeros((2, 2))},
        filtered_source=lambda source, _mask: {**source, "filtered": True}))
    monkeypatch.setattr(generation, "_native_frame", lambda *_args: (np.zeros((2, 2, 3)), tmp_path / "frame.png"))
    calls = []

    def fake_generate(source: dict, _direction: dict, _zone: object, _root: Path,
                      _helpers: ModuleType, budget: int) -> dict:
        calls.append((source.get("filtered", False), budget))
        return {"case_id": "case", "pairs": [], "entries": [], "line_winner_id": None,
                "paint_winner_id": None,
                "direction_screen": {"method": automatic_generation.SCREEN_METHOD, "budget": budget}}

    monkeypatch.setattr(generation, "generate", fake_generate)
    paths = generation.ensure_populations(tmp_path, context, runtime, tmp_path / "output", 12)
    assert calls == [(False, 12), (True, 12)]
    assert all(read(path)["direction_screen"]["budget"] == 12 for path in paths.values())
    with pytest.raises(ValueError, match="direction screen"):
        generation.ensure_populations(tmp_path, context, runtime, tmp_path / "output", 16)


def test_early_result_reuse_checks_budget(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from run_cases import write

    output = tmp_path / "output"
    result_path = output / "results/case.json.gz"
    write(result_path, {"case_record": "record.json.gz", "array_file": "arrays.npz",
                        "direction_screen": {"method": automatic_generation.SCREEN_METHOD, "budget": 12}})
    write(output / "record.json.gz", {"case_id": "case", "min_visible_lengthwise": 4,
                                       "min_visible_cross_court": 3})
    (output / "arrays.npz").touch()
    monkeypatch.setattr(run_cases, "load_runtime", lambda *_args: (
        None, SimpleNamespace(prepare_view=lambda *_args: None, read_json_gz=read), None))
    assert run_cases.run_case(tmp_path, output, "case", direction_budget=12)["status"] == "reused"
    with pytest.raises(ValueError, match="direction screen"):
        run_cases.run_case(tmp_path, output, "case", direction_budget=16)
