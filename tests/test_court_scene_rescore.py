"""Saved-fit rescoring: the combined score, ties, missing evidence and untouched inputs."""

from __future__ import annotations

import copy
import csv
import gzip
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from court_detector import net_choice
from court_detector.detect import NET_WEIGHT
from experiments.annotator.court_scene_sampling import rescore
from experiments.annotator.court_scene_sampling.sampling import METHODS

NATIVE_SIZE = [1920, 1080]
# A plausible court in TL TR BR BL order; its net projects inside the frame.
CORNERS = [[700.0, 300.0], [1220.0, 300.0], [1500.0, 900.0], [420.0, 900.0]]
FAR_SEGMENT = [[10.0, 1000.0, 60.0, 1000.0]]  # nowhere near the net posts


def post_segments(corners: list[list[float]]) -> list[list[float]]:
    """Native-pixel fragments along both net posts, so each post counts as supported."""
    context = SimpleNamespace(segments=np.empty((0, 4)), size=(960, 540), native_size=tuple(NATIVE_SIZE))
    pieces = net_choice.project_pieces(corners, context)["pieces_native_px"]
    return [pieces[2].ravel().tolist(), pieces[3].ravel().tolist()]


def accepted(role: str, paint: float, geometry: float | None = 0.5) -> dict:
    return {"role": role, "frame_index": {"first": 10, "middle": 20, "last": 30}[role], "view_id": f"v_{role}",
            "route": "full_search", "status": "court", "corners_native_px": CORNERS, "paint_score": paint,
            "evidence": {"q_paint10_span_weighted": paint, "q_geom_span_weighted": geometry}}


def outcome(frames: list[dict], chosen_position: int) -> dict:
    return {"status": "court", "frames": frames, "chosen_position": chosen_position,
            "seconds": {"detector_walltime_measured": 12.5}, "established": {"view_id": "kept"}}


def results_with(method_outcome: dict) -> dict:
    scene = {"scene_id": "s", "status": "analysed", "methods": {method: copy.deepcopy(method_outcome)
                                                                 for method in METHODS}}
    return {"schema": rescore.SOURCE_SCHEMA, "finished": True,
            "videos": [{"video_id": "v", "native_size": NATIVE_SIZE, "scenes": [scene], "later_scenes": []}]}


def cache_for(segments_by_view: dict[str, list]) -> dict:
    frames = {view_id: {"native_size": NATIVE_SIZE, "segments_native_px": segments}
              for view_id, segments in segments_by_view.items()}
    return {"schema": rescore.LINES_SCHEMA, "frames": frames, "recovery_seconds": 3.0}


def only_method(results: dict, cache: dict) -> dict:
    return rescore.rescore(results, cache)[0]["methods"]["full_three"]


def test_combined_score_uses_measured_net_posts() -> None:
    frames = [accepted("middle", 0.50, 0.60), accepted("last", 0.52, 0.40)]
    cache = cache_for({"v_middle": post_segments(CORNERS), "v_last": FAR_SEGMENT})
    method = only_method(results_with(outcome(frames, 1)), cache)
    middle, last = method["candidates"]
    assert middle["net_state"] == "measured" and middle["net_reward"] == 1.0
    assert last["net_state"] == "measured" and last["net_reward"] == 0.0
    assert middle["combined_score"] == pytest.approx(0.9 * 0.50 + 0.1 * 0.60 + NET_WEIGHT)
    assert last["combined_score"] == pytest.approx(0.9 * 0.52 + 0.1 * 0.40)
    assert method["selected"] == {"paint": 1, "ranked": 0}
    assert method["changes"] == {"ranked": True}


def test_exact_ties_go_middle_then_first_then_last() -> None:
    frames = [accepted("last", 0.5), accepted("first", 0.5), accepted("middle", 0.5)]
    cache = cache_for({"v_last": FAR_SEGMENT, "v_first": FAR_SEGMENT, "v_middle": FAR_SEGMENT})
    method = only_method(results_with(outcome(frames, 2)), cache)
    assert method["ranked_order"] == [2, 1, 0]
    assert method["selected"] == {"paint": 2, "ranked": 2}


def test_no_court_and_unevaluated_outcomes_stay_unchanged() -> None:
    results = results_with({"status": "no_court", "frames": [], "chosen_position": None})
    results["videos"][0]["scenes"][0]["methods"]["seed_refit"] = {"status": "detection_failed", "error": "x"}
    results["videos"][0]["later_scenes"] = [{"scene_id": "short", "status": "scene_too_short_for_feet"}]
    scenes = rescore.rescore(results, cache_for({}))
    assert scenes[0]["methods"]["full_three"] == {"state": "no_court", "original_status": "no_court"}
    assert scenes[0]["methods"]["seed_refit"] == {"state": "no_evaluation", "original_status": "detection_failed"}
    assert scenes[1]["methods"]["cheap_first"]["original_status"] == "scene_too_short_for_feet"


@pytest.mark.parametrize("gap", ["lines", "geometry_score"])
def test_missing_score_evidence_keeps_the_original_winner_for_the_whole_method(gap: str) -> None:
    # A gap disables the method's rescoring: no zero stands in for it and the frame is not dropped.
    frames = [accepted("middle", 0.50, None if gap == "geometry_score" else 0.5), accepted("last", 0.52)]
    lines = {"v_last": FAR_SEGMENT} if gap == "lines" else {"v_middle": post_segments(CORNERS), "v_last": FAR_SEGMENT}
    method = only_method(results_with(outcome(frames, 1)), cache_for(lines))
    assert method["state"] == "score_evidence_missing"
    assert method["missing_score_evidence"] == [{"position": 0, "view_id": "v_middle", "missing": [gap]}]
    assert method["selected"] == {"paint": 1, "ranked": 1}
    assert "combined_score" not in method["candidates"][0]
    assert method["candidates"][1]["combined_score"] == pytest.approx(0.9 * 0.52 + 0.1 * 0.5)


def test_command_writes_new_outputs_without_changing_its_inputs(tmp_path: Path, monkeypatch) -> None:
    frames = [accepted("middle", 0.50, 0.60), accepted("last", 0.52, 0.40)]
    results = results_with(outcome(frames, 1))
    cache = cache_for({"v_middle": post_segments(CORNERS), "v_last": FAR_SEGMENT})
    before = copy.deepcopy(results)
    rescore.rescore(results, cache)
    assert results == before

    results_path, cache_path = tmp_path / "results.json.gz", tmp_path / "lines.json.gz"
    for path, value in ((results_path, results), (cache_path, cache)):
        with gzip.open(path, "wt") as stream:
            json.dump(value, stream)
    source_bytes = results_path.read_bytes()
    output_dir = tmp_path / "rescored"
    monkeypatch.setattr(sys, "argv", ["rescore", "--results", str(results_path), "--lines-cache", str(cache_path),
                                      "--output-dir", str(output_dir)])
    assert rescore.main() == 0
    assert results_path.read_bytes() == source_bytes
    with gzip.open(output_dir / "results.json.gz", "rt") as stream:
        report = json.load(stream)
    assert report["line_recovery_seconds"] == 3.0
    with gzip.open(output_dir / "comparison.csv.gz", "rt") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == list(METHODS)
    assert (rows[0]["paint_role"], rows[0]["ranked_role"], rows[0]["ranked_changed"]) == ("last", "middle", "True")
