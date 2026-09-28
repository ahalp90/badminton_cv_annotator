"""A detector's with block keeps one worker pool for every view's search and scoring, then closes it."""

import json
import multiprocessing
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from court_detector import (
    detect,
    generation,
    scoring,
    search_records,
)
from court_detector.inputs import (
    ViewInputs,
    same_frame_provenance,
)
from scratch.court_det_fix.court_detector import run_views


def child_pids() -> set[int]:
    """This process's live child processes, such as pool workers."""
    return {child.pid for child in multiprocessing.active_children() if child.pid is not None}


def test_a_serial_detector_opens_no_pool() -> None:
    with detect.CourtDetector(detect.Switches()) as detector:
        assert detector.pool is None


def test_workers_close_when_the_block_fails_and_a_later_block_opens_new_ones() -> None:
    detector = detect.CourtDetector(detect.Switches(workers=2))
    before = child_pids()
    with pytest.raises(RuntimeError, match="view failed"), detector:
        assert detector.pool is not None
        first_worker = detector.pool.submit(os.getpid).result()
        assert first_worker in child_pids() - before
        raise RuntimeError("view failed")
    assert detector.pool is None and child_pids() == before
    with detector:
        assert detector.pool is not None
        assert detector.pool.submit(os.getpid).result() != first_worker
    assert detector.pool is None and child_pids() == before


def test_a_detector_refuses_to_open_a_second_pool() -> None:
    with detect.CourtDetector(detect.Switches(workers=2)) as detector:
        with pytest.raises(RuntimeError, match="already open"), detector:
            pass
        assert detector.pool is not None


def test_search_and_scoring_use_the_open_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    pools_seen = []

    def generate(*_args: object, pool: object, **_kwargs: object) -> dict:
        pools_seen.append(("search", pool))
        return {"entries": [], "pairs": []}

    def score_populations(*_args: object, pool: object, **_kwargs: object) -> SimpleNamespace:
        pools_seen.append(("scoring", pool))
        return SimpleNamespace(parents=[], children=[], fit_rows=[], c_rankings={}, identity_resolution={},
                               line_maps=None)

    # The detector's live modules are these module objects, so it calls the fakes.
    monkeypatch.setattr(generation, "generate", generate)
    monkeypatch.setattr(scoring, "score_populations", score_populations)
    monkeypatch.setattr(search_records, "direction_record", lambda *_args: {})
    monkeypatch.setattr(detect.search, "paint_mask", lambda *_args: None)
    monkeypatch.setattr(detect.search, "filtered_source", lambda source, _mask: source)
    monkeypatch.setattr(detect, "choose_court", lambda view_id, *_args: detect.CourtResult(
        view_id, None, "no_gated_court", None, None))
    detector = detect.CourtDetector(detect.Switches(workers=2, self_checks=False))
    context = SimpleNamespace(size=(20, 10))
    source = {"all_feet_px": [[[1., 2.]]], "dimensions": {"width": 20, "height": 10}}
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    view = ViewInputs("view", frame, 0, (0, 10), np.empty((0, 4)), np.empty((0, 4)), same_frame_provenance("view", 0))

    def search_and_score() -> None:
        populations = detector.search(context, source, frame, detect.Laps())
        detector.score_and_choose(view, context, populations, [], frame, detect.Laps(), {})

    search_and_score()
    with detector:
        open_pool = detector.pool
        search_and_score()
        search_and_score()
    assert open_pool is not None
    one_view = ["search", "search", "scoring"]
    assert pools_seen == [(stage, None) for stage in one_view] + [(stage, open_pool) for stage in one_view * 2]


def test_view_batch_replaces_a_dead_pool_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class CrashingFirstView(detect.CourtDetector):
        def detect(self, view: ViewInputs, *_args: object, **_kwargs: object) -> detect.CourtResult:
            assert self.pool is not None
            if view.view_id == "first":
                self.pool.submit(os._exit, 1).result()
            self.pool.submit(os.getpid).result()
            return detect.CourtResult(view.view_id, None, "no_gated_court", None, {})

    detector = CrashingFirstView(detect.Switches(workers=2, self_checks=False))
    monkeypatch.setattr(run_views, "CourtDetector", lambda _switches: detector)
    view_ids = ["first", "second", "third"]
    anchors = dict(zip(view_ids, (50, 100, 150), strict=True))
    view_path = tmp_path / "views.json"
    view_path.write_text(json.dumps([{"case_id": name, "anchor": anchors[name]} for name in view_ids]))
    shot_path = tmp_path / "shots.jsonl"
    shot_path.write_text("")
    monkeypatch.setattr(run_views, "VIEWS", view_path)
    monkeypatch.setattr(run_views, "SHOT_CHECK", shot_path)
    people_record = {"video_path": "test-video", "frame_count": 201, "fps": 25,
                     "video_size": [20, 10], "samples": []}
    monkeypatch.setattr(run_views, "read_json_gz", lambda path: (
        {"cases": []} if path == run_views.MANIFEST else people_record))
    monkeypatch.setattr(run_views, "DecodedVideo", lambda *_args: SimpleNamespace(fps=25, size=(20, 10)))
    monkeypatch.setattr(run_views, "pack_sources", lambda *_args: (
        {name: {"segments_px": [], "bbox_px": []} for name in view_ids},
        {name: same_frame_provenance(name, anchors[name]) for name in view_ids},
        {name: tmp_path / "frame.png" for name in view_ids},
    ))
    monkeypatch.setattr(run_views.cv2, "imread", lambda *_args: np.zeros((10, 20, 3), dtype=np.uint8))
    output = tmp_path / "output"
    monkeypatch.setattr(sys, "argv", ["run_views", "--people", str(tmp_path), "--output", str(output),
                                     "--workers", "2", "--no-self-checks", *view_ids])
    before = child_pids()

    assert run_views.main() == 1

    rows = [json.loads((output / "results" / f"{name}.json").read_text()) for name in view_ids]
    assert "BrokenProcessPool" in rows[0]["error"]
    assert [row["error"] for row in rows[1:]] == [None, None]
    assert detector.pool is None and child_pids() == before
