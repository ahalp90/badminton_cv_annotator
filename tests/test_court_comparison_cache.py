from __future__ import annotations

import base64
import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
import pytest

import annotator.court_evidence as evidence
from annotator.court_views import CourtView

prepare = import_module('experiments.annotator.court_geometry_repair.scripts.prepare_court_comparison')
rebuild = import_module('experiments.annotator.court_geometry_repair.scripts.rebuild_scene_courts')
candidate_fixtures = import_module('tests.test_court_candidate_selection')
court_fixtures = import_module('tests.test_court_evidence')


def _view() -> CourtView:
    generator = np.random.default_rng(42)
    image = generator.integers(0, 256, (540, 960), dtype=np.uint8)
    return CourtView(np.zeros((3, 16, 16), dtype=bool), image)


def _write_cache(
    root: Path,
    video_id: int,
    scenes: list[evidence.SceneEvidence],
) -> None:
    rebuild.write_json(
        root / f"video_{video_id:02d}_metadata.json.gz",
        {
            "video_id": video_id,
            "scene_count": len(scenes),
        },
    )
    for index, scene in enumerate(scenes):
        path = root / f"video_{video_id:02d}_scene_{index:04d}.json.gz"
        rebuild.write_json(
            path,
            rebuild.scene_payload(
                index,
                (scene.start_frame, scene.end_frame),
                list(scene.sampled_frame_indices),
                (1280, 720),
                scene.quad,
                scene.view,
                [],
                0.0,
            ),
        )


def _build(
    scenes: list[evidence.SceneEvidence],
    intervals: list[tuple[int, int]],
    *,
    bboxes: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    ndet: np.ndarray | None = None,
) -> evidence.CourtEvidenceResult:
    if bboxes is None or scores is None or ndet is None:
        bboxes, scores, ndet = court_fixtures._pose_inputs(intervals[-1][1])
    return evidence.build_detected_court_evidence(
        "cache-test",
        "detected",
        "7",
        (1280.0, 720.0),
        intervals,
        scenes,
        bboxes,
        scores,
        ndet,
        detector_resolution=(1280.0, 720.0),
    )


def test_cached_alternative_round_trip_matches_direct_acceptance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    quad = candidate_fixtures._candidate_scene(monkeypatch)
    view = _view()
    interval = (0, 10)
    direct_scene = evidence.SceneEvidence(*interval, (0,), quad, view)
    _write_cache(tmp_path, 7, [direct_scene])

    cached_scene = prepare._scene_evidence(tmp_path, [interval], 7, frame_wh=(1280, 720))[0]
    assert cached_scene.quad is not None
    np.testing.assert_allclose(cached_scene.quad.alternative_corners_px, quad.alternative_corners_px)
    np.testing.assert_array_equal(cached_scene.view.image, view.image)
    np.testing.assert_array_equal(cached_scene.view.hashes, view.hashes)

    bboxes = np.tile(
        np.array([[[600.0, 220.0, 650.0, 300.0], [600.0, 500.0, 650.0, 580.0]]]),
        (10, 1, 1),
    )
    scores = np.full((10, 2), 0.9)
    ndet = np.full(10, 2, dtype=int)
    direct = _build([direct_scene], [interval], bboxes=bboxes, scores=scores, ndet=ndet)
    cached = _build([cached_scene], [interval], bboxes=bboxes, scores=scores, ndet=ndet)
    np.testing.assert_array_equal(cached.keep_vote, direct.keep_vote)
    np.testing.assert_array_equal(cached.court_present, direct.court_present)
    np.testing.assert_allclose(
        cached.scene_records[0].active_corners_native_px,
        direct.scene_records[0].active_corners_native_px,
    )
    assert json.dumps([asdict(row) for row in cached.scene_records], default=rebuild.json_value) == json.dumps(
        [asdict(row) for row in direct.scene_records], default=rebuild.json_value,
    )


def test_cached_view_round_trip_preserves_three_scene_grouping(tmp_path: Path) -> None:
    corners = np.array([[100.0, 100.0], [1100.0, 100.0], [1200.0, 700.0], [0.0, 700.0]])
    intervals = [(0, 10), (10, 20), (20, 30)]
    view = _view()
    raw = [corners - 3.0, corners, corners + 3.0]
    direct_scenes = [
        evidence.SceneEvidence(start, end, (start,), court_fixtures._quad(quad), view)
        for (start, end), quad in zip(intervals, raw, strict=True)
    ]
    _write_cache(tmp_path, 7, direct_scenes)
    cached_scenes = list(prepare._scene_evidence(tmp_path, intervals, 7, frame_wh=(1280, 720)))

    direct = _build(direct_scenes, intervals)
    cached = _build(cached_scenes, intervals)
    assert [record.view_group_index for record in direct.scene_records] == [0, 0, 0]
    assert [record.view_group_index for record in cached.scene_records] == [0, 0, 0]
    np.testing.assert_array_equal(cached.keep_vote, direct.keep_vote)
    np.testing.assert_array_equal(cached.court_present, direct.court_present)
    np.testing.assert_array_equal(
        cached.inputs.homography_rows.to_numpy(),
        direct.inputs.homography_rows.to_numpy(),
    )
    assert json.dumps([asdict(row) for row in cached.scene_records], default=rebuild.json_value) == json.dumps(
        [asdict(row) for row in direct.scene_records], default=rebuild.json_value,
    )


def test_legacy_cache_without_current_evidence_fails_with_regeneration_guidance(tmp_path: Path) -> None:
    rebuild.write_json(tmp_path / "video_07_metadata.json.gz", {"video_id": 7, "scene_count": 1})
    rebuild.write_json(
        tmp_path / "video_07_scene_0000.json.gz",
        {"interval": [0, 10], "sampled_frame_indices": [0], "court_quad": None},
    )

    with pytest.raises(ValueError, match="regenerate with rebuild_scene_courts.py"):
        prepare._scene_evidence(tmp_path, [(0, 10)], 7, frame_wh=(1280, 720))


@pytest.mark.parametrize('field_name', ['alternative_corners_px', 'line_segments_px'])
def test_current_cache_without_quad_evidence_fails_with_regeneration_guidance(tmp_path: Path, field_name: str) -> None:
    interval = (0, 10)
    view = _view()
    payload = rebuild.scene_payload(
        0,
        interval,
        [0],
        (1280, 720),
        court_fixtures._quad(np.array([[100.0, 100.0], [1100.0, 100.0], [1200.0, 700.0], [0.0, 700.0]])),
        view,
        [],
        0.0,
    )
    payload["court_quad"] = asdict(payload["court_quad"])
    del payload["court_quad"][field_name]
    rebuild.write_json(
        tmp_path / "video_07_metadata.json.gz",
        {"video_id": 7, "scene_count": 1},
    )
    rebuild.write_json(tmp_path / "video_07_scene_0000.json.gz", payload)

    with pytest.raises(ValueError, match=field_name):
        prepare._scene_evidence(tmp_path, [interval], 7, frame_wh=(1280, 720))


def test_current_cache_quad_without_view_fails_with_regeneration_guidance(tmp_path: Path) -> None:
    interval = (0, 10)
    payload = rebuild.scene_payload(
        0,
        interval,
        [0],
        (1280, 720),
        court_fixtures._quad(np.array([[100.0, 100.0], [1100.0, 100.0], [1200.0, 700.0], [0.0, 700.0]])),
        None,
        [],
        0.0,
    )
    rebuild.write_json(tmp_path / "video_07_metadata.json.gz", {"video_id": 7, "scene_count": 1})
    rebuild.write_json(tmp_path / "video_07_scene_0000.json.gz", payload)

    with pytest.raises(ValueError, match="omits view"):
        prepare._scene_evidence(tmp_path, [interval], 7, frame_wh=(1280, 720))


def test_cached_geometry_requires_matching_source_dimensions(tmp_path: Path) -> None:
    scene = evidence.SceneEvidence(0, 10, (0,), None, None)
    _write_cache(tmp_path, 7, [scene])
    with pytest.raises(ValueError, match="cached frame dimensions differ"):
        prepare._scene_evidence(tmp_path, [(0, 10)], 7, frame_wh=(1920, 1080))
    restored = prepare._scene_evidence(tmp_path, [(0, 10)], 7, frame_wh=(1280, 720))[0]
    assert restored.quad is None and restored.view is None


def test_cached_view_rejects_lossy_image_encoding(tmp_path: Path) -> None:
    view = _view()
    payload = rebuild.view_payload(view)
    success, encoded = cv2.imencode('.jpg', view.image)
    assert success
    payload['image_png_base64'] = base64.b64encode(encoded.tobytes()).decode('ascii')
    with pytest.raises(ValueError, match="must be PNG"):
        prepare._view(payload, path=tmp_path / 'scene.json.gz')
