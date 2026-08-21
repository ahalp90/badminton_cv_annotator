from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.backends import BackendSpec, GenerationEvidence
from experiments.detail_schema import DetailArm, DetailCase, DetailManifest
from experiments.multiscale_schema import MultiscaleCase, Segment
from experiments.score_detail_trials import DETAIL_SCORE_SCHEMA, _aggregate

from experiments import combined_visual_trials as combined


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _combined_payload(tmp_path: Path) -> tuple[Path, dict]:
    clip = tmp_path / "clips/case.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"combined clip")
    segments = (Segment("S00001", 0, 300),)
    broad_frames = combined.storyboard_source_frames(
        segments,
        0,
        300,
        100,
        180,
        combined.BROAD_FRAMES,
    )
    assert broad_frames is not None
    case = {
        "case_id": "detail-context-sset_01-r000-S00001",
        "pair_id": "context-sset_01-r000",
        "context_case_id": "context-sset_01-r000--90",
        "video_id": "sset_01",
        "clip_path": "clips/case.mp4",
        "clip_sha256": _digest(b"combined clip"),
        "dense_reference_clip_sha256": _digest(b"dense clip"),
        "source_video_sha256": _digest(b"source video"),
        "source_fps": 25.0,
        "output_fps": 8.0,
        "broad_source_start_frame": 0,
        "broad_source_end_frame": 300,
        "broad_target_start_frame": 100,
        "broad_target_end_frame": 180,
        "broad_source_frames": list(broad_frames),
        "segments": [
            {
                "segment_id": "S00001",
                "source_start_frame": 0,
                "source_end_frame": 300,
            }
        ],
        "dense_source_frames": list(range(80, 200)),
        "target_start_frame": 100,
        "target_end_frame": 180,
        "target_segment_ids": ["S00001"],
    }
    payload = {
        "schema": combined.COMBINED_MANIFEST_SCHEMA,
        "expected_frames": 200,
        "broad_frames": 80,
        "dense_frames": 120,
        "width": 512,
        "height": 288,
        "output_fps": 8.0,
        "cases": [case],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_manifest_keeps_truth_out_and_preserves_required_frames(tmp_path: Path) -> None:
    path, payload = _combined_payload(tmp_path)

    case = combined.load_combined_manifest(path)[0]

    assert case.expected_frames == 200
    assert len(case.broad_source_frames) == 80
    assert len(case.dense_source_frames) == 120
    payload["cases"][0]["human_label"] = "live"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden truth key"):
        combined.load_combined_manifest(path)


def test_manifest_rejects_stale_clip_and_dropped_required_frame(tmp_path: Path) -> None:
    path, payload = _combined_payload(tmp_path)
    (tmp_path / "clips/case.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="clip hash differs"):
        combined.load_combined_manifest(path)

    (tmp_path / "clips/case.mp4").write_bytes(b"combined clip")
    payload["cases"][0]["broad_source_frames"][0] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sorted unique|dropped a required"):
        combined.load_combined_manifest(path)


def _context_case(index: int) -> MultiscaleCase:
    pair_id = f"context-sset_01-r{index:03d}"
    return MultiscaleCase(
        case_id=f"{pair_id}--90",
        pair_id=pair_id,
        video_id="sset_01",
        context_seconds=90,
        clip_path=Path(f"context-{index}.mp4"),
        source_start_frame=0,
        source_end_frame=300,
        target_start_frame=100,
        target_end_frame=180,
        sample_fps=8.0,
        expected_frames=96,
        width=512,
        height=288,
        source_frames=tuple(range(96)),
        segments=(Segment("S00001", 0, 300),),
        pipeline_priors={"span_id": index},
    )


def _detail_case(index: int, clip_hash: str) -> DetailCase:
    context_pair_id = f"context-sset_01-r{index:03d}"
    pair_id = f"{context_pair_id}-S00001"
    return DetailCase(
        case_id=f"detail-{pair_id}",
        pair_id=pair_id,
        context_case_id=f"{context_pair_id}--90",
        video_id="sset_01",
        clip_path=Path(f"dense-{index}.mp4"),
        clip_sha256=clip_hash,
        source_start_frame=80,
        source_end_frame=200,
        source_frames=tuple(range(80, 200)),
        target_start_frame=100,
        target_end_frame=180,
        boundary_frame=None,
        source_fps=25.0,
        sample_fps=25.0,
        expected_frames=120,
        width=512,
        height=288,
        target_segment_ids=("S00001",),
        deterministic_facts=None,
        broad_facts=None,
    )


def test_builder_scales_all_frozen_cases_and_hashes_source_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_path = tmp_path / "context.json"
    detail_path = tmp_path / "detail.json"
    source_path = tmp_path / "source.avi"
    context_path.write_bytes(b"context")
    detail_path.write_bytes(b"detail")
    source_path.write_bytes(b"source")
    context_cases = tuple(_context_case(index) for index in range(19))
    detail_cases = tuple(
        _detail_case(index, _digest(f"dense-{index}".encode())) for index in range(19)
    )
    detail_manifest = DetailManifest(
        arm=DetailArm.SHORT_ONLY,
        expected_frames=120,
        width=512,
        height=288,
        cases=detail_cases,
    )
    monkeypatch.setattr(combined, "load_manifest", lambda _path: context_cases)
    monkeypatch.setattr(
        combined, "load_detail_manifest", lambda *_args, **_kwargs: detail_manifest
    )
    monkeypatch.setattr(combined, "_source_info", lambda _path: (25.0, 300))

    source_hash_calls = 0
    real_sha256 = combined._sha256

    def counting_hash(path: Path) -> str:
        nonlocal source_hash_calls
        if path == source_path:
            source_hash_calls += 1
        return real_sha256(path)

    def fake_render(_source: Path, case: combined.CombinedVisualCase) -> None:
        case.clip_path.write_bytes(case.case_id.encode())

    monkeypatch.setattr(combined, "_sha256", counting_hash)
    monkeypatch.setattr(combined, "_render_combined_clip", fake_render)
    output = tmp_path / "output"

    combined.build_combined_cases(
        context_path,
        detail_path,
        {"sset_01": source_path},
        output,
    )

    cases = combined.load_combined_manifest(output / "inference/manifest.json")
    assert len(cases) == 19
    assert source_hash_calls == 1
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["cases"] == 19


def test_builder_fails_before_render_when_broad_map_does_not_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context_path = tmp_path / "context.json"
    detail_path = tmp_path / "detail.json"
    source_path = tmp_path / "source.avi"
    for path in (context_path, detail_path, source_path):
        path.write_bytes(b"input")
    detail_manifest = DetailManifest(
        arm=DetailArm.SHORT_ONLY,
        expected_frames=120,
        width=512,
        height=288,
        cases=(_detail_case(0, _digest(b"dense")),),
    )
    monkeypatch.setattr(combined, "load_manifest", lambda _path: (_context_case(0),))
    monkeypatch.setattr(
        combined, "load_detail_manifest", lambda *_args, **_kwargs: detail_manifest
    )
    monkeypatch.setattr(combined, "_source_info", lambda _path: (25.0, 300))
    monkeypatch.setattr(combined, "storyboard_source_frames", lambda *_args: None)

    with pytest.raises(ValueError, match="do not fit 80"):
        combined.build_combined_cases(
            context_path,
            detail_path,
            {"sset_01": source_path},
            tmp_path / "output",
        )


class _FakeBackend:
    spec = BackendSpec(
        key="fake",
        model_id="fake/model",
        model_revision="revision",
        backend_name="fake-runtime",
        backend_distribution="fake",
        expected_backend_version="1",
        cache_dtype="none",
        package_names=(),
    )
    backend_version = "1"

    def generate(self, *_args, **_kwargs) -> GenerationEvidence:
        return GenerationEvidence(
            raw_response="{maybe}",
            sampled_input_frames=tuple(range(200)),
            width=512,
            height=288,
            visual_tokens=14_400,
            total_input_tokens=15_000,
        )


def _parent_row() -> dict:
    return {
        "arm": "short_only",
        "case_id": "detail-context-sset_01-r000-S00001",
        "pair_id": "context-sset_01-r000",
        "context_case_id": "context-sset_01-r000--90",
        "video_id": "sset_01",
        "target_start_frame": 100,
        "target_end_frame": 180,
        "target_frames": 80,
        "valid_reply": True,
        "predicted_content": "live",
        "truth_content_frames": {"live": 80},
        "exact_scene_correct_frames": 80,
        "exact_scene_accuracy": 1.0,
        "binary_live_nonlive_correct_frames": 80,
        "binary_live_nonlive_accuracy": 1.0,
        "binary_scene_correct_frames": 80,
        "binary_scene_accuracy": 1.0,
        "predicted_route": "routine_live",
        "truth_route": "routine_live",
        "route_correct": True,
        "parser_error": None,
        "generation_error": None,
    }


def test_runner_uses_full_grid_and_scorer_keeps_invalid_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _payload = _combined_payload(tmp_path)
    attempts = tmp_path / "attempts"
    monkeypatch.setattr(combined, "_load_backend", lambda _backend: _FakeBackend())

    combined.run_combined_trials(manifest_path, "internvideo3", attempts)

    attempt_path = attempts / "internvideo3/detail-context-sset_01-r000-S00001.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert len(attempt["sampling"]["sampled_input_frames"]) == 200
    assert attempt["parsed_response"] is None
    assert attempt["parser_error"]

    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "schema": "vlm-multiscale-truth-v1",
                "cases": [
                    {
                        "pair_id": "context-sset_01-r000",
                        "video_id": "sset_01",
                        "truth_intervals": [
                            {
                                "source_start_frame": 100,
                                "source_end_frame": 180,
                                "truth": "live",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parent_row = _parent_row()
    parent_score = tmp_path / "parent.json"
    parent_score.write_text(
        json.dumps(
            {
                "schema": DETAIL_SCORE_SCHEMA,
                "backend": "internvideo3",
                "rows": [parent_row],
                "by_arm": {"short_only": _aggregate([parent_row])},
            }
        ),
        encoding="utf-8",
    )

    result = combined.score_combined_trials(
        manifest_path,
        truth_path,
        attempts,
        "internvideo3",
        parent_score,
    )

    assert result["combined_visual"]["valid_replies"] == 0
    assert result["combined_visual"]["routine_live_recall"] == 0.0
    assert result["changed_route_case_ids"] == ["detail-context-sset_01-r000-S00001"]


def test_score_output_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "score.json"
    combined._write_new_json(output, {"schema": combined.COMBINED_SCORE_SCHEMA})
    with pytest.raises(FileExistsError):
        combined._write_new_json(output, {})
