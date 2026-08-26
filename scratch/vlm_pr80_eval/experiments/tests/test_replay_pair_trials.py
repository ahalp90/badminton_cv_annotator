from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from experiments.backends import BackendSpec, GenerationEvidence
from experiments.detail_schema import DetailArm, DetailCase, DetailManifest

from experiments import replay_pair_trials as replay


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _detail_case(
    tmp_path: Path,
    case_id: str,
    *,
    span_id: int,
    span_start: int,
    span_end: int,
    target_start: int,
    target_end: int,
    filtered_contact_count: int = 2,
    track_visible_fraction: float = 0.8,
) -> DetailCase:
    clip = tmp_path / f"{case_id}.mp4"
    clip.write_bytes(case_id.encode())
    return DetailCase(
        case_id=case_id,
        pair_id=f"pair-{case_id}",
        context_case_id=f"context-{case_id}--90",
        video_id="sset_01",
        clip_path=clip,
        clip_sha256=_digest(clip.read_bytes()),
        source_start_frame=target_start - 40,
        source_end_frame=target_start + 80,
        source_frames=tuple(range(target_start - 40, target_start + 80)),
        target_start_frame=target_start,
        target_end_frame=target_end,
        boundary_frame=None,
        source_fps=25.0,
        sample_fps=25.0,
        expected_frames=120,
        width=512,
        height=288,
        target_segment_ids=("S0001",),
        deterministic_facts={
            "inspected_segment": {
                "segment_id": "S0001",
                "source_start_frame": target_start - 40,
                "source_end_frame": target_start + 80,
            },
            "proposed_span": {
                "source_start_frame": span_start,
                "source_end_frame": span_end,
            },
            "pipeline_priors": {
                "span_id": span_id,
                "filtered_contact_count": filtered_contact_count,
                "track_visible_fraction": track_visible_fraction,
            },
        },
        broad_facts=None,
    )


def _pair_case(tmp_path: Path, case_id: str = "target") -> replay.ReplayPairCase:
    clip = tmp_path / "clips" / f"{case_id}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"pair clip")
    return replay.ReplayPairCase(
        case_id=case_id,
        detail_case_id=case_id,
        pair_id=f"pair-{case_id}",
        context_case_id=f"context-{case_id}--90",
        video_id="sset_01",
        clip_path=clip,
        clip_sha256=_digest(clip.read_bytes()),
        reference_case_id="reference",
        reference_clip_sha256=_digest(b"reference clip"),
        source_video_sha256=_digest(b"source"),
        source_fps=25.0,
        output_fps=8.0,
        reference_source_frames=tuple(range(20, 140)),
        target_source_frames=tuple(range(200, 320)),
        target_start_frame=240,
        target_end_frame=250,
        target_segment_ids=("S0001",),
        candidate_span_id=3,
        candidate_start_frame=20,
        candidate_end_frame=140,
        candidate_filtered_contact_count=2,
        candidate_track_visible_fraction=0.8,
        candidate_gap_frames=100,
        expected_frames=240,
        width=512,
        height=288,
    )


def test_manifest_rejects_truth_and_keeps_exact_frame_maps(tmp_path: Path) -> None:
    case = _pair_case(tmp_path)
    payload = {
        "schema": replay.PAIR_MANIFEST_SCHEMA,
        "expected_frames": 240,
        "width": 512,
        "height": 288,
        "output_fps": 8.0,
        "cases": [replay._case_payload(case)],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = replay.load_replay_pair_manifest(path, verify_clip_hash=True)
    assert loaded[0].reference_source_frames == tuple(range(20, 140))
    assert loaded[0].target_source_frames == tuple(range(200, 320))

    payload["cases"][0]["truth"] = "live"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden truth key"):
        replay.load_replay_pair_manifest(path)


def test_pairing_uses_gap_then_contact_count_visibility_then_span_id() -> None:
    candidates = [
        {"span_id": 9, "start_frame": 0, "end_frame": 90, "filtered_contact_count": 2, "track_visible_fraction": 0.9},
        {"span_id": 4, "start_frame": 10, "end_frame": 90, "filtered_contact_count": 3, "track_visible_fraction": 0.2},
        {"span_id": 3, "start_frame": 20, "end_frame": 90, "filtered_contact_count": 3, "track_visible_fraction": 0.2},
        {"span_id": 1, "start_frame": 20, "end_frame": 90, "filtered_contact_count": 0, "track_visible_fraction": 1.0},
        {"span_id": 2, "start_frame": 20, "end_frame": 90, "filtered_contact_count": 1, "track_visible_fraction": 1.0},
    ]
    selected = replay.select_reference_span(
        candidates,
        target_start_frame=100,
        source_fps=25.0,
    )
    assert selected is not None
    assert selected["span_id"] == 3


def test_reference_case_selection_uses_gold_duration_then_later_start(tmp_path: Path) -> None:
    first = _detail_case(tmp_path, "first", span_id=3, span_start=0, span_end=140, target_start=20, target_end=60)
    second = _detail_case(tmp_path, "second", span_id=3, span_start=0, span_end=140, target_start=30, target_end=70)
    third = _detail_case(tmp_path, "third", span_id=3, span_start=0, span_end=140, target_start=31, target_end=71)
    assert replay._choose_reference_case(
        (first, second, third), 3, span_start_frame=0, span_end_frame=140
    ) is third


def test_builder_uses_manifest_facts_and_reports_unavailable_requested_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _detail_case(
        tmp_path,
        "reference",
        span_id=3,
        span_start=100,
        span_end=200,
        target_start=120,
        target_end=180,
    )
    target = _detail_case(
        tmp_path,
        "target",
        span_id=7,
        span_start=300,
        span_end=450,
        target_start=500,
        target_end=520,
    )
    detail = DetailManifest(
        arm=DetailArm.DETERMINISTIC,
        expected_frames=120,
        width=512,
        height=288,
        cases=(reference, target),
    )
    monkeypatch.setattr(replay, "load_detail_manifest", lambda *_args, **_kwargs: detail)
    monkeypatch.setattr(replay, "_source_info", lambda _path: (25.0, 1_000))
    monkeypatch.setattr(
        replay,
        "_render_pair_clip",
        lambda _source, case: case.clip_path.write_bytes(case.case_id.encode()),
    )
    detail_manifest_path = tmp_path / "detail.json"
    detail_manifest_path.write_bytes(b"frozen detail manifest")
    source = tmp_path / "source.avi"
    source.write_bytes(b"source")
    output = tmp_path / "pairs"

    replay.build_replay_pair_cases(
        detail_manifest_path,
        output,
        source_videos={"sset_01": source},
        case_ids={"target", "missing"},
    )

    loaded = replay.load_replay_pair_manifest(output / "inference/manifest.json", verify_clip_hash=True)
    assert len(loaded) == 1
    assert loaded[0].reference_case_id == "reference"
    provenance = json.loads((output / "scoring/provenance.json").read_text(encoding="utf-8"))
    assert provenance["unavailable_cases"] == {
        "missing": "case ID is absent from the deterministic detail manifest"
    }


def test_prompt_and_parser_are_strict_but_allow_one_json_fence(tmp_path: Path) -> None:
    case = _pair_case(tmp_path)
    prompt = replay.build_replay_pair_prompt(case)
    assert "REFERENCE" in prompt and "TARGET" in prompt
    assert replay.parse_relation_reply('{"target_relation":"repeated_action"}') is replay.TargetRelation.REPEATED_ACTION
    assert replay.parse_relation_reply('```json\n{"target_relation":"different_action"}\n```') is replay.TargetRelation.DIFFERENT_ACTION
    assert replay.parse_relation_reply("different_action") is replay.TargetRelation.DIFFERENT_ACTION
    assert replay.parse_relation_reply("{target_relation: unclear}") is replay.TargetRelation.UNCLEAR
    with pytest.raises(ValueError):
        replay.parse_relation_reply('{"target_relation":"different_action","extra":1}')
    with pytest.raises(ValueError):
        replay.parse_relation_reply("the clips show different_action")


class _FakeBackend:
    spec = BackendSpec("fake", "fake/model", "rev", "fake", "fake", "1", "none", ())
    backend_version = "1"

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, *_args, **_kwargs) -> GenerationEvidence:
        return GenerationEvidence(
            self.response,
            tuple(range(240)),
            512,
            288,
            20,
            40,
        )


def test_runner_records_identity_and_rejects_wrong_frame_count_on_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _pair_case(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": replay.PAIR_MANIFEST_SCHEMA,
                "expected_frames": 240,
                "width": 512,
                "height": 288,
                "output_fps": 8.0,
                "cases": [replay._case_payload(case)],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(replay, "_load_backend", lambda _name: _FakeBackend('{"target_relation":"different_action"}'))
    attempts = tmp_path / "attempts"
    replay.run_replay_pair_trials(manifest_path, "internvideo3", attempts)
    attempt_path = attempts / "internvideo3" / "target.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["parsed_response"] == {"target_relation": "different_action"}
    attempt["sampling"]["sampled_input_frames"] = list(range(239))
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    with pytest.raises(ValueError, match="all 240 frames"):
        replay._load_attempt(attempts, "internvideo3", case, _digest(manifest_path.read_bytes()))


def _write_attempt(
    attempts: Path,
    case: replay.ReplayPairCase,
    manifest_sha256: str,
    relation: str | None,
) -> None:
    prompt = replay.build_replay_pair_prompt(case)
    raw = "not json" if relation is None else json.dumps({"target_relation": relation})
    parsed = None if relation is None else {"target_relation": relation}
    payload = {
        "schema": replay.PAIR_ATTEMPT_SCHEMA,
        "backend": "internvideo3",
        "model": {},
        "manifest_sha256": manifest_sha256,
        "case": replay._case_payload(case),
        "prompt": prompt,
        "prompt_sha256": _digest(prompt.encode()),
        "raw_response": raw,
        "parsed_response": parsed,
        "parser_error": None if relation is not None else "replay-pair reply is not valid JSON: Expecting value",
        "generation_error": None,
        "elapsed_seconds": 0.1,
        "sampling": {
            "expected_input_frames": 240,
            "sampled_input_frames": list(range(240)),
            "requested_fps": 8.0,
            "requested_width": 512,
            "requested_height": 288,
            "max_new_tokens": 128,
            "width": 512,
            "height": 288,
            "visual_tokens": 20,
            "total_input_tokens": 40,
        },
    }
    (attempts / "internvideo3").mkdir(parents=True, exist_ok=True)
    (attempts / "internvideo3" / f"{case.case_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_score_reports_veto_rules_material_filter_and_subset_delta(tmp_path: Path) -> None:
    first = _pair_case(tmp_path, "live")
    second = replace(first, case_id="short", detail_case_id="short", pair_id="pair-short", context_case_id="context-short--90", target_start_frame=200, target_end_frame=202, target_source_frames=tuple(range(160, 280)), candidate_gap_frames=60)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": replay.PAIR_MANIFEST_SCHEMA,
                "expected_frames": 240,
                "width": 512,
                "height": 288,
                "output_fps": 8.0,
                "cases": [replay._case_payload(first), replay._case_payload(second)],
            }
        ),
        encoding="utf-8",
    )
    manifest_sha = _digest(manifest_path.read_bytes())
    attempts = tmp_path / "attempts"
    _write_attempt(attempts, first, manifest_sha, "repeated_action")
    _write_attempt(attempts, second, manifest_sha, None)
    parent_rows = [
        {"arm": "short_only", "case_id": case.case_id, "video_id": case.video_id, "pair_id": case.pair_id, "truth_route": "routine_live", "predicted_route": "routine_live", "valid_reply": True}
        for case in (first, second)
    ]
    parent = {
        "schema": "vlm-multiscale-detail-score-v1",
        "by_arm": {"short_only": {"route_accuracy": 1.0, "routine_live_recall": 1.0, "routine_live_precision": 1.0, "close_check_recall": None, "close_check_precision": None}},
        "by_arm_material_target": {"short_only": {"route_accuracy": 1.0, "routine_live_recall": 1.0, "routine_live_precision": 1.0}},
        "rows": parent_rows,
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(json.dumps({"schema": replay.TRUTH_SCHEMA, "cases": [
        {"pair_id": first.pair_id, "video_id": first.video_id, "truth_intervals": [{"source_start_frame": first.target_start_frame, "source_end_frame": first.target_end_frame, "truth": "live"}]},
        {"pair_id": second.pair_id, "video_id": second.video_id, "truth_intervals": [{"source_start_frame": second.target_start_frame, "source_end_frame": second.target_end_frame, "truth": "live"}]},
    ]}), encoding="utf-8")

    score = replay.score_replay_pair_trials(manifest_path, attempts, "internvideo3", parent_path, truth_path)
    assert score["relation_counts"] == {"invalid": 1, "repeated_action": 1}
    assert score["changed_case_ids"]["all_relations"] == ["live", "short"]
    assert score["changed_case_ids"]["repeat_only"] == ["live"]
    assert score["by_rule"]["all_relations"]["routine_live_recall"] == 0.0
    assert score["by_rule_material_target"]["all_relations"]["cases"] == 1
    assert score["parent_metrics"]["strict"]["cases"] == 2
