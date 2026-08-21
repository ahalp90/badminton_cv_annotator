from __future__ import annotations

import json
from pathlib import Path

import pytest
from experiments.multiscale_schema import (
    BroadContent,
    TargetRoute,
    load_manifest,
    parse_broad_reply,
    target_route,
    validate_context_pairs,
)


def _case(context_seconds: int) -> dict:
    source_start = 0 if context_seconds == 120 else 750
    source_end = 3_000
    source_frames = [
        source_start + index * (source_end - source_start) // 96
        for index in range(96)
    ]
    return {
        "case_id": f"sset_01-span-7--{context_seconds}",
        "pair_id": "sset_01-span-7",
        "video_id": "sset_01",
        "context_seconds": context_seconds,
        "clip_path": f"clips/context-{context_seconds}.mp4",
        "source_start_frame": source_start,
        "source_end_frame": source_end,
        "target_start_frame": 1_500,
        "target_end_frame": 1_750,
        "sample_fps": 8.0,
        "source_frames": source_frames,
        "segments": [
            {
                "segment_id": "S0040",
                "source_start_frame": source_start,
                "source_end_frame": 1_500,
            },
            {
                "segment_id": "S0041",
                "source_start_frame": 1_500,
                "source_end_frame": 1_800,
            },
            {
                "segment_id": "S0042",
                "source_start_frame": 1_800,
                "source_end_frame": source_end,
            },
        ],
        "pipeline_priors": {"span_id": 7, "court_fraction": 0.92},
    }


def _write_manifest(path: Path, cases: list[dict]) -> None:
    payload = {
        "schema": "vlm-multiscale-manifest-v1",
        "expected_frames": 96,
        "width": 512,
        "height": 288,
        "cases": cases,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_reply() -> str:
    return json.dumps(
        {
            "segments": [
                {
                    "segment_id": "S0040",
                    "content": "replay",
                    "repeat_of": None,
                    "needs_close_check": False,
                },
                {
                    "segment_id": "S0041",
                    "content": "live",
                    "repeat_of": None,
                    "needs_close_check": False,
                },
                {
                    "segment_id": "S0042",
                    "content": "cutaway",
                    "repeat_of": None,
                    "needs_close_check": False,
                },
            ]
        }
    )


def test_load_manifest_accepts_fair_truth_blind_pairs(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90), _case(120)])

    cases = load_manifest(path, require_clips=False)

    validate_context_pairs(cases)
    assert {case.context_seconds for case in cases} == {90, 120}
    assert {case.target_segment_ids for case in cases} == {("S0041",)}
    assert cases[0].clip_path.parent == path.parent / "clips"


def test_load_manifest_rejects_nested_truth(tmp_path: Path) -> None:
    cases = [_case(90)]
    cases[0]["pipeline_priors"]["truth"] = "replay"
    path = tmp_path / "manifest.json"
    _write_manifest(path, cases)

    with pytest.raises(ValueError, match="forbidden truth key"):
        load_manifest(path, require_clips=False)


def test_load_manifest_rejects_local_frames_in_a_source_global_map(tmp_path: Path) -> None:
    cases = [_case(90)]
    cases[0]["source_frames"][0] = 0
    path = tmp_path / "manifest.json"
    _write_manifest(path, cases)

    with pytest.raises(ValueError, match="leaves its source window"):
        load_manifest(path, require_clips=False)


def test_validate_context_pairs_rejects_a_different_target(tmp_path: Path) -> None:
    cases = [_case(90), _case(120)]
    cases[1]["target_start_frame"] = 1_200
    path = tmp_path / "manifest.json"
    _write_manifest(path, cases)

    loaded = load_manifest(path, require_clips=False)
    with pytest.raises(ValueError, match="target identity"):
        validate_context_pairs(loaded)


def test_parse_broad_reply_requires_every_known_segment(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]
    payload = json.loads(_valid_reply())
    payload["segments"].pop()

    with pytest.raises(ValueError, match="omits segment IDs"):
        parse_broad_reply(case, json.dumps(payload))


def test_parse_broad_reply_rejects_forward_repeat_links(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]
    payload = json.loads(_valid_reply())
    payload["segments"][0]["repeat_of"] = "S0041"

    with pytest.raises(ValueError, match="earlier segment"):
        parse_broad_reply(case, json.dumps(payload))


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda payload: payload["segments"][0].update(segment_id="UNKNOWN"), "unknown segment"),
        (lambda payload: payload["segments"][1].update(segment_id="S0040"), "duplicate"),
        (lambda payload: payload["segments"][1].update(content="match"), "not a valid BroadContent"),
    ],
)
def test_parse_broad_reply_rejects_bad_ids_or_content(
    tmp_path: Path,
    mutate,
    error: str,
) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]
    payload = json.loads(_valid_reply())
    mutate(payload)

    with pytest.raises(ValueError, match=error):
        parse_broad_reply(case, json.dumps(payload))


def test_parse_broad_reply_rejects_markdown_wrapping(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]

    with pytest.raises(ValueError, match="not valid JSON"):
        parse_broad_reply(case, f"```json\n{_valid_reply()}\n```")


def test_target_route_ignores_unrelated_replay_and_cutaway(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]
    replies = parse_broad_reply(case, _valid_reply())

    assert target_route(case, replies) is TargetRoute.ROUTINE_LIVE
    assert replies[0].content is BroadContent.REPLAY


def test_target_route_sends_unclear_or_invalid_answers_to_close_check(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write_manifest(path, [_case(90)])
    case = load_manifest(path, require_clips=False)[0]
    payload = json.loads(_valid_reply())
    payload["segments"][1]["content"] = "unclear"

    replies = parse_broad_reply(case, json.dumps(payload))

    assert target_route(case, replies) is TargetRoute.CLOSE_CHECK
    assert target_route(case, None) is TargetRoute.CLOSE_CHECK


def test_target_route_checks_a_mixed_target(tmp_path: Path) -> None:
    case_payload = _case(90)
    case_payload["target_start_frame"] = 1_400
    case_payload["target_end_frame"] = 1_900
    path = tmp_path / "manifest.json"
    _write_manifest(path, [case_payload])
    case = load_manifest(path, require_clips=False)[0]

    replies = parse_broad_reply(case, _valid_reply())

    assert case.target_segment_ids == ("S0040", "S0041", "S0042")
    assert target_route(case, replies) is TargetRoute.CLOSE_CHECK
