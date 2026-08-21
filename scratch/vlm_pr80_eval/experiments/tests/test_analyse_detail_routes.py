from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.analyse_detail_routes import (
    ROUTE_ANALYSIS_SCHEMA,
    TRACK_VISIBLE_FRACTION_THRESHOLD,
    analyse_detail_routes,
    main,
)
from experiments.detail_schema import DETAIL_MANIFEST_SCHEMA, DetailArm
from experiments.score_detail_trials import DETAIL_SCORE_SCHEMA


def _score_payload(backend: str, routes: dict[str, str]) -> dict:
    truth = {
        "live-case": ("routine_live", 4, {"live": 4}),
        "mixed-case": ("close_check", 4, {"live": 2, "replay": 2}),
    }
    rows = []
    for case_id, predicted_route in routes.items():
        truth_route, target_frames, frame_counts = truth[case_id]
        rows.append(
            {
                "arm": "short_only",
                "case_id": case_id,
                "target_frames": target_frames,
                "truth_route": truth_route,
                "truth_content_frames": frame_counts,
                "predicted_route": predicted_route,
            }
        )
    return {"schema": DETAIL_SCORE_SCHEMA, "backend": backend, "rows": rows}


def _write_score(tmp_path: Path, backend: str, routes: dict[str, str]) -> Path:
    path = tmp_path / f"{backend}.json"
    path.write_text(json.dumps(_score_payload(backend, routes)), encoding="utf-8")
    return path


def _broad_case(case_id: str, broad_facts: list[dict] | None) -> dict:
    return {
        "case_id": case_id,
        "pair_id": f"pair-{case_id}",
        "context_case_id": f"context-{case_id}",
        "video_id": "sset_01",
        "clip_path": "missing.mp4",
        "clip_sha256": hashlib.sha256(case_id.encode()).hexdigest(),
        "source_start_frame": 0,
        "source_end_frame": 4,
        "source_frames": [0, 1, 2, 3],
        "target_start_frame": 1,
        "target_end_frame": 3,
        "boundary_frame": None,
        "source_fps": 25.0,
        "sample_fps": 25.0,
        "target_segment_ids": ["S1"],
        "deterministic_facts": {"source": "automatic"},
        "broad_facts": broad_facts,
    }


def _write_broad_manifest(tmp_path: Path, *, null_case: bool = True) -> Path:
    facts = None if null_case else [
        {
            "segment_id": "S1",
            "content": "live",
            "repeat_of": None,
            "needs_close_check": True,
        }
    ]
    payload = {
        "schema": "vlm-multiscale-detail-manifest-v1",
        "arm": "broad_facts",
        "expected_frames": 4,
        "width": 32,
        "height": 24,
        "cases": [
            _broad_case("live-case", facts),
            _broad_case("mixed-case", facts),
        ],
    }
    path = tmp_path / "broad_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_priors_manifest(
    tmp_path: Path,
    fractions: dict[str, object],
    *,
    arm: str = "deterministic",
) -> Path:
    cases = []
    for case_id, fraction in fractions.items():
        case = _broad_case(case_id, None)
        case["deterministic_facts"] = {
            "pipeline_priors": {"track_visible_fraction": fraction}
        }
        cases.append(case)
    payload = {
        "schema": DETAIL_MANIFEST_SCHEMA,
        "arm": arm,
        "expected_frames": 4,
        "width": 32,
        "height": 24,
        "cases": cases,
    }
    path = tmp_path / "priors_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pairs_backends_and_applies_all_models_live_union(tmp_path: Path) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "routine_live"},
    )
    intern = _write_score(
        tmp_path,
        "internvideo3",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )

    report = analyse_detail_routes(
        {"qwen3-vl": qwen, "internvideo3": intern}
    )

    assert report["schema"] == ROUTE_ANALYSIS_SCHEMA
    assert report["case_ids"] == ["live-case", "mixed-case"]
    assert report["rules"]["all_models_live"]["routes"] == {
        "live-case": "routine_live",
        "mixed-case": "close_check",
    }
    assert report["rules"]["short_only"]["qwen3-vl"]["metrics"][
        "routine_live_precision"
    ] == 0.5
    assert report["rules"]["all_models_live"]["metrics"][
        "routine_live_precision"
    ] == 1.0
    changed = report["rules"]["all_models_live"]["changed_case_ids_vs_parent"]
    assert changed == {"internvideo3": [], "qwen3-vl": ["mixed-case"]}


def test_broad_null_is_conservative_and_ignores_needs_close_check(tmp_path: Path) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "routine_live"},
    )
    intern = _write_score(
        tmp_path,
        "internvideo3",
        {"live-case": "routine_live", "mixed-case": "routine_live"},
    )

    report = analyse_detail_routes(
        {"qwen3-vl": qwen, "internvideo3": intern},
        _write_broad_manifest(tmp_path, null_case=True),
    )

    broad_qwen = report["rules"]["short_only_plus_broad_content"]["qwen3-vl"]
    assert broad_qwen["routes"] == {
        "live-case": "close_check",
        "mixed-case": "close_check",
    }
    assert broad_qwen["metrics"]["target_frame_coverage_kept"] == 0.0

    non_null_report = analyse_detail_routes(
        {"qwen3-vl": qwen},
        _write_broad_manifest(tmp_path, null_case=False),
    )
    assert non_null_report["rules"]["short_only_plus_broad_content"]["qwen3-vl"][
        "routes"
    ] == {"live-case": "routine_live", "mixed-case": "routine_live"}


def test_frozen_track_visibility_rule_and_backend_intersections(
    tmp_path: Path,
) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "routine_live"},
    )
    intern = _write_score(
        tmp_path,
        "internvideo3",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    priors = _write_priors_manifest(
        tmp_path,
        {"live-case": 0.8, "mixed-case": 0.79},
    )

    report = analyse_detail_routes(
        {"qwen3-vl": qwen, "internvideo3": intern},
        priors_manifest=priors,
    )

    assert TRACK_VISIBLE_FRACTION_THRESHOLD == 0.8
    assert report["priors_manifest"] == {
        "path": str(priors),
        "sha256": hashlib.sha256(priors.read_bytes()).hexdigest(),
    }
    prior_rule = report["rules"]["track_visible_fraction_at_0_8"]
    assert prior_rule["name"] == "track_visible_fraction_at_0_8"
    assert prior_rule["routes"] == {
        "live-case": "routine_live",
        "mixed-case": "close_check",
    }
    assert prior_rule["metrics"]["routine_live_precision"] == 1.0
    assert prior_rule["metrics"]["target_frame_coverage_kept"] == 0.5

    intersections = report["rules"]["short_only_plus_track_visible_fraction"]
    assert intersections["qwen3-vl"]["routes"] == prior_rule["routes"]
    assert intersections["internvideo3"]["routes"] == {
        "live-case": "routine_live",
        "mixed-case": "close_check",
    }
    assert intersections["qwen3-vl"]["metrics"]["routine_live_precision"] == 1.0
    assert report["changed_case_ids"][
        "short_only_plus_track_visible_fraction_vs_short_only"
    ] == {"internvideo3": [], "qwen3-vl": ["mixed-case"]}


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (lambda case: case.pop("pipeline_priors"), "one pipeline_priors record"),
        (lambda case: case["pipeline_priors"].update(track_visible_fraction=True), "numeric"),
        (lambda case: case["pipeline_priors"].update(track_visible_fraction=1.01), "in \\[0, 1\\]"),
        (lambda case: case["pipeline_priors"].update(truth_route="routine_live"), "forbidden truth key"),
    ],
)
def test_rejects_malformed_priors_manifest(tmp_path: Path, mutator, error: str) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    priors = _write_priors_manifest(
        tmp_path,
        {"live-case": 0.9, "mixed-case": 0.1},
    )
    payload = json.loads(priors.read_text(encoding="utf-8"))
    mutator(payload["cases"][0]["deterministic_facts"])
    priors.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=error):
        analyse_detail_routes({"qwen3-vl": qwen}, priors_manifest=priors)


def test_rejects_priors_case_id_mismatch_and_wrong_arm(tmp_path: Path) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    priors = _write_priors_manifest(tmp_path, {"live-case": 0.9})
    with pytest.raises(ValueError, match="priors manifest case IDs differ"):
        analyse_detail_routes({"qwen3-vl": qwen}, priors_manifest=priors)

    broad_priors = _write_priors_manifest(
        tmp_path,
        {"live-case": 0.9, "mixed-case": 0.1},
        arm=DetailArm.BROAD_FACTS.value,
    )
    with pytest.raises(ValueError, match="priors manifest arm must be deterministic"):
        analyse_detail_routes({"qwen3-vl": qwen}, priors_manifest=broad_priors)


@pytest.mark.parametrize(
    "change",
    [
        "case_id",
        "truth_route",
        "target_frames",
        "truth_content_frames",
    ],
)
def test_rejects_mismatched_truth_side_pairing(tmp_path: Path, change: str) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    intern_payload = _score_payload(
        "internvideo3",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    changed_row = intern_payload["rows"][0]
    if change == "case_id":
        changed_row["case_id"] = "different-case"
    elif change == "truth_route":
        changed_row["truth_route"] = "close_check"
    elif change == "target_frames":
        changed_row["target_frames"] = 5
        changed_row["truth_content_frames"] = {"live": 5}
    else:
        changed_row["truth_content_frames"] = {"live": 3, "replay": 1}
    intern = tmp_path / "internvideo3.json"
    intern.write_text(json.dumps(intern_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="case IDs differ|truth-side fields differ"):
        analyse_detail_routes({"qwen3-vl": qwen, "internvideo3": intern})


def test_cli_writes_versioned_output_exclusively(tmp_path: Path) -> None:
    qwen = _write_score(
        tmp_path,
        "qwen3-vl",
        {"live-case": "routine_live", "mixed-case": "close_check"},
    )
    output = tmp_path / "report.json"
    argv = ["--score", f"qwen3-vl={qwen}", "--out", str(output)]

    assert main(argv) == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema"] == ROUTE_ANALYSIS_SCHEMA
    with pytest.raises(FileExistsError):
        main(argv)
