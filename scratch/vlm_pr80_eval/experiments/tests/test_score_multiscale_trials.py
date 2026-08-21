from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.score_multiscale_trials import (
    score_broad_attempts,
    score_prediction_side,
)

from annotator.calibration.gt_scoring import ColumnAgg, RallyRow, VideoScoring


def _row(*, mapped_span: int, rally: int, complete: bool = True) -> RallyRow:
    return RallyRow(
        gt_index=rally,
        set_id="set-1",
        rally=rally,
        n_gt_strokes=3,
        classification="covered",
        mapped_span=mapped_span,
        ball_round_gt=3,
        ball_round_pred=3 if complete else 2,
        ball_round_correct=complete,
        timing_matched_n=3 if complete else 2,
        timing_mean_abs_err=1.0,
        player_gt="top",
        player_pred="top" if complete else "bot",
        player_correct=complete,
        server_gt="top",
        server_pred="top" if complete else "bot",
        server_correct=complete,
        hit_height_eligible_n=0,
        hit_height_correct_n=0,
        getpoint_eligible=True,
        getpoint_gt="top",
        getpoint_pred="top" if complete else "bot",
        getpoint_correct=complete,
        landing_eligible=False,
        landing_gt=None,
        landing_pred=None,
        landing_correct=None,
    )


def _scoring(rows: list[RallyRow]) -> VideoScoring:
    empty_agg = ColumnAgg(0, 0, 0, 0)
    return VideoScoring(
        name="sset_01",
        rows=rows,
        boundary_metrics={},
        ball_round=empty_agg,
        timing_primary=(0, 0),
        timing_covered=(0, 0),
        player=empty_agg,
        server=empty_agg,
        hit_height=empty_agg,
        landing=empty_agg,
        getpoint=empty_agg,
        contact_matches=0,
        contact_filtered_total=0,
        contact_gt_total=0,
        n_raw_contacts=0,
        n_filtered_contacts=0,
        hit_height_failures=[],
        ball_round_diffs=[],
        timing_errs=[],
        geometric_verdict_rows={},
    )


def test_prediction_precision_counts_every_retained_span() -> None:
    scoring = _scoring(
        [
            _row(mapped_span=0, rally=1),
            _row(mapped_span=1, rally=2, complete=False),
            _row(mapped_span=2, rally=3),
            _row(mapped_span=2, rally=4),
        ]
    )

    result = score_prediction_side(
        scoring,
        predicted_span_count=4,
        retained_span_ids={0, 1, 2, 3},
        baseline_correct_span_ids={0, 1},
    )

    assert result["correct_complete_records"] == 1
    assert result["complete_record_precision"] == 0.25
    assert result["baseline_correct_still_usable"] == 1
    assert result["baseline_correct_coverage"] == 0.5
    assert result["passes_half_baseline_gate"] is True
    assert [record["status"] for record in result["records"]] == [
        "correct",
        "incorrect_record",
        "merged",
        "spurious_or_partial",
    ]


def test_prediction_precision_fails_coverage_gate_after_dropping_good_records() -> None:
    scoring = _scoring([_row(mapped_span=0, rally=1), _row(mapped_span=1, rally=2)])

    result = score_prediction_side(
        scoring,
        predicted_span_count=3,
        retained_span_ids={2},
        baseline_correct_span_ids={0, 1},
    )

    assert result["complete_record_precision"] == 0.0
    assert result["passes_half_baseline_gate"] is False


def test_prediction_precision_rejects_unknown_span_ids() -> None:
    with pytest.raises(ValueError, match="out of range"):
        score_prediction_side(
            _scoring([]),
            predicted_span_count=2,
            retained_span_ids={2},
        )


def test_prediction_precision_rejects_duplicate_span_ids() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        score_prediction_side(
            _scoring([]),
            predicted_span_count=2,
            retained_span_ids=[1, 1],
        )


def test_prediction_precision_requires_a_covered_mapping() -> None:
    row = _row(mapped_span=0, rally=1)._replace(classification="split")

    result = score_prediction_side(
        _scoring([row]),
        predicted_span_count=1,
        retained_span_ids=[0],
    )

    assert result["correct_complete_records"] == 0
    assert result["records"][0]["status"] == "incorrect_record"


def _write_broad_score_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    cases = []
    for seconds in (90, 120):
        clip = tmp_path / f"clip-{seconds}.mp4"
        clip.write_bytes(b"clip")
        cases.append(
            {
                "case_id": f"pair--{seconds}",
                "pair_id": "pair",
                "video_id": "sset_01",
                "context_seconds": seconds,
                "clip_path": str(clip),
                "source_start_frame": 0,
                "source_end_frame": 300,
                "target_start_frame": 100,
                "target_end_frame": 200,
                "sample_fps": 2.0,
                "source_frames": [0, 99, 100, 199, 200, 299],
                "segments": [
                    {"segment_id": "S0001", "source_start_frame": 0, "source_end_frame": 100},
                    {"segment_id": "S0002", "source_start_frame": 100, "source_end_frame": 200},
                    {"segment_id": "S0003", "source_start_frame": 200, "source_end_frame": 300},
                ],
                "pipeline_priors": {"definitive_mask_fraction": 0.0},
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "vlm-multiscale-manifest-v1",
                "expected_frames": 6,
                "width": 32,
                "height": 24,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "schema": "vlm-multiscale-truth-v1",
                "cases": [
                    {
                        "pair_id": "pair",
                        "truth_intervals": [
                            {
                                "source_start_frame": 100,
                                "source_end_frame": 200,
                                "truth": "live-non-standard",
                            }
                        ],
                    }
                ],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )
    attempts = tmp_path / "attempts/internvideo3"
    attempts.mkdir(parents=True)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    valid_segments = [
        {
            "segment_id": "S0001",
            "content": "replay",
            "repeat_of": None,
            "needs_close_check": False,
        },
        {
            "segment_id": "S0002",
            "content": "live",
            "repeat_of": None,
            "needs_close_check": False,
        },
        {
            "segment_id": "S0003",
            "content": "cutaway",
            "repeat_of": None,
            "needs_close_check": False,
        },
    ]
    for seconds in (90, 120):
        valid = seconds == 90
        attempt = {
            "schema": "vlm-multiscale-attempt-v1",
            "backend": "internvideo3",
            "manifest_sha256": manifest_hash,
            "case": {"case_id": f"pair--{seconds}"},
            "parsed_response": valid_segments if valid else None,
            "parser_error": None if valid else "invalid reply",
            "generation_error": None,
        }
        (attempts / f"pair--{seconds}.json").write_text(
            json.dumps(attempt),
            encoding="utf-8",
        )
    return manifest, truth, attempts.parent


def test_broad_score_is_target_only_and_keeps_invalid_replies_in_denominator(tmp_path: Path) -> None:
    manifest, truth, attempts = _write_broad_score_inputs(tmp_path)

    score = score_broad_attempts(manifest, truth, attempts, "internvideo3")

    assert score["selected_duration_seconds"] == 90
    assert score["by_duration"]["90"]["mean_case_exact_scene_accuracy"] == 1.0
    assert score["by_duration"]["120"]["mean_case_exact_scene_accuracy"] == 0.0
    assert score["by_duration"]["120"]["valid_replies"] == 0
    assert score["broad_gate_passed"] is False
    assert score["safe_bypass_gate_passed"] is False
    assert score["by_duration"]["90"]["routine_live_recall"] == 1.0
    assert score["by_duration"]["90"]["close_check_recall"] is None
