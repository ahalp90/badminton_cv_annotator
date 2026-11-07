from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.annotator.shuttleset_benchmarks.shuttleset_benchmark as benchmark
from annotator.rally_results.landing_and_winner import WinnerSource
from annotator.types import CourtHalf
from dataset_builder.vision import save_json_gz
from experiments.annotator.legacy_evaluation.gt_scoring import (
    ColumnAgg,
    RallyRow,
    _hit_height_gt_map,
    aggregate_rally_scores,
)
from experiments.annotator.shuttleset_benchmarks.shuttleset_benchmark import (
    CONTACT_TOLERANCES_BASE30,
    _aggregate_contact_curves,
    _aggregate_feature_outputs,
    _contact_ground_truth,
    _coordinate_rows_summary,
    _landing_coordinate_output,
    _merged_outcome_rows,
    _rally_output_rows,
    projection_to_annotator_result,
)


def _record(rally_id: int = 0) -> dict[str, object]:
    return {
        "key": {"rally_id": rally_id},
        "rally": {"start_frame": 10, "end_frame": 30},
        "contacts": {
            "raw_candidates": [
                {
                    "contact_frame": 15,
                    "wrist_near": True,
                    "temporal_proximity_suppressed": False,
                },
                {
                    "contact_frame": 24,
                    "wrist_near": True,
                    "temporal_proximity_suppressed": True,
                },
            ],
            "accepted": [
                {"stroke_idx": 0, "contact_frame": 15, "hit_height_code": 1, "player_half": "Top"},
            ],
            "stroke_count": 1,
            "hit_height_failures": [],
        },
        "outcomes": {
            "striker_half": "Top",
            "server_prediction": "Top",
            "next_server": None,
            "verdict": {
                "value": "won",
                "source": "next_server",
                "landing_margin_m": 0.2,
                "within_line_margin": False,
                "within_net_margin": False,
            },
            "landing": {
                "frame": 28,
                "normalized_court_position": [0.4, 0.8],
                "court_half": "Bot",
                "at_image_border": False,
                "net_ender": False,
            },
            "geometric_verdict": {
                "value": "won",
                "winner": "Top",
                "agreement": True,
                "window_closed_by_mask": False,
            },
        },
    }


def test_projection_to_annotator_result_restores_scored_fields() -> None:
    result = projection_to_annotator_result([_record()])

    rally, = result.rallies
    assert (rally.sequence.start_frame, rally.sequence.end_frame) == (10, 30)
    assert [row.contact_frame for row in result.heuristic_candidates] == [15, 24]
    assert [event.frame for event in rally.sequence.events] == [15]
    assert np.isnan(rally.sequence.events[0].probability)
    assert result.hit_heights == {15: 1}
    assert rally.final_half is CourtHalf.TOP
    assert rally.first_half is CourtHalf.TOP
    assert result.outcomes[0].winner is CourtHalf.TOP
    assert result.outcomes[0].winner_source is WinnerSource.NEXT_SERVER
    assert result.outcomes[0].landing.court_xy == (0.4, 0.8)
    assert result.outcomes[0].conservative_landing_winner is CourtHalf.TOP


def test_contact_curve_aggregation_uses_integer_populations() -> None:
    def curve(matched: int, gt: int, candidates: int, raw_candidates: int):
        return {
            str(base): {
                "matched": matched,
                "gt": gt,
                "candidates": candidates,
                "raw_matched": matched,
                "raw_candidates": raw_candidates,
            }
            for base in CONTACT_TOLERANCES_BASE30
        }

    curves = {
        "a": curve(2, 4, 3, 5),
        "b": curve(3, 6, 4, 7),
    }

    aggregate = _aggregate_contact_curves(curves)

    assert aggregate["5"]["matched"] == 5
    assert aggregate["5"]["gt"] == 10
    assert aggregate["5"]["recall"] == 0.5
    assert aggregate["5"]["raw_candidates"] == 12


def test_hit_height_ground_truth_is_keyed_by_frame_after_mismatch() -> None:
    tables = {
        "set1": pd.DataFrame(
            {
                "rally": [1, 1, 1],
                "frame_num": [10, 11, 20],
                "hit_height": [1, 9, 2],
            }
        )
    }

    heights = _hit_height_gt_map(tables)

    assert heights[("set1", 1, 10)] == 1
    assert heights[("set1", 1, 20)] == 2


def test_benchmark_rejects_modified_pinned_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records_path = tmp_path / "rally_records.json.gz"
    record = _record()
    payload = {"records": [record]}
    save_json_gz(records_path, payload)
    monkeypatch.setattr(
        benchmark, "EXPECTED_RALLY_RECORDS_SHA256", benchmark._sha256(records_path)
    )
    rally = record["rally"]
    assert isinstance(rally, dict)
    rally["start_frame"] = 11
    save_json_gz(records_path, payload)

    with pytest.raises(ValueError, match="rally records SHA-256"):
        benchmark.benchmark_records(
            records_path=records_path,
            configuration_path=tmp_path / "missing.toml",
            ground_truth_root=tmp_path,
        )


def test_coordinate_summary_reports_unusable_annotation_cases() -> None:
    rows = [
        {
            "ground_truth_available": True,
            "prediction_available": True,
            "error": 0.25,
        },
        {
            "ground_truth_available": True,
            "prediction_available": False,
            "error": None,
        },
        {
            "ground_truth_available": False,
            "prediction_available": True,
            "error": None,
        },
        {
            "mapping_eligible": False,
            "ground_truth_available": True,
            "prediction_available": True,
            "error": 0.1,
        },
    ]

    assert _coordinate_rows_summary(rows) == {
        "population": 4,
        "mapping_eligible": 3,
        "excluded_merged_mapping": 1,
        "ground_truth_available": 2,
        "prediction_available": 1,
        "eligible": 1,
        "excluded_ground_truth": 1,
        "excluded_prediction": 1,
        "mean_error": 0.25,
        "median_error": 0.25,
        "p90_error": 0.25,
    }
    output = _landing_coordinate_output(rows, include_rows=False)
    assert output["units"] == "normalized doubles-court Euclidean distance"
    assert output["matching"] == (
        "GT rally paired through one unique, unmerged covered predicted span; "
        "landing frames are not matched"
    )


def test_feature_aggregation_preserves_exact_coordinate_populations() -> None:
    def video(fps: float, error: float | None) -> dict[str, object]:
        coordinate_summary = {
            name: {
                "population": 2,
                "eligible": int(error is not None),
                "excluded_prediction": int(error is None),
                "excluded_ground_truth": 0,
            }
            for name in ("shuttle", "striker", "opponent")
        }
        coordinate_row = {
            f"{name}_error": error for name in ("shuttle", "striker", "opponent")
        }
        return {
            "fps": fps,
            "population": {"rallies": 3},
            "contact_coordinates": {
                "summary": coordinate_summary,
                "rows": [coordinate_row],
            },
            "court_corners": {
                "rows": [
                    {
                        "corner_errors_px": [1.0, 2.0, 3.0, 4.0],
                    }
                ]
            },
        }

    aggregate = _aggregate_feature_outputs(
        {"sset_01": video(25.0, 0.5), "sset_02": video(30.0, None)}
    )

    assert aggregate["fps_groups"] == {
        "25": {"videos": 1, "rallies": 3},
        "30": {"videos": 1, "rallies": 3},
    }
    contact = aggregate["contact_coordinates"]
    assert isinstance(contact, dict)
    shuttle = contact["summary"]["shuttle"]
    assert shuttle["population"] == 4
    assert shuttle["eligible"] == 1
    assert shuttle["excluded_prediction"] == 1
    assert shuttle["mean_error"] == 0.5
    corners = aggregate["court_corners"]
    assert isinstance(corners, dict)
    assert corners["summary"]["corners"] == 8
    assert corners["summary"]["median_error"] == 2.5


def test_contact_ground_truth_joins_complete_contact_key() -> None:
    tables = {
        "set1": pd.DataFrame(
            {
                "frame_num": [10, 10, 20],
                "rally": [1, 1, 1],
                "ball_round": [1, 2, 3],
                "player": ["A", "B", "A"],
                "flaw": [None, 1, None],
            }
        )
    }
    master = pd.DataFrame(
        {
            "vid": [1, 1, 2],
            "set_id": ["set1", "set1", "set1"],
            "rally": [1, 1, 1],
            "ball_round": [1, 3, 3],
            "frame_num": [10, 20, 20],
            "player_side": ["Top", "Bottom", "Top"],
        }
    )

    contacts, population = _contact_ground_truth(tables, master, 1)

    assert contacts["ball_round"].tolist() == [1, 3]
    assert contacts["player_side"].tolist() == ["Top", "Bottom"]
    assert population == {
        "source_rows": 3,
        "master_aligned_rows": 2,
        "unmatched_rows": 1,
        "flaw_marked_rows": 1,
        "flaw_marked_aligned_rows": 0,
        "flaw_marked_unmatched_rows": 1,
        "duplicate_frame_groups": 1,
        "duplicate_frame_extra_rows": 1,
    }


def _rally_row(gt_index: int, mapped_span: int | None, *, covered: bool) -> RallyRow:
    return RallyRow(
        gt_index=gt_index,
        set_id="set1",
        rally=gt_index + 1,
        n_gt_strokes=1,
        classification="covered" if covered else "missed",
        mapped_span=mapped_span,
        ball_round_gt=1,
        ball_round_pred=1 if covered else None,
        ball_round_correct=covered,
        timing_matched_n=int(covered),
        timing_mean_abs_err=0.0 if covered else None,
        player_gt="Top",
        player_pred="Top" if covered else None,
        player_correct=covered,
        server_gt="Top",
        server_pred="Top" if covered else None,
        server_correct=covered,
        hit_height_eligible_n=0,
        hit_height_correct_n=0,
        getpoint_eligible=True,
        getpoint_gt="Top",
        getpoint_pred="Top" if covered else None,
        getpoint_correct=covered,
        landing_eligible=True,
        landing_gt="Bot",
        landing_pred="Bot" if covered else None,
        landing_correct=covered,
    )


def test_rally_counts_keep_eligibility_and_covered_denominators() -> None:
    covered = _rally_row(0, 0, covered=True)._replace(
        n_gt_strokes=3, timing_matched_n=2, hit_height_eligible_n=2, hit_height_correct_n=1,
    )
    missed = _rally_row(1, None, covered=False)._replace(
        n_gt_strokes=4, hit_height_eligible_n=3,
        landing_eligible=False, landing_correct=None,
        getpoint_eligible=False, getpoint_correct=None,
    )

    counts = aggregate_rally_scores([covered, missed])

    assert counts["ball_round"] == ColumnAgg(1, 2, 1, 1)
    assert counts["timing"] == ColumnAgg(2, 7, 2, 3)
    assert counts["player"] == counts["server"] == ColumnAgg(1, 2, 1, 1)
    assert counts["hit_height"] == ColumnAgg(1, 5, 1, 2)
    assert counts["landing"] == counts["getpoint"] == ColumnAgg(1, 1, 1, 1)


def test_merged_span_outcomes_are_marked_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scoring = SimpleNamespace(
        rows=[
            _rally_row(0, 0, covered=True),
            _rally_row(1, 0, covered=True),
            _rally_row(2, None, covered=False),
        ]
    )

    assert _merged_outcome_rows(scoring) == {0, 1}
    rows = _rally_output_rows(scoring)
    assert [row["outcome_mapping_eligible"] for row in rows] == [False, False, True]
    assert rows[0]["ball_round_pred"] is None
    assert rows[1]["ball_round_correct"] is None
    assert rows[0]["player_pred"] is None
    assert rows[1]["landing_correct"] is None
    assert rows[2]["player_correct"] is False
    monkeypatch.setattr(benchmark, "flatten_metrics", lambda _scoring: {})
    metrics = benchmark._benchmark_metrics(scoring)
    assert metrics["outcome_mapping_excluded_merged_rallies"] == 2
    assert metrics["ball_round_primary_total"] == 1
    assert metrics["ball_round_primary_correct"] == 0
    assert metrics["ball_round_covered_total"] == 0
    assert metrics["player_primary_total"] == 1
    assert metrics["player_primary_correct"] == 0
    assert metrics["player_covered_total"] == 0
    assert metrics["landing_primary_total"] == 1
