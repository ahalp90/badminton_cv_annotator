from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scratch.contact_det_full_ds_fit.scripts.summarise_shuttleset22_sections import (
    build_summary,
    load_sections,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_json_gz(path: Path, value: object) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as file:
        json.dump(value, file)


def test_summary_counts_clean_partial_merged_and_missed_rallies(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json.gz"
    labels_path = tmp_path / "labels.json.gz"
    result_path = tmp_path / "result.json"
    _write_json_gz(
        predictions_path,
        {
            "videos": [
                {
                    "fixture": "8",
                    "spans": [
                        {"span_id": 0, "start_frame": 5, "end_frame": 25},
                        {"span_id": 1, "start_frame": 30, "end_frame": 45},
                        {"span_id": 2, "start_frame": 50, "end_frame": 80},
                        {"span_id": 3, "start_frame": 90, "end_frame": 100},
                    ],
                }
            ]
        },
    )
    _write_json_gz(
        labels_path,
        {
            "videos": [
                {
                    "fixture": "8",
                    "rallies": [
                        {
                            "set_id": "set1",
                            "rally": 1,
                            "contacts": [{"frame": 10}, {"frame": 20}],
                        },
                        {
                            "set_id": "set1",
                            "rally": 2,
                            "contacts": [{"frame": 35}, {"frame": 47}],
                        },
                        {
                            "set_id": "set1",
                            "rally": 3,
                            "contacts": [{"frame": 55}, {"frame": 65}],
                        },
                        {
                            "set_id": "set1",
                            "rally": 4,
                            "contacts": [{"frame": 70}],
                        },
                        {
                            "set_id": "set1",
                            "rally": 5,
                            "contacts": [{"frame": 110}],
                        },
                    ],
                }
            ]
        },
    )
    _write_json(
        result_path,
        {
            "whole_rallies": {
                "by_tolerance": {
                    "5": {
                        "sections": [
                            {
                                "fixture": "8",
                                "span_id": 0,
                                "outcome": "fully_correct",
                            },
                            {
                                "fixture": "8",
                                "span_id": 1,
                                "outcome": "fully_correct",
                            },
                        ]
                    },
                    "10": {"sections": []},
                }
            }
        },
    )

    summary = build_summary(predictions_path, labels_path, result_path)

    assert summary["predicted_sections"] == 4
    assert summary["labelled_rallies"] == 5
    assert summary["correct_rally_sections"] == 1
    assert summary["precision"] == 0.25
    assert summary["recall"] == 0.2
    assert summary["section_counts"] == {
        "one_complete_rally": 1,
        "one_partial_rally": 1,
        "no_labelled_rally": 1,
        "several_labelled_rallies": 1,
    }
    assert summary["rally_counts"] == {
        "clean_one_to_one": 1,
        "complete_but_merged": 2,
        "partial_or_split": 1,
        "missed": 1,
    }
    assert summary["whole_rally_contact_score"]["5"] == {
        "old_scorer_fully_correct": 2,
        "clean_span_and_fully_correct": 1,
        "contact_tolerance_crossed_section_edge": 1,
        "share_of_one_rally_sections": 1.0,
        "share_of_all_predicted_sections": 0.5,
        "clean_span_share_of_all_predicted_sections": 0.25,
    }


def test_overlapping_sections_fail_loudly() -> None:
    with pytest.raises(ValueError, match="Overlapping sections"):
        load_sections(
            {
                "videos": [
                    {
                        "fixture": "8",
                        "spans": [
                            {"span_id": 0, "start_frame": 5, "end_frame": 25},
                            {"span_id": 1, "start_frame": 20, "end_frame": 30},
                        ],
                    }
                ]
            }
        )
