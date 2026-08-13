"""Focused tests for the post-hoc first/last-stroke buffer analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from annotator.artifact_io import atomic_gzip_text_writer, write_json_object
from scripts.analyse_first_last_stroke_buffered_search import main


STRICT_COLUMNS = (
    "rally_id",
    "tolerance_base30",
    "tolerance_frames",
    "row_kind",
    "gt_frame",
    "candidate_frame",
    "offset_frames",
)
WIDE_COLUMNS = (
    "window_id",
    "rally_id",
    "edge",
    "window_start",
    "window_end",
    "row_kind",
    "gt_frame",
    "candidate_frame",
    "offset_frames",
)


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    if path.name.endswith(".csv.gz"):
        with atomic_gzip_text_writer(path, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


@pytest.mark.parametrize("compressed", [False, True])
def test_post_hoc_search_classifies_gt_matches_and_predicted_span_position(
    tmp_path: Path,
    capsys,
    compressed: bool,
) -> None:
    measurement_root = tmp_path / "measurement"
    configuration_id = "detected_ckn_opencv_consensus/sset_01/tracknet-stride-8"
    leaf_dir = measurement_root / configuration_id
    leaf_dir.mkdir(parents=True)
    json_suffix = ".json.gz" if compressed else ".json"
    csv_suffix = ".csv.gz" if compressed else ".csv"
    write_json_object(measurement_root / f"manifest{json_suffix}", {
        "status": "succeeded",
        "configurations": [{"configuration_id": configuration_id, "status": "succeeded"}],
    })
    write_json_object(leaf_dir / f"annotations{json_suffix}", {
        "spans": [[100, 200], [300, 309], [310, 400]],
        "filtered_contacts": [
            {"rally_id": 0, "contact_frame": 115},
            {"rally_id": 0, "contact_frame": 118},
            {"rally_id": 0, "contact_frame": 190},
            {"rally_id": 1, "contact_frame": 306},
        ],
    })
    _write_csv(leaf_dir / f"strict_contacts{csv_suffix}", STRICT_COLUMNS, [
        (0, 5, 5, "unmatched_gt", 110, "", ""),
        (0, 5, 5, "matched", 118, 118, 0),
        (0, 5, 5, "matched", 190, 190, 0),
        (1, 5, 5, "matched", 310, 306, -4),
        (2, 5, 5, "unmatched_gt", 500, "", ""),
    ])
    _write_csv(leaf_dir / f"wide_edge_contacts{csv_suffix}", WIDE_COLUMNS, [
        (0, 0, "first", 35, 151, "matched", 110, 118, 8),
        (0, 0, "first", 35, 151, "unmatched_candidate", "", 115, ""),
        (1, 0, "last", 151, 265, "matched", 190, 190, 0),
        (2, 1, "first", 235, 386, "matched", 310, 306, -4),
        (3, 2, "first", 425, 576, "unmatched_gt", 500, "", ""),
    ])

    output_directory = tmp_path / "analysis"
    assert main([str(measurement_root), str(output_directory)]) == 0
    output_paths = list(output_directory.glob("first_last_stroke_buffered_search_*.csv"))
    assert len(output_paths) == 1
    output_path = output_paths[0]

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[0]["candidate_gt_relation"] == "other_gt_stroke_match"
    assert rows[0]["matched_gt_frame"] == "118"
    assert rows[0]["additional_candidate_count"] == "1"
    assert rows[0]["candidate_span_relation"] == "inside_associated_span"

    assert rows[1]["candidate_gt_relation"] == "first_last_stroke_match"
    assert rows[1]["candidate_assigned_to_associated_span"] == "True"
    assert rows[1]["required_boundary_extension_frames"] == "0"

    assert rows[2]["candidate_gt_relation"] == "first_last_stroke_match"
    assert rows[2]["candidate_span_relation"] == "before_associated_span"
    assert rows[2]["required_boundary_extension_frames"] == "4"

    assert rows[3]["candidate_gt_relation"] == "no_candidate"
    assert rows[3]["predicted_boundary_classification"] == "missed"
    assert rows[3]["candidate_span_relation"] == "no_candidate"

    output = capsys.readouterr().out
    assert "first_last_buffers_with_candidate=3" in output
    assert "first_last_buffers_without_candidate=1" in output
    assert "selected_candidates_matching_first_last_stroke=2" in output
    assert "selected_candidates_matching_other_gt_stroke=1" in output
    assert "selected_candidates_without_gt_match=0" in output
    assert "first_last_matches_outside_associated_span=1" in output
    assert "additional_candidates_in_buffers=1" in output
