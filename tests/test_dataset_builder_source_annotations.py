"""Tests for the ShuttleSet source_contacts reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataset_builder import schema_v1
from dataset_builder.source_annotations import (
    SourceRally,
    load_source_annotations,
    set_number,
)


_COLUMNS = ["rally", "ball_round", "time", "frame_num", "player", "type", "flaw"]


def _write_set(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)


def _row(
    rally: int, ball_round: int, frame_num: object, type_: str, flaw: str = ""
) -> dict[str, object]:
    return {
        "rally": rally,
        "ball_round": ball_round,
        "time": "00:00:00",
        "frame_num": frame_num,
        "player": "A",
        "type": type_,
        "flaw": flaw,
    }


def test_main_path_builds_usable_rally_and_maps_contact_types(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [
            _row(1, 1, 10, "發長球"),  # maps to long_service
            _row(1, 2, 20, "神秘球"),  # no taxonomy mapping
            _row(1, 3, 30, "切球"),  # maps to drop
        ],
    )

    result = load_source_annotations(
        tmp_path, source_dataset="ShuttleSet", video_id="v1", frame_count=1000
    )

    assert result.rallies == (SourceRally(1, 1, 10, 31, (0, 1, 2)),)
    contacts = result.contacts
    assert list(contacts.columns) == list(schema_v1.SOURCE_CONTACTS.column_names())
    assert contacts.iloc[0]["contact_type_en"] == "long_service"
    assert pd.isna(contacts.iloc[1]["contact_type_en"])
    assert contacts.iloc[2]["contact_type_en"] == "drop"
    assert list(contacts["rally_id"]) == [0, 0, 0]
    assert result.population == {
        "source_contact_rows": 3,
        "usable_contact_rows": 3,
        "usable_rallies": 1,
        "excluded_flaw_rows": 0,
        "excluded_invalid_frame_rows": 0,
        "excluded_incomplete_rallies": 0,
        "excluded_incomplete_rally_rows": 0,
        "excluded_non_monotonic_rallies": 0,
        "excluded_non_monotonic_rally_rows": 0,
    }
    validated = schema_v1.validate_table(schema_v1.SOURCE_CONTACTS, contacts)
    assert validated.equals(contacts)


def test_flaw_marked_row_makes_rally_unusable(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(2, 1, 40, "長球"), _row(2, 2, 41, "長球", flaw="1")],
    )

    result = load_source_annotations(
        tmp_path, source_dataset="ShuttleSet", video_id="v1", frame_count=1000
    )

    assert result.rallies == ()
    assert result.contacts["rally_id"].isna().all()
    assert result.population["excluded_flaw_rows"] == 1
    assert result.population["excluded_incomplete_rallies"] == 1
    assert result.population["excluded_incomplete_rally_rows"] == 2


def test_non_monotonic_contacts_are_excluded(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(3, 1, 100, "長球"), _row(3, 2, 90, "長球")],
    )

    result = load_source_annotations(
        tmp_path, source_dataset="ShuttleSet", video_id="v1", frame_count=1000
    )

    assert result.rallies == ()
    assert result.contacts["rally_id"].isna().all()
    assert result.population["excluded_non_monotonic_rallies"] == 1
    assert result.population["excluded_non_monotonic_rally_rows"] == 2


def test_out_of_range_frame_excludes_rally_but_keeps_row(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(4, 1, 10, "長球"), _row(4, 2, 60, "長球")],
    )

    result = load_source_annotations(
        tmp_path, source_dataset="ShuttleSet", video_id="v1", frame_count=50
    )

    assert result.rallies == ()
    assert result.contacts["rally_id"].isna().all()
    assert list(result.contacts["frame_num"]) == [10, 60]
    assert result.population["excluded_invalid_frame_rows"] == 1
    assert result.population["excluded_incomplete_rallies"] == 1


def test_rally_id_continues_across_sets_sorted_by_set(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(1, 1, 10, "長球"), _row(1, 2, 11, "長球")],
    )
    _write_set(
        tmp_path / "set2.csv",
        [_row(1, 1, 5, "長球"), _row(1, 2, 6, "長球")],
    )

    result = load_source_annotations(
        tmp_path, source_dataset="ShuttleSet", video_id="v1", frame_count=1000
    )

    assert [rally.source_set for rally in result.rallies] == [1, 2]
    set2_rows = result.contacts[result.contacts["source_set"] == 2]
    assert set(set2_rows["rally_id"]) == {1}


def test_set_number_rejects_non_set_filenames() -> None:
    with pytest.raises(ValueError):
        set_number(Path("foo.csv"))
