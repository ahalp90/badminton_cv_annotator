"""Tests for the ShuttleSet source_contacts reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataset_builder import schema_v1
from dataset_builder.players import MatchPlayers, Player, SidePhase
from dataset_builder.source_annotations import (
    SourceRally,
    load_source_annotations,
    set_number,
)


_COLUMNS = [
    "rally", "ball_round", "time", "frame_num", "roundscore_A", "roundscore_B",
    "player", "type", "flaw",
]
# Player A is the match winner; downcourt 1 starts A on the top court in set 1.
_MATCH = MatchPlayers(
    Player("kento_momota", "Kento MOMOTA", "male"),
    Player("chou_tien_chen", "CHOU Tien Chen", "male"),
    first_a_is_top=True,
)


def _write_set(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=_COLUMNS).to_csv(path, index=False)


def _row(
    rally: int,
    ball_round: int,
    frame_num: object,
    type_: str,
    flaw: str = "",
    player: str = "A",
    score_a: int = 0,
    score_b: int = 0,
) -> dict[str, object]:
    return {
        "rally": rally,
        "ball_round": ball_round,
        "time": "00:00:00",
        "frame_num": frame_num,
        "roundscore_A": score_a,
        "roundscore_B": score_b,
        "player": player,
        "type": type_,
        "flaw": flaw,
    }


def _load(set_dir: Path, frame_count: int = 1000):
    return load_source_annotations(
        set_dir,
        source_dataset="ShuttleSet",
        video_id="v1",
        frame_count=frame_count,
        match=_MATCH,
    )


def test_main_path_builds_usable_rally_and_maps_contact_types(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [
            _row(1, 1, 10, "發長球"),  # maps to long_service
            _row(1, 2, 20, "神秘球"),  # no taxonomy mapping
            _row(1, 3, 30, "切球"),  # maps to drop
        ],
    )

    result = _load(tmp_path)

    assert result.rallies == (SourceRally(1, 1, 10, 31, (0, 1, 2), a_is_top=True),)
    contacts = result.contacts
    # This module owns every source_contacts column except the four
    # position-derived ones, which export_v1 adds once it has player positions.
    position_derived = {
        "recovery_distance", "recovery_frames_valid",
        "movement_inefficiency_top", "movement_inefficiency_bottom",
    }
    assert list(contacts.columns) == [
        name for name in schema_v1.SOURCE_CONTACTS.column_names() if name not in position_derived
    ]
    assert contacts.iloc[0]["contact_type_en"] == "long_service"
    assert pd.isna(contacts.iloc[1]["contact_type_en"])
    assert contacts.iloc[2]["contact_type_en"] == "drop"
    assert list(contacts["rally_id"]) == [0, 0, 0]
    assert not contacts["flaw_marked"].any()  # A clean rally: no row carries the flag.
    assert result.population == {
        "source_contact_rows": 3,
        "usable_contact_rows": 3,
        "usable_rallies": 1,
        "side_phases": 1,
        "kept_flaw_rows": 0,
        "excluded_invalid_frame_rows": 0,
        "excluded_incomplete_rallies": 0,
        "excluded_incomplete_rally_rows": 0,
        "excluded_non_monotonic_rallies": 0,
        "excluded_non_monotonic_rally_rows": 0,
    }
    # The owned columns already carry their frozen dtypes; export_v1 validates
    # the complete table once it adds the position-derived columns.
    frozen_dtypes = schema_v1.SOURCE_CONTACTS.pandas_dtypes()
    assert {name: str(dtype) for name, dtype in contacts.dtypes.items()} == {
        name: frozen_dtypes[name] for name in contacts.columns
    }


def test_flagged_serve_keeps_its_rally_and_marks_it(tmp_path: Path) -> None:
    # Issue #138: a flaw-marked row no longer drops its rally. The flag marks a
    # broken frame number, almost always the serve's, not a bad rally.
    _write_set(
        tmp_path / "set1.csv",
        [_row(2, 1, 40, "長球", flaw="1"), _row(2, 2, 65, "長球")],
    )

    result = _load(tmp_path)

    assert result.rallies == (SourceRally(1, 2, 40, 66, (0, 1), a_is_top=True),)
    assert list(result.contacts["rally_id"]) == [0, 0]
    assert result.contacts["flaw_marked"].tolist() == [True, False]
    assert result.population["kept_flaw_rows"] == 1
    assert result.population["excluded_incomplete_rallies"] == 0
    assert result.population["excluded_incomplete_rally_rows"] == 0


def test_non_monotonic_contacts_are_excluded(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(3, 1, 100, "長球"), _row(3, 2, 90, "長球")],
    )

    result = _load(tmp_path)

    assert result.rallies == ()
    assert result.contacts["rally_id"].isna().all()
    assert result.population["excluded_non_monotonic_rallies"] == 1
    assert result.population["excluded_non_monotonic_rally_rows"] == 2


def test_out_of_range_frame_excludes_rally_but_keeps_row(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [_row(4, 1, 10, "長球"), _row(4, 2, 60, "長球")],
    )

    result = _load(tmp_path, frame_count=50)

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

    result = _load(tmp_path)

    assert [rally.source_set for rally in result.rallies] == [1, 2]
    set2_rows = result.contacts[result.contacts["source_set"] == 2]
    assert set(set2_rows["rally_id"]) == {1}


def test_hitter_player_id_maps_the_source_player_letters(tmp_path: Path) -> None:
    _write_set(
        tmp_path / "set1.csv",
        [
            _row(1, 1, 10, "發長球", player="A"),
            _row(1, 2, 20, "挑球", player="B"),
            _row(1, 3, 30, "切球", player="X"),  # not a singles player letter
        ],
    )

    result = _load(tmp_path)

    assert result.contacts["player_id"].tolist()[:2] == ["kento_momota", "chou_tien_chen"]
    assert pd.isna(result.contacts["player_id"].iloc[2])


def test_set3_change_of_ends_splits_the_match_into_two_side_phases(tmp_path: Path) -> None:
    # One contact per rally; player A's score reaches 11 in rally 11, so rally 12
    # is the first played after the change of ends.
    _write_set(
        tmp_path / "set3.csv",
        [_row(rally, 1, rally * 10, "長球", score_a=rally) for rally in range(1, 13)],
    )

    result = _load(tmp_path)

    assert result.side_phases == (
        SidePhase(source_set=3, post_switch=False, a_is_top=True, start_frame=10, end_frame=111),
        SidePhase(source_set=3, post_switch=True, a_is_top=False, start_frame=120, end_frame=121),
    )
    assert [rally.a_is_top for rally in result.rallies] == [True] * 11 + [False]


def test_set_number_rejects_non_set_filenames() -> None:
    with pytest.raises(ValueError):
        set_number(Path("foo.csv"))


def test_english_label_maps_the_shuttleset22_spelling_variant() -> None:
    from dataset_builder.source_annotations import english_label

    assert english_label("過渡切球") == "passive_drop"
    assert english_label("過度切球") == "passive_drop"
    assert english_label("not a label") is pd.NA
    assert english_label(float("nan")) is pd.NA
