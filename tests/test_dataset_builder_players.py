"""Tests for the curated player table and the court-side rule (issue #18)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_builder.players import (
    DEFAULT_PLAYERS,
    MatchPlayers,
    Player,
    SidePhase,
    a_is_top,
    load_match_players,
    load_players,
    phase_for_span,
)


PLAYERS_CSV = """player_id,player_name,sex
kento_momota,Kento MOMOTA,male
chou_tien_chen,CHOU Tien Chen,male
akane_yamaguchi,Akane YAMAGUCHI,female
"""
# ShuttleSet22's 2022 rows store downcourt as a float, ShuttleSet as an integer.
MATCH_CSV = """id,video,winner,loser,downcourt
1,men_final,Kento MOMOTA,CHOU Tien Chen,1.0
2,mixed_pair,Kento MOMOTA,Akane YAMAGUCHI,0
3,unknown_name,Kento MOMOTA,Somebody ELSE,1
"""
_PHASES = (
    SidePhase(source_set=1, post_switch=False, a_is_top=True, start_frame=10, end_frame=100),
    SidePhase(source_set=2, post_switch=False, a_is_top=False, start_frame=200, end_frame=300),
)


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_curated_table_loads_and_is_keyed_by_display_name() -> None:
    players = load_players()

    assert DEFAULT_PLAYERS.is_file()
    assert players["Kento MOMOTA"] == Player("kento_momota", "Kento MOMOTA", "male")
    assert {player.sex for player in players.values()} == {"female", "male"}


def test_load_players_rejects_a_bad_sex_and_a_repeated_id(tmp_path: Path) -> None:
    bad_sex = _write(tmp_path, "sex.csv", PLAYERS_CSV.replace(",male\n", ",m\n", 1))
    with pytest.raises(ValueError, match="expected one of"):
        load_players(bad_sex)

    repeated = _write(tmp_path, "id.csv", PLAYERS_CSV + "kento_momota,Kento MOMOTA JR,male\n")
    with pytest.raises(ValueError, match="repeats player_id"):
        load_players(repeated)


def test_load_match_players_reads_a_float_downcourt(tmp_path: Path) -> None:
    players = load_players(_write(tmp_path, "players.csv", PLAYERS_CSV))

    match = load_match_players(_write(tmp_path, "match.csv", MATCH_CSV), "men_final", players)

    # The match winner is player A, and downcourt 1 starts A on the top court.
    assert match == MatchPlayers(
        players["Kento MOMOTA"], players["CHOU Tien Chen"], first_a_is_top=True
    )
    assert match.on_side("top", a_is_top=True) == players["Kento MOMOTA"]
    assert match.on_side("bottom", a_is_top=True) == players["CHOU Tien Chen"]
    assert match.on_side("top", a_is_top=False) == players["CHOU Tien Chen"]


def test_load_match_players_rejects_unusable_rows(tmp_path: Path) -> None:
    players = load_players(_write(tmp_path, "players.csv", PLAYERS_CSV))
    match_table = _write(tmp_path, "match.csv", MATCH_CSV)

    with pytest.raises(ValueError, match="add the name to configs/players.csv"):
        load_match_players(match_table, "unknown_name", players)
    with pytest.raises(ValueError, match="0 rows for video"):
        load_match_players(match_table, "no_such_video", players)
    with pytest.raises(ValueError, match="one BWF draw"):
        load_match_players(match_table, "mixed_pair", players)


def test_a_is_top_follows_the_set_number_and_the_change_of_ends() -> None:
    # Set 1 keeps the downcourt orientation, set 2 reverses it.
    assert a_is_top(True, 1, post_switch=False) is True
    assert a_is_top(False, 1, post_switch=False) is False
    assert a_is_top(True, 2, post_switch=False) is False
    assert a_is_top(False, 2, post_switch=False) is True
    # Set 3 starts like set 1 and reverses once a score first reaches 11.
    assert a_is_top(True, 3, post_switch=False) is True
    assert a_is_top(True, 3, post_switch=True) is False

    with pytest.raises(ValueError, match="set number must be"):
        a_is_top(True, 4, post_switch=False)


def test_phase_for_span_needs_exactly_one_overlapping_phase() -> None:
    assert phase_for_span(_PHASES, 50, 120) is _PHASES[0]
    assert phase_for_span(_PHASES, 120, 190) is None  # falls between the two phases
    assert phase_for_span(_PHASES, 50, 250) is None  # overlaps both
    assert phase_for_span((), 0, 10) is None
    # The envelopes are half-open: ending where a phase starts is not an overlap.
    assert phase_for_span(_PHASES, 0, 10) is None
