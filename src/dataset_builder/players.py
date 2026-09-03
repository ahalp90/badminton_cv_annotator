"""Curated player identity joined to the ShuttleSet and ShuttleSet22 match tables.

``configs/players.csv`` is the single source of truth for who a player is and
which BWF singles draw they compete in. A match table names the two people of
one match and carries the ``downcourt`` flag, which fixes who starts on the top
(far) court. Sides swap between sets and again at set 3's change of ends, so a
match splits into side phases, each with a fixed court orientation.

Added for issue #18 after Ari's review of PR #135: posture variability divides
by hip width, so the dataset must record whether the player is a man or a woman.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from dataset_builder.features import COURT_SIDES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAYERS = REPO_ROOT / "configs" / "players.csv"
MATCH_TABLE_FILENAME = "match.csv"
SEXES = ("female", "male")
# features.COURT_SIDES is ordered (top, bottom); top is the far court.
TOP_SIDE = COURT_SIDES[0]
FIRST_SET = 1
SWAPPED_SET = 2  # the set played with the sides of set 1 reversed
SWITCH_SET = 3  # the only set with a mid-set change of ends

PLAYER_COLUMNS = ("player_id", "player_name", "sex")
MATCH_COLUMNS = ("video", "winner", "loser", "downcourt")


class Player(NamedTuple):
    """One person from ``configs/players.csv``."""

    player_id: str
    player_name: str
    sex: str


class MatchPlayers(NamedTuple):
    """The two people of one match and their set-1 court orientation."""

    player_a: Player
    player_b: Player
    first_a_is_top: bool

    def on_side(self, side: str, a_is_top: bool) -> Player:
        """Return the person on ``side`` given the phase's court orientation."""
        if side not in COURT_SIDES:
            raise ValueError(f"side must be one of {COURT_SIDES}, got {side!r}")
        a_is_on_side = a_is_top == (side == TOP_SIDE)
        return self.player_a if a_is_on_side else self.player_b


class SidePhase(NamedTuple):
    """One stretch of a match played with a fixed court orientation.

    ``start_frame`` and ``end_frame`` are the half-open envelope of the phase's
    valid human-contact frames, so they bound the phase but do not prove that
    every frame between them belongs to it.
    """

    source_set: int
    post_switch: bool
    a_is_top: bool
    start_frame: int
    end_frame: int


def load_players(path: Path = DEFAULT_PLAYERS) -> dict[str, Player]:
    """Load the curated player table, keyed by the display name the match tables use."""
    table = _read_table(path, PLAYER_COLUMNS)
    players: dict[str, Player] = {}
    identifiers: set[str] = set()
    for record in table.to_dict("records"):
        player = Player(*(str(record[column]) for column in PLAYER_COLUMNS))
        if not all(player):
            raise ValueError(f"{path} has an empty cell in row {player}")
        if player.sex not in SEXES:
            raise ValueError(
                f"{path}: {player.player_name!r} has sex {player.sex!r}, expected one of {SEXES}"
            )
        if player.player_id in identifiers:
            raise ValueError(f"{path} repeats player_id {player.player_id!r}")
        if player.player_name in players:
            raise ValueError(f"{path} repeats player_name {player.player_name!r}")
        identifiers.add(player.player_id)
        players[player.player_name] = player
    if not players:
        raise ValueError(f"{path} has no players")
    return players


def load_match_players(
    match_table: Path, video: str, players: Mapping[str, Player]
) -> MatchPlayers:
    """Resolve one match's two people and whether player A starts on the top court."""
    table = _read_table(match_table, MATCH_COLUMNS)
    rows = table[table["video"] == video]
    if len(rows) != 1:
        raise ValueError(
            f"{match_table} has {len(rows)} rows for video {video!r}, expected exactly one"
        )
    row = rows.iloc[0]
    # ShuttleSet stores downcourt as "1"; ShuttleSet22's 2022 rows store it as "1.0".
    downcourt = pd.to_numeric(row["downcourt"], errors="coerce")
    if downcourt != 0 and downcourt != 1:
        raise ValueError(
            f"{match_table}: {video!r} has downcourt {row['downcourt']!r}, expected 0 or 1"
        )
    winner = _named_player(players, str(row["winner"]), match_table, video)
    loser = _named_player(players, str(row["loser"]), match_table, video)
    if winner.sex != loser.sex:
        raise ValueError(
            f"{match_table}: {video!r} pairs {winner.player_name} ({winner.sex}) with "
            f"{loser.player_name} ({loser.sex}); a singles match is one BWF draw"
        )
    # ShuttleSet labels the match winner A, and downcourt says where A starts.
    return MatchPlayers(player_a=winner, player_b=loser, first_a_is_top=bool(downcourt))


def a_is_top(first_a_is_top: bool, set_number: int, post_switch: bool) -> bool:
    """Return whether player A is on the top court during this side phase.

    The same rule as ``classifier_shared.player_mapping.map_players``: every
    phase is played either with the sides of set 1 or with them reversed, and
    set 3's change of ends reverses them again.
    """
    if not FIRST_SET <= set_number <= SWITCH_SET:
        raise ValueError(f"set number must be {FIRST_SET}..{SWITCH_SET}, got {set_number}")
    phase_set = SWAPPED_SET if set_number == SWAPPED_SET or post_switch else FIRST_SET
    return first_a_is_top ^ (phase_set == SWAPPED_SET)


def phase_for_span(phases: Sequence[SidePhase], start: int, end: int) -> SidePhase | None:
    """Return the one phase overlapping ``[start, end)``, or None when it is ambiguous."""
    overlapping = [
        phase for phase in phases if phase.start_frame < end and start < phase.end_frame
    ]
    return overlapping[0] if len(overlapping) == 1 else None


def _read_table(path: Path, required: Sequence[str]) -> pd.DataFrame:
    """Read a curated CSV as text, with empty fields kept as empty strings."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return table


def _named_player(
    players: Mapping[str, Player], name: str, match_table: Path, video: str
) -> Player:
    player = players.get(name)
    if player is None:
        raise ValueError(
            f"{match_table}: {video!r} names {name!r}, who has no row in the curated "
            f"player table; add the name to configs/players.csv"
        )
    return player
