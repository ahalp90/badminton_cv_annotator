"""Player-disjoint splits for the feedback-evaluation harness.

Proposal task 3.2, and the functional requirement that a reported score reflects
generalisation rather than memorisation.

The unit of assignment is the *player*, never the clip. Assigning clips at
random puts the same player on both sides, and a model that has learnt one
player's habits -- their grip, their footwork, the way a commentator talks about
them -- then scores well on the test set for the wrong reason. Whole players go
to one side or the other.

A singles rally involves two players, which makes the construction less obvious
than it looks. Two tempting shortcuts are both wrong:

- Naming one "subject" player per rally needs a per-rally judgement about whose
  error it was. Commentary rarely says, so the call would come from a winner
  heuristic and its error would land straight in the split.
- Keying on the pair is not disjoint at all. Pair AB in train and pair AC in
  test both contain A, so A trains and tests -- the exact leak this module
  exists to stop.

So: partition the players, then keep a clip only when *every* player in it falls
on the same side. Clips spanning the boundary are discarded and counted. That
costs clips -- more at an even partition, fewer at 70/30 -- and paying it is the
only version where "player-disjoint" is true as written.

A split is a durable artefact, not something recomputed per run. Versions A, B
and C have to be scored on the *same* test clips or their means are not
comparable, so the split is built once, written to disk, and passed to every
subsequent run.
"""
from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

from .contracts import ReferenceRecord


class HasClipId(Protocol):
    """Anything keyed by clip -- a `ReferenceRecord` or a `PredictionRecord`."""

    clip_id: str


RecordT = TypeVar("RecordT", bound=HasClipId)


logger = logging.getLogger(__name__)

SPLIT_SCHEMA = "feedback-eval-split/1"

# Sides are named rather than boolean so a caller cannot silently invert them.
SIDES = ("train", "test")


class LeakageError(RuntimeError):
    """Raised when a player appears on both sides of a split."""


class SplitError(ValueError):
    """Raised when a split cannot be built or does not fit the records given."""


@dataclass(frozen=True)
class Split:
    """Which players, and therefore which clips, belong to each side.

    Both the player assignment and the resolved clip lists are stored. The
    players are the decision; the clips are what that decision selected from one
    particular reference set, and keeping them lets a later run detect that the
    reference set changed underneath it.
    """

    train_players: tuple[str, ...]
    test_players: tuple[str, ...]
    train_clips: tuple[str, ...]
    test_clips: tuple[str, ...]
    # Clips whose players straddle the partition. Neither side may use them, and
    # they are named rather than merely counted so the loss is auditable.
    discarded_clips: tuple[str, ...] = ()
    seed: int | None = None

    def __post_init__(self) -> None:
        shared = set(self.train_players) & set(self.test_players)
        if shared:
            raise LeakageError(f"player(s) on both sides of the split: {sorted(shared)}")
        if not self.train_players or not self.test_players:
            raise SplitError("a split needs at least one player on each side")

    def players(self, side: str) -> tuple[str, ...]:
        return self.train_players if _side(side) == "train" else self.test_players

    def clips(self, side: str) -> tuple[str, ...]:
        return self.train_clips if _side(side) == "train" else self.test_clips


def _side(side: str) -> str:
    if side not in SIDES:
        raise SplitError(f"side {side!r} is not one of {list(SIDES)}")
    return side


def clips_by_player(records: tuple[ReferenceRecord, ...]) -> dict[str, tuple[str, ...]]:
    """Map each player to their clips, in reference-file order.

    A clip appears under every player it involves, so the lists overlap and the
    counts do not sum to the number of clips.
    """
    grouped: dict[str, list[str]] = {}
    for record in records:
        for player in record.player_ids:
            grouped.setdefault(player, []).append(record.clip_id)
    return {player: tuple(clips) for player, clips in sorted(grouped.items())}


def players_by_clip(records: tuple[ReferenceRecord, ...]) -> dict[str, frozenset[str]]:
    """Map each clip to the set of players it involves."""
    return {record.clip_id: frozenset(record.player_ids) for record in records}


def assert_player_disjoint(
    train: tuple[ReferenceRecord, ...],
    test: tuple[ReferenceRecord, ...],
) -> None:
    """Refuse two record sets that share a player.

    This is the guard the harness exists to make unavoidable. Call it on the
    records actually scored, not on the split file -- the split can be correct
    while the records handed to the scorer are not.
    """
    train_players = {player for record in train for player in record.player_ids}
    test_players = {player for record in test for player in record.player_ids}
    shared = train_players & test_players
    if shared:
        raise LeakageError(
            f"{len(shared)} player(s) appear in both train and test: {sorted(shared)}. "
            "Scores from this split would reflect memorisation, not generalisation."
        )


def build_split(
    records: tuple[ReferenceRecord, ...],
    *,
    test_players: tuple[str, ...] | None = None,
    test_fraction: float = 0.3,
    seed: int | None = None,
) -> Split:
    """Assign whole players to train or test.

    `test_players` pins the split by name and is what a reported result should
    use: it survives a change of seed, of Python version, and of this function.
    Without it, players are shuffled under `seed` and taken into test until they
    account for at least `test_fraction` of the clips -- so the fraction is over
    *clips*, since that is what the mean is computed over, while the thing being
    assigned is still the player.

    The realised test share is always smaller than `test_fraction`, because
    clips shared with a train player are then discarded. Read the actual counts
    off the returned `Split` rather than assuming the target was met.
    """
    if not records:
        raise SplitError("cannot split an empty reference set")
    grouped = clips_by_player(records)
    if len(grouped) < 2:
        raise SplitError(
            f"a player-disjoint split needs at least 2 players, got {len(grouped)}: "
            f"{sorted(grouped)}"
        )

    if test_players is not None:
        chosen = tuple(dict.fromkeys(test_players))
        unknown = sorted(set(chosen) - grouped.keys())
        if unknown:
            raise SplitError(f"test player(s) not present in the reference set: {unknown}")
        if len(chosen) == len(grouped):
            raise SplitError("every player was assigned to test, leaving nothing to train on")
        seed = None
    else:
        if not 0.0 < test_fraction < 1.0:
            raise SplitError(f"test_fraction must be strictly between 0 and 1, got {test_fraction}")
        if seed is None:
            raise SplitError("building a split by fraction requires an explicit seed")
        # Sort before shuffling so the seed alone determines the outcome; dict
        # order would otherwise leak the reference file's ordering into it.
        shuffled = sorted(grouped)
        random.Random(seed).shuffle(shuffled)
        target = test_fraction * len(records)
        chosen_list: list[str] = []
        taken = 0
        for player in shuffled:
            # Stop before consuming the last player: train must not be empty.
            if len(chosen_list) == len(grouped) - 1:
                break
            if taken >= target:
                break
            chosen_list.append(player)
            taken += len(grouped[player])
        if not chosen_list:
            chosen_list = [shuffled[0]]
        chosen = tuple(chosen_list)

    test_set = set(chosen)
    train_players = tuple(player for player in grouped if player not in test_set)
    test_ordered = tuple(player for player in grouped if player in test_set)

    # A clip belongs to a side only when every player in it is on that side.
    # Anything straddling the partition is discarded: using it on either side
    # would put the crossing player on both.
    train_clips: list[str] = []
    test_clips: list[str] = []
    discarded: list[str] = []
    for record in records:
        players = set(record.player_ids)
        if players <= test_set:
            test_clips.append(record.clip_id)
        elif not players & test_set:
            train_clips.append(record.clip_id)
        else:
            discarded.append(record.clip_id)

    if not test_clips:
        raise SplitError(
            f"test players {sorted(test_set)} share every one of their clips with a train "
            "player, so the test side is empty. Pin a different set of test players."
        )
    if not train_clips:
        raise SplitError(
            "every clip crosses the partition or sits in test, leaving nothing to train on"
        )
    if discarded:
        logger.warning(
            "%d of %d clip(s) span the player partition and were discarded from both sides",
            len(discarded),
            len(records),
        )
    return Split(
        train_players=train_players,
        test_players=test_ordered,
        train_clips=tuple(train_clips),
        test_clips=tuple(test_clips),
        discarded_clips=tuple(discarded),
        seed=seed,
    )


def select(records: Sequence[RecordT], split: Split, side: str) -> tuple[RecordT, ...]:
    """Take one side's records, in file order. Works on references or predictions.

    Requires every clip the split names to be present. A split built against a
    larger record set will otherwise silently score a subset, which is the same
    broken-denominator failure `records.align` exists to prevent -- and it is
    worth being strict on predictions too: a model that produced output for only
    some test clips has not been evaluated on the test set.
    """
    wanted = set(split.clips(_side(side)))
    present = {record.clip_id for record in records}
    missing = sorted(wanted - present)
    if missing:
        raise SplitError(
            f"{len(missing)} clip(s) in the {side} split are absent from the records given "
            f"{missing[:5]}; the split and the records disagree"
        )
    return tuple(record for record in records if record.clip_id in wanted)


def save_split(path: Path, split: Split) -> Path:
    """Write the split as the durable artefact every later run is pinned to."""
    payload = {
        "schema": SPLIT_SCHEMA,
        "seed": split.seed,
        "train": {"players": list(split.train_players), "clips": list(split.train_clips)},
        "test": {"players": list(split.test_players), "clips": list(split.test_clips)},
        "discarded_clips": list(split.discarded_clips),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_split(path: Path) -> Split:
    """Read a split back, rejecting anything this version cannot honour."""
    if not path.is_file():
        raise SplitError(f"no such split file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SplitError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise SplitError(f"{path}: expected a JSON object")
    schema = payload.get("schema")
    if schema != SPLIT_SCHEMA:
        raise SplitError(f"{path}: schema {schema!r}, expected {SPLIT_SCHEMA!r}")

    def _side_payload(side: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        block = payload.get(side)
        if not isinstance(block, dict):
            raise SplitError(f"{path}: {side!r} must be an object")
        out = []
        for key in ("players", "clips"):
            value = block.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SplitError(f"{path}: {side}.{key} must be a list of strings")
            out.append(tuple(value))
        return out[0], out[1]

    train_players, train_clips = _side_payload("train")
    test_players, test_clips = _side_payload("test")
    discarded = payload.get("discarded_clips", [])
    if not isinstance(discarded, list) or not all(isinstance(item, str) for item in discarded):
        raise SplitError(f"{path}: discarded_clips must be a list of strings")
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise SplitError(f"{path}: seed must be an integer or null, got {seed!r}")
    return Split(
        train_players=train_players,
        test_players=test_players,
        train_clips=train_clips,
        test_clips=test_clips,
        discarded_clips=tuple(discarded),
        seed=seed,
    )
