"""Derive per-clip faults from ShuttleSet's terminal-shot annotations.

Proposal task 3.1. `template_references.py` needs a `clip_faults.csv` saying
which fault each clip shows, and its docstring is blunt that the call "is a
human judgement and this module does not make it". That judgement was the last
thing standing between the harness and a real reference set: somebody had to
watch several thousand rallies.

They already did. ShuttleSet annotates, for every rally, the shot that ended it
(`type`), which player played it (`player`), and how the rally was lost
(`lose_reason`). That is a human expert's per-rally record of *what went wrong*,
published with the dataset. This module joins it to the template library.

**What this derivation claims, and what it does not.** ShuttleSet records the
*outcome* of the terminal shot -- the shuttle went out, or into the net -- not
its *technical cause*. Mapping "smash hit out" back to "no trunk rotation" would
be an inference this data cannot support, and the resulting reference set would
look rigorous while laundering a guess. So the templates this module targets are
keyed to the observed outcome (`badminton_singles_shuttleset_v1.jsonl`), not to
the cause-keyed library in `badminton_singles_v1.jsonl`. The fault statement is
annotated; only the correction text is drafted.

Three classes of rally are deliberately excluded, and each is counted:

`opponent_winner`
    `lose_reason` 對手落地致勝 -- the rally ended in a clean winner, so the
    losing player made no recorded error. Coaching feedback here would have to
    invent a cause ("your recovery was late"), which the annotation does not
    record. 1189 of 3508 rallies, and dropping them is the honest call.

`ambiguous_reason`
    落點判斷失誤 (misjudged the landing) and 犯規 (foul). For the misjudgement
    label the annotation does not identify who erred consistently: the row
    carries a shot played by one player while `getpoint_player` sometimes names
    that same player and sometimes the other. 48 rallies; not worth guessing.

`attribution_inconsistent`
    An out/net error where `getpoint_player` names the player who played the
    shot -- the invariant "you lose the point you erred on" fails. 14 rallies,
    treated as annotation noise rather than silently kept.

`player_ids` is both players of the match, from `match.csv`. That is not
laziness about who erred: `splits.build_split` needs every player in the rally
so that it can keep a clip only when they all fall on the same side, and a
singles rally genuinely involves two players regardless of whose mistake it was.
See `contracts.ReferenceRecord`.

Run as `python -m feedback_eval.shuttleset_faults` (see --help).
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from classifier_shared.taxonomy import ZH_TO_EN

logger = logging.getLogger(__name__)


class ShuttleSetFaultError(ValueError):
    """Raised when the ShuttleSet annotations are missing or malformed."""


# lose_reason -> error mode. 掛網 (caught the net cord) and 未過網 (never
# reached the net) are merged: both are "into the net" to a coach, and the
# distinction is about where the shuttle stopped, not about what to fix.
SELF_ERROR_MODES: dict[str, str] = {
    "出界": "hit_out",
    "掛網": "into_net",
    "未過網": "into_net",
}

# The rally ended in a winner, so the loser has no recorded error of their own.
OPPONENT_WINNER = "對手落地致勝"

# Stroke family per ShuttleSet English stroke name (see classifier_shared.taxonomy).
# Families are coaching groupings, not the classifier's taxonomy: what a coach
# says about a drop that finds the net is the same whether the annotation called
# it 切球 or 過渡切球.
STROKE_FAMILIES: dict[str, str] = {
    "short_service": "serve",
    "long_service": "serve",
    "net_shot": "net_play",
    "cross_court_net_shot": "net_play",
    "rush": "net_play",
    "return_net": "block",
    "defensive_return_drive": "block",
    "lob": "lift",
    "defensive_return_lob": "lift",
    "drive": "drive",
    "driven_flight": "drive",
    "back_court_drive": "drive",
    "push": "drive",
    "clear": "clear",
    "smash": "smash",
    "wrist_smash": "smash",
    "drop": "drop",
    "passive_drop": "drop",
}


@dataclass
class DeriveReport:
    """What was derived, and every rally that was not.

    Every exclusion is counted. A reference set quietly missing a third of its
    rallies still scores, and the mean it produces looks perfectly reasonable.
    """

    kept: int = 0
    opponent_winner: int = 0
    ambiguous_reason: int = 0
    attribution_inconsistent: int = 0
    unmapped_stroke: Counter[str] = field(default_factory=Counter)
    missing_match: Counter[str] = field(default_factory=Counter)
    by_template: Counter[str] = field(default_factory=Counter)

    @property
    def total(self) -> int:
        return (
            self.kept
            + self.opponent_winner
            + self.ambiguous_reason
            + self.attribution_inconsistent
            + sum(self.unmapped_stroke.values())
            + sum(self.missing_match.values())
        )

    def render(self) -> str:
        lines = [
            f"rallies seen             : {self.total}",
            f"clips written            : {self.kept}",
            "",
            "excluded:",
            f"  opponent winner        : {self.opponent_winner}",
            f"  ambiguous lose_reason  : {self.ambiguous_reason}",
            f"  attribution inconsistent: {self.attribution_inconsistent}",
            f"  unmapped stroke type   : {sum(self.unmapped_stroke.values())}"
            + (f" {dict(self.unmapped_stroke)}" if self.unmapped_stroke else ""),
            f"  no match.csv row       : {sum(self.missing_match.values())}"
            + (f" {sorted(self.missing_match)}" if self.missing_match else ""),
            "",
            "faults by template:",
        ]
        for template_id, count in sorted(self.by_template.items()):
            lines.append(f"  {template_id:<22} {count:>5}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ClipFault:
    """One rally, its derived fault, and who was on court."""

    clip_id: str
    template_id: str
    erring_player: str
    player_ids: tuple[str, ...]


def load_match_players(path: Path) -> dict[str, tuple[str, ...]]:
    """Read `match.csv` into video -> (winner, loser).

    Both are carried because the split partitions players, not clips.
    """
    if not path.is_file():
        raise ShuttleSetFaultError(f"no such match file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {"video", "winner", "loser"} - set(reader.fieldnames or ())
        if missing:
            raise ShuttleSetFaultError(f"{path}: missing column(s) {sorted(missing)}")
        players: dict[str, tuple[str, ...]] = {}
        for row in reader:
            video = (row["video"] or "").strip()
            winner = (row["winner"] or "").strip()
            loser = (row["loser"] or "").strip()
            if not (video and winner and loser):
                raise ShuttleSetFaultError(
                    f"{path}: blank video/winner/loser in row {row!r}"
                )
            players[video] = (winner, loser)
    if not players:
        raise ShuttleSetFaultError(f"{path}: no rows")
    return players


def _set_number(path: Path) -> str:
    """`set3.csv` -> `3`. The rally number restarts each set, so it is part of the id."""
    stem = path.stem
    if not stem.startswith("set") or not stem[3:].isdigit():
        raise ShuttleSetFaultError(f"unexpected set file name: {path}")
    return stem[3:]


def derive_faults(
    set_root: Path,
    match_players: dict[str, tuple[str, ...]],
) -> tuple[tuple[ClipFault, ...], DeriveReport]:
    """Walk the ShuttleSet set files and derive one fault per usable rally."""
    set_files = sorted(set_root.glob("*/set*.csv"))
    if not set_files:
        raise ShuttleSetFaultError(f"{set_root}: no */set*.csv files")

    faults: list[ClipFault] = []
    report = DeriveReport()
    for set_file in set_files:
        video = set_file.parent.name
        players = match_players.get(video)
        set_number = _set_number(set_file)
        with set_file.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                lose_reason = (row.get("lose_reason") or "").strip()
                if not lose_reason:
                    continue  # not the rally-ending shot
                if players is None:
                    report.missing_match[video] += 1
                    continue
                if lose_reason == OPPONENT_WINNER:
                    report.opponent_winner += 1
                    continue
                mode = SELF_ERROR_MODES.get(lose_reason)
                if mode is None:
                    report.ambiguous_reason += 1
                    continue
                striker = (row.get("player") or "").strip()
                if (row.get("getpoint_player") or "").strip() == striker:
                    # The player who hit the losing shot also won the point.
                    report.attribution_inconsistent += 1
                    continue
                family = STROKE_FAMILIES.get(ZH_TO_EN.get((row.get("type") or "").strip(), ""))
                if family is None:
                    report.unmapped_stroke[(row.get("type") or "").strip()] += 1
                    continue

                template_id = f"{family}_{mode}"
                report.by_template[template_id] += 1
                report.kept += 1
                faults.append(
                    ClipFault(
                        clip_id=f"{video}_s{set_number}_r{(row.get('rally') or '').strip()}",
                        template_id=template_id,
                        erring_player=striker,
                        player_ids=players,
                    )
                )

    duplicates = [
        clip_id for clip_id, count in Counter(f.clip_id for f in faults).items() if count > 1
    ]
    if duplicates:
        raise ShuttleSetFaultError(
            f"{len(duplicates)} clip_id(s) derived more than once, e.g. {duplicates[:3]}; "
            "one rally must produce at most one terminal shot"
        )
    return tuple(faults), report


def write_clip_faults(path: Path, faults: Sequence[ClipFault]) -> Path:
    """Write `clip_id,template_id,erring_player` for template_references.

    `erring_player` is surplus to that reader's required columns and is carried
    for audit: it is the one thing the derivation knows that the reference set
    cannot express, and without it the attribution is unreviewable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "template_id", "erring_player"])
        for fault in faults:
            writer.writerow([fault.clip_id, fault.template_id, fault.erring_player])
    return path


def write_clip_players(path: Path, faults: Sequence[ClipFault]) -> Path:
    """Write `clip_id,player_id` -- one row per player, so two per singles rally."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "player_id"])
        for fault in faults:
            for player_id in fault.player_ids:
                writer.writerow([fault.clip_id, player_id])
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--set-root",
        type=Path,
        default=Path("data/shuttleset/set"),
        help="directory holding one subdirectory per match, each with set*.csv",
    )
    parser.add_argument(
        "--match",
        type=Path,
        default=None,
        help="match.csv mapping video -> winner/loser (default: <set-root>/match.csv)",
    )
    parser.add_argument(
        "--out-faults", type=Path, required=True, help="clip_faults CSV to write"
    )
    parser.add_argument(
        "--out-players", type=Path, required=True, help="clip_players CSV to write"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    match_path = args.match or args.set_root / "match.csv"
    faults, report = derive_faults(args.set_root, load_match_players(match_path))
    print(report.render())
    if not faults:
        raise ShuttleSetFaultError("no faults derived; refusing to write an empty fault set")

    write_clip_faults(args.out_faults, faults)
    write_clip_players(args.out_players, faults)
    players = len({player for fault in faults for player in fault.player_ids})
    print(f"\nwrote {args.out_faults} and {args.out_players}")
    print(f"{len(faults)} clip(s) over {players} player(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
