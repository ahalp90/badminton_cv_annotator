"""Build a reference set from the COSC595 commentary lane.

Proposal task 3.1. Turns the scraper's rally-paired commentary into the
harness's `references.jsonl`, so the reference set stops being eight
hand-written stubs.

The join is small because the two lanes already agree in shape.
`scraper.commentary_cleaning` writes, per chunk, a cleaned `text_clean` plus
`ALT_PHRASINGS_K` meaning-preserving `alt_phrasings` -- which is exactly the
multi-phrasing reference list BERTScore wants, for the same reason: one
phrasing systematically under-scores a correct answer worded differently.
`scraper.commentary_pairing` writes `rally_commentary_pairs.csv`, mapping each
rally to the chunk that follows it.

    rally_commentary_pairs.csv  video_id, rally_id -> chunk_id
    chunks/<video_id>.json      chunk_id -> text_clean, alt_phrasings, clean_pass
    players.csv                 video_id, rally_id -> player_id

**What this reference set is, and is not.** Broadcast commentary is descriptive
and reactive: it says what happened and how good it looked, not what the player
should change. It is not coaching feedback and must never be recorded as
`expert`. Records are written with `source="commentary"` so the report can say
which claim the numbers support -- scoring against commentary measures whether
generated text matches how an informed observer described the rally, which is a
weaker and different claim than matching a coach's correction. This is the route
Rai & Kovashka (2026) take deliberately, pairing competition commentary with
coaching literature, so it is defensible; it is not interchangeable with expert
assessment.

**Every player in the rally is recorded, not one of them.** A singles rally has
two, and `players.csv` carries one row per (rally, player) -- so two rows for
one rally is the normal case, not a conflict. Reducing that to a single
"subject" player would need a per-rally judgement about whose error it was that
the commentary rarely supports, and a combined pair key is not player-disjoint
at all. `splits` handles the two-player case directly by partitioning players
and discarding the clips that straddle the partition.

Run as `python -m feedback_eval.commentary_references` (see --help).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .contracts import ReferenceRecord

logger = logging.getLogger(__name__)

# The pairing stage leaves these blank for a rally it held out (replay-masked)
# or found no chunk for. Both are "no reference exists", not an error.
UNPAIRED_CHUNK_ID = ""


class CommentaryReferenceError(ValueError):
    """Raised when the commentary inputs are malformed or disagree."""


@dataclass(frozen=True)
class BuildReport:
    """What was kept and, for everything else, why it was dropped.

    Counted rather than logged in passing: a reference set silently missing a
    third of its rallies still scores, and the mean it produces looks fine.
    """

    kept: int
    unpaired: int
    chunk_missing: int
    not_clean_pass: int
    blank_text: int
    no_player: int

    @property
    def dropped(self) -> int:
        return (
            self.unpaired
            + self.chunk_missing
            + self.not_clean_pass
            + self.blank_text
            + self.no_player
        )

    def render(self) -> str:
        return "\n".join(
            [
                f"kept            : {self.kept}",
                f"dropped         : {self.dropped}",
                f"  unpaired rally: {self.unpaired}",
                f"  chunk missing : {self.chunk_missing}",
                f"  clean_pass=0  : {self.not_clean_pass}",
                f"  blank text    : {self.blank_text}",
                f"  no player_id  : {self.no_player}",
            ]
        )


def clip_id_for(video_id: str, rally_id: int, run_id: str | None = None) -> str:
    """Build the clip key the model's predictions must also use.

    `rally_id` is a list position within one extraction run, and the rally
    dataset contract is explicit that it is not stable across runs: a changed
    config can insert or remove an earlier span and shift every id after it.
    Pass `run_id` whenever references and predictions might come from different
    runs, so a mismatch surfaces as an `align` error rather than as a quietly
    wrong pairing.
    """
    stem = f"{video_id}_r{rally_id}"
    return f"{run_id}_{stem}" if run_id else stem


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise CommentaryReferenceError(f"no such file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise CommentaryReferenceError(
                f"{path}: missing column(s) {missing}; found {sorted(columns)}"
            )
        return list(reader)


def _rally_id(path: Path, raw: str) -> int:
    try:
        return int(raw)
    except (TypeError, ValueError) as error:
        raise CommentaryReferenceError(f"{path}: rally_id {raw!r} is not an integer") from error


def load_player_map(path: Path) -> dict[tuple[str, int], tuple[str, ...]]:
    """Read `video_id,rally_id,player_id` into every player each rally involves.

    One row per (rally, player), so a singles rally contributes two rows. Order
    is preserved and repeats are collapsed, because a player named twice for one
    rally would otherwise count twice in that rally's per-player mean.

    Curtis's v1 schema is the source: `rallies.top_player_id` and
    `bottom_player_id`, resolved against `configs/players.csv`. Both go in.
    """
    mapping: dict[tuple[str, int], dict[str, None]] = {}
    for row in _read_csv(path, {"video_id", "rally_id", "player_id"}):
        video_id = (row["video_id"] or "").strip()
        player_id = (row["player_id"] or "").strip()
        if not video_id or not player_id:
            raise CommentaryReferenceError(
                f"{path}: video_id and player_id must be non-blank, got "
                f"{row['video_id']!r} / {row['player_id']!r}"
            )
        key = (video_id, _rally_id(path, row["rally_id"]))
        mapping.setdefault(key, {}).setdefault(player_id, None)
    if not mapping:
        raise CommentaryReferenceError(f"{path}: no player rows")
    return {key: tuple(players) for key, players in mapping.items()}


def load_chunks(chunks_dir: Path, video_id: str) -> dict[str, dict]:
    """Index one video's cleaned commentary chunks by chunk_id."""
    path = chunks_dir / f"{video_id}.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommentaryReferenceError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(payload, list):
        raise CommentaryReferenceError(f"{path}: expected a JSON list of chunks")
    indexed: dict[str, dict] = {}
    for position, chunk in enumerate(payload):
        if not isinstance(chunk, dict):
            raise CommentaryReferenceError(f"{path}[{position}]: expected an object")
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise CommentaryReferenceError(f"{path}[{position}]: chunk_id must be a non-empty string")
        indexed[chunk_id] = chunk
    return indexed


def references_from_chunk(chunk: dict) -> tuple[str, ...]:
    """Cleaned text first, then its alternate phrasings, de-duplicated.

    Order matters only for readability -- BERTScore takes the best match -- but
    the cleaned text leads because it is the one phrasing the cleaning stage
    actually vouched for with a score.
    """
    candidates = [chunk.get("text_clean")]
    alternates = chunk.get("alt_phrasings")
    if isinstance(alternates, list):
        candidates.extend(alternates)
    seen: dict[str, None] = {}
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            seen.setdefault(candidate.strip(), None)
    return tuple(seen)


def build_references(
    pairs_csv: Path,
    chunks_dir: Path,
    player_map: dict[tuple[str, int], tuple[str, ...]],
    *,
    run_id: str | None = None,
    require_clean_pass: bool = True,
) -> tuple[tuple[ReferenceRecord, ...], BuildReport]:
    """Join pairs, chunks and players into reference records."""
    rows = _read_csv(pairs_csv, {"video_id", "rally_id", "chunk_id"})
    chunk_cache: dict[str, dict[str, dict]] = {}
    counts: Counter[str] = Counter()
    records: list[ReferenceRecord] = []
    seen_clips: dict[str, tuple[str, int]] = {}

    for row in rows:
        video_id = (row["video_id"] or "").strip()
        if not video_id:
            raise CommentaryReferenceError(f"{pairs_csv}: blank video_id")
        rally_id = _rally_id(pairs_csv, row["rally_id"])
        chunk_id = (row["chunk_id"] or "").strip()

        if chunk_id == UNPAIRED_CHUNK_ID:
            counts["unpaired"] += 1
            continue
        if video_id not in chunk_cache:
            chunk_cache[video_id] = load_chunks(chunks_dir, video_id)
        chunk = chunk_cache[video_id].get(chunk_id)
        if chunk is None:
            counts["chunk_missing"] += 1
            continue
        if require_clean_pass and not chunk.get("clean_pass"):
            counts["not_clean_pass"] += 1
            continue
        references = references_from_chunk(chunk)
        if not references:
            counts["blank_text"] += 1
            continue
        player_ids = player_map.get((video_id, rally_id))
        if not player_ids:
            counts["no_player"] += 1
            continue

        clip_id = clip_id_for(video_id, rally_id, run_id)
        if clip_id in seen_clips:
            raise CommentaryReferenceError(
                f"{pairs_csv}: duplicate rally {(video_id, rally_id)} produces clip_id {clip_id!r} twice"
            )
        seen_clips[clip_id] = (video_id, rally_id)
        records.append(
            ReferenceRecord(
                clip_id=clip_id,
                player_ids=player_ids,
                references=references,
                source="commentary",
            )
        )

    report = BuildReport(
        kept=len(records),
        unpaired=counts["unpaired"],
        chunk_missing=counts["chunk_missing"],
        not_clean_pass=counts["not_clean_pass"],
        blank_text=counts["blank_text"],
        no_player=counts["no_player"],
    )
    return tuple(records), report


def write_references(path: Path, records: Sequence[ReferenceRecord]) -> Path:
    """Write records in the JSONL shape `records.load_references` reads back."""
    if not records:
        raise CommentaryReferenceError("refusing to write an empty reference set")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "clip_id": record.clip_id,
                "player_ids": list(record.player_ids),
                "references": list(record.references),
                "source": record.source,
            }
        )
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pairs", type=Path, required=True, help="rally_commentary_pairs.csv")
    parser.add_argument("--chunks-dir", type=Path, required=True, help="scrape chunks/ directory")
    parser.add_argument(
        "--players",
        type=Path,
        required=True,
        help="CSV of video_id,rally_id,player_id -- one row per player, so two per singles rally",
    )
    parser.add_argument("--out", type=Path, required=True, help="references JSONL to write")
    parser.add_argument(
        "--run-id",
        default=None,
        help="prefix clip ids with the extraction run; rally_id is only stable within one run",
    )
    parser.add_argument(
        "--keep-failed-clean",
        action="store_true",
        help="keep chunks the cleaning stage scored below its BERTScore gate (not for a reported run)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.keep_failed_clean:
        logger.warning(
            "--keep-failed-clean: including chunks that failed the cleaning BERTScore gate; "
            "these references were not vouched for by the cleaning stage"
        )
    records, report = build_references(
        args.pairs,
        args.chunks_dir,
        load_player_map(args.players),
        run_id=args.run_id,
        require_clean_pass=not args.keep_failed_clean,
    )
    print(report.render())
    write_references(args.out, records)
    players = len({player for record in records for player in record.player_ids})
    print(f"\nwrote {args.out} -- {len(records)} clip(s) over {players} player(s)")
    if players < 2:
        logger.warning("only %d player(s): a player-disjoint split needs at least 2", players)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
