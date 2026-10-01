"""Draw a fixed subset of one side of a pinned split.

Generating predictions for all 543 held-out clips means cutting and running the
model on every one of them. A seeded subset of ~100 is enough to separate a
model version from the `random_template` floor, and is a few hours of GPU time
rather than a day.

The subset is written as a *split file*, not as a smaller reference file. That
is what `score_cli` needs: `splits.select` insists every clip the split names is
present in the records it is given, so a 100-clip reference file scored against
the full split is refused (443 clips missing), while the full reference file
scored against a 100-clip split selects exactly those 100. It also keeps the
player-disjoint guard running, because the train side is carried over unchanged.

Only the sampled side changes. Its player list is narrowed to the players who
actually appear in the sampled clips, so the per-player means in a score run
name only players that were scored. The output records what it is a subset of,
its size and its seed, so the 100 clips can be regenerated from the arguments
alone.

Every version and every baseline compared in the report must be scored against
the same subset file. Like `split_cli`, this refuses to overwrite an existing
file without --force: redrawing the subset after seeing a score is the same
mistake as re-splitting after seeing one.

Run as `python -m feedback_eval.subset_split` (see --help).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

from .records import load_references
from .splits import SPLIT_SCHEMA, Split, SplitError, load_split, players_by_clip

logger = logging.getLogger(__name__)


def subset_split(
    split: Split,
    players_of: dict[str, frozenset[str]],
    *,
    size: int,
    seed: int,
    side: str = "test",
) -> Split:
    """Return `split` with one side cut down to `size` seeded clips.

    `players_of` maps clip_id -> its players, from the reference set the split
    was built on. The sampled clips keep their original file order, so the
    subset reads the same way the full split does.
    """
    if side not in ("train", "test"):
        raise SplitError(f"side must be 'train' or 'test', got {side!r}")
    clips = split.clips(side)
    if size <= 0:
        raise SplitError(f"size must be positive, got {size}")
    if size > len(clips):
        raise SplitError(f"asked for {size} clip(s) but the {side} side has only {len(clips)}")
    unknown = [clip_id for clip_id in clips if clip_id not in players_of]
    if unknown:
        raise SplitError(
            f"{len(unknown)} {side} clip(s) are not in the reference set {unknown[:5]}; "
            "the split and the references disagree"
        )

    chosen = set(random.Random(seed).sample(list(clips), size))
    kept = tuple(clip_id for clip_id in clips if clip_id in chosen)
    present = {player for clip_id in kept for player in players_of[clip_id]}
    players = tuple(player for player in split.players(side) if player in present)

    if side == "test":
        return Split(
            train_players=split.train_players,
            test_players=players,
            train_clips=split.train_clips,
            test_clips=kept,
            discarded_clips=split.discarded_clips,
            seed=split.seed,
        )
    return Split(
        train_players=players,
        test_players=split.test_players,
        train_clips=kept,
        test_clips=split.test_clips,
        discarded_clips=split.discarded_clips,
        seed=split.seed,
    )


def save_subset(path: Path, split: Split, *, source: Path, side: str, size: int, seed: int) -> Path:
    """Write the subset in the ordinary split schema, plus where it came from.

    `load_split` ignores the extra `subset` block, so the file is a valid split
    for every existing tool.
    """
    payload = {
        "schema": SPLIT_SCHEMA,
        "seed": split.seed,
        "subset": {"of": str(source), "side": side, "size": size, "seed": seed},
        "train": {"players": list(split.train_players), "clips": list(split.train_clips)},
        "test": {"players": list(split.test_players), "clips": list(split.test_clips)},
        "discarded_clips": list(split.discarded_clips),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split", type=Path, required=True, help="the pinned split to draw from")
    parser.add_argument(
        "--references", type=Path, required=True, help="references JSONL the split was built on"
    )
    parser.add_argument("--size", type=int, default=100, help="clips to keep (default: 100)")
    parser.add_argument("--seed", type=int, required=True, help="recorded in the file, so pin it")
    parser.add_argument("--side", choices=("train", "test"), default="test")
    parser.add_argument("--out", type=Path, required=True, help="subset split JSON to write")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing subset (see module docstring)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.out.exists() and not args.force:
        raise SplitError(
            f"{args.out} already exists; redrawing a subset after scoring against it makes "
            "the comparison meaningless. Pass --force only if nothing has been scored on it."
        )

    split = load_split(args.split)
    players_of = players_by_clip(load_references(args.references))
    subset = subset_split(split, players_of, size=args.size, seed=args.seed, side=args.side)
    save_subset(args.out, subset, source=args.split, side=args.side, size=args.size, seed=args.seed)

    clips = subset.clips(args.side)
    videos = {clip_id.rsplit("_s", 1)[0] for clip_id in clips}
    print(
        f"wrote {args.out} -- {len(clips)} {args.side} clip(s) over "
        f"{len(subset.players(args.side))} player(s) from {len(videos)} match(es)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
