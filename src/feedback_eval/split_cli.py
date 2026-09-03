"""Build and inspect the player-disjoint split.

Build it once, commit the JSON, and pass it to every scoring run:

    python -m feedback_eval.split_cli \
        --references data/feedback_eval/references.jsonl \
        --seed 20260903 --test-fraction 0.3 \
        --out data/feedback_eval/split.json

Or pin the test players by name, which is what a reported result should use --
it survives a change of seed, of Python version, and of `build_split` itself:

    python -m feedback_eval.split_cli \
        --references data/feedback_eval/references.jsonl \
        --test-player "Viktor Axelsen" --test-player "An Se Young" \
        --out data/feedback_eval/split.json

Rebuilding a split with a different seed after seeing a score is how a
player-disjoint split stops meaning anything, so the command refuses to
overwrite an existing file unless --force is given.
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .records import load_references
from .splits import SplitError, build_split, clips_by_player, save_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--references", type=Path, required=True, help="references JSONL")
    parser.add_argument("--out", type=Path, required=True, help="split JSON to write")
    parser.add_argument(
        "--test-player",
        action="append",
        dest="test_players",
        help="pin a player to the test side; repeatable. Overrides --seed/--test-fraction",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="target share of CLIPS in test, filled by assigning whole players (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="required unless --test-player is given; recorded in the split file",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing split file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    if args.out.exists() and not args.force:
        raise SplitError(
            f"{args.out} already exists. Re-splitting after seeing a score invalidates the "
            "comparison; pass --force only if no result has been reported from it."
        )

    references = load_references(args.references)
    split = build_split(
        references,
        test_players=tuple(args.test_players) if args.test_players else None,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    save_split(args.out, split)

    grouped = clips_by_player(references)
    print(f"players  : {len(grouped)} ({len(split.train_players)} train / {len(split.test_players)} test)")
    print(f"clips    : {len(split.train_clips)} train / {len(split.test_clips)} test")
    print(f"discarded: {len(split.discarded_clips)} clip(s) spanning the partition")
    print(f"seed     : {split.seed}")
    print("\ntest players:")
    for player in split.test_players:
        print(f"  {player:<24} {len(grouped[player]):>4} clip(s)")
    print(f"\nwrote {args.out}")

    if split.discarded_clips:
        logging.warning(
            "%d of %d clip(s) involve players on both sides and belong to neither. "
            "This is the cost of a truly player-disjoint split; a more uneven "
            "partition usually discards fewer.",
            len(split.discarded_clips),
            len(references),
        )
    if len(split.test_clips) < 20:
        logging.warning(
            "only %d test clip(s): a mean over this few is dominated by individual clips, "
            "and one atypical player can move it more than the model version does",
            len(split.test_clips),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
