"""Score one model version's feedback against the reference set.

    python -m feedback_eval.score_cli \
        --references experiments/feedback_eval/stub/references.jsonl \
        --predictions experiments/feedback_eval/stub/predictions_version_a_stub.jsonl \
        --model-version A-stub \
        --out experiments/feedback_eval/stub/scores_version_a_stub.json

Add --dry-run to validate the inputs and exercise the whole path with a
deterministic fake scorer, without installing bert-score or downloading a model.

Pass --split to score one side of a player-disjoint split. Without it the run
covers every clip in the reference file, which is a wiring check rather than a
reportable result: a mean over clips whose players the model was trained on
measures memorisation.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .contracts import ScorerConfig
from .records import align, load_predictions, load_references
from .report import as_dict, render_summary
from .scoring import Scorer, build_bert_scorer, score_run
from .splits import assert_player_disjoint, load_split, select


def _fake_scorer(
    candidates: Sequence[str],
    references: Sequence[Sequence[str]],
) -> tuple[list[float], list[float], list[float]]:
    """Deterministic stand-in for BERTScore: token overlap with the best reference.

    For wiring checks only. It is a bag-of-words overlap, so it cannot see
    meaning -- which is the entire reason the real harness uses BERTScore. Never
    report a number that came out of this.
    """
    precision, recall, f1 = [], [], []
    for candidate, candidate_references in zip(candidates, references, strict=True):
        candidate_tokens = set(candidate.lower().split())
        best = 0.0
        best_pair = (0.0, 0.0)
        for reference in candidate_references:
            reference_tokens = set(reference.lower().split())
            if not candidate_tokens or not reference_tokens:
                continue
            shared = len(candidate_tokens & reference_tokens)
            clip_precision = shared / len(candidate_tokens)
            clip_recall = shared / len(reference_tokens)
            if clip_precision + clip_recall == 0:
                continue
            clip_f1 = 2 * clip_precision * clip_recall / (clip_precision + clip_recall)
            if clip_f1 >= best:
                best = clip_f1
                best_pair = (clip_precision, clip_recall)
        precision.append(best_pair[0])
        recall.append(best_pair[1])
        f1.append(best)
    return precision, recall, f1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", type=Path, required=True, help="references JSONL")
    parser.add_argument("--predictions", type=Path, required=True, help="predictions JSONL")
    parser.add_argument("--model-version", required=True, help="label for this run, e.g. A or B")
    parser.add_argument("--out", type=Path, help="write the full JSON payload here")
    parser.add_argument(
        "--model-type",
        default=None,
        help="embedding model for BERTScore; pin this for any run that goes in the report",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument(
        "--rescale-with-baseline",
        action="store_true",
        help="rescale against bert-score's baseline; spreads out the compressed raw range",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="use a deterministic token-overlap fake instead of BERTScore (wiring check only)",
    )
    parser.add_argument(
        "--split",
        type=Path,
        help="player-disjoint split JSON from split_cli; scores one side only",
    )
    parser.add_argument(
        "--split-side",
        default="test",
        choices=("test", "train"),
        help="which side of --split to score (default: test)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    config = ScorerConfig(
        model_type=args.model_type,
        lang=args.lang,
        rescale_with_baseline=args.rescale_with_baseline,
        batch_size=args.batch_size,
    )
    references = load_references(args.references)
    predictions = load_predictions(args.predictions)

    split_provenance: dict | None = None
    if args.split:
        split = load_split(args.split)
        held_out = select(references, split, "train" if args.split_side == "test" else "test")
        references = select(references, split, args.split_side)
        predictions = select(predictions, split, args.split_side)
        # The split file could be right while the records handed to the scorer
        # are not, so the guard runs on the selected records themselves.
        assert_player_disjoint(held_out, references)
        split_provenance = {
            "path": str(args.split),
            "side": args.split_side,
            "seed": split.seed,
            "players": list(split.players(args.split_side)),
        }
        logging.info(
            "split %s side=%s: %d clip(s) over %d player(s)",
            args.split,
            args.split_side,
            len(references),
            len(split.players(args.split_side)),
        )
    else:
        logging.warning(
            "no --split: scoring every clip in the reference file. Not a reportable "
            "result -- a mean over trained-on players measures memorisation."
        )

    pairs = align(references, predictions)

    scorer: Scorer
    if args.dry_run:
        logging.warning("--dry-run: token-overlap fake scorer, NOT BERTScore. Do not report these numbers.")
        scorer = _fake_scorer
    else:
        scorer = build_bert_scorer(config)

    run = score_run(pairs, model_version=args.model_version, scorer=scorer, config=config)
    print(render_summary(run))

    if args.out:
        payload = as_dict(run)
        payload["dry_run"] = bool(args.dry_run)
        payload["split"] = split_provenance
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
