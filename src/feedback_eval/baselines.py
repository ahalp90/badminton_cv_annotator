"""Non-model baselines that calibrate the BERTScore band.

The harness's own README is emphatic that "BERTScore is not an absolute
quantity": raw F1 sits in a compressed high band even for unrelated text, so a
single number from Version A says nothing on its own. The usual fix is to
compare A against B -- but that only arrives once B is trained, and it still
leaves "is 0.62 good?" unanswered.

These baselines answer it without a model. They need no GPU, no checkpoint and
no clips; they are text generators over the reference set, scored through
exactly the same path as a real model version. Read together they bracket the
range any reported score has to be read against:

`empty`
    Returns nothing for every clip. Scores a hard zero by construction (see
    `scoring.score_run`), and exists to show that the floor is 0.0 rather than
    whatever BERTScore assigns to an empty string.

`constant`
    One generic, fault-independent piece of coaching advice, repeated for every
    clip. This is the number to beat. It is fluent, in-domain English about
    badminton, and it is *identical* on every clip, so whatever it scores is
    what a model earns for saying plausible coaching words while paying no
    attention to the rally. A Version A that does not clear this has not
    demonstrated anything.

`random_template`
    A correction drawn at random from the template library, almost always for
    the wrong fault. Harder than `constant`: the text is real coaching feedback
    of exactly the right register and length, just about the wrong thing. The
    gap between this and a model version is the part of the score attributable
    to identifying the fault rather than to sounding like a coach.

`oracle`
    The clip's own first reference, copied. The ceiling. Not 1.0 in general,
    because the clip's other phrasings are also in the reference list and
    BERTScore keeps the best match -- which makes it a useful check that the
    scoring path is wired up at all.

Run as `python -m feedback_eval.baselines` (see --help).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

from .contracts import ReferenceRecord
from .records import load_references
from .templates import load_templates

logger = logging.getLogger(__name__)

# Deliberately generic: true of almost any rally, specific to none. The point of
# the constant baseline is that it cannot be responding to the clip.
CONSTANT_FEEDBACK = (
    "Move to the shuttle earlier so you can take it in front of the body, and "
    "recover to your base position after every shot. Keep the racket head up "
    "and stay balanced through the stroke."
)

BASELINES = ("empty", "constant", "random_template", "oracle")


class BaselineError(ValueError):
    """Raised when a baseline cannot be generated as asked."""


def generate(
    references: Sequence[ReferenceRecord],
    baseline: str,
    *,
    corrections: Sequence[str] = (),
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Produce `(clip_id, feedback)` for one baseline over the reference set."""
    if baseline not in BASELINES:
        raise BaselineError(f"unknown baseline {baseline!r}; expected one of {list(BASELINES)}")
    if baseline == "random_template" and not corrections:
        raise BaselineError("the random_template baseline needs a template library")

    # Seeded per baseline so the file is reproducible from the CLI arguments
    # alone; a reported baseline that cannot be regenerated is not a baseline.
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for record in references:
        if baseline == "empty":
            feedback = ""
        elif baseline == "constant":
            feedback = CONSTANT_FEEDBACK
        elif baseline == "random_template":
            feedback = rng.choice(list(corrections))
        else:
            feedback = record.references[0]
        rows.append((record.clip_id, feedback))
    return rows


def write_predictions(path: Path, rows: Sequence[tuple[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for clip_id, feedback in rows:
            handle.write(json.dumps({"clip_id": clip_id, "feedback": feedback}) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--references", type=Path, required=True, help="references JSONL")
    parser.add_argument(
        "--baseline", required=True, choices=BASELINES, help="which baseline to generate"
    )
    parser.add_argument(
        "--templates",
        type=Path,
        help="template library JSONL; required for the random_template baseline",
    )
    parser.add_argument("--seed", type=int, default=0, help="recorded in the run, so pin it")
    parser.add_argument("--out", type=Path, required=True, help="predictions JSONL to write")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    references = load_references(args.references)
    corrections: tuple[str, ...] = ()
    if args.baseline == "random_template":
        if args.templates is None:
            raise BaselineError("--templates is required for the random_template baseline")
        corrections = tuple(
            correction
            for template in load_templates(args.templates).values()
            for correction in template.corrections
        )

    rows = generate(references, args.baseline, corrections=corrections, seed=args.seed)
    write_predictions(args.out, rows)
    print(f"wrote {args.out} -- {len(rows)} prediction(s) for baseline {args.baseline!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
