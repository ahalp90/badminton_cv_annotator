"""Turn coaching templates into a per-clip reference set.

Proposal task 3.1, the template half. `templates.py` holds feedback keyed by
*fault*; this attaches those templates to clips and writes the harness's
`references.jsonl` with `source="template"`.

    templates.jsonl   template_id -> fault, corrections, provenance
    clip_faults.csv   clip_id -> template_id   (one row per fault on the clip)
    players.csv       clip_id -> player_id     (one row per player in the rally)

The clip-to-fault call is a human judgement and this module does not make it.
Something has to watch the clip and say which faults it shows; that is the real
cost of the template route, and pretending otherwise would produce a reference
set that scores without meaning anything.

A clip may carry more than one fault, in which case its references are the union
of those templates' corrections. Feedback naming a different real fault on the
same clip is not wrong, and BERTScore keeps the best-matching reference.

**What a template reference is, and is not.** A template says what a coach says
about a *fault*, not about this *player* in this *rally*. It is the proposal's
deliberate fallback so that evaluation is "never blocked on securing an expert",
and it is weaker than an expert assessment of the clip itself. Records are
written `source="template"`; only genuine per-clip assessment may be recorded as
`expert`.

Run as `python -m feedback_eval.template_references` (see --help).
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .commentary_references import CommentaryReferenceError, write_references
from .contracts import ReferenceRecord
from .templates import FeedbackTemplate, load_templates

logger = logging.getLogger(__name__)


class TemplateReferenceError(ValueError):
    """Raised when the template inputs are malformed or disagree."""


@dataclass(frozen=True)
class BuildReport:
    """What was built, and every clip that could not be."""

    kept: int
    no_player: int
    unknown_template: tuple[str, ...]
    provenance_mix: tuple[tuple[str, int], ...]

    def render(self) -> str:
        lines = [
            f"clips written    : {self.kept}",
            f"clips w/o player : {self.no_player}",
            "",
            "reference provenance (templates actually used):",
        ]
        for kind, count in self.provenance_mix:
            lines.append(f"  {kind:<12} {count:>4}")
        return "\n".join(lines)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise TemplateReferenceError(f"no such file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = sorted(required - columns)
        if missing:
            raise TemplateReferenceError(
                f"{path}: missing column(s) {missing}; found {sorted(columns)}"
            )
        return list(reader)


def _grouped_pairs(path: Path, key: str, value: str) -> dict[str, tuple[str, ...]]:
    """Read a two-column CSV into ordered, de-duplicated groups."""
    grouped: dict[str, dict[str, None]] = {}
    for row in _read_csv(path, {key, value}):
        key_value = (row[key] or "").strip()
        item = (row[value] or "").strip()
        if not key_value or not item:
            raise TemplateReferenceError(
                f"{path}: {key} and {value} must be non-blank, got "
                f"{row[key]!r} / {row[value]!r}"
            )
        grouped.setdefault(key_value, {}).setdefault(item, None)
    if not grouped:
        raise TemplateReferenceError(f"{path}: no rows")
    return {name: tuple(items) for name, items in grouped.items()}


def load_clip_faults(path: Path) -> dict[str, tuple[str, ...]]:
    """Read `clip_id,template_id`; a clip may show more than one fault."""
    return _grouped_pairs(path, "clip_id", "template_id")


def load_clip_players(path: Path) -> dict[str, tuple[str, ...]]:
    """Read `clip_id,player_id`; a singles rally contributes two rows."""
    return _grouped_pairs(path, "clip_id", "player_id")


def build_references(
    templates: dict[str, FeedbackTemplate],
    clip_faults: dict[str, tuple[str, ...]],
    clip_players: dict[str, tuple[str, ...]],
) -> tuple[tuple[ReferenceRecord, ...], BuildReport]:
    """Join templates, faults and players into reference records."""
    unknown = sorted(
        {
            template_id
            for template_ids in clip_faults.values()
            for template_id in template_ids
            if template_id not in templates
        }
    )
    if unknown:
        raise TemplateReferenceError(
            f"{len(unknown)} fault(s) name a template that is not in the library: {unknown[:5]}"
        )

    records: list[ReferenceRecord] = []
    used_provenance: Counter[str] = Counter()
    no_player = 0
    for clip_id, template_ids in clip_faults.items():
        player_ids = clip_players.get(clip_id)
        if not player_ids:
            no_player += 1
            continue
        references: dict[str, None] = {}
        for template_id in template_ids:
            template = templates[template_id]
            used_provenance[template.provenance.kind] += 1
            for correction in template.corrections:
                references.setdefault(correction, None)
        records.append(
            ReferenceRecord(
                clip_id=clip_id,
                player_ids=player_ids,
                references=tuple(references),
                source="template",
            )
        )
    if no_player:
        logger.warning("%d clip(s) had faults but no player and were skipped", no_player)

    return tuple(records), BuildReport(
        kept=len(records),
        no_player=no_player,
        unknown_template=(),
        provenance_mix=tuple(sorted(used_provenance.items())),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--templates", type=Path, required=True, help="template library JSONL")
    parser.add_argument(
        "--clip-faults",
        type=Path,
        required=True,
        help="CSV of clip_id,template_id -- one row per fault shown on the clip",
    )
    parser.add_argument(
        "--players",
        type=Path,
        required=True,
        help="CSV of clip_id,player_id -- one row per player, so two per singles rally",
    )
    parser.add_argument("--out", type=Path, required=True, help="references JSONL to write")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    records, report = build_references(
        load_templates(args.templates),
        load_clip_faults(args.clip_faults),
        load_clip_players(args.players),
    )
    print(report.render())
    try:
        write_references(args.out, records)
    except CommentaryReferenceError as error:
        raise TemplateReferenceError(str(error)) from error
    players = len({player for record in records for player in record.player_ids})
    print(f"\nwrote {args.out} -- {len(records)} clip(s) over {players} player(s)")
    if players < 2:
        logger.warning("only %d player(s): a player-disjoint split needs at least 2", players)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
