"""Coaching-feedback templates: the fault-keyed half of the reference set.

Proposal task 3.1. The proposal's primary plan for references is "templates of
good feedback compiled from coaching literature -- so evaluation is never
blocked on securing an expert". This module holds those templates.

A template is not feedback about a clip. It is feedback about a *fault*: what a
coach says when a player meets the shuttle behind the body, whoever that player
is. Attaching a template to a clip needs a separate judgement about which faults
that clip shows, which is what `template_references.py` consumes.

**Provenance is mandatory and is not decoration.** A template lifted from a
coaching video transcript, one quoted from a textbook, and one the team wrote
from general knowledge support different claims in the report, and the weakest
of the three is the one that quietly poses as the other two. `Provenance.kind`
forces the distinction at write time:

    transcript  -- from a coaching video; detail names channel, video and time
    publication -- from a book, manual or paper; detail is the citation
    drafted     -- written by the team from general coaching knowledge, traced
                   to no source. Honest, usable, and the weakest of the three.

`library_summary` counts templates by kind so the report can state the mix
rather than implying every template came from the literature.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# Written at the point a template is created, never inferred afterwards.
PROVENANCE_KINDS = ("transcript", "publication", "drafted")


class TemplateError(ValueError):
    """Raised when a template library is malformed."""


@dataclass(frozen=True)
class Provenance:
    """Where one template came from."""

    kind: str
    detail: str

    def __post_init__(self) -> None:
        if self.kind not in PROVENANCE_KINDS:
            raise TemplateError(
                f"provenance kind {self.kind!r} is not one of {list(PROVENANCE_KINDS)}"
            )
        if not self.detail.strip():
            raise TemplateError(
                f"provenance of kind {self.kind!r} needs a detail: a citation, a "
                "channel and video id, or who drafted it and when"
            )


@dataclass(frozen=True)
class FeedbackTemplate:
    """One fault and the accepted ways of coaching it.

    `corrections` carries several phrasings for the same reason the harness
    accepts several references: BERTScore keeps the best match, so a single
    phrasing systematically under-scores a correct answer worded differently.
    """

    template_id: str
    skill_area: str
    fault: str
    corrections: tuple[str, ...]
    provenance: Provenance


def _require_str(path: Path, lineno: int, row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TemplateError(f"{path}:{lineno}: {key!r} must be a non-blank string, got {value!r}")
    return value.strip()


def load_templates(path: Path) -> dict[str, FeedbackTemplate]:
    """Read a template library, keyed by `template_id`."""
    if not path.is_file():
        raise TemplateError(f"no such template library: {path}")
    templates: dict[str, FeedbackTemplate] = {}
    seen: dict[str, int] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise TemplateError(f"{path}:{lineno}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise TemplateError(f"{path}:{lineno}: expected a JSON object")

        template_id = _require_str(path, lineno, row, "template_id")
        if template_id in seen:
            raise TemplateError(
                f"{path}:{lineno}: duplicate template_id {template_id!r} "
                f"(first seen on line {seen[template_id]})"
            )
        seen[template_id] = lineno

        raw_corrections = row.get("corrections")
        if not isinstance(raw_corrections, list) or not raw_corrections:
            raise TemplateError(f"{path}:{lineno}: 'corrections' must be a non-empty list")
        corrections: dict[str, None] = {}
        for index, correction in enumerate(raw_corrections):
            if not isinstance(correction, str) or not correction.strip():
                raise TemplateError(
                    f"{path}:{lineno}: corrections[{index}] must be a non-blank string, "
                    f"got {correction!r}"
                )
            corrections.setdefault(correction.strip(), None)

        raw_provenance = row.get("provenance")
        if not isinstance(raw_provenance, dict):
            raise TemplateError(f"{path}:{lineno}: 'provenance' must be an object")
        try:
            provenance = Provenance(
                kind=str(raw_provenance.get("kind", "")),
                detail=str(raw_provenance.get("detail", "")),
            )
        except TemplateError as error:
            raise TemplateError(f"{path}:{lineno}: {error}") from error

        templates[template_id] = FeedbackTemplate(
            template_id=template_id,
            skill_area=_require_str(path, lineno, row, "skill_area"),
            fault=_require_str(path, lineno, row, "fault"),
            corrections=tuple(corrections),
            provenance=provenance,
        )
    if not templates:
        raise TemplateError(f"{path}: no templates")
    return templates


def library_summary(templates: dict[str, FeedbackTemplate]) -> str:
    """Counts by provenance kind and skill area, for the report.

    The provenance mix is a finding, not bookkeeping: a library that is mostly
    `drafted` supports a weaker claim than one grounded in coaching material,
    and the report has to be able to state which it had.
    """
    by_kind = Counter(template.provenance.kind for template in templates.values())
    by_area = Counter(template.skill_area for template in templates.values())
    lines = [f"templates      : {len(templates)}", "", "by provenance:"]
    for kind in PROVENANCE_KINDS:
        lines.append(f"  {kind:<12} {by_kind.get(kind, 0):>4}")
    lines += ["", "by skill area:"]
    for area, count in sorted(by_area.items()):
        lines.append(f"  {area:<24} {count:>4}")
    return "\n".join(lines)
