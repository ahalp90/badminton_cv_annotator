"""Read and validate the harness's JSONL inputs.

Two files, joined on `clip_id`:

references.jsonl   {"clip_id", "player_ids": [...], "references": [...], "source"}
predictions.jsonl  {"clip_id", "feedback"}

One reference set is scored against many prediction files (one per model
version), which is what keeps A, B and C on the same footing.

Every validation here fails loudly. A silently dropped clip is the failure mode
that matters: it changes the denominator, and a mean over a different subset of
clips is not a comparable number.
"""
from __future__ import annotations

import json
from pathlib import Path

from .contracts import PredictionRecord, REFERENCE_SOURCES, ReferenceRecord


class RecordError(ValueError):
    """Raised when an input file is malformed or the two files disagree."""


def _read_jsonl(path: Path) -> list[tuple[int, dict]]:
    """Return (line number, object) pairs, skipping blank lines.

    Line numbers are 1-based and kept so every later error can name the line the
    reader has to open.
    """
    if not path.is_file():
        raise RecordError(f"no such file: {path}")
    rows: list[tuple[int, dict]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecordError(f"{path}:{lineno}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise RecordError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
        rows.append((lineno, row))
    if not rows:
        raise RecordError(f"{path}: no records")
    return rows


def _require_str(path: Path, lineno: int, row: dict, key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecordError(f"{path}:{lineno}: {key!r} must be a non-blank string, got {value!r}")
    return value.strip()


def _require_str_list(path: Path, lineno: int, row: dict, key: str) -> tuple[str, ...]:
    """Read a non-empty list of non-blank strings, preserving order.

    Duplicates are dropped rather than rejected: the same player named twice, or
    the same phrasing supplied twice, is a harmless input quirk, but a repeated
    player would double that player's weight in a per-player mean.
    """
    value = row.get(key)
    if not isinstance(value, list) or not value:
        raise RecordError(f"{path}:{lineno}: {key!r} must be a non-empty list")
    seen: dict[str, None] = {}
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RecordError(
                f"{path}:{lineno}: {key}[{index}] must be a non-blank string, got {item!r}"
            )
        seen.setdefault(item.strip(), None)
    return tuple(seen)


def load_references(path: Path) -> tuple[ReferenceRecord, ...]:
    """Load the gold feedback set."""
    records: list[ReferenceRecord] = []
    seen: dict[str, int] = {}
    for lineno, row in _read_jsonl(path):
        clip_id = _require_str(path, lineno, row, "clip_id")
        if clip_id in seen:
            raise RecordError(
                f"{path}:{lineno}: duplicate clip_id {clip_id!r} (first seen on line {seen[clip_id]})"
            )
        seen[clip_id] = lineno

        player_ids = _require_str_list(path, lineno, row, "player_ids")
        source = _require_str(path, lineno, row, "source")
        if source not in REFERENCE_SOURCES:
            raise RecordError(
                f"{path}:{lineno}: source {source!r} is not one of {list(REFERENCE_SOURCES)}"
            )

        references = _require_str_list(path, lineno, row, "references")

        records.append(
            ReferenceRecord(
                clip_id=clip_id,
                player_ids=player_ids,
                references=references,
                source=source,
            )
        )
    return tuple(records)


def load_predictions(path: Path) -> tuple[PredictionRecord, ...]:
    """Load one model version's generated feedback.

    Unlike references, `feedback` may be blank: a model that returns nothing is a
    real result and is scored as zero rather than quietly dropped.
    """
    records: list[PredictionRecord] = []
    seen: dict[str, int] = {}
    for lineno, row in _read_jsonl(path):
        clip_id = _require_str(path, lineno, row, "clip_id")
        if clip_id in seen:
            raise RecordError(
                f"{path}:{lineno}: duplicate clip_id {clip_id!r} (first seen on line {seen[clip_id]})"
            )
        seen[clip_id] = lineno

        feedback = row.get("feedback")
        if not isinstance(feedback, str):
            raise RecordError(f"{path}:{lineno}: 'feedback' must be a string, got {feedback!r}")
        records.append(PredictionRecord(clip_id=clip_id, feedback=feedback.strip()))
    return tuple(records)


def align(
    references: tuple[ReferenceRecord, ...],
    predictions: tuple[PredictionRecord, ...],
) -> tuple[tuple[ReferenceRecord, PredictionRecord], ...]:
    """Pair each reference with its prediction, in reference-file order.

    Requires an exact clip_id match in both directions. Scoring a subset would
    still produce a mean, and that mean would look like a comparable result while
    resting on different clips -- so a mismatch is an error, not a warning.
    """
    by_clip = {prediction.clip_id: prediction for prediction in predictions}
    reference_ids = {reference.clip_id for reference in references}

    missing = sorted(reference_ids - by_clip.keys())
    extra = sorted(by_clip.keys() - reference_ids)
    if missing or extra:
        raise RecordError(
            f"clip sets differ: {len(missing)} reference clip(s) have no prediction "
            f"{missing[:5]}, {len(extra)} prediction(s) have no reference {extra[:5]}"
        )
    return tuple((reference, by_clip[reference.clip_id]) for reference in references)
