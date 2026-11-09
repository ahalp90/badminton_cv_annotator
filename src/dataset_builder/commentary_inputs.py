"""Load retimed commentary chunks and check their alignment with cleaned text."""
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scraper.commentary_retiming import MIN_MATCH_RATIO, SEARCH_PAD_S, AlignStatus

RETIMED_RELATIVE_DIR = "commentary/retimed_chunks"


RETIMED_ALIGN_STATUSES = tuple(status.value for status in AlignStatus)


RETIMED_START_SLACK_S = 2.0


RETIMED_SHIFT_TOLERANCE_S = 1e-6


RETIMED_PRESERVED_FIELDS = (
    "text",
    "text_clean",
    "bert_f1",
    "clean_pass",
    "alt_phrasings",
)


RETIMED_COARSE_STATUSES = (AlignStatus.UNMATCHED.value, AlignStatus.COLLISION.value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def _validate_timed_rows(
    rows: Sequence[Mapping[str, Any]],
    duration_seconds: float,
    name: str,
) -> None:
    if not rows:
        raise ValueError(f"{name} must be non-empty")
    previous_start = -math.inf
    tolerance = 1e-6
    for index, row in enumerate(rows):
        start = float(row["start"])
        end = float(row["end"])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"{name} row {index} contains non-finite timestamps")
        if (
            start < -tolerance
            or end + tolerance < start
            or end > duration_seconds + tolerance
        ):
            raise ValueError(f"{name} row {index} is outside the canonical timeline")
        if start + tolerance < previous_start:
            raise ValueError(f"{name} row {index} is not timestamp-ordered")
        if not str(row["text"]).strip():
            raise ValueError(f"{name} row {index} has empty text")
        previous_start = start


def _validate_unique_chunk_starts(
    chunks: Sequence[Mapping[str, object]],
    name: str,
) -> None:
    """Reject overlap-window variants with the same coarse source start."""
    seen: set[float] = set()
    for index, chunk in enumerate(chunks):
        start = float(chunk["start"])
        if start in seen:
            raise ValueError(f"{name} has duplicate start time at row {index}")
        seen.add(start)


def _validate_retimed_row(
    video_id: str,
    index: int,
    row: Mapping[str, Any],
    cleaned: Mapping[str, Any],
) -> None:
    """Check that one re-timed row only moved its own start and end.

    :param video_id: the video the row belongs to.
    :param index: the row's position, used in error messages.
    :param row: the re-timed row.
    :param cleaned: the cleaned chunk carrying the same ``chunk_id``.
    :return: None; raises ValueError when the row departs from its cleaned source.
    """
    status = str(row["align_status"])
    if status not in RETIMED_ALIGN_STATUSES:
        raise ValueError(f"{video_id} retimed row {index} has unknown status {status}")
    if float(row["coarse_start"]) != float(cleaned["start"]) or float(
        row["coarse_end"]
    ) != float(cleaned["end"]):
        raise ValueError(f"{video_id} retimed row {index} coarse times differ from cleaned")
    if any(row[field] != cleaned[field] for field in RETIMED_PRESERVED_FIELDS):
        raise ValueError(f"{video_id} retimed row {index} altered a cleaned field")
    if status in RETIMED_COARSE_STATUSES and (
        float(row["start"]) != float(row["coarse_start"])
        or float(row["end"]) != float(row["coarse_end"])
    ):
        raise ValueError(f"{video_id} retimed row {index} is {status} but was re-timed")
    if status == AlignStatus.ALIGNED.value:
        _validate_aligned_row(video_id, index, row)


def _validate_aligned_row(video_id: str, index: int, row: Mapping[str, Any]) -> None:
    """Check that an aligned row's new time is credible.

    The sidecars are not hash-pinned, so this is the only defence against a producer
    bug that re-times a chunk to somewhere it could not have been matched.

    :param video_id: the video the row belongs to.
    :param index: the row's position, used in error messages.
    :param row: an aligned re-timed row.
    :return: None; raises ValueError when the start left its search window, the
        recorded shift disagrees with the start, or the match ratio is below the floor.
    """
    start = float(row["start"])
    coarse_start = float(row["coarse_start"])
    low = coarse_start - SEARCH_PAD_S - RETIMED_START_SLACK_S
    high = float(row["coarse_end"]) + SEARCH_PAD_S + RETIMED_START_SLACK_S
    if not low <= start <= high:
        raise ValueError(f"{video_id} retimed row {index} moved outside its search window")
    shift = row["align_shift_s"]
    if shift is None or abs(float(shift) - (start - coarse_start)) > RETIMED_SHIFT_TOLERANCE_S:
        raise ValueError(f"{video_id} retimed row {index} shift disagrees with its start")
    ratio = row["align_match_ratio"]
    if ratio is None or float(ratio) < MIN_MATCH_RATIO:
        raise ValueError(f"{video_id} retimed row {index} is aligned below the match floor")


def load_retimed_chunks_for_video(
    commentary_root: Path,
    video_id: str,
    chunks: Sequence[Mapping[str, Any]],
    duration_seconds: float,
) -> tuple[tuple[Mapping[str, Any], ...] | None, str | None]:
    """Load one video's optional WhisperX-aligned re-timed sidecar (issue #136).

    These sidecars post-date the pinned commentary manifest, so they are hash-recorded
    rather than hash-checked. Their content is instead tied to the cleaned chunks: the
    same chunk population, the same cleaned fields, and coarse times matching the
    cleaned start and end.

    :param commentary_root: root of the commentary artifact tree.
    :param video_id: the video to load.
    :param chunks: the video's validated cleaned chunks.
    :param duration_seconds: the video's canonical duration, for timeline validation.
    :return: ``(rows, sha256)``, or ``(None, None)`` when no sidecar exists.
    """
    path = commentary_root / RETIMED_RELATIVE_DIR / f"{video_id}.json"
    if not path.is_file():
        return None, None
    if not chunks:
        raise ValueError(f"{video_id} has a retimed sidecar but no cleaned chunks")
    rows = [
        _mapping(row, f"{video_id} retimed chunk")
        for row in _sequence(
            json.loads(path.read_text(encoding="utf-8")), f"{video_id} retimed chunks"
        )
    ]
    cleaned_by_id = {str(chunk["chunk_id"]): chunk for chunk in chunks}
    if len(rows) != len(chunks) or {str(row["chunk_id"]) for row in rows} != set(
        cleaned_by_id
    ):
        raise ValueError(f"{video_id} retimed chunk population differs from cleaned")
    for index, row in enumerate(rows):
        _validate_retimed_row(
            video_id, index, row, cleaned_by_id[str(row["chunk_id"])]
        )
    _validate_timed_rows(rows, duration_seconds, f"{video_id} retimed chunks")
    _validate_unique_chunk_starts(rows, f"{video_id} retimed chunks")
    return tuple(rows), _sha256(path)
