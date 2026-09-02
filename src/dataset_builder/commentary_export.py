"""Auxiliary commentary tables for the v1 export.

Reads the commentary preparation layout used by the issue #104 commentary
benchmark: ``transcripts/<video_id>.json``, ``commentary/cleaned_chunks/
<video_id>.json``, and ``status/commentary_per_video_status.json``. The rows
stay tied to the video. Rally association is cut from v1.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

import pandas as pd

from dataset_builder.schema_v1 import COMMENTARY_CHUNKS, TRANSCRIPT_SEGMENTS, TableSpec


TRANSCRIPTS_DIRECTORY = "transcripts"
CLEANED_CHUNKS_DIRECTORY = "commentary/cleaned_chunks"
STATUS_PATH = "status/commentary_per_video_status.json"
VALID_CLEAN_STATUS = "valid"
TIMESTAMP_PRECISION_BY_SOURCE = {"youtube_asr": "caption", "whisper": "whisperx_coarse"}


def empty_table(table: TableSpec) -> pd.DataFrame:
    """Return a zero-row frame with the table's frozen columns."""
    return pd.DataFrame(columns=list(table.column_names()))


def commentary_tables(
    commentary_root: Path,
    source_dataset: str,
    video_ids: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return unvalidated transcript-segment and commentary-chunk frames."""
    root = Path(commentary_root)
    status_by_id = _load_status(root / STATUS_PATH)
    segment_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    for video_id in video_ids:
        status = status_by_id.get(video_id)
        if status is None:
            raise ValueError(f"commentary status has no row for {video_id!r}")
        precision = _timestamp_precision(status, video_id)
        transcript = _load_transcript(root / TRANSCRIPTS_DIRECTORY / f"{video_id}.json", video_id)
        for index, segment in enumerate(transcript):
            segment_rows.append(
                {
                    "source_dataset": source_dataset,
                    "video_id": video_id,
                    "segment_index": index,
                    "timestamp_precision": precision,
                    "start_seconds": float(segment["start"]),
                    "end_seconds": float(segment["end"]),
                    "text": str(segment["text"]),
                }
            )
        for chunk in _load_chunks(root, video_id, status):
            chunk_rows.append(
                {
                    "source_dataset": source_dataset,
                    "video_id": video_id,
                    "chunk_id": str(chunk["chunk_id"]),
                    "timestamp_precision": precision,
                    "start_seconds": float(chunk["start"]),
                    "end_seconds": float(chunk["end"]),
                    "text": str(chunk["text"]),
                    "text_clean": str(chunk["text_clean"]),
                    "bert_f1": _optional_float(chunk.get("bert_f1")),
                    "clean_pass": _optional_bool(chunk.get("clean_pass")),
                }
            )
    segments = (
        pd.DataFrame(segment_rows) if segment_rows else empty_table(TRANSCRIPT_SEGMENTS)
    )
    chunks = pd.DataFrame(chunk_rows) if chunk_rows else empty_table(COMMENTARY_CHUNKS)
    return segments, chunks


def _load_status(path: Path) -> dict[str, Mapping[str, object]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("records"), list):
        raise ValueError(f"commentary status must contain a records list: {path}")
    status_by_id: dict[str, Mapping[str, object]] = {}
    for row in payload["records"]:
        if not isinstance(row, Mapping) or "video_id" not in row:
            raise ValueError(f"commentary status rows must name a video_id: {path}")
        video_id = str(row["video_id"])
        if video_id in status_by_id:
            raise ValueError(f"commentary status repeats {video_id!r}")
        status_by_id[video_id] = row
    return status_by_id


def _timestamp_precision(status: Mapping[str, object], video_id: str) -> str:
    source = status.get("transcript_source")
    precision = TIMESTAMP_PRECISION_BY_SOURCE.get(str(source))
    if precision is None:
        raise ValueError(f"{video_id} has unsupported transcript_source {source!r}")
    return precision


def _load_transcript(path: Path, video_id: str) -> list[Mapping[str, object]]:
    payload = _read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("segments"), list):
        raise ValueError(f"{video_id} transcript must contain a segments list")
    return [_timed_row(segment, f"{video_id} transcript segment") for segment in payload["segments"]]


def _load_chunks(
    root: Path, video_id: str, status: Mapping[str, object]
) -> list[Mapping[str, object]]:
    if status.get("clean_status") != VALID_CLEAN_STATUS:
        return []
    payload = _read_json(root / CLEANED_CHUNKS_DIRECTORY / f"{video_id}.json")
    if not isinstance(payload, list):
        raise ValueError(f"{video_id} cleaned chunks must be a list")
    chunks = [_timed_row(chunk, f"{video_id} cleaned chunk") for chunk in payload]
    for chunk in chunks:
        if "chunk_id" not in chunk or "text_clean" not in chunk:
            raise ValueError(f"{video_id} cleaned chunk lacks chunk_id or text_clean")
    return chunks


def _timed_row(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not {"start", "end", "text"} <= set(value):
        raise ValueError(f"{name} must contain start, end, and text")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"bert_f1 must be a number: {value!r}")
    return float(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"clean_pass must be boolean: {value!r}")
    return value


def _read_json(path: Path) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))
