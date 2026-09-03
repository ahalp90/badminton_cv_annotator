"""Tests for dataset_builder.commentary_export (issue #18 auxiliary commentary tables)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dataset_builder.commentary_export import commentary_tables, empty_table
from dataset_builder.schema_v1 import (
    COMMENTARY_CHUNKS,
    TRANSCRIPT_SEGMENTS,
    read_table,
    validate_table,
    write_table,
)


def _write_status(root: Path, records: list[dict[str, object]]) -> None:
    path = root / "status" / "commentary_per_video_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"records": records}, handle)


def _write_transcript(
    root: Path, video_id: str, source: str, segments: list[dict[str, object]]
) -> None:
    path = root / "transcripts" / f"{video_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"source": source, "segments": segments}, handle)


def _write_chunks(root: Path, video_id: str, chunks: list[dict[str, object]]) -> None:
    path = root / "commentary" / "cleaned_chunks" / f"{video_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(chunks, handle)


def test_commentary_tables_main_path(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root,
        [
            {"video_id": "sset_01", "transcript_source": "youtube_asr", "clean_status": "valid"},
            {
                "video_id": "sset_02",
                "transcript_source": "whisper",
                "clean_status": "dropped_by_relevance_triage",
            },
        ],
    )
    _write_transcript(
        root,
        "sset_01",
        "youtube_asr",
        [
            {"start": 1.0, "end": 2.5, "text": "NA"},
            {"start": 2.5, "end": 4.0, "text": "great rally here"},
        ],
    )
    _write_transcript(
        root, "sset_02", "whisper", [{"start": 0.0, "end": 1.0, "text": "only segment"}]
    )
    _write_chunks(
        root,
        "sset_01",
        [
            {
                "chunk_id": "sset_01_c0",
                "start": 3.0,
                "end": 4.0,
                "text": "raw0",
                "text_clean": "clean0",
                "alt_phrasings": [],
                "bert_f1": 0.95,
                "clean_pass": True,
            },
            {
                "chunk_id": "sset_01_c1",
                "start": 4.0,
                "end": 5.0,
                "text": "raw1",
                "text_clean": "clean1",
                "alt_phrasings": ["alt"],
                "bert_f1": 0.80,
                "clean_pass": False,
            },
        ],
    )
    # sset_02 has no cleaned_chunks file: clean_status is not "valid", so none is read.

    segments, chunks = commentary_tables(root, "ShuttleSet", ["sset_01", "sset_02"])

    assert len(segments) == 3
    assert len(chunks) == 2
    validated_segments = validate_table(TRANSCRIPT_SEGMENTS, segments)
    validated_chunks = validate_table(COMMENTARY_CHUNKS, chunks)

    precision = validated_segments.set_index(["video_id", "segment_index"])["timestamp_precision"]
    assert precision["sset_01", 0] == "caption"
    assert precision["sset_01", 1] == "caption"
    assert precision["sset_02", 0] == "whisperx_coarse"
    for _video_id, group in validated_segments.groupby("video_id", sort=False):
        assert list(group["segment_index"]) == list(range(len(group)))

    chunks_by_id = validated_chunks.set_index("chunk_id")
    assert chunks_by_id.loc["sset_01_c0", "text"] == "raw0"
    assert chunks_by_id.loc["sset_01_c0", "text_clean"] == "clean0"
    assert chunks_by_id.loc["sset_01_c0", "bert_f1"] == pytest.approx(0.95)
    assert chunks_by_id.loc["sset_01_c0", "clean_pass"]
    assert chunks_by_id.loc["sset_01_c1", "bert_f1"] == pytest.approx(0.80)
    assert not chunks_by_id.loc["sset_01_c1", "clean_pass"]

    # A literal "NA" text value must not be corrupted to null through a write/read round trip.
    write_table(tmp_path / "export", TRANSCRIPT_SEGMENTS, segments)
    reread = read_table(tmp_path / "export", TRANSCRIPT_SEGMENTS)
    na_row = reread[(reread["video_id"] == "sset_01") & (reread["segment_index"] == 0)].iloc[0]
    assert na_row["text"] == "NA"


def test_commentary_chunk_missing_optional_fields_yields_na(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_03", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, "sset_03", "youtube_asr", [{"start": 0.0, "end": 1.0, "text": "hello"}])
    _write_chunks(
        root,
        "sset_03",
        [{"chunk_id": "sset_03_c0", "start": 0.0, "end": 1.0, "text": "raw", "text_clean": "clean"}],
    )

    _, chunks = commentary_tables(root, "ShuttleSet", ["sset_03"])

    assert pd.isna(chunks.loc[0, "bert_f1"])
    assert pd.isna(chunks.loc[0, "clean_pass"])
    validated = validate_table(COMMENTARY_CHUNKS, chunks)
    assert pd.isna(validated.loc[0, "bert_f1"])
    assert pd.isna(validated.loc[0, "clean_pass"])


def test_unknown_transcript_source_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_04", "transcript_source": "srt_file", "clean_status": "valid"}]
    )

    with pytest.raises(ValueError, match="unsupported transcript_source"):
        commentary_tables(root, "ShuttleSet", ["sset_04"])


def test_video_id_missing_from_status_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_05", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )

    with pytest.raises(ValueError, match="no row for 'sset_06'"):
        commentary_tables(root, "ShuttleSet", ["sset_06"])


def test_cleaned_chunk_missing_text_clean_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_07", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, "sset_07", "youtube_asr", [{"start": 0.0, "end": 1.0, "text": "hello"}])
    _write_chunks(root, "sset_07", [{"chunk_id": "sset_07_c0", "start": 0.0, "end": 1.0, "text": "raw"}])

    with pytest.raises(ValueError, match="lacks chunk_id or text_clean"):
        commentary_tables(root, "ShuttleSet", ["sset_07"])


def test_transcript_segment_missing_end_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_08", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, "sset_08", "youtube_asr", [{"start": 0.0, "text": "hello"}])

    with pytest.raises(ValueError, match="must contain start, end, and text"):
        commentary_tables(root, "ShuttleSet", ["sset_08"])


def test_empty_table_has_frozen_columns_and_validates() -> None:
    empty = empty_table(TRANSCRIPT_SEGMENTS)

    assert list(empty.columns) == list(TRANSCRIPT_SEGMENTS.column_names())
    assert len(empty) == 0
    validated = validate_table(TRANSCRIPT_SEGMENTS, empty)
    assert len(validated) == 0
    assert list(validated.columns) == list(TRANSCRIPT_SEGMENTS.column_names())
