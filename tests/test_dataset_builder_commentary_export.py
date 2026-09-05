"""Tests for dataset_builder.commentary_export (issue #18 auxiliary commentary tables
and the issue #138 commentary-to-rally link)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset_builder.commentary_export import LAG_SECONDS, commentary_tables, empty_table
from dataset_builder.schema_v1 import (
    COMMENTARY_CHUNKS,
    COMMENTARY_RALLY_LINKS,
    TRANSCRIPT_SEGMENTS,
    read_table,
    validate_table,
    write_table,
)
from dataset_builder.vision import save_npy_xz
from scraper.commentary_retiming import AlignStatus


RUN_ID = "run1"
SOURCE_DATASET = "ShuttleSet"
EMPTY_RALLIES = pd.DataFrame()


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


def _chunk(chunk_id: str, start: float, end: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "chunk_id": chunk_id, "start": start, "end": end,
        "text": f"text {chunk_id}", "text_clean": f"clean {chunk_id}",
        "bert_f1": 0.9, "clean_pass": True, "alt_phrasings": [],
    }
    row.update(overrides)
    return row


def _retimed_row(chunk: dict[str, object], *, status: AlignStatus, **overrides: object) -> dict[str, object]:
    """Build one re-timed sidecar row from a cleaned chunk, coarse by default."""
    row = dict(chunk)
    row["coarse_start"] = chunk["start"]
    row["coarse_end"] = chunk["end"]
    row["align_status"] = status.value
    row["align_shift_s"] = None
    row["align_match_ratio"] = None
    row.update(overrides)
    return row


def _write_retimed(root: Path, video_id: str, rows: list[dict[str, object]]) -> None:
    path = root / "commentary" / "retimed_chunks" / f"{video_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle)


def _rallies_frame(spans: dict[str, list[tuple[int, float, float]]]) -> pd.DataFrame:
    """One source_contacts rallies frame for _source_contact_spans, keyed as it expects."""
    return pd.DataFrame(
        [
            {
                "run_id": RUN_ID,
                "source_dataset": SOURCE_DATASET,
                "video_id": video_id,
                "rally_origin": "source_contacts",
                "rally_id": rally_id,
                "start_seconds": start,
                "end_seconds": end,
            }
            for video_id, rows in spans.items()
            for rally_id, start, end in rows
        ]
    )


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

    result = commentary_tables(
        root, RUN_ID, "ShuttleSet", ["sset_01", "sset_02"], EMPTY_RALLIES, {}, {}
    )
    segments, chunks = result.segments, result.chunks

    assert len(segments) == 3
    assert len(chunks) == 2
    assert result.links.empty
    # No sidecar for either video: behaviour is exactly the coarse-only path.
    assert result.masked_start_chunks == {"sset_01": None, "sset_02": None}
    validated_segments = validate_table(TRANSCRIPT_SEGMENTS, segments)
    validated_chunks = validate_table(COMMENTARY_CHUNKS, chunks)
    validate_table(COMMENTARY_RALLY_LINKS, result.links)

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
    assert chunks_by_id.loc["sset_01_c0", "timestamp_precision"] == "caption"
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

    result = commentary_tables(root, RUN_ID, "ShuttleSet", ["sset_03"], EMPTY_RALLIES, {}, {})
    chunks = result.chunks

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
        commentary_tables(root, RUN_ID, "ShuttleSet", ["sset_04"], EMPTY_RALLIES, {}, {})


def test_video_id_missing_from_status_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_05", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )

    with pytest.raises(ValueError, match="no row for 'sset_06'"):
        commentary_tables(root, RUN_ID, "ShuttleSet", ["sset_06"], EMPTY_RALLIES, {}, {})


def test_cleaned_chunk_missing_text_clean_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_07", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, "sset_07", "youtube_asr", [{"start": 0.0, "end": 1.0, "text": "hello"}])
    _write_chunks(root, "sset_07", [{"chunk_id": "sset_07_c0", "start": 0.0, "end": 1.0, "text": "raw"}])

    with pytest.raises(ValueError, match="lacks chunk_id or text_clean"):
        commentary_tables(root, RUN_ID, "ShuttleSet", ["sset_07"], EMPTY_RALLIES, {}, {})


def test_transcript_segment_missing_end_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    _write_status(
        root, [{"video_id": "sset_08", "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, "sset_08", "youtube_asr", [{"start": 0.0, "text": "hello"}])

    with pytest.raises(ValueError, match="must contain start, end, and text"):
        commentary_tables(root, RUN_ID, "ShuttleSet", ["sset_08"], EMPTY_RALLIES, {}, {})


def test_empty_table_has_frozen_columns_and_validates() -> None:
    empty = empty_table(TRANSCRIPT_SEGMENTS)

    assert list(empty.columns) == list(TRANSCRIPT_SEGMENTS.column_names())
    assert len(empty) == 0
    validated = validate_table(TRANSCRIPT_SEGMENTS, empty)
    assert len(validated) == 0
    assert list(validated.columns) == list(TRANSCRIPT_SEGMENTS.column_names())


# ---------------------------------------------------------------------------
# Issue #138 rally links
# ---------------------------------------------------------------------------


def _setup_video(root: Path, video_id: str, chunks: list[dict[str, object]]) -> None:
    _write_status(
        root, [{"video_id": video_id, "transcript_source": "youtube_asr", "clean_status": "valid"}]
    )
    _write_transcript(root, video_id, "youtube_asr", [{"start": 0.0, "end": 1.0, "text": "x"}])
    _write_chunks(root, video_id, chunks)


def test_rally_links_inside_post_rally_and_out_of_window(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    video_id = "sset_10"
    _setup_video(
        root,
        video_id,
        [
            _chunk("c_in", 5.0, 6.0),  # starts inside the rally [0, 10)
            _chunk("c_post", 15.0, 16.0),  # 5 s after the rally ends: within LAG_SECONDS
            _chunk("c_none", 21.0, 22.0),  # 11 s after the rally ends: outside the window
        ],
    )
    assert LAG_SECONDS == 10.0
    rallies = _rallies_frame({video_id: [(0, 0.0, 10.0)]})

    result = commentary_tables(root, RUN_ID, SOURCE_DATASET, [video_id], rallies, {}, {})
    links = result.links.set_index("chunk_id")

    assert set(links.index) == {"c_in", "c_post"}  # c_none links to nothing
    assert links.loc["c_in", "relation"] == "inside"
    assert links.loc["c_in", "lag_seconds"] == pytest.approx(0.0)
    assert not links.loc["c_in", "ambiguous"]
    assert links.loc["c_in", "starts_on_masked_frame"] is None  # no replay mask root given

    assert links.loc["c_post", "relation"] == "post_rally"
    assert links.loc["c_post", "lag_seconds"] == pytest.approx(5.0)
    assert not links.loc["c_post", "ambiguous"]
    assert result.ambiguous_links[video_id] == 0
    assert result.multi_rally_chunks[video_id] == 0


def test_rally_link_ambiguous_when_chunk_matches_two_rallies(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    video_id = "sset_11"
    # Rally A ends at 10 s; rally B runs [14, 20). A chunk starting at 14 s starts
    # inside B and also lands within LAG_SECONDS (4 s) of A's end.
    _setup_video(root, video_id, [_chunk("c_both", 14.0, 15.0)])
    rallies = _rallies_frame({video_id: [(0, 0.0, 10.0), (1, 14.0, 20.0)]})

    result = commentary_tables(root, RUN_ID, SOURCE_DATASET, [video_id], rallies, {}, {})
    links = result.links.set_index("rally_id")

    assert len(links) == 2
    assert links.loc[0, "ambiguous"]
    assert links.loc[1, "ambiguous"]
    assert links.loc[0, "relation"] == "post_rally"
    assert links.loc[0, "lag_seconds"] == pytest.approx(4.0)
    assert links.loc[1, "relation"] == "inside"
    assert links.loc[1, "lag_seconds"] == pytest.approx(0.0)
    # One ambiguous chunk contributes one link row per rally it matches.
    assert result.ambiguous_links[video_id] == 2
    assert result.multi_rally_chunks[video_id] == 1


def test_retimed_chunks_get_aligned_precision_unmatched_do_not(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    video_id = "sset_12"
    aligned_chunk = _chunk("c_aligned", 10.0, 12.0)
    unmatched_chunk = _chunk("c_unmatched", 20.0, 22.0)
    _write_status(
        root,
        [
            {
                "video_id": video_id,
                "transcript_source": "youtube_asr",
                "clean_status": "valid",
                "local_duration_s": 120.0,
            }
        ],
    )
    _write_transcript(root, video_id, "youtube_asr", [{"start": 0.0, "end": 1.0, "text": "x"}])
    _write_chunks(root, video_id, [aligned_chunk, unmatched_chunk])
    _write_retimed(
        root,
        video_id,
        [
            _retimed_row(
                aligned_chunk, status=AlignStatus.ALIGNED,
                start=10.5, end=12.5, align_shift_s=0.5, align_match_ratio=0.9,
            ),
            _retimed_row(unmatched_chunk, status=AlignStatus.UNMATCHED),
        ],
    )

    result = commentary_tables(root, RUN_ID, SOURCE_DATASET, [video_id], EMPTY_RALLIES, {}, {})
    chunks = result.chunks.set_index("chunk_id")

    assert chunks.loc["c_aligned", "timestamp_precision"] == "whisperx_aligned"
    assert chunks.loc["c_aligned", "start_seconds"] == pytest.approx(10.5)
    assert chunks.loc["c_aligned", "end_seconds"] == pytest.approx(12.5)

    # Unmatched: coarse times and the original caption precision survive unchanged.
    assert chunks.loc["c_unmatched", "timestamp_precision"] == "caption"
    assert chunks.loc["c_unmatched", "start_seconds"] == pytest.approx(20.0)
    assert chunks.loc["c_unmatched", "end_seconds"] == pytest.approx(22.0)


def test_missing_retimed_sidecar_leaves_timing_unchanged(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    video_id = "sset_13"
    _setup_video(root, video_id, [_chunk("c0", 10.0, 12.0)])
    # No commentary/retimed_chunks/sset_13.json is written.

    result = commentary_tables(root, RUN_ID, SOURCE_DATASET, [video_id], EMPTY_RALLIES, {}, {})
    chunk = result.chunks.set_index("chunk_id").loc["c0"]

    assert chunk["timestamp_precision"] == "caption"
    assert chunk["start_seconds"] == pytest.approx(10.0)
    assert chunk["end_seconds"] == pytest.approx(12.0)


def test_replay_mask_flags_masked_start_when_given(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    mask_root = tmp_path / "replay_masks"
    video_id = "sset_14"
    fps = 25.0
    # Chunk at 2.0 s is frame 50; chunk at 8.0 s is frame 200. Only frame 50 is masked.
    _setup_video(
        root,
        video_id,
        [_chunk("c_masked", 2.0, 3.0), _chunk("c_clear", 8.0, 9.0)],
    )
    rallies = _rallies_frame({video_id: [(0, 0.0, 100.0)]})  # both chunks link "inside"
    mask = np.zeros(500, dtype=np.bool_)
    mask[50] = True
    save_npy_xz(mask_root / video_id / "definitive_exclusion_mask.npy.xz", mask)

    result = commentary_tables(
        root, RUN_ID, SOURCE_DATASET, [video_id], rallies, {video_id: fps}, {video_id: 500},
        replay_mask_root=mask_root,
    )
    links = result.links.set_index("chunk_id")

    assert links.loc["c_masked", "starts_on_masked_frame"]
    assert not links.loc["c_clear", "starts_on_masked_frame"]
    assert result.masked_start_chunks[video_id] == 1


def test_replay_mask_root_given_but_video_file_missing_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    mask_root = tmp_path / "replay_masks"
    mask_root.mkdir()
    video_id = "sset_15"
    _setup_video(root, video_id, [_chunk("c0", 2.0, 3.0)])

    with pytest.raises(FileNotFoundError):
        commentary_tables(
            root, RUN_ID, SOURCE_DATASET, [video_id], EMPTY_RALLIES, {video_id: 25.0}, {},
            replay_mask_root=mask_root,
        )


def test_replay_mask_length_mismatch_raises(tmp_path: Path) -> None:
    root = tmp_path / "commentary_root"
    mask_root = tmp_path / "replay_masks"
    video_id = "sset_16"
    _setup_video(root, video_id, [_chunk("c0", 2.0, 3.0)])
    save_npy_xz(
        mask_root / video_id / "definitive_exclusion_mask.npy.xz",
        np.zeros(500, dtype=np.bool_),
    )

    with pytest.raises(ValueError, match=f"{video_id} replay mask length differs"):
        commentary_tables(
            root, RUN_ID, SOURCE_DATASET, [video_id], EMPTY_RALLIES, {video_id: 25.0},
            {video_id: 400},  # video's real frame_count does not match the 500-frame mask
            replay_mask_root=mask_root,
        )
