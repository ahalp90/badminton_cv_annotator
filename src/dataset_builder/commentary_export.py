"""Auxiliary commentary tables for the v1 export.

Reads the commentary preparation layout used by the issue #104 commentary
benchmark: ``transcripts/<video_id>.json``, ``commentary/cleaned_chunks/
<video_id>.json``, ``status/commentary_per_video_status.json``, and the
optional issue #136 re-timed sidecars under ``commentary/retimed_chunks/``.
The segments and chunks stay tied to the video. ``commentary_rally_links``
additionally links each chunk to the ``source_contacts`` rallies it plausibly
refers to, under Ari's issue #138 pairing rule (see ``LAG_SECONDS``). That
link's coverage is measured; its accuracy is not, so issue #104 keeps the
disposition at unresolved rather than keep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.calibration.commentary_benchmark_inputs import (
    RETIMED_RELATIVE_DIR,
    load_retimed_chunks_for_video,
)
from dataset_builder.schema_v1 import (
    COMMENTARY_CHUNKS,
    COMMENTARY_RALLY_LINKS,
    TRANSCRIPT_SEGMENTS,
    CommentaryRelation,
    RallyOrigin,
    TableSpec,
)
from dataset_builder.vision import load_npy_xz
from scraper.commentary_pairing import chunk_start_on_mask
from scraper.commentary_retiming import AlignStatus


TRANSCRIPTS_DIRECTORY = "transcripts"
CLEANED_CHUNKS_DIRECTORY = "commentary/cleaned_chunks"
STATUS_PATH = "status/commentary_per_video_status.json"
VALID_CLEAN_STATUS = "valid"
TIMESTAMP_PRECISION_BY_SOURCE = {"youtube_asr": "caption", "whisper": "whisperx_coarse"}
ALIGNED_TIMESTAMP_PRECISION = "whisperx_aligned"
REPLAY_MASK_FILENAME = "definitive_exclusion_mask.npy.xz"

# Ari's issue #138 rule: a chunk links to the rally it starts inside, plus every rally
# that ended within this many seconds before it. Swept on the issue #136 aligned times
# over 6,156 combined human rallies (docs/dataset_builder/issue_104_shuttleset_benchmark.md,
# "Aligned rerun, issue 136"): 8 s covers 34.4% with 0 ambiguous chunks, 10 s covers 36.1%
# with 0 ambiguous, 15 s covers 39.6% with 8 ambiguous, 20 s covers 42.5% with 97
# ambiguous. 10 s is the widest window with zero ambiguity, and coverage keeps climbing
# past it with no natural knee, so wider is a judgement call this project does not need
# to make yet.
#
# That zero-ambiguity count comes from commentary_benchmark.py's _multi_rally_counts,
# which skips any chunk starting on a masked frame and removes masked rallies from the
# candidate set before counting ambiguity. This exporter keeps both: a masked chunk still
# gets a link, and a masked rally still counts as a candidate. So the swept figure
# describes the benchmark's filtered population, not the population this table ships;
# see the per-video ambiguity counts this module adds to the manifest for the shipped
# number.
LAG_SECONDS = 10.0


@dataclass(frozen=True)
class CommentaryTables:
    """The three auxiliary commentary frames plus per-video link counts."""

    segments: pd.DataFrame
    chunks: pd.DataFrame
    links: pd.DataFrame
    # video_id -> masked-start chunk count, or None when no replay mask was given.
    masked_start_chunks: dict[str, int | None]
    # video_id -> count of commentary_rally_links rows with ambiguous == True, on the
    # shipped population (masked chunks and masked rallies both still counted).
    ambiguous_links: dict[str, int]
    # video_id -> count of distinct chunks that link to more than one rally, same
    # population as ambiguous_links but counted per chunk rather than per link row.
    multi_rally_chunks: dict[str, int]


def empty_table(table: TableSpec) -> pd.DataFrame:
    """Return a zero-row frame with the table's frozen columns."""
    return pd.DataFrame(columns=list(table.column_names()))


def commentary_tables(
    commentary_root: Path,
    run_id: str,
    source_dataset: str,
    video_ids: Sequence[str],
    rallies: pd.DataFrame,
    fps_by_video: Mapping[str, float],
    frame_count_by_video: Mapping[str, int],
    *,
    replay_mask_root: Path | None = None,
) -> CommentaryTables:
    """Return unvalidated segment, chunk, and rally-link frames for one export.

    :param commentary_root: root of the commentary preparation tree.
    :param run_id: the export's run identifier, carried by every link row.
    :param source_dataset: dataset label, for example ShuttleSet.
    :param video_ids: videos to read, in the export's order.
    :param rallies: the export's already-assembled rallies frame, used to find each
        video's source_contacts rally spans.
    :param fps_by_video: each video's frame rate, used to test a chunk start against
        the optional replay mask.
    :param frame_count_by_video: each video's decoded frame count, used to check the
        optional replay mask's length.
    :param replay_mask_root: root with one ``<video_id>/definitive_exclusion_mask.npy.xz``
        per video, or None to leave every chunk's mask flag null.
    """
    root = Path(commentary_root)
    status_by_id = _load_status(root / STATUS_PATH)
    rallies_by_video = _source_contact_spans(rallies, run_id, source_dataset)
    segment_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    masked_start_chunks: dict[str, int | None] = {}
    ambiguous_links: dict[str, int] = {}
    multi_rally_chunks: dict[str, int] = {}
    for video_id in video_ids:
        status = status_by_id.get(video_id)
        if status is None:
            raise ValueError(f"commentary status has no row for {video_id!r}")
        video_rows = _video_commentary_rows(
            root,
            run_id,
            source_dataset,
            video_id,
            status,
            rallies_by_video.get(video_id, {}),
            fps_by_video,
            frame_count_by_video,
            replay_mask_root,
        )
        segment_rows.extend(video_rows.segment_rows)
        chunk_rows.extend(video_rows.chunk_rows)
        link_rows.extend(video_rows.link_rows)
        masked_start_chunks[video_id] = video_rows.masked_count
        ambiguous_links[video_id] = video_rows.ambiguous_row_count
        multi_rally_chunks[video_id] = video_rows.multi_rally_chunk_count

    segments = (
        pd.DataFrame(segment_rows) if segment_rows else empty_table(TRANSCRIPT_SEGMENTS)
    )
    chunks_table = pd.DataFrame(chunk_rows) if chunk_rows else empty_table(COMMENTARY_CHUNKS)
    links_table = (
        pd.DataFrame(link_rows) if link_rows else empty_table(COMMENTARY_RALLY_LINKS)
    )
    return CommentaryTables(
        segments, chunks_table, links_table, masked_start_chunks, ambiguous_links, multi_rally_chunks
    )


@dataclass(frozen=True)
class _VideoCommentaryRows:
    """One video's segment, chunk, and rally-link rows, plus its link counts."""

    segment_rows: list[dict[str, object]]
    chunk_rows: list[dict[str, object]]
    link_rows: list[dict[str, object]]
    masked_count: int | None
    ambiguous_row_count: int
    multi_rally_chunk_count: int


def _video_commentary_rows(
    root: Path,
    run_id: str,
    source_dataset: str,
    video_id: str,
    status: Mapping[str, object],
    spans: Mapping[int, tuple[float, float]],
    fps_by_video: Mapping[str, float],
    frame_count_by_video: Mapping[str, int],
    replay_mask_root: Path | None,
) -> _VideoCommentaryRows:
    """Build one video's rows and per-video link counts.

    :param spans: the video's source_contacts rally spans, from ``_source_contact_spans``.
    """
    precision = _timestamp_precision(status, video_id)
    transcript = _load_transcript(root / TRANSCRIPTS_DIRECTORY / f"{video_id}.json", video_id)
    segment_rows = [
        {
            "source_dataset": source_dataset,
            "video_id": video_id,
            "segment_index": index,
            "timestamp_precision": precision,
            "start_seconds": float(segment["start"]),
            "end_seconds": float(segment["end"]),
            "text": str(segment["text"]),
        }
        for index, segment in enumerate(transcript)
    ]

    chunks = _load_chunks(root, video_id, status)
    retimed_by_id = _retimed_chunks_by_id(root, video_id, chunks, status)
    mask = (
        None
        if replay_mask_root is None
        else _load_replay_mask(replay_mask_root, video_id, frame_count_by_video.get(video_id))
    )
    masked_count: int | None = 0 if mask is not None else None
    ambiguous_row_count = 0
    multi_rally_chunk_count = 0
    chunk_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])
        start, end, chunk_precision = _effective_timing(
            chunk, retimed_by_id.get(chunk_id), precision
        )
        chunk_rows.append(
            {
                "source_dataset": source_dataset,
                "video_id": video_id,
                "chunk_id": chunk_id,
                "timestamp_precision": chunk_precision,
                "start_seconds": start,
                "end_seconds": end,
                "text": str(chunk["text"]),
                "text_clean": str(chunk["text_clean"]),
                "bert_f1": _optional_float(chunk.get("bert_f1")),
                "clean_pass": _optional_bool(chunk.get("clean_pass")),
            }
        )

        on_mask: bool | None = None
        if mask is not None:
            fps = fps_by_video.get(video_id)
            if fps is None:
                raise ValueError(f"{video_id} has a replay mask but no known fps")
            on_mask = chunk_start_on_mask(start, fps, mask)
            masked_count = (masked_count or 0) + int(on_mask)

        links = _rally_links(start, spans)
        ambiguous = len(links) > 1
        if ambiguous:
            multi_rally_chunk_count += 1
            ambiguous_row_count += len(links)
        for rally_id, relation, lag_seconds in links:
            link_rows.append(
                {
                    "run_id": run_id,
                    "source_dataset": source_dataset,
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "rally_origin": RallyOrigin.SOURCE_CONTACTS.value,
                    "rally_id": rally_id,
                    "relation": relation,
                    "lag_seconds": lag_seconds,
                    "ambiguous": ambiguous,
                    "starts_on_masked_frame": on_mask,
                }
            )

    return _VideoCommentaryRows(
        segment_rows, chunk_rows, link_rows, masked_count, ambiguous_row_count, multi_rally_chunk_count
    )


def _rally_links(
    start_seconds: float, spans: Mapping[int, tuple[float, float]]
) -> list[tuple[int, str, float]]:
    """Rallies one chunk links to under the issue #138 rule.

    :param start_seconds: the chunk's effective start on the video timeline.
    :param spans: ``{rally_id: (start_seconds, end_seconds)}`` for the video's
        source_contacts rallies.
    :return: ``[(rally_id, relation, lag_seconds), ...]``; empty when the chunk
        matches no rally.
    """
    links = [
        (rally_id, CommentaryRelation.INSIDE.value, 0.0)
        for rally_id, (span_start, span_end) in spans.items()
        if span_start <= start_seconds < span_end
    ]
    links.extend(
        (rally_id, CommentaryRelation.POST_RALLY.value, start_seconds - span_end)
        for rally_id, (_span_start, span_end) in spans.items()
        if span_end < start_seconds <= span_end + LAG_SECONDS
    )
    return links


def _source_contact_spans(
    rallies: pd.DataFrame, run_id: str, source_dataset: str
) -> dict[str, dict[int, tuple[float, float]]]:
    """Return ``{video_id: {rally_id: (start_seconds, end_seconds)}}`` for one run's
    source_contacts rallies."""
    if rallies.empty:
        return {}
    subset = rallies[
        (rallies["run_id"] == run_id)
        & (rallies["source_dataset"] == source_dataset)
        & (rallies["rally_origin"] == RallyOrigin.SOURCE_CONTACTS.value)
    ]
    spans: dict[str, dict[int, tuple[float, float]]] = {}
    for row in subset.itertuples(index=False):
        spans.setdefault(str(row.video_id), {})[int(row.rally_id)] = (
            float(row.start_seconds),
            float(row.end_seconds),
        )
    return spans


def _effective_timing(
    chunk: Mapping[str, object],
    retimed: Mapping[str, object] | None,
    base_precision: str,
) -> tuple[float, float, str]:
    """Return a chunk's ``(start_seconds, end_seconds, timestamp_precision)``.

    Without a re-timed row, the coarse chunk times and base precision are unchanged.
    An aligned row's start and end already hold the aligned times; an unmatched or
    collision row's start and end already hold the original coarse times, so using
    ``retimed`` there changes nothing but the precision label, which also stays put.
    """
    if retimed is None:
        return float(chunk["start"]), float(chunk["end"]), base_precision
    precision = (
        ALIGNED_TIMESTAMP_PRECISION
        if str(retimed["align_status"]) == AlignStatus.ALIGNED.value
        else base_precision
    )
    return float(retimed["start"]), float(retimed["end"]), precision


def _retimed_chunks_by_id(
    root: Path, video_id: str, chunks: Sequence[Mapping[str, object]], status: Mapping[str, object]
) -> dict[str, Mapping[str, object]]:
    """Return the video's validated re-timed rows by chunk_id, or ``{}`` with no sidecar."""
    if not (root / RETIMED_RELATIVE_DIR / f"{video_id}.json").is_file():
        return {}
    duration = status.get("local_duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise ValueError(
            f"{video_id} has a retimed sidecar but no numeric local_duration_s in its status row"
        )
    rows, _sha256 = load_retimed_chunks_for_video(root, video_id, chunks, float(duration))
    return {} if rows is None else {str(row["chunk_id"]): row for row in rows}


def _load_replay_mask(replay_mask_root: Path, video_id: str, frame_count: int | None) -> np.ndarray:
    """Load one video's replay mask; raise if the root was given but the file is not,
    or if the mask's length does not match the video's frame count."""
    path = Path(replay_mask_root) / video_id / REPLAY_MASK_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{video_id} has no replay mask at {path}")
    mask = load_npy_xz(path)
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError(f"{video_id} replay mask must be one-dimensional boolean")
    if frame_count is None:
        raise ValueError(f"{video_id} has a replay mask but no known frame_count")
    if len(mask) != frame_count:
        raise ValueError(f"{video_id} replay mask length differs from its frame_count")
    return mask


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
