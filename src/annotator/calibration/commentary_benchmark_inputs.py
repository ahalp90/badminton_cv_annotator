"""Load and validate the pinned issue #104 commentary and rally inputs.

This module owns the input side of the commentary benchmark: hash-checking
the pinned commentary, issue #103, and ShuttleSet/ShuttleSet22 artifacts, then
assembling one validated :class:`VideoInputs` per canonical video.
``commentary_benchmark`` consumes :func:`_load_video_inputs` and evaluates the
result; it does not read these artifacts itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from annotator.calibration.scoring import load_gt_rallies
from annotator.calibration.shuttleset22_features import load_annotation_rallies
from dataset_builder.vision import load_json_gz, load_npy_xz
from scraper.commentary_retiming import MIN_MATCH_RATIO, SEARCH_PAD_S, AlignStatus


RETIMED_RELATIVE_DIR = "commentary/retimed_chunks"
RETIMED_ALIGN_STATUSES = tuple(status.value for status in AlignStatus)
# An aligned start may sit up to SEARCH_PAD_S before the coarse start or after the coarse
# end, plus a little slack for words the aligner timed slightly out of order.
RETIMED_START_SLACK_S = 2.0
RETIMED_SHIFT_TOLERANCE_S = 1e-6
# A re-timed row may only move start/end; every other cleaned field must survive intact.
RETIMED_PRESERVED_FIELDS = (
    "text",
    "text_clean",
    "bert_f1",
    "clean_pass",
    "alt_phrasings",
)
# Statuses that kept the coarse times because alignment failed or collided.
RETIMED_COARSE_STATUSES = (AlignStatus.UNMATCHED.value, AlignStatus.COLLISION.value)

COMMENTARY_CODE_COMMIT = "819d3075e72966a3d80eb454202b83b3810225ae"
COMMENTARY_PROVIDER = "openrouter"
COMMENTARY_MODEL = "google/gemma-4-31b-it"
COMMENTARY_MANIFEST_SHA256 = (
    "96ac531c8312bf52ed0946e46ac6ca4441cae0d9d385840bdde69fd0cbb8b167"
)
COMMENTARY_STATUS_SHA256 = (
    "bedea7dcac94783625d75350ae75f9f2975ef33c5acde4154bf20963a6f7ca36"
)
COMMENTARY_SOURCE_MANIFEST_SHA256 = (
    "52a9933bcfd8d4d1cf7c032473181fa637bd296f073fbf09ff69ab0f5334342c"
)
COMMENTARY_REMOVED_OVERLAP_ROWS = 533
COMMENTARY_INVENTORY_SHA256 = (
    "d6adc338cd7a568eca83d82745edd34ba1a761181c9e2d828d6975194369a65a"
)
ISSUE103_RALLY_RECORDS_SHA256 = (
    "71c54a7a7521871c152acedd46b399c86e78969b24949b35f6f4bda59567409c"
)
ISSUE103_RUN_MANIFEST_SHA256 = (
    "84f91c139decdc4fe29957b8dd56cdd400491ba2b5aa190684fd3aa0e84a55db"
)
SHUTTLESET_SHOTS_MASTER_SHA256 = (
    "569dc74bbbb5d015a1e0be93b2c9a0885603eb320555028f11b9d259c79ee79f"
)
EXPECTED_SHUTTLESET_IDS = {
    "sset_01",
    "sset_02",
    "sset_03",
    "sset_04",
    "sset_05",
    "sset_06",
    "sset_07",
    "sset_08",
    "sset_11",
    "sset_13",
    "sset_14",
    "sset_15",
    "sset_16",
    "sset_17",
    "sset_18",
    "sset_19",
    "sset_20",
    "sset_21",
    "sset_22",
    "sset_23",
    "sset_24",
    "sset_25",
    "sset_26",
    "sset_28",
    "sset_29",
    "sset_30",
    "sset_31",
    "sset_32",
    "sset_33",
    "sset_34",
    "sset_35",
    "sset_36",
    "sset_37",
    "sset_38",
    "sset_39",
    "sset_40",
    "sset_41",
    "sset_42",
    "sset_43",
    "sset_44",
}
EXPECTED_SHUTTLESET22_IDS = {
    "ss22_08",
    "ss22_09",
    "ss22_10",
    "ss22_11",
    "ss22_12",
    "ss22_13",
    "ss22_15",
    "ss22_16",
    "ss22_17",
    "ss22_18",
    "ss22_19",
    "ss22_20",
    "ss22_21",
    "ss22_22",
    "ss22_23",
    "ss22_24",
    "ss22_25",
    "ss22_26",
    "ss22_27",
    "ss22_28",
    "ss22_29",
    "ss22_30",
    "ss22_31",
    "ss22_32",
    "ss22_33",
    "ss22_34",
    "ss22_35",
    "ss22_36",
    "ss22_37",
    "ss22_38",
    "ss22_39",
    "ss22_40",
    "ss22_41",
    "ss22_42",
    "ss22_43",
    "ss22_44",
    "ss22_46",
    "ss22_47",
    "ss22_48",
    "ss22_49",
    "ss22_50",
    "ss22_51",
    "ss22_52",
    "ss22_53",
    "ss22_54",
    "ss22_55",
    "ss22_57",
}


@dataclass(frozen=True)
class VideoInputs:
    """Validated commentary and rally inputs for one canonical video."""

    dataset: str
    video_id: str
    fps: float
    frame_count: int
    transcript_source: str
    transcript_segments: int
    chunks: tuple[Mapping[str, object], ...]
    rallies: tuple[tuple[int, int, int], ...]
    human_rallies: tuple[tuple[int, int, int], ...]
    replay_mask: np.ndarray | None
    annotation_population: Mapping[str, int] | None
    # WhisperX-aligned re-timing (issue #136); None when no sidecar exists for the video.
    retimed_chunks: tuple[Mapping[str, object], ...] | None = None
    retimed_sha256: str | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(name: str, path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{name} SHA-256 differs: expected {expected}, found {actual}")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return value


def _manifest_index(commentary_root: Path) -> dict[str, Mapping[str, object]]:
    manifest_path = commentary_root / "status" / "commentary_artifact_manifest.json"
    _require_sha256(
        "commentary artifact manifest", manifest_path, COMMENTARY_MANIFEST_SHA256
    )
    manifest = _mapping(
        json.loads(manifest_path.read_text(encoding="utf-8")), "commentary manifest"
    )
    _validate_repair_metadata(manifest, "commentary manifest")
    artifacts = _sequence(manifest["artifacts"], "commentary manifest artifacts")
    index: dict[str, Mapping[str, object]] = {}
    for raw_artifact in artifacts:
        artifact = _mapping(raw_artifact, "commentary manifest artifact")
        relative = str(artifact["path"])
        if relative in index:
            raise ValueError(f"duplicate commentary manifest path: {relative}")
        path = commentary_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(artifact["size_bytes"]):
            raise ValueError(f"commentary artifact size differs: {relative}")
        if _sha256(path) != artifact["sha256"]:
            raise ValueError(f"commentary artifact SHA-256 differs: {relative}")
        index[relative] = artifact
    if len(index) != int(manifest["artifact_count"]):
        raise ValueError("commentary manifest artifact count differs")
    return index


def _validate_repair_metadata(payload: Mapping[str, Any], name: str) -> None:
    repair = _mapping(payload.get("repair"), f"{name} repair")
    expected: dict[str, object] = {
        "source_manifest_sha256": COMMENTARY_SOURCE_MANIFEST_SHA256,
        "identity_policy": "one chunk per video and coarse source start",
        "removed_overlap_row_count": COMMENTARY_REMOVED_OVERLAP_ROWS,
        "affected_video_count": 86,
        "paid_request_count": 0,
    }
    if any(repair.get(field) != value for field, value in expected.items()):
        raise ValueError(f"{name} repair identity differs")


def _require_manifest_path(index: Mapping[str, object], relative: str) -> None:
    if relative not in index:
        raise ValueError(f"commentary manifest omits {relative}")


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


def _validate_rallies(
    video_id: str,
    rallies: Sequence[tuple[int, int, int]],
    frame_count: int,
    name: str,
) -> None:
    rally_ids = set()
    for rally_id, start_frame, end_frame in rallies:
        if rally_id in rally_ids:
            raise ValueError(f"{video_id} {name} repeats rally {rally_id}")
        if not 0 <= start_frame < end_frame <= frame_count:
            raise ValueError(
                f"{video_id} {name} rally {rally_id} is outside the local source"
            )
        rally_ids.add(rally_id)


def _canonical_inventory(commentary_root: Path) -> list[Mapping[str, Any]]:
    inventory_path = commentary_root / "inventory" / "source_inventory.json"
    _require_sha256(
        "commentary source inventory", inventory_path, COMMENTARY_INVENTORY_SHA256
    )
    payload = _mapping(
        json.loads(inventory_path.read_text(encoding="utf-8")), "source inventory"
    )
    records = [
        _mapping(row, "source inventory record")
        for row in _sequence(payload["records"], "source inventory records")
        if _mapping(row, "source inventory record").get("process_once") is True
    ]
    ids_by_dataset = {
        "ShuttleSet": {
            str(row["video_id"]) for row in records if row["dataset"] == "ShuttleSet"
        },
        "ShuttleSet22": {
            str(row["video_id"]) for row in records if row["dataset"] == "ShuttleSet22"
        },
    }
    if ids_by_dataset["ShuttleSet"] != EXPECTED_SHUTTLESET_IDS:
        raise ValueError("ShuttleSet commentary population differs")
    if ids_by_dataset["ShuttleSet22"] != EXPECTED_SHUTTLESET22_IDS:
        raise ValueError("ShuttleSet22 commentary population differs")
    youtube_ids = {str(row["youtube_id"]) for row in records}
    if len(records) != 87 or len(youtube_ids) != 87:
        raise ValueError("canonical commentary source identities are not unique")
    return records


def _issue103_inputs(
    rally_records_path: Path,
    issue103_artifacts: Path,
) -> tuple[
    dict[str, list[tuple[int, int, int]]], dict[str, np.ndarray], dict[str, object]
]:
    _require_sha256(
        "issue #103 rally records", rally_records_path, ISSUE103_RALLY_RECORDS_SHA256
    )
    run_manifest_path = issue103_artifacts / "run_manifest.json.gz"
    _require_sha256(
        "issue #103 run manifest", run_manifest_path, ISSUE103_RUN_MANIFEST_SHA256
    )
    collection = load_json_gz(rally_records_path)
    if collection.get("schema") != "rally-record-collection/0.2":
        raise ValueError("issue #103 rally-record schema differs")
    rallies: dict[str, list[tuple[int, int, int]]] = {
        video_id: [] for video_id in EXPECTED_SHUTTLESET_IDS
    }
    for raw_record in _sequence(collection["records"], "issue #103 rally records"):
        record = _mapping(raw_record, "issue #103 rally record")
        key = _mapping(record["key"], "issue #103 rally key")
        rally = _mapping(record["rally"], "issue #103 rally")
        video_id = str(key["video_id"])
        if video_id not in rallies:
            raise ValueError(f"issue #103 rally references unexpected video {video_id}")
        rallies[video_id].append(
            (int(rally["rally_id"]), int(rally["start_frame"]), int(rally["end_frame"]))
        )
    if sum(len(rows) for rows in rallies.values()) != 3527 or any(
        not rows for rows in rallies.values()
    ):
        raise ValueError("issue #103 rally population differs")

    run_manifest = load_json_gz(run_manifest_path)
    stages = {
        str(_mapping(stage, "run stage")["name"]): _mapping(stage, "run stage")
        for stage in _sequence(run_manifest["stages"], "run stages")
    }
    masks: dict[str, np.ndarray] = {}
    for video_id in sorted(EXPECTED_SHUTTLESET_IDS):
        stage = stages[f"annotation:{video_id}"]
        outputs = {
            str(_mapping(output, "stage output")["name"]): _mapping(
                output, "stage output"
            )
            for output in _sequence(stage["outputs"], "stage outputs")
        }
        identity = outputs["raw_replay_mask"]
        mask_path = issue103_artifacts / str(identity["path"])
        if not mask_path.is_file() or mask_path.stat().st_size != int(
            identity["size_bytes"]
        ):
            raise ValueError(f"{video_id} replay-mask identity differs")
        if _md5(mask_path) != identity["md5"]:
            raise ValueError(f"{video_id} replay-mask MD5 differs")
        mask = load_npy_xz(mask_path)
        if mask.ndim != 1 or mask.dtype != np.bool_:
            raise ValueError(f"{video_id} replay mask must be one-dimensional boolean")
        masks[video_id] = mask
    provenance = {
        "rally_records_sha256": ISSUE103_RALLY_RECORDS_SHA256,
        "run_manifest_sha256": ISSUE103_RUN_MANIFEST_SHA256,
        "run_id": collection["run_id"],
        "production_source_commit": collection["code_version"],
    }
    return rallies, masks, provenance


def _load_status_index(commentary_root: Path) -> dict[str, Mapping[str, object]]:
    """Load and validate the per-video commentary status population."""
    status_path = commentary_root / "status" / "commentary_per_video_status.json"
    _require_sha256(
        "commentary per-video status", status_path, COMMENTARY_STATUS_SHA256
    )
    status_payload = _mapping(
        json.loads(status_path.read_text(encoding="utf-8")), "commentary status"
    )
    _validate_repair_metadata(status_payload, "commentary status")
    status_rows = [
        _mapping(row, "commentary status row")
        for row in _sequence(status_payload["records"], "commentary status rows")
        if "transcript_source" in _mapping(row, "commentary status row")
    ]
    status_by_id = {str(row["video_id"]): row for row in status_rows}
    expected_status_ids = EXPECTED_SHUTTLESET_IDS | EXPECTED_SHUTTLESET22_IDS
    if (
        len(status_rows) != len(expected_status_ids)
        or len(status_by_id) != len(status_rows)
        or set(status_by_id) != expected_status_ids
    ):
        raise ValueError("commentary status population differs")
    return status_by_id


def _load_shots_master(shuttleset_ground_truth_root: Path) -> pd.DataFrame:
    """Load the pinned ShuttleSet shots table used for human-contact rallies."""
    shots_master_path = shuttleset_ground_truth_root / "shots_master.csv"
    _require_sha256(
        "ShuttleSet shots_master.csv", shots_master_path, SHUTTLESET_SHOTS_MASTER_SHA256
    )
    return pd.read_csv(shots_master_path)


def _check_video_contract(
    video_id: str, row: Mapping[str, Any], status: Mapping[str, Any]
) -> None:
    """Validate that a video's inventory row, status row, and code identity agree."""
    association_fields = {
        "dataset": str(row["dataset"]),
        "public_source_url": str(row["public_source_url"]),
        "youtube_id": str(row["youtube_id"]),
        "local_frame_count": int(row["local_frame_count"]),
        "local_fps": str(row["local_fps"]),
    }
    if any(status[field] != value for field, value in association_fields.items()):
        raise ValueError(f"{video_id} status-to-inventory association differs")
    if (
        status["timeline_status"] != "compatible"
        or status["transcript_status"] != "valid"
    ):
        raise ValueError(f"{video_id} status is not benchmark-eligible")
    if (
        status["code_commit"] != COMMENTARY_CODE_COMMIT
        or status["provider"] != COMMENTARY_PROVIDER
        or status["model"] != COMMENTARY_MODEL
        or status["transcript_source"] not in ("youtube_asr", "whisper")
    ):
        raise ValueError(f"{video_id} commentary contract identity differs")


def _load_transcript_for_video(
    commentary_root: Path,
    manifest_index: Mapping[str, object],
    video_id: str,
    status: Mapping[str, Any],
) -> tuple[str, list[Mapping[str, Any]]]:
    """Load, hash-check, and range-validate one video's normalized transcript."""
    transcript_relative = f"transcripts/{video_id}.json"
    _require_manifest_path(manifest_index, transcript_relative)
    transcript_path = commentary_root / transcript_relative
    transcript = _mapping(
        json.loads(transcript_path.read_text(encoding="utf-8")),
        f"{video_id} transcript",
    )
    segments = [
        _mapping(segment, f"{video_id} transcript segment")
        for segment in _sequence(
            transcript["segments"], f"{video_id} transcript segments"
        )
    ]
    if transcript["source"] != status["transcript_source"]:
        raise ValueError(f"{video_id} transcript source differs from status")
    if len(segments) != int(status["transcript_segment_count"]):
        raise ValueError(f"{video_id} transcript population differs from status")
    if _sha256(transcript_path) != status["transcript_sha256"]:
        raise ValueError(f"{video_id} transcript SHA-256 differs from status")
    _validate_timed_rows(
        segments,
        float(status["local_duration_s"]),
        f"{video_id} transcript",
    )
    return str(transcript["source"]), segments


def _load_cleaned_chunks_for_video(
    commentary_root: Path,
    manifest_index: Mapping[str, object],
    video_id: str,
    status: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Load one video's cleaned chunks, or apply the explicit ss22_17 zero case."""
    cleaned_relative = f"commentary/cleaned_chunks/{video_id}.json"
    if status["clean_status"] == "valid":
        _require_manifest_path(manifest_index, cleaned_relative)
        cleaned_path = commentary_root / cleaned_relative
        chunks = [
            _mapping(chunk, f"{video_id} cleaned chunk")
            for chunk in _sequence(
                json.loads(cleaned_path.read_text(encoding="utf-8")),
                f"{video_id} cleaned chunks",
            )
        ]
        if len(chunks) != int(status["clean_chunk_count"]):
            raise ValueError(f"{video_id} cleaned population differs from status")
        if _sha256(cleaned_path) != status["clean_chunk_sha256"]:
            raise ValueError(f"{video_id} cleaned SHA-256 differs from status")
        _validate_timed_rows(
            chunks,
            float(status["local_duration_s"]),
            f"{video_id} cleaned chunks",
        )
        _validate_unique_chunk_starts(chunks, f"{video_id} cleaned chunks")
        chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
        if len(set(chunk_ids)) != len(chunk_ids) or any(
            not chunk_id.startswith(f"{video_id}_c") for chunk_id in chunk_ids
        ):
            raise ValueError(f"{video_id} cleaned chunk association differs")
        return chunks
    if video_id == "ss22_17" and status["clean_status"] == "dropped_by_relevance_triage":
        if status["clean_chunk_count"] != 0 or status["clean_chunk_sha256"] is not None:
            raise ValueError("ss22_17 dropped-cleaning identity differs")
        return []
    raise ValueError(f"{video_id} has unexpected clean status {status['clean_status']}")


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


def _load_retimed_chunks_for_video(
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


def _video_rally_views(
    dataset: str,
    video_id: str,
    row: Mapping[str, Any],
    frame_count: int,
    shots_master: pd.DataFrame,
    issue103_rallies: Mapping[str, list[tuple[int, int, int]]],
    issue103_masks: Mapping[str, np.ndarray],
    shuttleset22_root: Path,
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    np.ndarray | None,
    Mapping[str, int] | None,
]:
    """Return one video's primary rallies, human-contact rallies, mask, and population."""
    if dataset == "ShuttleSet":
        rallies = issue103_rallies[video_id]
        replay_mask = issue103_masks[video_id]
        if len(replay_mask) != frame_count:
            raise ValueError(f"{video_id} replay mask length differs from inventory")
        human_rows = load_gt_rallies(shots_master, int(video_id.removeprefix("sset_")))
        human_rallies = [
            (rally_id, rally.stroke_frames[0], rally.stroke_frames[-1] + 1)
            for rally_id, rally in enumerate(human_rows)
        ]
        return rallies, human_rallies, replay_mask, None

    annotation_dir = shuttleset22_root / "annotations" / "set" / str(row["title"])
    records, annotation_population = load_annotation_rallies(annotation_dir, frame_count)
    rallies = [
        (
            int(_mapping(record["rally"], "ShuttleSet22 rally")["rally_id"]),
            int(_mapping(record["rally"], "ShuttleSet22 rally")["start_frame"]),
            int(_mapping(record["rally"], "ShuttleSet22 rally")["end_frame"]),
        )
        for record in records
    ]
    return rallies, rallies, None, annotation_population


def _load_video_inputs(
    commentary_root: Path,
    shuttleset_ground_truth_root: Path,
    shuttleset22_root: Path,
    inventory_records: Sequence[Mapping[str, Any]],
    manifest_index: Mapping[str, object],
    issue103_rallies: Mapping[str, list[tuple[int, int, int]]],
    issue103_masks: Mapping[str, np.ndarray],
) -> list[VideoInputs]:
    status_by_id = _load_status_index(commentary_root)
    shots_master = _load_shots_master(shuttleset_ground_truth_root)
    inputs = []
    for row in sorted(
        inventory_records,
        key=lambda item: (str(item["dataset"]), str(item["video_id"])),
    ):
        dataset = str(row["dataset"])
        video_id = str(row["video_id"])
        status = status_by_id[video_id]
        _check_video_contract(video_id, row, status)
        transcript_source, segments = _load_transcript_for_video(
            commentary_root, manifest_index, video_id, status
        )
        chunks = _load_cleaned_chunks_for_video(
            commentary_root, manifest_index, video_id, status
        )
        retimed_chunks, retimed_sha256 = _load_retimed_chunks_for_video(
            commentary_root, video_id, chunks, float(status["local_duration_s"])
        )

        fps = float(row["local_fps"].split("/", maxsplit=1)[0]) / float(
            row["local_fps"].split("/", maxsplit=1)[1]
        )
        frame_count = int(row["local_frame_count"])
        rallies, human_rallies, replay_mask, annotation_population = _video_rally_views(
            dataset,
            video_id,
            row,
            frame_count,
            shots_master,
            issue103_rallies,
            issue103_masks,
            shuttleset22_root,
        )
        _validate_rallies(video_id, rallies, frame_count, "primary")
        _validate_rallies(video_id, human_rallies, frame_count, "human-contact")
        inputs.append(
            VideoInputs(
                dataset=dataset,
                video_id=video_id,
                fps=fps,
                frame_count=frame_count,
                transcript_source=transcript_source,
                transcript_segments=len(segments),
                chunks=tuple(_mapping(chunk, f"{video_id} chunk") for chunk in chunks),
                rallies=tuple(rallies),
                human_rallies=tuple(human_rallies),
                replay_mask=replay_mask,
                annotation_population=annotation_population,
                retimed_chunks=retimed_chunks,
                retimed_sha256=retimed_sha256,
            )
        )
    return inputs
