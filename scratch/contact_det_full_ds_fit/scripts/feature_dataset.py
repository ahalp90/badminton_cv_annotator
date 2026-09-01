"""Check and load a complete set of saved contact features before labels are read."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from scratch.contact_det.scripts.freeze_tree_contact_features import (
    FEATURE_SCHEMA,
    IDENTITY_FIELDS,
)
from scratch.contact_det_full_ds_fit.scripts.experiment_config import (
    DevelopmentSplit,
    VideoSpec,
    load_development_split,
    verify_accepted_development_split,
)
from scratch.contact_det_full_ds_fit.scripts.freeze_contact_features import (
    FEATURE_FILENAME,
    RUN_RECORD_SCHEMA,
    VIDEO_RECORD_SCHEMA,
    _feature_family_names,
    _validate_feature_rows,
)

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
RUN_RECORD_FIELDS = {
    "schema",
    "status",
    "feature_schema",
    "source_commit",
    "split_file",
    "split_sha256",
    "motion_mode",
    "contact_labels_read",
    "video_names",
    "video_count",
    "completed_video_names",
    "completed_video_count",
    "row_count",
    "feature_families",
    "identity_fields",
    "videos",
}
VIDEO_RECORD_FIELDS = {
    "schema",
    "feature_schema",
    "source_commit",
    "split_file",
    "split_sha256",
    "motion_mode",
    "contact_labels_read",
    "video",
    "feature_file",
    "feature_sha256",
    "row_count",
    "feature_summary",
    "input_files",
}
VIDEO_FIELDS = {"name", "video_id", "fps", "width", "height", "role"}
INPUT_FILE_FIELDS = {"role", "filename", "size_bytes", "sha256"}


@dataclass(frozen=True)
class VerifiedFeatureDataset:
    """Feature rows and video ranges checked against one complete run record."""

    record_path: Path
    record: dict[str, Any]
    split: DevelopmentSplit
    rows: np.ndarray
    video_ranges: dict[str, tuple[int, int]]
    model_input_fields: tuple[str, ...]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_file(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty relative filename")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise ValueError(f"{name} must stay inside the feature directory")
    return path


def _checked_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 file hash")
    return value


def _check_input_file_records(value: object, video_name: str) -> None:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{video_name}: input files must be a non-empty list")
    roles: set[str] = set()
    for index, raw_file in enumerate(value):
        file_record = _mapping(raw_file, f"{video_name}.input_files[{index}]")
        if set(file_record) != INPUT_FILE_FIELDS:
            raise ValueError(f"{video_name}: input file fields differ")
        role = file_record["role"]
        filename = file_record["filename"]
        size_bytes = file_record["size_bytes"]
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError(f"{video_name}: input file roles must be non-empty and unique")
        if not isinstance(filename, str) or PurePosixPath(filename).name != filename:
            raise ValueError(f"{video_name}: input filenames must not contain a path")
        if type(size_bytes) is not int or size_bytes <= 0:
            raise ValueError(f"{video_name}: input file sizes must be positive integers")
        _checked_sha256(file_record["sha256"], f"{video_name}.{role}.sha256")
        roles.add(role)


def _check_video_identity(value: object, video: VideoSpec) -> None:
    identity = _mapping(value, f"{video.fixture}.video")
    if set(identity) != VIDEO_FIELDS:
        raise ValueError(f"{video.fixture}: video fields differ")
    expected = {
        "name": video.fixture,
        "video_id": video.video_id,
        "fps": video.fps,
        "width": video.width,
        "height": video.height,
        "role": video.role.value,
    }
    if dict(identity) != expected:
        raise ValueError(f"{video.fixture}: video details differ from the accepted split")


def _load_video_rows(
    output_root: Path,
    raw_record: object,
    video: VideoSpec,
    run_record: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    record = _mapping(raw_record, f"{video.fixture}.record")
    if set(record) != VIDEO_RECORD_FIELDS:
        raise ValueError(f"{video.fixture}: feature record fields differ")
    shared_fields = (
        "feature_schema",
        "source_commit",
        "split_file",
        "split_sha256",
        "motion_mode",
        "contact_labels_read",
    )
    if record.get("schema") != VIDEO_RECORD_SCHEMA or any(
        record.get(field) != run_record.get(field) for field in shared_fields
    ):
        raise ValueError(f"{video.fixture}: feature record differs from the full run")
    _check_video_identity(record.get("video"), video)
    _check_input_file_records(record.get("input_files"), video.fixture)

    relative_path = _relative_file(record.get("feature_file"), f"{video.fixture}.feature_file")
    expected_relative_path = PurePosixPath("videos") / video.fixture / FEATURE_FILENAME
    if relative_path != expected_relative_path:
        raise ValueError(f"{video.fixture}: feature filename differs")
    feature_path = output_root.joinpath(*relative_path.parts)
    expected_sha256 = _checked_sha256(
        record.get("feature_sha256"),
        f"{video.fixture}.feature_sha256",
    )
    if _sha256(feature_path) != expected_sha256:
        raise ValueError(f"{video.fixture}: feature file hash differs")

    from dataset_builder.vision import load_npy_xz

    rows = load_npy_xz(feature_path)
    row_count = _integer(record.get("row_count"), f"{video.fixture}.row_count")
    if row_count != len(rows):
        raise ValueError(f"{video.fixture}: feature row count differs")
    summary = dict(_mapping(record.get("feature_summary"), f"{video.fixture}.feature_summary"))
    _validate_feature_rows(video, rows, summary)
    return rows, dict(record)


def load_verified_feature_dataset(
    run_record_path: Path,
    split_path: Path,
    expected_motion_mode: str,
) -> VerifiedFeatureDataset:
    """Load all 40 feature files after checking their run and split records."""
    split_file = Path(split_path)
    split = load_development_split(split_file)
    verify_accepted_development_split(split)
    split_sha256 = _sha256(split_file)

    record_path = Path(run_record_path)
    record = dict(_mapping(json.loads(record_path.read_text(encoding="utf-8")), "run record"))
    if set(record) != RUN_RECORD_FIELDS:
        raise ValueError("run record fields differ")
    if record.get("schema") != RUN_RECORD_SCHEMA or record.get("status") != "complete":
        raise ValueError("feature run is not complete")
    if record.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("feature row version differs")
    source_commit = record.get("source_commit")
    if not isinstance(source_commit, str) or HEX_COMMIT.fullmatch(source_commit) is None:
        raise ValueError("feature source commit is invalid")
    if record.get("split_file") != split_file.name or record.get("split_sha256") != split_sha256:
        raise ValueError("feature split file differs")
    if record.get("motion_mode") != expected_motion_mode:
        raise ValueError("feature motion values differ from the requested run")
    if record.get("contact_labels_read") is not False:
        raise ValueError("feature preparation must not read ShuttleSet contact labels")

    video_names = [video.fixture for video in split.videos]
    if record.get("video_names") != video_names or record.get("completed_video_names") != video_names:
        raise ValueError("feature videos differ from the accepted split")
    if record.get("video_count") != len(video_names) or record.get("completed_video_count") != len(
        video_names
    ):
        raise ValueError("feature video counts differ")
    if record.get("feature_families") != _feature_family_names():
        raise ValueError("feature fields differ from the feature-saving code")
    if record.get("identity_fields") != list(IDENTITY_FIELDS):
        raise ValueError("feature identity fields differ")

    raw_video_records = record.get("videos")
    if not isinstance(raw_video_records, list) or len(raw_video_records) != len(split.videos):
        raise ValueError("feature run has the wrong number of video records")
    output_root = record_path.parent
    chunks: list[np.ndarray] = []
    video_ranges: dict[str, tuple[int, int]] = {}
    row_start = 0
    expected_dtype: np.dtype[Any] | None = None
    for video, raw_video_record in zip(split.videos, raw_video_records, strict=True):
        rows, _video_record = _load_video_rows(output_root, raw_video_record, video, record)
        if expected_dtype is None:
            expected_dtype = rows.dtype
        elif rows.dtype != expected_dtype:
            raise ValueError(f"{video.fixture}: feature field types differ from earlier videos")
        row_end = row_start + len(rows)
        video_ranges[video.fixture] = (row_start, row_end)
        chunks.append(rows)
        row_start = row_end

    rows = np.concatenate(chunks)
    if len(rows) != _integer(record.get("row_count"), "run row_count"):
        raise ValueError("full feature row count differs")
    family_names = _feature_family_names()
    expected_fields = set(IDENTITY_FIELDS)
    for names in family_names.values():
        expected_fields.update(names)
    if rows.dtype.names is None or set(rows.dtype.names) != expected_fields:
        raise ValueError("feature row fields differ")
    model_input_fields = tuple(family_names["physics"] + family_names["missingness"])
    return VerifiedFeatureDataset(
        record_path,
        record,
        split,
        rows,
        video_ranges,
        model_input_fields,
    )
