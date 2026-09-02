"""Export the frozen v1 dataset from one completed dataset-builder run.

The export never reruns extraction or annotation. It reads the validated
rally-record collection and the pinned shuttle, pose, and court artifacts,
derives the kept issue #22 features, and writes the ``schema_v1`` tables plus
a dataset manifest. ShuttleSet human contacts and commentary are optional
inputs and produce source-scoped, auxiliary rows.

``build_video_tables`` and ``write_dataset`` are shared with the ShuttleSet22
export, which has primitives but no production run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from annotator.shuttle_track import validate_shuttle_track
from annotator.video_metadata import VideoMetadata
from dataset_builder.commentary_export import commentary_tables, empty_table
from dataset_builder.features import (
    PlayerFeatureInputs,
    derive_player_feature_inputs,
    player_rally_features,
)
from dataset_builder.fixed_sources import load_fixed_source_manifest
from dataset_builder.manifest import artifact_integrity, load_run_manifest, run_manifest_sha256
from dataset_builder.models import ArtifactIntegrity, RunManifest, StageOutcome
from dataset_builder.records import RALLY_RECORDS_FILENAME, load_rally_records
from dataset_builder.schema_v1 import (
    COMMENTARY_CHUNKS,
    DATASET_MANIFEST_FILENAME,
    DATASET_SCHEMA,
    FEATURE_DISPOSITIONS,
    PLAYER_RALLIES,
    PLAYER_SIGNALS,
    PLAYER_SIGNALS_DIRECTORY,
    PRIMITIVE_ARTIFACT_NOTES,
    PRIMITIVE_ARTIFACTS,
    RALLIES,
    SCHEMA_FROZEN_ON,
    SOURCE_CONTACTS,
    TABLES,
    TRANSCRIPT_SEGMENTS,
    RallyOrigin,
    TableSpec,
    write_table,
)
from dataset_builder.source_annotations import load_source_annotations
from dataset_builder.vision import (
    TRACK_FILENAME,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
    save_json_gz,
    save_npy_xz,
)


STAGES_DIRECTORY = "stages"
PRIMITIVE_STAGE_BASES = ("shuttle", "pose", "court", "annotation")
LOCATION_INPUT = "input_dir"
LOCATION_EXPORT = "export_dir"
ARTIFACT_NOTES = {note.artifact: note for note in PRIMITIVE_ARTIFACT_NOTES}


@dataclass(frozen=True)
class ExportInputs:
    """Explicit inputs for one run-directory export; nothing is discovered."""

    run_dir: Path
    output_dir: Path
    fixed_sources_manifest: Path | None = None
    ground_truth_root: Path | None = None
    commentary_root: Path | None = None

    def __post_init__(self) -> None:
        if (self.fixed_sources_manifest is None) != (self.ground_truth_root is None):
            raise ValueError(
                "fixed_sources_manifest and ground_truth_root must be given together"
            )


@dataclass(frozen=True)
class VideoInputs:
    """Everything ``build_video_tables`` needs for one video, already loaded."""

    run_id: str
    source_dataset: str
    video_id: str
    metadata: VideoMetadata
    player_inputs: PlayerFeatureInputs
    annotator_spans: tuple[tuple[int, int], ...]
    input_artifacts: tuple[ArtifactIntegrity, ...]
    annotation_dir: Path | None = None
    annotation_root: Path | None = None
    identity: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoTables:
    """Rows and manifest entry produced for one video."""

    rallies: list[dict[str, object]]
    player_rallies: list[dict[str, object]]
    source_contacts: pd.DataFrame | None
    artifacts: list[dict[str, object]]
    manifest: dict[str, object]


@dataclass(frozen=True)
class DatasetIdentity:
    """Manifest-level provenance shared by every export kind."""

    run_id: str
    source_dataset: str
    input_root: Path
    code_version: str | None = None
    input_manifest_sha256: str | None = None
    run_manifest_sha256: str | None = None
    fixed_source_manifest: dict[str, object] | None = None
    sources_manifest: dict[str, object] | None = None
    ground_truth_root: Path | None = None
    commentary_root: Path | None = None


def export_dataset_v1(inputs: ExportInputs) -> dict[str, object]:
    """Write every v1 table and the dataset manifest; return the manifest."""
    run_dir = Path(inputs.run_dir).resolve(strict=True)
    output_dir = Path(inputs.output_dir)
    records_path = run_dir / RALLY_RECORDS_FILENAME
    collection = load_json_gz(records_path)
    records = load_rally_records(records_path)
    manifest = load_run_manifest(run_dir)
    sources = [dict(_mapping(source, "record source")) for source in _list(collection["sources"])]
    source_dataset = single_source_dataset(sources)
    video_ids = [str(source["video_id"]) for source in sources]
    annotation_dirs, fixed_manifest = _annotation_directories(inputs, video_ids)
    annotation_root = _optional_resolved(inputs.ground_truth_root)
    spans_by_video = annotator_spans_by_video(records)
    run_id = str(collection["run_id"])

    videos: list[VideoTables] = []
    for source in sources:
        video_id = str(source["video_id"])
        metadata = VideoMetadata.from_dict(source["video_metadata"])
        stages = run_dir / STAGES_DIRECTORY
        video_inputs = VideoInputs(
            run_id=run_id,
            source_dataset=source_dataset,
            video_id=video_id,
            metadata=metadata,
            player_inputs=derive_player_inputs(
                stages / "shuttle" / video_id / TRACK_FILENAME,
                stages / "pose" / video_id,
                stages / "court" / video_id,
                court_video_id=video_id,
                metadata=metadata,
            ),
            annotator_spans=spans_by_video.get(video_id, ()),
            input_artifacts=_run_artifacts(run_dir, manifest, video_id),
            annotation_dir=annotation_dirs.get(video_id),
            annotation_root=annotation_root,
        )
        videos.append(build_video_tables(output_dir, video_inputs))

    identity = DatasetIdentity(
        run_id=run_id,
        source_dataset=source_dataset,
        input_root=run_dir,
        code_version=str(collection["code_version"]),
        input_manifest_sha256=str(collection["input_manifest_sha256"]),
        run_manifest_sha256=run_manifest_sha256(manifest),
        fixed_source_manifest=fixed_manifest,
        ground_truth_root=annotation_root,
        commentary_root=_optional_resolved(inputs.commentary_root),
    )
    return write_dataset(output_dir, identity, videos, video_ids)


def write_dataset(
    output_dir: Path,
    identity: DatasetIdentity,
    videos: Sequence[VideoTables],
    video_ids: Sequence[str],
) -> dict[str, object]:
    """Write the tables and manifest for already-built per-video tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = _assemble_tables(videos)
    if identity.commentary_root is not None:
        segments, chunks = commentary_tables(
            identity.commentary_root, identity.source_dataset, video_ids
        )
        tables[TRANSCRIPT_SEGMENTS.name] = segments
        tables[COMMENTARY_CHUNKS.name] = chunks
    table_entries = {
        table.name: _table_entry(output_dir, table, tables[table.name]) for table in TABLES
    }
    dataset_manifest: dict[str, object] = {
        "schema": DATASET_SCHEMA,
        "frozen_on": SCHEMA_FROZEN_ON,
        "run_id": identity.run_id,
        "source_dataset": identity.source_dataset,
        "input_root": str(identity.input_root),
        "code_version": identity.code_version,
        "input_manifest_sha256": identity.input_manifest_sha256,
        "run_manifest_sha256": identity.run_manifest_sha256,
        "fixed_source_manifest": identity.fixed_source_manifest,
        "sources_manifest": identity.sources_manifest,
        "ground_truth_root": _optional_text(identity.ground_truth_root),
        "commentary_root": _optional_text(identity.commentary_root),
        "videos": [video.manifest for video in videos],
        "tables": table_entries,
        "dispositions": [
            {
                "feature": disposition.feature,
                "disposition": disposition.disposition.value,
                "columns": list(disposition.columns),
                "reason": disposition.reason,
            }
            for disposition in FEATURE_DISPOSITIONS
        ],
    }
    save_json_gz(output_dir / DATASET_MANIFEST_FILENAME, dataset_manifest)
    return dataset_manifest


def build_video_tables(output_dir: Path, inputs: VideoInputs) -> VideoTables:
    """Derive the v1 rows for one video and save its player-signal arrays."""
    signal_files = _save_player_signals(output_dir, inputs.video_id, inputs.player_inputs)
    identity = (inputs.run_id, inputs.source_dataset, inputs.video_id)

    rallies: list[dict[str, object]] = []
    player_rallies: list[dict[str, object]] = []
    for rally_id, (start, end) in enumerate(inputs.annotator_spans):
        rallies.append(
            _rally_row(
                identity, RallyOrigin.ANNOTATOR, rally_id, inputs.metadata, start, end, None, None
            )
        )
        player_rallies.extend(
            _player_rows(identity, RallyOrigin.ANNOTATOR, rally_id, inputs.player_inputs, start, end)
        )

    source_contacts = None
    source_rallies = None
    source_population = None
    annotation_files: list[dict[str, object]] = []
    if inputs.annotation_dir is not None:
        annotations = load_source_annotations(
            inputs.annotation_dir,
            source_dataset=inputs.source_dataset,
            video_id=inputs.video_id,
            frame_count=inputs.metadata.frame_count,
        )
        source_contacts = annotations.contacts
        source_rallies = len(annotations.rallies)
        source_population = dict(annotations.population)
        annotation_files = [
            artifact_integrity(
                f"{inputs.video_id}.{path.stem}", path, relative_to=inputs.annotation_root
            ).to_dict()
            for path in sorted(Path(inputs.annotation_dir).glob("set*.csv"))
        ]
        for rally_id, source_rally in enumerate(annotations.rallies):
            rallies.append(
                _rally_row(
                    identity,
                    RallyOrigin.SOURCE_CONTACTS,
                    rally_id,
                    inputs.metadata,
                    source_rally.start_frame,
                    source_rally.end_frame,
                    source_rally.source_set,
                    source_rally.source_rally,
                )
            )
            player_rallies.extend(
                _player_rows(
                    identity,
                    RallyOrigin.SOURCE_CONTACTS,
                    rally_id,
                    inputs.player_inputs,
                    source_rally.start_frame,
                    source_rally.end_frame,
                )
            )

    artifacts = [
        _artifact_row(inputs.source_dataset, inputs.video_id, LOCATION_INPUT, integrity)
        for integrity in inputs.input_artifacts
    ]
    artifacts.extend(
        _artifact_row(inputs.source_dataset, inputs.video_id, LOCATION_EXPORT, integrity)
        for integrity in signal_files
    )
    return VideoTables(
        rallies=rallies,
        player_rallies=player_rallies,
        source_contacts=source_contacts,
        artifacts=artifacts,
        manifest={
            "video_id": inputs.video_id,
            "source_dataset": inputs.source_dataset,
            "fps": inputs.metadata.to_dict()["fps"],
            "frame_count": inputs.metadata.frame_count,
            "annotator_rallies": len(inputs.annotator_spans),
            "source_rallies": source_rallies,
            "source_population": source_population,
            "source_annotation_files": annotation_files,
            "player_signals": [integrity.to_dict() for integrity in signal_files],
            **inputs.identity,
        },
    )


def derive_player_inputs(
    track_path: Path,
    pose_dir: Path,
    court_dir: Path,
    *,
    court_video_id: str,
    metadata: VideoMetadata,
) -> PlayerFeatureInputs:
    """Load pinned shuttle, pose, and court artifacts and derive player signals."""
    track = load_npy_xz(track_path)
    validate_shuttle_track(track, metadata.frame_count)
    pose = load_pose_arrays(pose_dir, metadata.frame_count)
    court = load_court_vision(
        court_dir,
        video_id=court_video_id,
        frame_count=metadata.frame_count,
        resolution=(float(metadata.width), float(metadata.height)),
    )
    return derive_player_feature_inputs(track, pose, court, court_video_id)


def annotator_spans_by_video(
    records: Sequence[Mapping[str, object]],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Return each video's half-open rally spans in rally_id order."""
    grouped: dict[str, list[tuple[int, int, int]]] = {}
    for record in records:
        key = _mapping(record["key"], "record key")
        rally = _mapping(record["rally"], "rally")
        grouped.setdefault(str(key["video_id"]), []).append(
            (int(rally["rally_id"]), int(rally["start_frame"]), int(rally["end_frame"]))
        )
    spans: dict[str, tuple[tuple[int, int], ...]] = {}
    for video_id, rows in grouped.items():
        if [row[0] for row in rows] != list(range(len(rows))):
            raise ValueError(f"rally records for {video_id!r} are not contiguous from zero")
        spans[video_id] = tuple((start, end) for _, start, end in rows)
    return spans


def single_source_dataset(sources: Sequence[Mapping[str, object]]) -> str:
    """Return the one source dataset label a collection may carry."""
    datasets = {str(source["source_dataset"]) for source in sources}
    if len(datasets) != 1:
        raise ValueError(f"export expects one source dataset per run, found {sorted(datasets)}")
    return datasets.pop()


def _save_player_signals(
    output_dir: Path, video_id: str, player_inputs: PlayerFeatureInputs
) -> tuple[ArtifactIntegrity, ...]:
    arrays = {
        "posture": np.asarray(player_inputs.posture, dtype=np.float64),
        "court_position": np.asarray(player_inputs.court_positions, dtype=np.float64),
        "posture_interpolation": np.asarray(player_inputs.posture_interpolation, dtype=np.int8),
        "position_interpolation": np.asarray(
            player_inputs.position_interpolation, dtype=np.int8
        ),
    }
    directory = Path(output_dir) / PLAYER_SIGNALS_DIRECTORY / video_id
    files = []
    for signal in PLAYER_SIGNALS:
        path = save_npy_xz(directory / signal.filename, arrays[signal.name])
        files.append(artifact_integrity(signal.name, path, relative_to=output_dir))
    return tuple(files)


def _rally_row(
    identity: tuple[str, str, str],
    origin: RallyOrigin,
    rally_id: int,
    metadata: VideoMetadata,
    start: int,
    end: int,
    source_set: int | None,
    source_rally: int | None,
) -> dict[str, object]:
    run_id, source_dataset, video_id = identity
    return {
        "run_id": run_id,
        "source_dataset": source_dataset,
        "video_id": video_id,
        "rally_origin": origin.value,
        "rally_id": rally_id,
        "fps": float(metadata.fps),
        "frame_count": metadata.frame_count,
        "start_frame": start,
        "end_frame": end,
        "duration_frames": end - start,
        "start_seconds": float(Fraction(start) / metadata.fps),
        "end_seconds": float(Fraction(end) / metadata.fps),
        "duration_seconds": float(Fraction(end - start) / metadata.fps),
        "source_set": source_set,
        "source_rally": source_rally,
    }


def _player_rows(
    identity: tuple[str, str, str],
    origin: RallyOrigin,
    rally_id: int,
    player_inputs: PlayerFeatureInputs,
    start: int,
    end: int,
) -> list[dict[str, object]]:
    run_id, source_dataset, video_id = identity
    rows = []
    for features in player_rally_features(player_inputs, start, end):
        rows.append(
            {
                "run_id": run_id,
                "source_dataset": source_dataset,
                "video_id": video_id,
                "rally_origin": origin.value,
                "rally_id": rally_id,
                **features._asdict(),
            }
        )
    return rows


def _run_artifacts(
    run_dir: Path, manifest: RunManifest, video_id: str
) -> tuple[ArtifactIntegrity, ...]:
    artifacts = []
    for base in PRIMITIVE_STAGE_BASES:
        stage = next((s for s in manifest.stages if s.name == f"{base}:{video_id}"), None)
        if stage is None or stage.outcome is not StageOutcome.PROCESSED:
            raise ValueError(f"run manifest has no processed {base} stage for {video_id!r}")
        for output in stage.outputs:
            if output.name not in ARTIFACT_NOTES:
                continue
            stored = run_dir / output.path
            if not stored.is_file() or stored.stat().st_size != output.size_bytes:
                raise ValueError(f"run artifact differs from its manifest record: {output.path}")
            artifacts.append(output)
    return tuple(artifacts)


def _artifact_row(
    source_dataset: str, video_id: str, location: str, integrity: ArtifactIntegrity
) -> dict[str, object]:
    note = ARTIFACT_NOTES[integrity.name]
    return {
        "source_dataset": source_dataset,
        "video_id": video_id,
        "artifact": integrity.name,
        "location": location,
        "relative_path": integrity.path,
        "md5": integrity.md5,
        "size_bytes": integrity.size_bytes,
        "reliability": note.reliability.value,
        "note": note.note,
    }


def _assemble_tables(videos: Sequence[VideoTables]) -> dict[str, pd.DataFrame]:
    rallies = [row for video in videos for row in video.rallies]
    players = [row for video in videos for row in video.player_rallies]
    artifacts = [row for video in videos for row in video.artifacts]
    contacts = [video.source_contacts for video in videos if video.source_contacts is not None]
    return {
        RALLIES.name: pd.DataFrame(rallies) if rallies else empty_table(RALLIES),
        PLAYER_RALLIES.name: pd.DataFrame(players) if players else empty_table(PLAYER_RALLIES),
        SOURCE_CONTACTS.name: (
            pd.concat(contacts, ignore_index=True) if contacts else empty_table(SOURCE_CONTACTS)
        ),
        PRIMITIVE_ARTIFACTS.name: (
            pd.DataFrame(artifacts) if artifacts else empty_table(PRIMITIVE_ARTIFACTS)
        ),
        TRANSCRIPT_SEGMENTS.name: empty_table(TRANSCRIPT_SEGMENTS),
        COMMENTARY_CHUNKS.name: empty_table(COMMENTARY_CHUNKS),
    }


def _table_entry(output_dir: Path, table: TableSpec, frame: pd.DataFrame) -> dict[str, object]:
    path = write_table(output_dir, table, frame)
    integrity = artifact_integrity(table.name, path, relative_to=output_dir)
    return {
        "filename": table.filename,
        "rows": int(len(frame)),
        "md5": integrity.md5,
        "size_bytes": integrity.size_bytes,
    }


def _annotation_directories(
    inputs: ExportInputs, video_ids: Sequence[str]
) -> tuple[dict[str, Path], dict[str, object] | None]:
    if inputs.fixed_sources_manifest is None or inputs.ground_truth_root is None:
        return {}, None
    fixed = load_fixed_source_manifest(inputs.fixed_sources_manifest)
    entries = fixed.entries_by_video_id()
    root = Path(inputs.ground_truth_root).resolve(strict=True)
    directories: dict[str, Path] = {}
    for video_id in video_ids:
        entry = entries.get(video_id)
        if entry is None:
            raise ValueError(f"fixed source manifest has no entry for {video_id!r}")
        directory = root / entry.ground_truth.annotation_directory
        if not directory.is_dir():
            raise FileNotFoundError(f"annotation directory is missing: {directory}")
        directories[video_id] = directory
    identity = {"path": str(fixed.path), "md5": fixed.md5, "size_bytes": fixed.size_bytes}
    return directories, identity


def _optional_resolved(path: Path | None) -> Path | None:
    return None if path is None else Path(path).resolve(strict=True)


def _optional_text(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value
