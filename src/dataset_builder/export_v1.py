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
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.shuttle_track import validate_shuttle_track
from annotator.video_metadata import VideoMetadata
from dataset_builder.commentary_export import CommentaryTables, commentary_tables, empty_table
from dataset_builder.degradation import player_trend_rows
from dataset_builder.features import (
    COURT_SIDES,
    PlayerFeatureInputs,
    clip_frames,
    derive_player_feature_inputs,
    movement_inefficiency,
    player_rally_features,
    recovery_at_opponent_contacts,
)
from dataset_builder.fixed_sources import load_fixed_source_manifest
from dataset_builder.manifest import artifact_integrity, load_run_manifest, run_manifest_sha256
from dataset_builder.models import ArtifactIntegrity, RunManifest, StageOutcome
from dataset_builder.players import (
    DEFAULT_PLAYERS,
    MATCH_TABLE_FILENAME,
    MatchPlayers,
    Player,
    load_match_players,
    load_players,
    phase_for_span,
)
from dataset_builder.records import RALLY_RECORDS_FILENAME, load_rally_records
from dataset_builder.schema_v1 import (
    COMMENTARY_CHUNKS,
    COMMENTARY_RALLY_LINKS,
    DATASET_MANIFEST_FILENAME,
    DATASET_SCHEMA,
    FEATURE_DISPOSITIONS,
    PLAYER_RALLIES,
    PLAYER_SIGNALS,
    PLAYER_SIGNALS_DIRECTORY,
    PLAYER_TRENDS,
    PLAYERS,
    PRIMITIVE_ARTIFACT_NOTES,
    PRIMITIVE_ARTIFACTS,
    RALLIES,
    SCHEMA_FROZEN_ON,
    SOURCE_CONTACTS,
    TABLES,
    TRANSCRIPT_SEGMENTS,
    RallyOrigin,
    TableSpec,
    validate_table,
    write_table,
)
from dataset_builder.source_annotations import SourceAnnotations, load_source_annotations
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
TOP_SIDE, BOTTOM_SIDE = COURT_SIDES


@dataclass(frozen=True)
class ExportInputs:
    """Explicit inputs for one run-directory export; nothing is discovered."""

    run_dir: Path
    output_dir: Path
    fixed_sources_manifest: Path | None = None
    ground_truth_root: Path | None = None
    commentary_root: Path | None = None
    replay_mask_root: Path | None = None
    video_ids: tuple[str, ...] | None = None
    players: Path = DEFAULT_PLAYERS

    def __post_init__(self) -> None:
        if (self.fixed_sources_manifest is None) != (self.ground_truth_root is None):
            raise ValueError(
                "fixed_sources_manifest and ground_truth_root must be given together"
            )
        if self.video_ids is not None and (
            not self.video_ids or len(set(self.video_ids)) != len(self.video_ids)
        ):
            raise ValueError("video_ids must be a non-empty tuple without repeats")


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
    match_players: MatchPlayers | None = None
    identity: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoTables:
    """Rows and manifest entry produced for one video."""

    rallies: list[dict[str, object]]
    player_rallies: list[dict[str, object]]
    player_trends: list[dict[str, object]]
    source_contacts: pd.DataFrame | None
    players: tuple[Player, ...]
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
    inpainted_root: Path | None = None
    replay_mask_root: Path | None = None
    players_table: dict[str, object] | None = None


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
    sources = _select_sources(sources, inputs.video_ids)
    video_ids = [str(source["video_id"]) for source in sources]
    annotation_dirs, fixed_manifest = _annotation_directories(inputs, video_ids)
    annotation_root = _optional_resolved(inputs.ground_truth_root)
    spans_by_video = annotator_spans_by_video(records)
    run_id = str(collection["run_id"])
    players_path = Path(inputs.players)
    players = load_players(players_path)

    videos: list[VideoTables] = []
    for source in sources:
        video_id = str(source["video_id"])
        metadata = VideoMetadata.from_dict(source["video_metadata"])
        stages = run_dir / STAGES_DIRECTORY
        annotation_dir = annotation_dirs.get(video_id)
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
            annotation_dir=annotation_dir,
            annotation_root=annotation_root,
            match_players=_match_players(annotation_dir, players),
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
        replay_mask_root=_optional_resolved(inputs.replay_mask_root),
        players_table=artifact_integrity("players", players_path).to_dict(),
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
        fps_by_video = {
            str(video.manifest["video_id"]): float(Fraction(str(video.manifest["fps"])))
            for video in videos
        }
        frame_count_by_video = {
            str(video.manifest["video_id"]): int(video.manifest["frame_count"]) for video in videos
        }
        commentary: CommentaryTables = commentary_tables(
            identity.commentary_root,
            identity.run_id,
            identity.source_dataset,
            video_ids,
            tables[RALLIES.name],
            fps_by_video,
            frame_count_by_video,
            replay_mask_root=identity.replay_mask_root,
        )
        tables[TRANSCRIPT_SEGMENTS.name] = commentary.segments
        tables[COMMENTARY_CHUNKS.name] = commentary.chunks
        tables[COMMENTARY_RALLY_LINKS.name] = commentary.links
        for video in videos:
            video_id = str(video.manifest["video_id"])
            video.manifest["commentary_masked_start_chunks"] = (
                commentary.masked_start_chunks.get(video_id)
            )
            video.manifest["commentary_ambiguous_links"] = commentary.ambiguous_links.get(video_id)
            video.manifest["commentary_multi_rally_chunks"] = (
                commentary.multi_rally_chunks.get(video_id)
            )
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
        "inpainted_root": _optional_text(identity.inpainted_root),
        "replay_mask_root": _optional_text(identity.replay_mask_root),
        "players_table": identity.players_table,
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
    match = inputs.match_players
    annotations = _load_annotations(inputs)

    # Annotator spans need the side phases, so the source annotations load first.
    rallies: list[dict[str, object]] = []
    player_rallies: list[dict[str, object]] = []
    for rally_id, (start, end) in enumerate(inputs.annotator_spans):
        player_ids = _annotator_player_ids(annotations, match, start, end)
        rallies.append(
            _rally_row(
                identity,
                RallyOrigin.ANNOTATOR,
                rally_id,
                inputs.metadata,
                start,
                end,
                None,
                None,
                None,
                player_ids,
                False,  # No contact rows to carry the ShuttleSet flaw flag.
            )
        )
        player_rallies.extend(
            _player_rows(
                identity,
                RallyOrigin.ANNOTATOR,
                rally_id,
                inputs.player_inputs,
                start,
                end,
                player_ids,
                (None, None),
                (None, None),
            )
        )

    source_contacts = None
    source_rallies = None
    source_population = None
    if annotations is not None and match is not None:
        contact_features = _source_contact_features(
            annotations, match, inputs.player_inputs, float(inputs.metadata.fps)
        )
        source_contacts = contact_features.contacts
        source_rallies = len(annotations.rallies)
        source_population = dict(annotations.population)
        source_population["unmatched_hitters"] = contact_features.unmatched_hitters
        for rally_id, source_rally in enumerate(annotations.rallies):
            player_ids = _side_player_ids(match, source_rally.a_is_top)
            in_rally = source_contacts["rally_id"].eq(rally_id)
            flaw_marked = bool(source_contacts.loc[in_rally, "flaw_marked"].any())
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
                    len(source_rally.contact_rows),
                    player_ids,
                    flaw_marked,
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
                    player_ids,
                    contact_features.recovery_medians[rally_id],
                    contact_features.movement_medians[rally_id],
                )
            )

    trend_rows, trend_population = player_trend_rows(identity, rallies, player_rallies)

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
        player_trends=trend_rows,
        source_contacts=source_contacts,
        players=() if match is None else (match.player_a, match.player_b),
        artifacts=artifacts,
        manifest={
            "video_id": inputs.video_id,
            "source_dataset": inputs.source_dataset,
            "fps": inputs.metadata.to_dict()["fps"],
            "frame_count": inputs.metadata.frame_count,
            "annotator_rallies": len(inputs.annotator_spans),
            "source_rallies": source_rallies,
            "source_population": source_population,
            "trend_population": trend_population,
            "source_annotation_files": _annotation_files(inputs),
            "match_players": _match_players_entry(match),
            "player_signals": [integrity.to_dict() for integrity in signal_files],
            **inputs.identity,
        },
    )


def _match_players(
    annotation_dir: Path | None, players: Mapping[str, Player]
) -> MatchPlayers | None:
    """Resolve the two people of the match whose set CSVs live in ``annotation_dir``."""
    if annotation_dir is None:
        return None
    return load_match_players(
        annotation_dir.parent / MATCH_TABLE_FILENAME, annotation_dir.name, players
    )


def _annotation_files(inputs: VideoInputs) -> list[dict[str, object]]:
    """Pin the set CSVs the video's source annotations were read from."""
    if inputs.annotation_dir is None:
        return []
    return [
        artifact_integrity(
            f"{inputs.video_id}.{path.stem}", path, relative_to=inputs.annotation_root
        ).to_dict()
        for path in sorted(Path(inputs.annotation_dir).glob("set*.csv"))
    ]


def _load_annotations(inputs: VideoInputs) -> SourceAnnotations | None:
    """Read the video's ShuttleSet set CSVs, or None when it has none."""
    if inputs.annotation_dir is None:
        return None
    if inputs.match_players is None:
        raise ValueError(
            f"{inputs.video_id!r} has source annotations but no match_players; "
            f"the match table must name both people"
        )
    return load_source_annotations(
        inputs.annotation_dir,
        source_dataset=inputs.source_dataset,
        video_id=inputs.video_id,
        frame_count=inputs.metadata.frame_count,
        match=inputs.match_players,
    )


def _side_player_ids(match: MatchPlayers, a_is_top: bool) -> dict[str, str]:
    """Return the player_id on each court side for one side phase."""
    return {side: match.on_side(side, a_is_top).player_id for side in COURT_SIDES}


def _annotator_player_ids(
    annotations: SourceAnnotations | None, match: MatchPlayers | None, start: int, end: int
) -> dict[str, str] | None:
    """Return the side-to-player map for an annotator span, or None when unresolved."""
    if annotations is None or match is None:
        return None
    phase = phase_for_span(annotations.side_phases, start, end)
    return None if phase is None else _side_player_ids(match, phase.a_is_top)


class SourceContactFeatures(NamedTuple):
    """source_contacts with its position-derived columns filled and validated
    against the frozen schema, and their medians.

    ``recovery_medians`` and ``movement_medians`` are keyed by rally_id, each a
    ``(top, bottom)`` pair for ``player_rallies.recovery_distance_median`` and
    ``.movement_inefficiency_median``.
    """

    contacts: pd.DataFrame
    recovery_medians: dict[int, tuple[float | None, float | None]]
    movement_medians: dict[int, tuple[float | None, float | None]]
    unmatched_hitters: int


def _source_contact_features(
    annotations: SourceAnnotations,
    match: MatchPlayers,
    player_inputs: PlayerFeatureInputs,
    fps: float,
) -> SourceContactFeatures:
    """Fill source_contacts' recovery and movement columns from human contact order.

    Per rally, a contact's striker slot is its hitter matched against the
    rally's top and bottom player ids. A null or unmatched hitter leaves that
    contact's recovery values null rather than guessing which player recovered;
    movement inefficiency does not need a striker, so it is unaffected.

    ``annotations.contacts`` has every source_contacts column except these
    four; adding them completes the table, so this is also where it is
    validated against the frozen schema.
    """
    contacts = annotations.contacts.copy()
    # Relied on below: a fresh copy of source_annotations' freshly built table
    # keeps its RangeIndex, so an index label is also its row position.
    positions = np.asarray(player_inputs.court_positions, dtype=float)
    row_count = len(contacts)
    recovery_distance: list[float | None] = [None] * row_count
    recovery_frames_valid: list[int] = [0] * row_count
    movement_top: list[float | None] = [None] * row_count
    movement_bottom: list[float | None] = [None] * row_count
    recovery_medians: dict[int, tuple[float | None, float | None]] = {}
    movement_medians: dict[int, tuple[float | None, float | None]] = {}
    unmatched_hitters = 0

    for rally_id, source_rally in enumerate(annotations.rallies):
        in_rally = contacts["rally_id"].eq(rally_id).fillna(False)
        rally_contacts = contacts.loc[in_rally].sort_values("frame_num", kind="stable")
        row_positions = rally_contacts.index.to_list()
        frames = rally_contacts["frame_num"].to_numpy(dtype=int)
        player_ids = _side_player_ids(match, source_rally.a_is_top)

        striker_slots: list[int | None] = []
        for hitter in rally_contacts["player_id"]:
            if not pd.isna(hitter) and hitter == player_ids[TOP_SIDE]:
                striker_slots.append(0)
            elif not pd.isna(hitter) and hitter == player_ids[BOTTOM_SIDE]:
                striker_slots.append(1)
            else:
                striker_slots.append(None)
                unmatched_hitters += 1

        resolved = [index for index, slot in enumerate(striker_slots) if slot is not None]
        recovery_by_side: dict[int, list[float]] = {0: [], 1: []}
        if resolved:
            recovery_rows = recovery_at_opponent_contacts(
                positions,
                [int(frames[index]) for index in resolved],
                [striker_slots[index] for index in resolved],
                fps,
                frame_range=(source_rally.start_frame, source_rally.end_frame),
            )
            for local_index, recovery_row in zip(resolved, recovery_rows, strict=True):
                row = row_positions[local_index]
                recovery_frames_valid[row] = recovery_row["valid_frames"]
                distance = recovery_row["mean_distance"]
                recovery_distance[row] = distance
                if distance is not None:
                    recovery_by_side[recovery_row["measured_slot"]].append(distance)
        recovery_medians[rally_id] = _median_pair(recovery_by_side)

        movement_by_side: dict[int, list[float]] = {0: [], 1: []}
        if len(frames) >= 2:
            movement = movement_inefficiency(positions, frames)
            for interval in range(len(frames) - 1):
                row = row_positions[interval]
                top_value, bottom_value = movement[interval]
                movement_top[row] = float(top_value) if math.isfinite(top_value) else None
                movement_bottom[row] = float(bottom_value) if math.isfinite(bottom_value) else None
                if math.isfinite(top_value):
                    movement_by_side[0].append(float(top_value))
                if math.isfinite(bottom_value):
                    movement_by_side[1].append(float(bottom_value))
        movement_medians[rally_id] = _median_pair(movement_by_side)

    contacts["recovery_distance"] = recovery_distance
    contacts["recovery_frames_valid"] = recovery_frames_valid
    contacts["movement_inefficiency_top"] = movement_top
    contacts["movement_inefficiency_bottom"] = movement_bottom
    contacts = validate_table(SOURCE_CONTACTS, contacts)
    return SourceContactFeatures(contacts, recovery_medians, movement_medians, unmatched_hitters)


def _median_pair(values_by_slot: Mapping[int, Sequence[float]]) -> tuple[float | None, float | None]:
    """Return the (top, bottom) median, or None per side with no values."""
    top, bottom = values_by_slot[0], values_by_slot[1]
    return (
        None if not top else float(np.median(top)),
        None if not bottom else float(np.median(bottom)),
    )


def _match_players_entry(match: MatchPlayers | None) -> dict[str, object] | None:
    if match is None:
        return None
    return {
        "player_a": match.player_a.player_id,
        "player_b": match.player_b.player_id,
        "first_a_is_top": match.first_a_is_top,
    }


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


def _select_sources(
    sources: Sequence[Mapping[str, object]], video_ids: Sequence[str] | None
) -> list[Mapping[str, object]]:
    if video_ids is None:
        return list(sources)
    by_id = {str(source["video_id"]): source for source in sources}
    unknown = [video_id for video_id in video_ids if video_id not in by_id]
    if unknown:
        raise ValueError(f"run has no rally records for video_ids {unknown}")
    return [by_id[video_id] for video_id in video_ids]


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
    shots_per_rally: int | None,
    player_ids: Mapping[str, str] | None,
    flaw_marked: bool,
) -> dict[str, object]:
    run_id, source_dataset, video_id = identity
    clip_start, clip_end = clip_frames(start, end, metadata.fps, metadata.frame_count)
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
        "clip_start_frame": clip_start,
        "clip_end_frame": clip_end,
        "source_set": source_set,
        "source_rally": source_rally,
        "top_player_id": None if player_ids is None else player_ids[TOP_SIDE],
        "bottom_player_id": None if player_ids is None else player_ids[BOTTOM_SIDE],
        "shots_per_rally": shots_per_rally,
        "flaw_marked": flaw_marked,
    }


def _player_rows(
    identity: tuple[str, str, str],
    origin: RallyOrigin,
    rally_id: int,
    player_inputs: PlayerFeatureInputs,
    start: int,
    end: int,
    player_ids: Mapping[str, str] | None,
    recovery_distance_median: tuple[float | None, float | None],
    movement_inefficiency_median: tuple[float | None, float | None],
) -> list[dict[str, object]]:
    run_id, source_dataset, video_id = identity
    rows = []
    for slot, features in enumerate(player_rally_features(player_inputs, start, end)):
        rows.append(
            {
                "run_id": run_id,
                "source_dataset": source_dataset,
                "video_id": video_id,
                "rally_origin": origin.value,
                "rally_id": rally_id,
                **features._asdict(),
                "player_id": None if player_ids is None else player_ids[features.court_side],
                "recovery_distance_median": recovery_distance_median[slot],
                "movement_inefficiency_median": movement_inefficiency_median[slot],
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
    player_rows = [row for video in videos for row in video.player_rallies]
    trend_rows = [row for video in videos for row in video.player_trends]
    artifacts = [row for video in videos for row in video.artifacts]
    contacts = [video.source_contacts for video in videos if video.source_contacts is not None]
    people = {player.player_id: player for video in videos for player in video.players}
    player_rallies = pd.DataFrame(player_rows) if player_rows else empty_table(PLAYER_RALLIES)
    player_trends = pd.DataFrame(trend_rows) if trend_rows else empty_table(PLAYER_TRENDS)
    source_contacts = (
        pd.concat(contacts, ignore_index=True) if contacts else empty_table(SOURCE_CONTACTS)
    )
    _check_player_ids(people, player_rallies, player_trends, source_contacts)
    return {
        RALLIES.name: pd.DataFrame(rallies) if rallies else empty_table(RALLIES),
        PLAYER_RALLIES.name: player_rallies,
        PLAYER_TRENDS.name: player_trends,
        PLAYERS.name: (
            pd.DataFrame([player._asdict() for player in people.values()])
            if people
            else empty_table(PLAYERS)
        ),
        SOURCE_CONTACTS.name: source_contacts,
        PRIMITIVE_ARTIFACTS.name: (
            pd.DataFrame(artifacts) if artifacts else empty_table(PRIMITIVE_ARTIFACTS)
        ),
        TRANSCRIPT_SEGMENTS.name: empty_table(TRANSCRIPT_SEGMENTS),
        COMMENTARY_CHUNKS.name: empty_table(COMMENTARY_CHUNKS),
        COMMENTARY_RALLY_LINKS.name: empty_table(COMMENTARY_RALLY_LINKS),
    }


def _check_player_ids(people: Mapping[str, Player], *frames: pd.DataFrame) -> None:
    """Fail before writing if a player_id foreign key has no row in the players table."""
    for frame in frames:
        used = {value for value in frame["player_id"] if not pd.isna(value)}
        unknown = sorted(used - set(people))
        if unknown:
            raise ValueError(f"player_id values missing from the players table: {unknown}")


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
