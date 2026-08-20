"""Persistence and restoration helpers for the default coordinator runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import tomllib

import numpy as np

from annotator.config import BaseAnnotatorConfig
from annotator.point_winner import (
    GeometricVerdictRow,
    Half,
    Landing,
    Verdict,
    VerdictRow,
    VerdictSource,
)
from annotator.run_video import AnnotatorResult
from annotator.types import ContactCandidate
from annotator.video_metadata import VideoMetadata
from dataset_builder._commentary_status import (
    load_cleaning_statuses,
    save_cleaning_statuses,
)
from dataset_builder.cli import BuilderConfig, SemanticValidator, StageExecution, StagePlan
from dataset_builder.models import InterpreterIdentity, StageOutcome
from dataset_builder.records import RallyRecordProjection, SourceReference, load_rally_records
from dataset_builder.selection import (
    COMMENTARY_AVAILABLE,
    COMMENTARY_FAILED,
    COMMENTARY_INELIGIBLE,
    COMMENTARY_NO_PAIR,
    COMMENTARY_STATUSES,
    SelectionDecision,
    load_selection,
    selected_video_ids,
)
from dataset_builder.shuttle_evidence import (
    ShuttleEvidence,
    ShuttleEvidenceArtifacts,
    load_shuttle_evidence,
)
from dataset_builder.vision import (
    ANNOTATOR_RESULT_FILENAME,
    COURT_EVIDENCE_FILENAME,
    COURT_KEEP_VOTE_FILENAME,
    COURT_PRESENT_FILENAME,
    DEFINITIVE_EXCLUSION_MASK_FILENAME,
    POSE_FILENAMES,
    RAW_REPLAY_MASK_FILENAME,
    SHUTTLE_QUALITY_FILENAME,
    AnnotationArtifacts,
    AnnotationOutput,
    AnnotationRun,
    CourtVision,
    PoseArrays,
    load_court_vision,
    load_json_gz,
    load_npy_xz,
    load_pose_arrays,
)
from dataset_builder.shuttle_quality import (
    ShuttleQualitySummary,
    summarize_shuttle_quality,
)
from dataset_builder.tracknet_input import TrackNetInput
from scraper import config as scraper_config
from scraper.commentary_pairing import CanonicalPairing


@dataclass
class RuntimeState:
    """Mutable values reconstructed stage-by-stage inside one coordinator run."""

    candidates: list[dict[str, object]] = field(default_factory=list)
    transcript_ids: set[str] = field(default_factory=set)
    decisions: tuple[SelectionDecision, ...] = ()
    selected_ids: tuple[str, ...] = ()
    videos: dict[str, Path] = field(default_factory=dict)
    sources: dict[str, SourceReference] = field(default_factory=dict)
    metadata: dict[str, VideoMetadata] = field(default_factory=dict)
    tracknet_inputs: dict[str, TrackNetInput] = field(default_factory=dict)
    active_ids: set[str] = field(default_factory=set)
    chunks: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    shuttles: dict[str, ShuttleEvidence] = field(default_factory=dict)
    poses: dict[str, PoseArrays] = field(default_factory=dict)
    courts: dict[str, CourtVision] = field(default_factory=dict)
    annotations: dict[str, AnnotationOutput] = field(default_factory=dict)
    pairings: dict[str, CanonicalPairing | None] = field(default_factory=dict)
    commentary_outcomes: dict[str, StageOutcome] = field(default_factory=dict)
    commentary_reasons: dict[str, str | None] = field(default_factory=dict)
    commentary_statuses: dict[str, str] = field(default_factory=dict)
    exclusions: dict[str, str] = field(default_factory=dict)
    projections: dict[str, RallyRecordProjection] = field(default_factory=dict)
    records: list[dict[str, object]] = field(default_factory=list)


class RuntimeSupport:
    """Own shared runtime state and deterministic restoration helpers."""

    def __init__(self, config: BuilderConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.workspace = self.run_dir / "workspace"
        self.state = RuntimeState()
        self.current_interpreter: InterpreterIdentity | None = None
        self.tracknet_interpreter: InterpreterIdentity | None = None
        self.pose_interpreter: InterpreterIdentity | None = None
        self.ffmpeg_interpreter: InterpreterIdentity | None = None

    def _plan(
        self,
        *,
        name: str,
        dependencies: tuple[str, ...],
        command: tuple[str, ...],
        configuration: Mapping[str, object],
        execute: Callable[[], StageExecution],
        restore: Callable[[], None],
        validators: Mapping[str, SemanticValidator],
        interpreter: InterpreterIdentity | None = None,
        model_weights: Mapping[str, Path] | None = None,
        inputs: Mapping[str, Path] | None = None,
        secret_values: tuple[str, ...] = (),
        failure_outcome: StageOutcome = StageOutcome.FAILED,
        blocks_pipeline: bool = False,
        on_failure: Callable[[str], None] | None = None,
    ) -> StagePlan:
        return StagePlan(
            name=name,
            contract_version=f"{name.partition(':')[0]}/0.1",
            dependencies=dependencies,
            command=command,
            configuration=configuration,
            interpreter=self._current() if interpreter is None else interpreter,
            model_weights={} if model_weights is None else model_weights,
            inputs={} if inputs is None else inputs,
            execute=execute,
            restore=restore,
            semantic_validators=validators,
            secret_values=secret_values,
            failure_outcome=failure_outcome,
            blocks_pipeline=blocks_pipeline,
            on_failure=on_failure,
        )

    def _current(self) -> InterpreterIdentity:
        if self.current_interpreter is None:
            raise RuntimeError("runtime preflight did not resolve the current interpreter")
        return self.current_interpreter

    def _tracknet(self) -> InterpreterIdentity:
        if self.tracknet_interpreter is None:
            raise RuntimeError("runtime preflight did not resolve the TrackNet interpreter")
        return self.tracknet_interpreter

    def _pose(self) -> InterpreterIdentity:
        if self.pose_interpreter is None:
            raise RuntimeError("runtime preflight did not resolve the pose interpreter")
        return self.pose_interpreter

    def _ffmpeg(self) -> InterpreterIdentity:
        if self.ffmpeg_interpreter is None:
            raise RuntimeError("runtime preflight did not resolve FFmpeg")
        return self.ffmpeg_interpreter

    def _required_environment(self, name: str) -> str:
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(f"required environment variable is unset: {name}")
        return value

    def _commentary_ready(self) -> bool:
        return (
            self.config.commentary_enabled
            and bool(os.environ.get(self.config.commentary_api_key_environment))
        )

    def _commentary_unavailable_reason(self) -> str:
        if not self.config.commentary_enabled:
            return "commentary processing is disabled by configuration"
        return (
            "optional commentary environment variable is unset: "
            f"{self.config.commentary_api_key_environment}"
        )

    def _commentary_secret_values(self) -> tuple[str, ...]:
        value = os.environ.get(self.config.commentary_api_key_environment)
        return () if not value else (value,)

    def _validate_mutable_roots(self) -> None:
        """Reject mutable run roots that could redirect writes outside the run."""
        run_root = self.run_dir.resolve(strict=False)
        _mutable_child(self.run_dir / "stages", run_root, "stage root")
        workspace = _mutable_child(self.workspace, run_root, "scraper workspace")
        for name, path in (
            ("transcript directory", scraper_config.TRANSCRIPTS_DIR),
            ("chunk directory", scraper_config.CHUNKS_DIR),
            ("video directory", scraper_config.VIDEOS_DIR),
        ):
            _mutable_child(path, workspace, name)

    def _stage_dir(self, phase: str) -> Path:
        return self.run_dir / "stages" / phase

    def _video_dir(self, phase: str, video_id: str) -> Path:
        return self._stage_dir(phase) / video_id

    def _reset_stage_dir(self, phase: str, video_id: str | None = None) -> Path:
        root = _mutable_child(
            self.run_dir / "stages",
            self.run_dir.resolve(strict=False),
            "stage root",
        )
        phase_root = self._stage_dir(phase)
        if phase_root.is_symlink():
            raise ValueError(f"stage phase directory must not be a symlink: {phase_root}")
        if video_id is not None and phase_root.exists() and not phase_root.is_dir():
            raise ValueError(f"stage phase root is not a directory: {phase_root}")
        destination = self._stage_dir(phase) if video_id is None else self._video_dir(
            phase,
            video_id,
        )
        if destination.is_symlink():
            raise ValueError(f"stage output directory must not be a symlink: {destination}")
        resolved = destination.resolve(strict=False)
        if root not in resolved.parents:
            raise ValueError(f"stage output directory escapes the run: {destination}")
        if destination.is_file():
            destination.unlink()
        elif destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True)
        return destination

    def _candidate_video_ids(self) -> list[str]:
        return [
            _video_id(row.get("video_id"), "candidate video_id")
            for row in self.state.candidates
        ]

    def _clear_transcript_workspace(self) -> None:
        _clear_sidecars(scraper_config.TRANSCRIPTS_DIR, self._candidate_video_ids())

    def _clear_chunk_workspace(self, video_ids: Sequence[str]) -> None:
        _clear_sidecars(scraper_config.CHUNKS_DIR, video_ids)

    @staticmethod
    def _video_stage(phase: str, video_id: str) -> str:
        return f"{phase}:{video_id}"

    def _active_video_ids(self) -> list[str]:
        return [video_id for video_id in self.state.selected_ids if video_id in self.state.active_ids]

    def _exclude(self, video_id: str, reason: str) -> None:
        self.state.active_ids.discard(video_id)
        self.state.exclusions[video_id] = reason

    def _restore_candidates(self, path: Path) -> None:
        self.state.candidates = _read_candidates(path)

    def _snapshot_transcripts(self) -> dict[str, Path]:
        outputs: dict[str, Path] = {}
        self.state.transcript_ids.clear()
        for row in self.state.candidates:
            video_id = _video_id(row.get("video_id"), "candidate video_id")
            source = scraper_config.TRANSCRIPTS_DIR / f"{video_id}.json"
            if not source.is_file():
                continue
            destination = self._stage_dir("transcript") / f"{video_id}.json"
            _atomic_copy(source, destination)
            _load_transcript(destination, video_id)
            outputs[f"transcript.{video_id}"] = destination
            self.state.transcript_ids.add(video_id)
        return outputs

    def _transcript_files(self) -> dict[str, Path]:
        return {
            f"transcript.{video_id}": self._stage_dir("transcript") / f"{video_id}.json"
            for video_id in self.state.transcript_ids
        }

    def _restore_transcripts(self) -> None:
        self.state.transcript_ids.clear()
        self._clear_transcript_workspace()
        for row in self.state.candidates:
            video_id = _video_id(row.get("video_id"), "candidate video_id")
            path = self._stage_dir("transcript") / f"{video_id}.json"
            if path.is_file():
                _load_transcript(path, video_id)
                self.state.transcript_ids.add(video_id)
        self._sync_transcripts()

    def _validate_transcripts(self) -> bool:
        for video_id, path in (
            (video_id, self._stage_dir("transcript") / f"{video_id}.json")
            for video_id in self.state.transcript_ids
        ):
            _load_transcript(path, video_id)
        return True

    def _sync_transcripts(self) -> None:
        for video_id in self.state.transcript_ids:
            source = self._stage_dir("transcript") / f"{video_id}.json"
            _atomic_copy(source, scraper_config.TRANSCRIPTS_DIR / source.name)

    def _snapshot_chunks(
        self,
        phase: str,
        video_ids: Sequence[str] | None = None,
    ) -> dict[str, Path]:
        selected = (
            self._candidate_video_ids()
            if video_ids is None
            else list(video_ids)
        )
        outputs: dict[str, Path] = {}
        self.state.chunks.clear()
        for video_id in selected:
            source = scraper_config.CHUNKS_DIR / f"{video_id}.json"
            if not source.is_file():
                continue
            destination = self._stage_dir(phase) / "chunks" / source.name
            _atomic_copy(source, destination)
            self.state.chunks[video_id] = _load_chunks(destination, video_id)
            outputs[f"chunks.{video_id}"] = destination
        return outputs

    def _restore_chunks(self, phase: str) -> None:
        self.state.chunks.clear()
        selected = (
            self._candidate_video_ids()
            if phase == "triage"
            else list(self.state.selected_ids)
        )
        self._clear_chunk_workspace(selected)
        root = self._stage_dir(phase) / "chunks"
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            video_id = path.stem
            chunks = _load_chunks(path, video_id)
            self.state.chunks[video_id] = chunks
            _atomic_copy(path, scraper_config.CHUNKS_DIR / path.name)

    def _restore_triage(self, candidates_path: Path) -> None:
        self._restore_candidates(candidates_path)
        self._restore_chunks("triage")

    def _triage_chunk_files(self) -> dict[str, Path]:
        root = self._stage_dir("triage") / "chunks"
        if not root.is_dir():
            return {}
        return {f"chunks.{path.stem}": path for path in sorted(root.glob("*.json"))}

    def _validate_chunks(self, phase: str) -> bool:
        root = self._stage_dir(phase) / "chunks"
        for path in sorted(root.glob("*.json")) if root.is_dir() else ():
            _load_chunks(path, path.stem)
        return True

    def _persist_commentary_statuses(self, path: Path) -> Path:
        return save_cleaning_statuses(
            path,
            self.state.commentary_statuses,
            self.state.selected_ids,
        )

    def _restore_commentary_cleaning(self, statuses_path: Path) -> None:
        cleaning_root = self._stage_dir("commentary_cleaning") / "chunks"
        phase = "commentary_cleaning" if any(cleaning_root.glob("*.json")) else "triage"
        self._restore_chunks(phase)
        for video_id in self.state.selected_ids:
            self.state.commentary_statuses.pop(video_id, None)
        self.state.commentary_statuses.update(
            load_cleaning_statuses(statuses_path, self.state.selected_ids)
        )

    def _validate_commentary_statuses(self, path: Path) -> bool:
        load_cleaning_statuses(path, self.state.selected_ids)
        return True

    def _mark_failed_commentary(self, rows: Sequence[Mapping[str, object]]) -> None:
        for row in rows:
            video_id = _video_id(row.get("video_id"), "candidate video_id")
            self.state.commentary_statuses[video_id] = COMMENTARY_FAILED

    def _restore_selection(self, path: Path) -> None:
        self.state.decisions = load_selection(path)
        self.state.selected_ids = selected_video_ids(self.state.decisions)

    def _restore_downloads(self, sources_path: Path) -> None:
        with sources_path.open("rb") as handle:
            payload = tomllib.load(handle)
        if payload.get("dataset") != self.config.source_dataset:
            raise ValueError("download source dataset differs from the configured dataset")
        videos = payload.get("videos")
        if not isinstance(videos, dict):
            raise ValueError("download sources manifest has no videos table")
        self.state.videos.clear()
        self.state.sources.clear()
        selected = set(self.state.selected_ids)
        for basename, raw_entry in videos.items():
            if not isinstance(basename, str) or not isinstance(raw_entry, dict):
                raise ValueError("download sources manifest contains an invalid video entry")
            video_id = str(raw_entry.get("video_id", ""))
            if video_id not in selected:
                continue
            path = scraper_config.VIDEOS_DIR / basename
            if not path.is_file():
                continue
            if video_id in self.state.videos:
                raise ValueError(f"downloaded source for {video_id!r} is duplicated")
            reference = SourceReference(
                video_id=video_id,
                basename=basename,
                title=_string(raw_entry.get("title"), "source title"),
                url=_string(raw_entry.get("url"), "source url"),
                commentary_eligible=_boolean(
                    raw_entry.get("commentary_eligible"), "source commentary_eligible",
                ),
            )
            self.state.videos[video_id] = path
            self.state.sources[video_id] = reference
        for video_id in selected - set(self.state.videos):
            self.state.exclusions[video_id] = "selected video was not downloaded"

    def _validate_downloads(self, sources_path: Path) -> bool:
        self._restore_downloads(sources_path)
        return True

    def _restore_metadata(self, video_id: str, path: Path) -> None:
        metadata = VideoMetadata.from_dict(load_json_gz(path))
        expected = self.state.videos[video_id].resolve(strict=True)
        if metadata.source_path.resolve(strict=True) != expected:
            raise ValueError("persisted metadata source differs from the selected video")
        self.state.metadata[video_id] = metadata
        self.state.active_ids.add(video_id)

    def _validate_metadata(self, video_id: str, path: Path) -> bool:
        self._restore_metadata(video_id, path)
        return True

    def _restore_shuttle(
        self,
        video_id: str,
        artifacts: ShuttleEvidenceArtifacts,
    ) -> None:
        logical_input = self.state.tracknet_inputs[video_id].metadata
        canonical = self.state.metadata[video_id]
        self.state.shuttles[video_id] = load_shuttle_evidence(
            artifacts=artifacts,
            input_video=canonical.source_path,
            input_height=logical_input.height,
            frame_count=logical_input.frame_count,
            stride=self.config.tracknet_stride,
            tracknet_model=self.config.tracknet_model,
            inpaint_model=self.config.inpaint_model,
        )

    def _validate_shuttle(
        self,
        video_id: str,
        artifacts: ShuttleEvidenceArtifacts,
    ) -> bool:
        self._restore_shuttle(video_id, artifacts)
        return True

    def _pose_files(self, video_id: str) -> dict[str, Path]:
        root = self._video_dir("pose", video_id)
        return {f"pose_{name}": root / filename for name, filename in POSE_FILENAMES.items()}

    def _restore_pose(self, video_id: str, output_dir: Path) -> None:
        self.state.poses[video_id] = load_pose_arrays(
            output_dir,
            self.state.metadata[video_id].frame_count,
        )

    def _validate_pose(self, video_id: str, output_dir: Path) -> bool:
        self._restore_pose(video_id, output_dir)
        return True

    def _court_files(self, video_id: str) -> dict[str, Path]:
        root = self._video_dir("court", video_id)
        return {
            "court_evidence": root / COURT_EVIDENCE_FILENAME,
            "court_keep_vote": root / COURT_KEEP_VOTE_FILENAME,
            "court_present": root / COURT_PRESENT_FILENAME,
        }

    def _restore_court(self, video_id: str, output_dir: Path) -> None:
        metadata = self.state.metadata[video_id]
        self.state.courts[video_id] = load_court_vision(
            output_dir,
            video_id=video_id,
            frame_count=metadata.frame_count,
            resolution=(float(metadata.width), float(metadata.height)),
        )

    def _validate_court(self, video_id: str, output_dir: Path) -> bool:
        self._restore_court(video_id, output_dir)
        return True

    def _annotation_files(self, video_id: str) -> dict[str, Path]:
        root = self._video_dir("annotation", video_id)
        return {
            "annotator_result": root / ANNOTATOR_RESULT_FILENAME,
            "raw_replay_mask": root / RAW_REPLAY_MASK_FILENAME,
            "definitive_exclusion_mask": root / DEFINITIVE_EXCLUSION_MASK_FILENAME,
            "shuttle_quality": root / SHUTTLE_QUALITY_FILENAME,
        }

    def _restore_annotation(self, video_id: str, output_dir: Path) -> None:
        shuttle = self.state.shuttles[video_id]
        expected_quality = summarize_shuttle_quality(
            shuttle.track,
            shuttle.inpaint_fill_mask,
            shuttle.guard_codes,
            BaseAnnotatorConfig().rejected_grades,
        )
        self.state.annotations[video_id] = _load_annotation(
            output_dir,
            video_id,
            self.state.metadata[video_id].frame_count,
            expected_quality,
        )

    def _validate_annotation(self, video_id: str, output_dir: Path) -> bool:
        self._restore_annotation(video_id, output_dir)
        return True

    def _chunk_file(self, video_id: str) -> Path | None:
        for phase in ("commentary_cleaning", "triage"):
            path = self._stage_dir(phase) / "chunks" / f"{video_id}.json"
            if path.is_file():
                return path
        return None

    def _restore_pairing(self, video_id: str, path: Path) -> None:
        payload = load_json_gz(path)
        if set(payload) != {"schema", "video_id", "rows"}:
            raise ValueError("commentary pairing payload fields differ")
        if payload["schema"] != "commentary-pairing/0.1" or payload["video_id"] != video_id:
            raise ValueError("commentary pairing identity differs")
        rows = payload["rows"]
        if not isinstance(rows, list):
            raise ValueError("commentary pairing rows must be a list")
        pairing = CanonicalPairing(video_id, self.state.metadata[video_id], tuple(rows))
        outcome = (
            StageOutcome.PROCESSED
            if self.config.commentary_enabled
            else StageOutcome.SKIPPED
        )
        self.state.pairings[video_id] = pairing
        self.state.commentary_outcomes[video_id] = outcome
        self.state.commentary_reasons[video_id] = (
            None if outcome is StageOutcome.PROCESSED else self._commentary_unavailable_reason()
        )

    def _validate_pairing(self, video_id: str, path: Path) -> bool:
        self._restore_pairing(video_id, path)
        return True

    def _unavailable_pairing(self, video_id: str, reason: str) -> None:
        self.state.pairings[video_id] = None
        self.state.commentary_outcomes[video_id] = StageOutcome.UNAVAILABLE
        self.state.commentary_reasons[video_id] = reason
        self.state.commentary_statuses[video_id] = COMMENTARY_FAILED

    def _restore_records(self, path: Path) -> None:
        self.state.records = load_rally_records(path)

    def _validate_records(self, path: Path) -> bool:
        self._restore_records(path)
        return True

    def _missing_commentary_reasons(
        self,
        video_id: str,
        annotation: AnnotationOutput,
        pairing: CanonicalPairing | None,
    ) -> dict[int, str]:
        if pairing is None:
            reason = self.state.commentary_reasons.get(video_id) or "commentary unavailable"
            return {rally_id: reason for rally_id in range(len(annotation.run.result.spans))}
        base_status = self._commentary_status(video_id)
        return {
            rally_id: (
                base_status
                if base_status != COMMENTARY_AVAILABLE
                else COMMENTARY_NO_PAIR
            )
            for rally_id, row in enumerate(pairing.rows)
            if row["chunk_id"] in (None, "")
        }

    def _commentary_provenance(self, video_id: str) -> dict[str, object]:
        transcript_path = self._stage_dir("transcript") / f"{video_id}.json"
        transcript_method = "unavailable"
        if transcript_path.is_file():
            transcript_method = str(_load_transcript(transcript_path, video_id)["source"])
        chunks = self.state.chunks.get(video_id, [])
        cleaned = any("text_clean" in chunk for chunk in chunks)
        return {
            "transcript": {"method": transcript_method, "configuration": {}},
            "cleaning": {
                "method": "commentary_cleaning" if cleaned else "unavailable",
                "configuration": {"model": scraper_config.CLEAN_MODEL} if cleaned else {},
            },
            "pairing": {
                "method": "first_chunk_after_rally",
                "configuration": {"window_seconds": scraper_config.PAIR_WINDOW_S},
            },
        }

    def _selection_status(self, video_id: str) -> str:
        decision = next(item for item in self.state.decisions if item.video_id == video_id)
        return decision.commentary_status

    def _commentary_status(self, video_id: str) -> str:
        source = self.state.sources.get(video_id)
        if source is not None and not source.commentary_eligible:
            return COMMENTARY_INELIGIBLE
        if video_id in self.state.commentary_statuses:
            return self.state.commentary_statuses[video_id]
        pairing = self.state.pairings.get(video_id)
        if pairing is not None and any(row["chunk_id"] not in (None, "") for row in pairing.rows):
            return COMMENTARY_AVAILABLE
        base = self._selection_status(video_id)
        return base if base != COMMENTARY_AVAILABLE else COMMENTARY_NO_PAIR


def _atomic_copy(source: Path, destination: Path) -> Path:
    if not Path(source).is_file():
        raise FileNotFoundError(f"stage source is not a regular file: {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(Path(source).read_bytes())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _mutable_child(path: Path, parent: Path, name: str) -> Path:
    """Return one direct mutable child without following a redirecting symlink."""
    candidate = Path(path)
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {candidate}")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"{name} must be a directory: {candidate}")
    resolved_parent = Path(parent).resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    if resolved.parent != resolved_parent:
        raise ValueError(f"{name} must be a direct child of {resolved_parent}: {candidate}")
    return resolved


def _clear_sidecars(directory: Path, video_ids: Sequence[str]) -> None:
    for video_id in video_ids:
        path = directory / f"{_video_id(video_id, 'sidecar video_id')}.json"
        if path.exists() and not (path.is_file() or path.is_symlink()):
            raise ValueError(f"sidecar path is not a file: {path}")
        path.unlink(missing_ok=True)


def _read_candidates(path: Path) -> list[dict[str, object]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != scraper_config.CANDIDATES_COLUMNS:
            raise ValueError("candidate snapshot header differs from the scraper contract")
        rows = [dict(row) for row in reader]
    ids = [_video_id(row.get("video_id"), "candidate video_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate snapshot contains duplicate video IDs")
    return rows


def _valid_candidates(path: Path) -> bool:
    _read_candidates(path)
    return True


def _write_candidates_snapshot(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=scraper_config.CANDIDATES_COLUMNS)
            writer.writeheader()
            writer.writerows({name: row.get(name, "") for name in writer.fieldnames} for row in rows)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _load_transcript(path: Path, video_id: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"source", "segments"}:
        raise ValueError(f"transcript fields differ for {video_id}")
    _string(payload["source"], "transcript source")
    segments = payload["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"transcript segments are empty for {video_id}")
    for segment in segments:
        if not isinstance(segment, dict) or set(segment) != {"start", "end", "text"}:
            raise ValueError(f"transcript segment is malformed for {video_id}")
        start = _finite_float(segment["start"], "transcript segment start")
        end = _finite_float(segment["end"], "transcript segment end")
        _string(segment["text"], "transcript segment text")
        if start < 0 or end < start:
            raise ValueError(f"transcript segment times are invalid for {video_id}")
    return payload


def _load_chunks(path: Path, video_id: str) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"commentary chunks must be a list for {video_id}")
    chunks = [dict(chunk) for chunk in payload if isinstance(chunk, dict)]
    if len(chunks) != len(payload):
        raise ValueError(f"commentary chunks contain non-objects for {video_id}")
    ids = [_string(chunk.get("chunk_id"), "commentary chunk_id") for chunk in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"commentary chunks contain duplicate IDs for {video_id}")
    for chunk in chunks:
        if not {"chunk_id", "start", "end", "text"}.issubset(chunk):
            raise ValueError(f"commentary chunk fields are incomplete for {video_id}")
        start = _finite_float(chunk["start"], "commentary chunk start")
        end = _finite_float(chunk["end"], "commentary chunk end")
        _string(chunk["text"], "commentary chunk text")
        if start < 0 or end < start:
            raise ValueError(f"commentary chunk times are invalid for {video_id}")
    return chunks


def _valid_selection(path: Path) -> bool:
    load_selection(path)
    return True


def _load_report(path: Path, run_id: str) -> None:
    payload = load_json_gz(path)
    expected = {
        "schema", "run_id", "source_commit", "selected_video_ids",
        "processed_video_ids", "exclusions", "rally_count", "commentary_status",
        "stage_outcomes",
    }
    if set(payload) != expected:
        raise ValueError("dataset-builder report fields differ")
    if payload["schema"] != "dataset-builder-report/0.1" or payload["run_id"] != run_id:
        raise ValueError("dataset-builder report identity differs")
    _string(payload["source_commit"], "report source_commit")
    selected = _report_video_ids(payload["selected_video_ids"], "selected_video_ids")
    processed = _report_video_ids(payload["processed_video_ids"], "processed_video_ids")
    if not set(processed).issubset(selected):
        raise ValueError("report processed videos must be selected")
    exclusions = _dict(payload["exclusions"])
    if any(not isinstance(reason, str) or not reason for reason in exclusions.values()):
        raise ValueError("report exclusions must contain non-empty reasons")
    rally_count = _integer(payload["rally_count"], "report rally_count")
    if rally_count < 0:
        raise ValueError("report rally_count must be non-negative")
    statuses = _dict(payload["commentary_status"])
    if set(statuses) != set(selected) or any(
        status not in COMMENTARY_STATUSES for status in statuses.values()
    ):
        raise ValueError("report commentary statuses differ from selected videos")
    for name, raw_outcome in _dict(payload["stage_outcomes"]).items():
        _string(name, "report stage name")
        outcome = _dict(raw_outcome)
        if set(outcome) != {"outcome", "reason"}:
            raise ValueError("report stage outcome fields differ")
        StageOutcome(_string(outcome["outcome"], "report stage outcome"))
        if outcome["reason"] is not None:
            _string(outcome["reason"], "report stage reason")


def _valid_report(path: Path, run_id: str) -> bool:
    _load_report(path, run_id)
    return True


def _report_video_ids(payload: object, name: str) -> list[str]:
    values = [_video_id(value, f"report {name}") for value in _list(payload)]
    if len(values) != len(set(values)):
        raise ValueError(f"report {name} contains duplicates")
    return values


def _load_annotation(
    output_dir: Path,
    video_id: str,
    frame_count: int,
    expected_quality: ShuttleQualitySummary,
) -> AnnotationOutput:
    payload = load_json_gz(output_dir / ANNOTATOR_RESULT_FILENAME)
    if set(payload) != {"schema", "video_id", "result"}:
        raise ValueError("annotator result payload fields differ")
    if payload["schema"] != "annotator-result/0.1" or payload["video_id"] != video_id:
        raise ValueError("annotator result identity differs")
    result = _annotation_result(payload["result"])
    raw_mask = _boolean_mask(output_dir / RAW_REPLAY_MASK_FILENAME, frame_count)
    definitive_mask = _boolean_mask(
        output_dir / DEFINITIVE_EXCLUSION_MASK_FILENAME,
        frame_count,
    )
    quality = ShuttleQualitySummary.from_payload(
        load_json_gz(output_dir / SHUTTLE_QUALITY_FILENAME),
    )
    if quality != expected_quality:
        raise ValueError("persisted shuttle quality differs from annotation inputs")
    run = AnnotationRun(video_id, result, raw_mask, definitive_mask, quality)
    artifacts = AnnotationArtifacts(
        output_dir / ANNOTATOR_RESULT_FILENAME,
        output_dir / RAW_REPLAY_MASK_FILENAME,
        output_dir / DEFINITIVE_EXCLUSION_MASK_FILENAME,
        output_dir / SHUTTLE_QUALITY_FILENAME,
    )
    return AnnotationOutput(run, artifacts)


def _annotation_result(payload: object) -> AnnotatorResult:
    if not isinstance(payload, dict) or set(payload) != set(AnnotatorResult._fields):
        raise ValueError("persisted AnnotatorResult fields differ")
    return AnnotatorResult(
        spans=[tuple(_integer_list(row, 2, "rally span")) for row in _list(payload["spans"])],
        contacts=[_contact(row) for row in _list(payload["contacts"])],
        filtered_contacts=[_contact(row) for row in _list(payload["filtered_contacts"])],
        filtered_by_rally={
            _integer_key(key): _integer_list(value, None, "filtered contacts")
            for key, value in _dict(payload["filtered_by_rally"]).items()
        },
        striker_halves=_half_list(payload["striker_halves"]),
        n_strokes_list=_integer_list(payload["n_strokes_list"], None, "stroke counts"),
        next_servers=_half_list(payload["next_servers"]),
        fitted_first_all=_half_list(payload["fitted_first_all"]),
        verdict_rows={
            _integer_key(key): _verdict_row(value)
            for key, value in _dict(payload["verdict_rows"]).items()
        },
        landings={
            _integer_key(key): None if value is None else _landing(value)
            for key, value in _dict(payload["landings"]).items()
        },
        geometric_verdict_rows={
            _integer_key(key): _geometric_row(value)
            for key, value in _dict(payload["geometric_verdict_rows"]).items()
        },
        hit_height_by_frame={
            _integer_key(key): _integer(value, "hit height")
            for key, value in _dict(payload["hit_height_by_frame"]).items()
        },
        hit_height_failures=_hit_height_failures(payload["hit_height_failures"]),
    )


def _contact(payload: object) -> ContactCandidate:
    row = _dict(payload)
    if set(row) != set(ContactCandidate._fields):
        raise ValueError("persisted contact fields differ")
    return ContactCandidate(
        _integer(row["rally_id"], "contact rally_id"),
        _integer(row["contact_frame"], "contact frame"),
        _optional_boolean(row["proximity_ok"], "contact proximity_ok"),
        _optional_boolean(row["wrist_near"], "contact wrist_near"),
        _optional_boolean(row["suppressed"], "contact suppressed"),
    )


def _verdict_row(payload: object) -> VerdictRow:
    row = _dict(payload)
    if set(row) != set(VerdictRow._fields):
        raise ValueError("persisted verdict fields differ")
    return VerdictRow(
        _integer(row["rally_id"], "verdict rally_id"),
        Half(_string(row["striker_half"], "verdict striker_half")),
        None if row["verdict"] is None else Verdict(row["verdict"]),
        None if row["verdict_source"] is None else VerdictSource(row["verdict_source"]),
        _optional_float(row["margin_m"], "verdict margin_m"),
        _boolean(row["within_line_margin"], "within_line_margin"),
        _boolean(row["within_net_margin"], "within_net_margin"),
    )


def _landing(payload: object) -> Landing:
    row = _dict(payload)
    if set(row) != set(Landing._fields):
        raise ValueError("persisted landing fields differ")
    norm = _list(row["norm"])
    if len(norm) != 2:
        raise ValueError("persisted landing norm must contain two values")
    return Landing(
        _integer(row["frame"], "landing frame"),
        (_finite_float(norm[0], "landing x"), _finite_float(norm[1], "landing y")),
        Half(_string(row["half"], "landing half")),
        _boolean(row["at_border"], "landing at_border"),
        _boolean(row["net_ender"], "landing net_ender"),
    )


def _geometric_row(payload: object) -> GeometricVerdictRow:
    row = _dict(payload)
    if set(row) != set(GeometricVerdictRow._fields):
        raise ValueError("persisted geometric verdict fields differ")
    return GeometricVerdictRow(
        _integer(row["rally_id"], "geometric rally_id"),
        None if row["geometric_verdict"] is None else Verdict(row["geometric_verdict"]),
        None if row["geometric_winner"] is None else Half(row["geometric_winner"]),
        _optional_boolean(row["agreement"], "geometric agreement"),
        _boolean(row["window_closed_by_mask"], "geometric mask flag"),
    )


def _boolean_mask(path: Path, frame_count: int) -> np.ndarray:
    values = load_npy_xz(path)
    if values.shape != (frame_count,) or values.dtype != np.bool_:
        raise ValueError(f"persisted annotation mask is invalid: {path}")
    return values


def _half_list(payload: object) -> list[Half | None]:
    return [None if value is None else Half(value) for value in _list(payload)]


def _hit_height_failures(payload: object) -> list[tuple[int, int, int, str]]:
    failures: list[tuple[int, int, int, str]] = []
    for value in _list(payload):
        row = _list(value)
        if len(row) != 4:
            raise ValueError("hit-height failure must contain four values")
        failures.append((
            _integer(row[0], "failure rally_id"),
            _integer(row[1], "failure stroke_idx"),
            _integer(row[2], "failure frame"),
            _string(row[3], "failure reason"),
        ))
    return failures


def _integer_list(payload: object, length: int | None, name: str) -> list[int]:
    values = [_integer(value, name) for value in _list(payload)]
    if length is not None and len(values) != length:
        raise ValueError(f"{name} must contain {length} integers")
    return values


def _dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("persisted value must be an object with string keys")
    return value


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("persisted value must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _video_id(value: object, name: str) -> str:
    result = _string(value, name)
    if result in {".", ".."} or "/" in result or "\\" in result:
        raise ValueError(f"{name} must be a path-safe basename")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _integer_key(value: str) -> int:
    if not value.isdecimal() or str(int(value)) != value:
        raise ValueError("persisted mapping keys must be canonical non-negative integers")
    return int(value)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _optional_boolean(value: object, name: str) -> bool | None:
    return None if value is None else _boolean(value, name)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)
