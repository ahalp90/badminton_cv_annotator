"""Default external-stage runtime for the dataset-builder coordinator."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path, PurePosixPath
import shutil
import sys

from annotator.config import BaseAnnotatorConfig
from annotator.point_winner import SHIPPED_LANDING_FILTER_OPTIONS
from annotator.video_metadata import probe_video_metadata
from dataset_builder._runtime_support import (
    RuntimeSupport,
    _atomic_copy,
    _load_report,
    _valid_candidates,
    _valid_report,
    _valid_selection,
    _write_candidates_snapshot,
)
from dataset_builder._vision_plans import pose_plans, shuttle_plans, tracknet_input_plans
from dataset_builder.artifact_index import (
    VIDEO_ARTIFACT_INDEX_SCHEMA,
    artifact_index_input_manifest,
    artifact_index_path,
    artifact_index_stage_records,
    load_video_artifact_index,
    require_replayable_vision,
    write_video_artifact_index,
)
from dataset_builder.cli import BuilderConfig, StageExecution, StagePlan
from dataset_builder.fixed_sources import (
    FIXED_ACQUISITION_FILENAME,
    FixedSourceEntry,
    FixedSourceManifest,
    ResolvedFixedSource,
    load_fixed_acquisition,
    load_fixed_source_manifest,
    preflight_fixed_sources,
    save_fixed_acquisition,
    select_fixed_source_entries,
)
from dataset_builder.manifest import load_run_manifest, resolve_interpreter, run_manifest_sha256
from dataset_builder.models import RunManifest, StageOutcome
from dataset_builder.records import (
    RALLY_RECORD_COLLECTION_SCHEMA,
    RALLY_RECORD_PROJECTION_SCHEMA,
    RALLY_RECORD_SCHEMA,
    RallyRecordProjection,
    SourceReference,
    assemble_rally_records,
    load_rally_record_projection,
    write_rally_record_projection,
    write_rally_records,
)
from dataset_builder.selection import (
    COMMENTARY_INELIGIBLE,
    SELECTED_VIDEOS_FILENAME,
    SelectionDecision,
    load_selection,
    resolve_visual_selection,
    selected_video_ids,
    with_commentary_statuses,
    write_selection,
)
from dataset_builder.shuttle_evidence import shuttle_evidence_artifacts
from dataset_builder.tracknet_input import load_tracknet_input
from dataset_builder.vision import RAW_REPLAY_MASK_FILENAME, run_full_annotation_stage, save_json_gz
from scraper import commentary_cleaning, config as scraper_config
from scraper import download_scraped_videos, relevance_triage, search_index
from scraper import transcript_acquisition
from scraper.commentary_pairing import pair_video_with_metadata


METADATA_FILENAME = "video_metadata.json.gz"
PAIRING_FILENAME = "commentary_pairing.json.gz"
PROJECTION_FILENAME = "primitive_projection.json.gz"
REPORT_FILENAME = "dataset_builder_report.json.gz"
COMMENTARY_STATUS_FILENAME = "commentary_status.json.gz"


class DefaultPipelineRuntime(RuntimeSupport):
    """Concrete plans around the repository's existing producer functions."""

    def __init__(self, config: BuilderConfig, run_dir: Path, source_commit: str) -> None:
        super().__init__(config, run_dir)
        self.source_commit = source_commit
        self.detector: object | None = None
        self.fixed_manifest: FixedSourceManifest | None = None
        self.fixed_entries: tuple[FixedSourceEntry, ...] = ()
        self.fixed_source_root: Path | None = None
        self.fixed_resolved: dict[str, ResolvedFixedSource] = {}
        self.fixed_sources_verified = False

    def preflight(self) -> None:
        """Validate all required local boundaries before search mutates state."""
        self._validate_mutable_roots()
        bound_workspace = scraper_config.SCRAPE_DIR.resolve(strict=False)
        if bound_workspace != self.workspace.resolve(strict=False):
            raise RuntimeError(
                "scraper modules were imported before BADMINTON_SCRAPE_DIR was bound "
                "to this run workspace"
            )
        executables = ["ffmpeg", "ffprobe"]
        if self.config.fixed_sources is None:
            executables.insert(0, scraper_config.YTDLP_BIN)
        for executable in executables:
            if shutil.which(executable) is None:
                raise FileNotFoundError(f"required executable is unavailable: {executable}")
        self.current_interpreter = resolve_interpreter(sys.executable)
        self.tracknet_interpreter = resolve_interpreter(
            self._required_environment(self.config.tracknet_python_environment),
        )
        self.pose_interpreter = resolve_interpreter(
            self._required_environment(self.config.pose_python_environment),
        )
        self.ffmpeg_interpreter = resolve_interpreter("ffmpeg", version_option="-version")
        required_files = {
            "TrackNet batch predictor": self.config.tracknet_dir / "batch_predict.py",
            "TrackNet weights": self.config.tracknet_model,
            "CourtKeyNet weights": self.config.court_model,
        }
        if self.config.inpaint_model is not None:
            required_files["InpaintNet weights"] = self.config.inpaint_model
        missing = [name for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"required model files are unavailable: {missing}")
        self._prepare_fixed_sources()

    def preflight_replay(self) -> None:
        """Validate the CPU-only boundaries needed for downstream replay."""
        self._validate_mutable_roots()
        bound_workspace = scraper_config.SCRAPE_DIR.resolve(strict=False)
        if bound_workspace != self.workspace.resolve(strict=False):
            raise RuntimeError(
                "scraper modules were imported before BADMINTON_SCRAPE_DIR was bound "
                "to this run workspace"
            )
        if self.config.fixed_sources is None:
            raise ValueError("annotation replay requires fixed_sources configuration")
        self.current_interpreter = resolve_interpreter(sys.executable)
        required_files = {
            "TrackNet weights": self.config.tracknet_model,
            "CourtKeyNet weights": self.config.court_model,
        }
        if self.config.inpaint_model is not None:
            required_files["InpaintNet weights"] = self.config.inpaint_model
        missing = [name for name, path in required_files.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"required replay model files are unavailable: {missing}")
        self._prepare_fixed_sources()

    def prepare_annotation_replay(self, manifest: RunManifest) -> tuple[str, ...]:
        """Reload pinned expensive artifacts through production restore functions."""
        fixed_manifest, _, _, _ = self._fixed_context()
        self._restore_candidates(self._stage_dir("search") / "candidates.csv")
        self._restore_selection(self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME)
        self._restore_fixed_acquisition(
            self._stage_dir("download") / FIXED_ACQUISITION_FILENAME,
            validate_source_integrity=True,
        )
        self._restore_commentary_cleaning(
            self._stage_dir("commentary_cleaning") / COMMENTARY_STATUS_FILENAME
        )
        for video_id in self.state.selected_ids:
            resolved = self.fixed_resolved[video_id]
            index = load_video_artifact_index(
                artifact_index_path(self.run_dir, video_id),
                run_dir=self.run_dir,
                manifest=manifest,
                source_dataset=self.config.source_dataset,
                fixed_manifest=fixed_manifest,
                resolved_source=resolved,
                source_reference=self.state.sources[video_id],
                artifact_scope="replay",
                validate_models=True,
            )
            require_replayable_vision(index)
            self._restore_metadata(
                video_id,
                self._video_dir("metadata", video_id) / METADATA_FILENAME,
            )
            tracknet_input = load_tracknet_input(
                source=self.state.metadata[video_id],
                output_dir=self._video_dir("tracknet_input", video_id),
                mode=self.config.tracknet_input_mode,
            )
            self.state.tracknet_inputs[video_id] = tracknet_input
            shuttle_artifacts = shuttle_evidence_artifacts(
                self._video_dir("shuttle", video_id),
                input_video=self.state.metadata[video_id].source_path,
                stride=self.config.tracknet_stride,
            )
            self._restore_shuttle(video_id, shuttle_artifacts)
            self._restore_pose(video_id, self._video_dir("pose", video_id))
            self._restore_court(video_id, self._video_dir("court", video_id))
        return self.state.selected_ids

    def plans(self, phase: str, manifest: RunManifest) -> Sequence[StagePlan]:
        """Return global or per-video plans for one ordered phase."""
        dispatch: dict[str, Callable[[RunManifest], Sequence[StagePlan]]] = {
            "search": self._search_plans,
            "transcript": self._transcript_plans,
            "triage": self._triage_plans,
            "selection": self._selection_plans,
            "download": self._download_plans,
            "metadata": self._metadata_plans,
            "commentary_cleaning": self._cleaning_plans,
            "tracknet_input": lambda current: tracknet_input_plans(self, current),
            "shuttle": lambda current: shuttle_plans(self, current),
            "pose": lambda current: pose_plans(self, current),
            "court": self._court_plans,
            "annotation": self._annotation_plans,
            "commentary_pairing": self._pairing_plans,
            "primitive_projection": self._projection_plans,
            "artifact_index": self._artifact_index_plans,
            "assembly": self._assembly_plans,
            "report": self._report_plans,
        }
        if self.config.fixed_sources is not None:
            dispatch.update({
                "search": self._fixed_search_plans,
                "transcript": self._fixed_transcript_plans,
                "triage": self._fixed_triage_plans,
                "selection": self._fixed_selection_plans,
                "download": self._fixed_download_plans,
                "commentary_cleaning": self._fixed_cleaning_plans,
            })
        if phase not in dispatch:
            raise ValueError(f"unsupported dataset-builder phase: {phase!r}")
        return dispatch[phase](manifest)

    def _prepare_fixed_sources(self) -> None:
        fixed = self.config.fixed_sources
        if fixed is None:
            return
        source_root = Path(self._required_environment(fixed.source_root_environment))
        if not source_root.is_absolute():
            raise ValueError(
                "fixed source root environment must contain an absolute path: "
                f"{fixed.source_root_environment}"
            )
        self.fixed_manifest = load_fixed_source_manifest(fixed.manifest)
        self.fixed_entries = select_fixed_source_entries(
            self.fixed_manifest,
            fixed.video_ids,
        )
        self.fixed_source_root = source_root

    def _fixed_context(self) -> tuple[FixedSourceManifest, Path, Path, tuple[str, ...]]:
        fixed = self.config.fixed_sources
        if fixed is None or self.fixed_manifest is None or self.fixed_source_root is None:
            raise RuntimeError("fixed-source runtime preflight is incomplete")
        return (
            self.fixed_manifest,
            self.fixed_source_root,
            fixed.ground_truth_root,
            fixed.video_ids,
        )

    def _fixed_search_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        manifest, _, _, video_ids = self._fixed_context()
        output = self._stage_dir("search") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("search")
            self.state.candidates = [_fixed_candidate(entry) for entry in self.fixed_entries]
            _write_candidates_snapshot(output, self.state.candidates)
            return StageExecution(
                StageOutcome.SKIPPED,
                {"candidates": output},
                {"candidates": len(self.state.candidates)},
                "fixed-source mode bypasses search",
            )

        return (self._plan(
            name="search",
            dependencies=(),
            command=(self._current().path, "dataset_builder", "fixed-search-bypass"),
            configuration={
                "mode": "fixed",
                "manifest_md5": manifest.md5,
                "video_ids": list(video_ids),
            },
            inputs={"fixed_source_manifest": manifest.path},
            execute=execute,
            restore=lambda: self._restore_candidates(output),
            validators={"candidate_schema": lambda _root: _valid_candidates(output)},
            blocks_pipeline=True,
        ),)

    def _fixed_transcript_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        candidates = self._stage_dir("search") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("transcript")
            self._clear_transcript_workspace()
            self.state.transcript_ids.clear()
            return StageExecution(
                StageOutcome.SKIPPED,
                {},
                {"transcripts": 0, "unavailable": len(self.state.candidates)},
                "fixed-source mode bypasses transcript acquisition",
            )

        return (self._plan(
            name="transcript",
            dependencies=("search",),
            command=(self._current().path, "dataset_builder", "fixed-transcript-bypass"),
            configuration={"mode": "fixed", "enabled": False},
            inputs={"candidates": candidates},
            execute=execute,
            restore=self._restore_transcripts,
            validators={},
        ),)

    def _fixed_triage_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        search_output = self._stage_dir("search") / "candidates.csv"
        output = self._stage_dir("triage") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("triage")
            self._clear_chunk_workspace(self._candidate_video_ids())
            self.state.chunks.clear()
            _write_candidates_snapshot(output, self.state.candidates)
            return StageExecution(
                StageOutcome.SKIPPED,
                {"candidates": output},
                {"triaged": 0},
                "fixed-source mode bypasses commentary triage",
            )

        return (self._plan(
            name="triage",
            dependencies=("transcript",),
            command=(self._current().path, "dataset_builder", "fixed-triage-bypass"),
            configuration={"mode": "fixed", "enabled": False},
            inputs={"candidates": search_output},
            execute=execute,
            restore=lambda: self._restore_triage(output),
            validators={"triage_schema": lambda _root: _valid_candidates(output)},
            failure_outcome=StageOutcome.UNAVAILABLE,
        ),)

    def _fixed_selection_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        manifest, _, _, video_ids = self._fixed_context()
        output = self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME
        published = self.run_dir / SELECTED_VIDEOS_FILENAME
        candidate_input = self._stage_dir("triage") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("selection")
            decisions_list: list[SelectionDecision] = []
            for index, video_id in enumerate(video_ids):
                decisions_list.append(SelectionDecision(
                    video_id=video_id,
                    visual_selected=True,
                    selection_source="fixed_manifest",
                    selection_reason="fixed_source_requested",
                    source_order=index,
                    commentary_status=COMMENTARY_INELIGIBLE,
                ))
            decisions = tuple(decisions_list)
            write_selection(output, decisions)
            write_selection(published, decisions)
            self.state.decisions = decisions
            self.state.selected_ids = selected_video_ids(decisions)
            return StageExecution(
                StageOutcome.PROCESSED,
                {"selected_videos": output},
                {"candidates": len(decisions), "selected": len(decisions)},
            )

        return (self._plan(
            name="selection",
            dependencies=("search",),
            command=(self._current().path, "dataset_builder", "fixed-selection"),
            configuration={
                "mode": "fixed",
                "manifest_md5": manifest.md5,
                "video_ids": list(video_ids),
            },
            inputs={"candidates": candidate_input},
            execute=execute,
            restore=lambda: self._restore_selection(output),
            validators={"selection_schema": lambda _root: _valid_selection(output)},
            blocks_pipeline=True,
        ),)

    def _fixed_download_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        manifest, source_root, ground_truth_root, video_ids = self._fixed_context()
        fixed = self.config.fixed_sources
        if fixed is None:
            raise RuntimeError("fixed-source configuration is unavailable")
        output = self._stage_dir("download") / FIXED_ACQUISITION_FILENAME

        def execute() -> StageExecution:
            self._reset_stage_dir("download")
            resolved = preflight_fixed_sources(
                manifest,
                source_root=source_root,
                ground_truth_root=ground_truth_root,
                requested_video_ids=video_ids,
                metadata_probe=probe_video_metadata,
            )
            self._set_fixed_source_state(resolved)
            self.fixed_sources_verified = True
            save_fixed_acquisition(output, manifest, resolved)
            return StageExecution(
                StageOutcome.SKIPPED,
                {"fixed_acquisition": output},
                {"selected": len(video_ids), "resolved": len(resolved)},
                "fixed-source mode bypasses download",
            )

        return (self._plan(
            name="download",
            dependencies=("selection",),
            command=(self._current().path, "dataset_builder", "fixed-source-preflight"),
            configuration={
                "mode": "fixed",
                "manifest_md5": manifest.md5,
                "source_root_environment": fixed.source_root_environment,
                "video_ids": list(video_ids),
            },
            inputs={
                "fixed_source_manifest": manifest.path,
                "selection": self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME,
            },
            execute=execute,
            restore=lambda: self._restore_fixed_acquisition(
                output,
                validate_source_integrity=False,
            ),
            validators={
                "fixed_source_integrity": lambda _root: self._validate_fixed_acquisition(output)
            },
            blocks_pipeline=True,
        ),)

    def _fixed_cleaning_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        statuses_path = self._stage_dir("commentary_cleaning") / COMMENTARY_STATUS_FILENAME

        def execute() -> StageExecution:
            self._reset_stage_dir("commentary_cleaning")
            self.state.chunks.clear()
            statuses = self._persist_commentary_statuses(statuses_path)
            return StageExecution(
                StageOutcome.SKIPPED,
                {"commentary_statuses": statuses},
                {"cleaned": 0, "videos": 0},
                "fixed-source mode bypasses commentary cleaning",
            )

        return (self._plan(
            name="commentary_cleaning",
            dependencies=("download", "triage"),
            command=(self._current().path, "dataset_builder", "fixed-cleaning-bypass"),
            configuration={"mode": "fixed", "enabled": False},
            inputs={
                "fixed_acquisition": self._stage_dir("download") / FIXED_ACQUISITION_FILENAME
            },
            execute=execute,
            restore=lambda: self._restore_commentary_cleaning(statuses_path),
            validators={
                "commentary_status_schema": lambda _root: (
                    self._validate_commentary_statuses(statuses_path)
                )
            },
        ),)

    def _set_fixed_source_state(
        self,
        resolved_sources: Sequence[ResolvedFixedSource],
    ) -> None:
        self.state.videos.clear()
        self.state.sources.clear()
        self.state.metadata.clear()
        self.state.active_ids.clear()
        self.fixed_resolved.clear()
        for resolved in resolved_sources:
            entry = resolved.entry
            self.state.videos[entry.video_id] = resolved.metadata.source_path
            self.state.sources[entry.video_id] = SourceReference(
                video_id=entry.video_id,
                basename=entry.source_basename,
                title=PurePosixPath(entry.ground_truth.annotation_directory).name,
                url=entry.source_url,
                commentary_eligible=False,
            )
            self.state.metadata[entry.video_id] = resolved.metadata
            self.fixed_resolved[entry.video_id] = resolved
            self.state.exclusions.pop(entry.video_id, None)

    def _restore_fixed_acquisition(
        self,
        path: Path,
        *,
        validate_source_integrity: bool,
    ) -> None:
        manifest, source_root, ground_truth_root, video_ids = self._fixed_context()
        resolved = load_fixed_acquisition(
            path,
            manifest,
            source_root=source_root,
            ground_truth_root=ground_truth_root,
            requested_video_ids=video_ids,
            validate_source_integrity=validate_source_integrity,
        )
        self._set_fixed_source_state(resolved)
        self.fixed_sources_verified = True

    def _validate_fixed_acquisition(self, path: Path) -> bool:
        self._restore_fixed_acquisition(
            path,
            validate_source_integrity=not self.fixed_sources_verified,
        )
        return True

    def _search_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        output = self._stage_dir("search") / "candidates.csv"
        configuration = {
            "search_terms": self.config.search_terms,
            "result_count": self.config.search_count,
        }

        def execute() -> StageExecution:
            self._reset_stage_dir("search")
            rows = search_index.build_candidates(
                deepcopy(self.config.search_terms),
                search_count=self.config.search_count,
            )
            _atomic_copy(scraper_config.CANDIDATES_CSV, output)
            self.state.candidates = [dict(row) for row in rows]
            return StageExecution(
                StageOutcome.PROCESSED,
                {"candidates": output},
                {"candidates": len(rows)},
            )

        return (self._plan(
            name="search",
            dependencies=(),
            command=(
                self._current().path, "-m", "scraper.search_index",
                "--search-count", str(self.config.search_count),
            ),
            configuration=configuration,
            execute=execute,
            restore=lambda: self._restore_candidates(output),
            validators={"candidate_schema": lambda _root: _valid_candidates(output)},
            blocks_pipeline=True,
        ),)

    def _transcript_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        search_output = self._stage_dir("search") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("transcript")
            self._clear_transcript_workspace()
            if not self.config.commentary_enabled:
                return StageExecution(
                    StageOutcome.SKIPPED,
                    {},
                    {
                        "transcripts": 0,
                        "unavailable": len(self.state.candidates),
                    },
                    self._commentary_unavailable_reason(),
                )
            error: Exception | None = None
            try:
                transcript_acquisition.run_transcript_acquisition(
                    rows=deepcopy(self.state.candidates),
                )
            except Exception as caught:  # noqa: BLE001 - optional external boundary.
                error = caught
            outputs = self._snapshot_transcripts()
            outcome = StageOutcome.PROCESSED if outputs and error is None else StageOutcome.UNAVAILABLE
            reason = None
            if outcome is StageOutcome.UNAVAILABLE:
                reason = (
                    f"{type(error).__name__}: {error}"
                    if error is not None
                    else "no candidate transcript was available"
                )
            return StageExecution(
                outcome,
                outputs,
                {
                    "transcripts": len(outputs),
                    "unavailable": len(self.state.candidates) - len(outputs),
                },
                reason,
            )

        return (self._plan(
            name="transcript",
            dependencies=("search",),
            command=(self._current().path, "-m", "scraper.transcript_acquisition"),
            configuration={
                "enabled": self.config.commentary_enabled,
                "optional": True,
                "subtitle_languages": scraper_config.SUB_LANGS,
                "subtitle_format": scraper_config.SUB_FORMAT,
                "whisperx_model": scraper_config.WHISPERX_COARSE_MODEL,
            },
            inputs={"candidates": search_output},
            execute=execute,
            restore=self._restore_transcripts,
            validators={"transcript_schema": lambda _root: self._validate_transcripts()},
            failure_outcome=StageOutcome.UNAVAILABLE,
        ),)

    def _triage_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        llm_settings = self._triage_llm_settings()
        output = self._stage_dir("triage") / "candidates.csv"
        inputs = {"candidates": self._stage_dir("search") / "candidates.csv"}
        inputs.update(self._transcript_files())

        def execute() -> StageExecution:
            self._reset_stage_dir("triage")
            self._clear_chunk_workspace(self._candidate_video_ids())
            if not self._commentary_ready():
                _write_candidates_snapshot(output, self.state.candidates)
                return StageExecution(
                    (
                        StageOutcome.SKIPPED
                        if not self.config.commentary_enabled
                        else StageOutcome.UNAVAILABLE
                    ),
                    {"candidates": output},
                    {"triaged": 0},
                    self._commentary_unavailable_reason(),
                )
            self._sync_transcripts()
            rows = deepcopy(self.state.candidates)
            try:
                keep_by_id = relevance_triage.run_relevance_triage(
                    rows=rows,
                    llm_settings=llm_settings,
                )
            except Exception as error:  # noqa: BLE001 - optional external boundary.
                _write_candidates_snapshot(output, self.state.candidates)
                return StageExecution(
                    StageOutcome.UNAVAILABLE,
                    {"candidates": output},
                    {"triaged": 0},
                    f"{type(error).__name__}: {error}",
                )
            _atomic_copy(scraper_config.CANDIDATES_CSV, output)
            self._restore_candidates(output)
            outputs = {"candidates": output, **self._snapshot_chunks("triage")}
            return StageExecution(
                StageOutcome.PROCESSED,
                outputs,
                {"triaged": len(keep_by_id)},
            )

        return (self._plan(
            name="triage",
            dependencies=("transcript",),
            command=(
                self._current().path,
                "-m",
                "scraper.relevance_triage",
                "--provider",
                llm_settings.provider.value,
                "--model",
                llm_settings.model,
                "--api-key-environment",
                llm_settings.api_key_environment,
            ),
            configuration={
                "enabled": self.config.commentary_enabled,
                **llm_settings.provenance(),
                "max_output_tokens": scraper_config.TRIAGE_MAX_TOKENS,
                "request_timeout_seconds": scraper_config.LLM_REQUEST_TIMEOUT_S,
                "chunk_window_seconds": scraper_config.CHUNK_WINDOW_S,
                "chunk_overlap_seconds": scraper_config.CHUNK_OVERLAP_S,
            },
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_triage(output),
            validators={"triage_schema": lambda _root: _valid_candidates(output)},
            secret_values=self._commentary_secret_values(),
            failure_outcome=StageOutcome.UNAVAILABLE,
        ),)

    def _selection_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        output = self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME
        published = self.run_dir / SELECTED_VIDEOS_FILENAME
        candidate_input = self._stage_dir("triage") / "candidates.csv"

        def execute() -> StageExecution:
            self._reset_stage_dir("selection")
            decisions = resolve_visual_selection(
                self.state.candidates,
                max_videos=self.config.max_videos,
                transcript_video_ids=self.state.transcript_ids,
            )
            write_selection(output, decisions)
            write_selection(published, decisions)
            self.state.decisions = decisions
            self.state.selected_ids = selected_video_ids(decisions)
            return StageExecution(
                StageOutcome.PROCESSED,
                {"selected_videos": output},
                {
                    "candidates": len(decisions),
                    "selected": len(self.state.selected_ids),
                },
            )

        return (self._plan(
            name="selection",
            dependencies=("search",),
            command=(self._current().path, "-m", "dataset_builder", "run", "selection"),
            configuration={
                "max_videos": self.config.max_videos,
                "transcript_video_ids": sorted(self.state.transcript_ids),
            },
            inputs={"candidates": candidate_input},
            execute=execute,
            restore=lambda: self._restore_selection(output),
            validators={"selection_schema": lambda _root: _valid_selection(output)},
            blocks_pipeline=True,
        ),)

    def _download_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        candidate_input = self._stage_dir("triage") / "candidates.csv"
        sources_path = scraper_config.VIDEOS_DIR / scraper_config.SOURCES_MANIFEST_NAME

        def execute() -> StageExecution:
            outcomes = download_scraped_videos.download_all_videos(
                candidates_path=candidate_input,
                output_dir=scraper_config.VIDEOS_DIR,
                max_workers=self.config.download_workers,
                selected_video_ids=self.state.selected_ids,
                dataset=self.config.source_dataset,
                accept_silent_video=True,
            )
            self._restore_downloads(sources_path)
            outputs = {"sources": sources_path}
            outputs.update({f"video.{video_id}": path for video_id, path in self.state.videos.items()})
            failed_ids = {outcome.video_id for outcome in outcomes if outcome.failed}
            failed_ids.update(set(self.state.selected_ids) - set(self.state.videos))
            counts = {
                "selected": len(self.state.selected_ids),
                "downloaded": len(self.state.videos),
                "failed": len(failed_ids),
            }
            if failed_ids:
                return StageExecution(
                    StageOutcome.FAILED,
                    outputs,
                    counts,
                    "not every selected video was downloaded and verified",
                )
            outcome = StageOutcome.PROCESSED if self.state.selected_ids else StageOutcome.EXCLUDED
            reason = None if outcome is StageOutcome.PROCESSED else "visual selection was empty"
            return StageExecution(outcome, outputs, counts, reason)

        return (self._plan(
            name="download",
            dependencies=("selection",),
            command=(self._current().path, "-m", "scraper.download_scraped_videos"),
            configuration={
                "dataset": self.config.source_dataset,
                "workers": self.config.download_workers,
                "selected_video_ids": list(self.state.selected_ids),
                "accept_silent_video": True,
            },
            inputs={"candidates": candidate_input, "selection": (
                self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME
            )},
            execute=execute,
            restore=lambda: self._restore_downloads(sources_path),
            validators={"download_sources": lambda _root: self._validate_downloads(sources_path)},
            blocks_pipeline=True,
        ),)

    def _metadata_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        return tuple(
            self._metadata_plan(video_id)
            for video_id in self.state.selected_ids
            if video_id in self.state.videos
        )

    def _metadata_plan(self, video_id: str) -> StagePlan:
        video_path = self.state.videos[video_id]
        output = self._video_dir("metadata", video_id) / METADATA_FILENAME
        fixed_mode = self.config.fixed_sources is not None

        def execute() -> StageExecution:
            self._reset_stage_dir("metadata", video_id)
            metadata = self.state.metadata[video_id] if fixed_mode else probe_video_metadata(video_path)
            if metadata.source_path.resolve(strict=True) != video_path.resolve(strict=True):
                raise ValueError("canonical metadata source differs from the selected video")
            save_json_gz(output, metadata.to_dict())
            self.state.metadata[video_id] = metadata
            self.state.active_ids.add(video_id)
            return StageExecution(
                StageOutcome.PROCESSED,
                {"video_metadata": output},
                {"frames": metadata.frame_count},
            )

        command = (
            (self._current().path, "dataset_builder", "restore-fixed-metadata", video_id)
            if fixed_mode
            else ("ffprobe", os.fspath(video_path))
        )
        inputs = (
            {"fixed_acquisition": self._stage_dir("download") / FIXED_ACQUISITION_FILENAME}
            if fixed_mode
            else {"source_video": video_path}
        )
        configuration: dict[str, object] = {
            "reject_vfr": True,
            "exact_frame_count": True,
        }
        if fixed_mode:
            configuration["source"] = "fixed_preflight"
        return self._plan(
            name=self._video_stage("metadata", video_id),
            dependencies=("download",),
            command=command,
            configuration=configuration,
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_metadata(video_id, output),
            validators={
                "canonical_metadata": lambda _root: self._validate_metadata(video_id, output),
            },
            on_failure=lambda reason: self._exclude(video_id, reason),
        )

    def _cleaning_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        llm_settings = self._clean_llm_settings()
        inputs = self._triage_chunk_files()
        inputs.update({f"video.{key}": value for key, value in self.state.videos.items()})
        statuses_path = self._stage_dir("commentary_cleaning") / COMMENTARY_STATUS_FILENAME

        def execute() -> StageExecution:
            self._reset_stage_dir("commentary_cleaning")
            self._restore_chunks("triage")
            selected = set(self.state.selected_ids)
            selected_rows: list[dict[str, object]] = []
            for row in self.state.candidates:
                video_id = str(row.get("video_id"))
                source = self.state.sources.get(video_id)
                if video_id in selected and source is not None and source.commentary_eligible:
                    selected_rows.append(deepcopy(row))
            if not self._commentary_ready():
                statuses = self._persist_commentary_statuses(statuses_path)
                return StageExecution(
                    (
                        StageOutcome.SKIPPED
                        if not self.config.commentary_enabled
                        else StageOutcome.UNAVAILABLE
                    ),
                    {"commentary_statuses": statuses},
                    {"cleaned": 0},
                    self._commentary_unavailable_reason(),
                )
            try:
                cleaned = commentary_cleaning.run_clean(
                    rows=selected_rows,
                    llm_settings=llm_settings,
                )
                commentary_cleaning.run_fine(scraper_config.VIDEOS_DIR, rows=selected_rows)
            except Exception as error:  # noqa: BLE001 - optional external boundary.
                self._mark_failed_commentary(selected_rows)
                statuses = self._persist_commentary_statuses(statuses_path)
                return StageExecution(
                    StageOutcome.UNAVAILABLE,
                    {"commentary_statuses": statuses},
                    {"cleaned": 0},
                    f"{type(error).__name__}: {error}",
                )
            outputs = self._snapshot_chunks("commentary_cleaning", self.state.selected_ids)
            outputs["commentary_statuses"] = self._persist_commentary_statuses(statuses_path)
            return StageExecution(
                StageOutcome.PROCESSED,
                outputs,
                {"cleaned": sum(cleaned.values()), "videos": len(self.state.chunks)},
            )

        dependencies = ("download", "triage")
        return (self._plan(
            name="commentary_cleaning",
            dependencies=dependencies,
            command=(
                self._current().path,
                "-m",
                "scraper.commentary_cleaning",
                "--provider",
                llm_settings.provider.value,
                "--model",
                llm_settings.model,
                "--api-key-environment",
                llm_settings.api_key_environment,
            ),
            configuration={
                "enabled": self.config.commentary_enabled,
                "provider": llm_settings.provider.value,
                "clean_model": llm_settings.model,
                "key_environment": llm_settings.api_key_environment,
                "fine_model": scraper_config.WHISPERX_FINE_MODEL,
                "alternative_phrasings": scraper_config.ALT_PHRASINGS_K,
                "bert_score_minimum": scraper_config.CLEAN_BERTSCORE_MIN,
                "request_timeout_seconds": scraper_config.LLM_REQUEST_TIMEOUT_S,
            },
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_commentary_cleaning(statuses_path),
            validators={
                "cleaned_chunk_schema": lambda _root: self._validate_chunks(
                    "commentary_cleaning",
                ),
                "commentary_status_schema": lambda _root: (
                    self._validate_commentary_statuses(statuses_path)
                ),
            },
            secret_values=self._commentary_secret_values(),
            failure_outcome=StageOutcome.UNAVAILABLE,
        ),)

    def _court_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        return tuple(self._court_plan(video_id) for video_id in self._active_video_ids())

    def _court_plan(self, video_id: str) -> StagePlan:
        from dataset_builder.vision import build_detected_court_stage

        metadata = self.state.metadata[video_id]
        output_dir = self._video_dir("court", video_id)

        def execute() -> StageExecution:
            self._reset_stage_dir("court", video_id)
            if self.detector is None:
                from courtkeynet.wrapper import CourtKeyNetDetector

                self.detector = CourtKeyNetDetector(
                    weights_path=self.config.court_model,
                    device=self.config.court_device,
                    resize_mode=self.config.court_resize_mode,
                )
            court = build_detected_court_stage(
                video_id=video_id,
                metadata=metadata,
                pose=self.state.poses[video_id],
                detector=self.detector,
                output_dir=output_dir,
            )
            if court.artifacts is None:
                raise RuntimeError("court stage did not persist its operational artifacts")
            self.state.courts[video_id] = court
            return StageExecution(
                StageOutcome.PROCESSED,
                court.artifacts.as_mapping(),
                {"scenes": len(court.raw_cuts)},
            )

        return self._plan(
            name=self._video_stage("court", video_id),
            dependencies=(self._video_stage("pose", video_id),),
            command=(self._current().path, "CourtKeyNet", os.fspath(metadata.source_path)),
            configuration={
                "device": self.config.court_device,
                "resize_mode": self.config.court_resize_mode,
            },
            model_weights={"courtkeynet": self.config.court_model},
            inputs={"source_video": metadata.source_path, **self._pose_files(video_id)},
            execute=execute,
            restore=lambda: self._restore_court(video_id, output_dir),
            validators={"court_schema": lambda _root: self._validate_court(video_id, output_dir)},
            on_failure=lambda reason: self._exclude(video_id, reason),
        )

    def _annotation_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        return tuple(self._annotation_plan(video_id) for video_id in self._active_video_ids())

    def _annotation_plan(self, video_id: str) -> StagePlan:
        metadata = self.state.metadata[video_id]
        output_dir = self._video_dir("annotation", video_id)

        def execute() -> StageExecution:
            self._reset_stage_dir("annotation", video_id)
            shuttle = self.state.shuttles[video_id]
            annotation = run_full_annotation_stage(
                video_id=video_id,
                metadata=metadata,
                track=shuttle.track,
                inpaint_fill_mask=shuttle.inpaint_fill_mask,
                guard_codes=shuttle.guard_codes,
                pose=self.state.poses[video_id],
                court=self.state.courts[video_id],
                output_dir=output_dir,
            )
            self.state.annotations[video_id] = annotation
            return StageExecution(
                StageOutcome.PROCESSED,
                annotation.artifacts.as_mapping(),
                {
                    "rallies": len(annotation.run.result.spans),
                    "guard_rejected_frames": (
                        annotation.run.shuttle_quality.guard_rejected_frames
                    ),
                    "inpaint_filled_frames": (
                        annotation.run.shuttle_quality.inpaint_filled_frames
                    ),
                },
            )

        inputs = {
            **self.state.shuttles[video_id].artifacts.as_mapping(),
            **self._pose_files(video_id),
            **self._court_files(video_id),
        }
        return self._plan(
            name=self._video_stage("annotation", video_id),
            dependencies=(
                self._video_stage("shuttle", video_id),
                self._video_stage("pose", video_id),
                self._video_stage("court", video_id),
            ),
            command=(self._current().path, "-m", "annotator.run_video", video_id),
            configuration=_annotation_configuration(),
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_annotation(video_id, output_dir),
            validators={
                "annotation_schema": lambda _root: self._validate_annotation(video_id, output_dir),
            },
            on_failure=lambda reason: self._exclude(video_id, reason),
        )

    def _pairing_plans(self, _manifest: RunManifest) -> Sequence[StagePlan]:
        return tuple(self._pairing_plan(video_id) for video_id in self._active_video_ids())

    def _pairing_plan(self, video_id: str) -> StagePlan:
        metadata = self.state.metadata[video_id]
        annotation = self.state.annotations[video_id]
        output = self._video_dir("commentary_pairing", video_id) / PAIRING_FILENAME

        def execute() -> StageExecution:
            self._reset_stage_dir("commentary_pairing", video_id)
            spans = [
                (rally_id, start, end)
                for rally_id, (start, end) in enumerate(annotation.run.result.spans)
            ]
            pairing = pair_video_with_metadata(
                video_id,
                spans,
                (
                    self.state.chunks.get(video_id, [])
                    if self.state.sources[video_id].commentary_eligible
                    else []
                ),
                annotation.run.raw_replay_mask,
                metadata,
            )
            save_json_gz(output, {
                "schema": "commentary-pairing/0.1",
                "video_id": video_id,
                "rows": [dict(row) for row in pairing.rows],
            })
            outcome = (
                StageOutcome.PROCESSED
                if self.config.commentary_enabled
                else StageOutcome.SKIPPED
            )
            reason = None if outcome is StageOutcome.PROCESSED else (
                self._commentary_unavailable_reason()
            )
            self.state.pairings[video_id] = pairing
            self.state.commentary_outcomes[video_id] = outcome
            self.state.commentary_reasons[video_id] = reason
            paired = sum(row["chunk_id"] not in (None, "") for row in pairing.rows)
            return StageExecution(
                outcome,
                {"commentary_pairing": output},
                {"rallies": len(pairing.rows), "paired": paired},
                reason,
            )

        inputs = {
            "raw_replay_mask": (
                self._video_dir("annotation", video_id) / RAW_REPLAY_MASK_FILENAME
            ),
            "video_metadata": self._video_dir("metadata", video_id) / METADATA_FILENAME,
        }
        chunk = self._chunk_file(video_id)
        if chunk is not None:
            inputs["commentary_chunks"] = chunk
        return self._plan(
            name=self._video_stage("commentary_pairing", video_id),
            dependencies=(self._video_stage("annotation", video_id), "commentary_cleaning"),
            command=(self._current().path, "-m", "scraper.commentary_pairing", video_id),
            configuration={
                "enabled": self.config.commentary_enabled,
                "pair_window_seconds": scraper_config.PAIR_WINDOW_S,
            },
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_pairing(video_id, output),
            validators={"pairing_schema": lambda _root: self._validate_pairing(video_id, output)},
            failure_outcome=StageOutcome.UNAVAILABLE,
            on_failure=lambda reason: self._unavailable_pairing(video_id, reason),
        )

    def _projection_plans(self, manifest: RunManifest) -> Sequence[StagePlan]:
        input_manifest = _projection_input_manifest(manifest)
        return tuple(
            self._projection_plan(video_id, input_manifest)
            for video_id in self._active_video_ids()
        )

    def _projection_plan(self, video_id: str, input_manifest: RunManifest) -> StagePlan:
        output = self._video_dir("primitive_projection", video_id) / PROJECTION_FILENAME

        def execute() -> StageExecution:
            self._reset_stage_dir("primitive_projection", video_id)
            projection = self._assemble_one(input_manifest, video_id)
            write_rally_record_projection(output, input_manifest, projection)
            self.state.projections[video_id] = projection
            return StageExecution(
                StageOutcome.PROCESSED,
                {"primitive_projection": output},
                {"rallies": len(projection.records)},
            )

        inputs = self._annotation_files(video_id)
        pairing_path = self._video_dir("commentary_pairing", video_id) / PAIRING_FILENAME
        if pairing_path.is_file():
            inputs["commentary_pairing"] = pairing_path
        return self._plan(
            name=self._video_stage("primitive_projection", video_id),
            dependencies=(
                self._video_stage("annotation", video_id),
                self._video_stage("commentary_pairing", video_id),
            ),
            command=(self._current().path, "dataset_builder.records", "project", video_id),
            configuration={
                "projection_schema": RALLY_RECORD_PROJECTION_SCHEMA,
                "record_schema": RALLY_RECORD_SCHEMA,
            },
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_projection(video_id, output, input_manifest),
            validators={
                "projection_schema": lambda _root: self._validate_projection(
                    video_id,
                    output,
                    input_manifest,
                ),
            },
            on_failure=lambda reason: self._exclude(video_id, reason),
        )

    def _artifact_index_plans(self, manifest: RunManifest) -> Sequence[StagePlan]:
        if self.config.fixed_sources is None:
            return ()
        input_manifest = artifact_index_input_manifest(manifest)
        return tuple(
            self._artifact_index_plan(video_id, input_manifest)
            for video_id in self.state.selected_ids
        )

    def _artifact_index_plan(
        self,
        video_id: str,
        input_manifest: RunManifest,
    ) -> StagePlan:
        fixed_manifest, _, _, _ = self._fixed_context()
        resolved = self.fixed_resolved[video_id]
        source_reference = self.state.sources[video_id]
        output = artifact_index_path(self.run_dir, video_id)
        records = artifact_index_stage_records(input_manifest, video_id)
        input_manifest_digest = run_manifest_sha256(input_manifest)

        def execute() -> StageExecution:
            self._reset_stage_dir("artifact_index", video_id)
            index = write_video_artifact_index(
                output,
                manifest=input_manifest,
                source_dataset=self.config.source_dataset,
                fixed_manifest=fixed_manifest,
                resolved_source=resolved,
                source_reference=source_reference,
            )
            return StageExecution(
                StageOutcome.PROCESSED,
                {"video_artifact_index": output},
                {
                    "indexed_stages": len(index.stages),
                    "indexed_artifacts": sum(len(stage.outputs) for stage in index.stages),
                },
            )

        def restore() -> None:
            load_video_artifact_index(
                output,
                run_dir=self.run_dir,
                manifest=input_manifest,
                source_dataset=self.config.source_dataset,
                fixed_manifest=fixed_manifest,
                resolved_source=resolved,
                source_reference=source_reference,
            )

        def validate() -> bool:
            restore()
            return True

        return self._plan(
            name=self._video_stage("artifact_index", video_id),
            dependencies=tuple(stage.name for stage in records),
            command=(self._current().path, "dataset_builder", "index", video_id),
            configuration={
                "schema": VIDEO_ARTIFACT_INDEX_SCHEMA,
                "input_manifest_sha256": input_manifest_digest,
                "video_id": video_id,
            },
            execute=execute,
            restore=restore,
            validators={"artifact_index_schema": lambda _root: validate()},
        )

    def _assembly_plans(self, manifest: RunManifest) -> Sequence[StagePlan]:
        active = self._active_video_ids()
        projection_manifest = _projection_input_manifest(manifest)
        dependencies = tuple(
            self._video_stage("primitive_projection", video_id)
            for video_id in active
        ) or ("selection",)
        inputs = {
            f"projection.{video_id}": (
                self._video_dir("primitive_projection", video_id) / PROJECTION_FILENAME
            )
            for video_id in active
        }
        records_path = self.run_dir / "rally_records.json.gz"

        def execute() -> StageExecution:
            input_manifest = load_run_manifest(self.run_dir)
            projections = [
                self.state.projections[video_id]
                for video_id in active
            ]
            records = [record for projection in projections for record in projection.records]
            artifacts = write_rally_records(
                self.run_dir,
                input_manifest,
                projections,
                code_version=self.source_commit,
                assembly_configuration={"record_mode": "primitive"},
                projection_manifest=(
                    projection_manifest if projections else input_manifest
                ),
            )
            self.state.records = records
            return StageExecution(
                StageOutcome.PROCESSED,
                {"rally_records": artifacts.records},
                {"videos": len(active), "rallies": len(records)},
            )

        return (self._plan(
            name="assembly",
            dependencies=dependencies,
            command=(self._current().path, "dataset_builder.records", "assemble"),
            configuration={
                "collection_schema": RALLY_RECORD_COLLECTION_SCHEMA,
                "projection_schema": RALLY_RECORD_PROJECTION_SCHEMA,
                "record_schema": RALLY_RECORD_SCHEMA,
                "validation_only": True,
            },
            inputs=inputs,
            execute=execute,
            restore=lambda: self._restore_records(records_path),
            validators={"record_schema": lambda _root: self._validate_records(records_path)},
            blocks_pipeline=True,
        ),)

    def _report_plans(self, manifest: RunManifest) -> Sequence[StagePlan]:
        output = self.run_dir / REPORT_FILENAME
        records_path = self.run_dir / "rally_records.json.gz"
        selection_path = self.run_dir / SELECTED_VIDEOS_FILENAME
        dependencies = tuple(
            stage.name for stage in manifest.stages if stage.name != "report"
        )

        def final_decisions() -> tuple[SelectionDecision, ...]:
            statuses = {
                video_id: self._commentary_status(video_id)
                for video_id in self.state.selected_ids
            }
            return with_commentary_statuses(self.state.decisions, statuses)

        def execute() -> StageExecution:
            input_manifest = load_run_manifest(self.run_dir)
            commentary = {
                video_id: self._commentary_status(video_id)
                for video_id in self.state.selected_ids
            }
            write_selection(selection_path, final_decisions())
            save_json_gz(output, {
                "schema": "dataset-builder-report/0.1",
                "run_id": manifest.run_id,
                "source_commit": self.source_commit,
                "selected_video_ids": list(self.state.selected_ids),
                "processed_video_ids": self._active_video_ids(),
                "exclusions": self.state.exclusions,
                "rally_count": len(self.state.records),
                "commentary_status": commentary,
                "stage_outcomes": {
                    stage.name: {"outcome": stage.outcome.value, "reason": stage.reason}
                    for stage in input_manifest.stages
                },
            })
            return StageExecution(
                StageOutcome.PROCESSED,
                {
                    "dataset_builder_report": output,
                    "selected_videos": selection_path,
                },
                {
                    "selected_videos": len(self.state.selected_ids),
                    "processed_videos": len(self._active_video_ids()),
                    "rallies": len(self.state.records),
                },
            )

        def restore() -> None:
            _load_report(output, manifest.run_id)
            if tuple(final_decisions()) != tuple(load_selection(selection_path)):
                raise ValueError("published commentary statuses differ from the run state")

        return (self._plan(
            name="report",
            dependencies=dependencies,
            command=(self._current().path, "dataset_builder", "report"),
            configuration={"schema": "dataset-builder-report/0.1"},
            inputs={
                "rally_records": records_path,
                "selection": self._stage_dir("selection") / SELECTED_VIDEOS_FILENAME,
            },
            execute=execute,
            restore=restore,
            validators={
                "report_schema": lambda _root: _valid_report(output, manifest.run_id),
                "selection_statuses": lambda _root: (
                    tuple(final_decisions()) == tuple(load_selection(selection_path))
                ),
            },
            blocks_pipeline=True,
        ),)

    def _assemble_one(
        self,
        manifest: RunManifest,
        video_id: str,
    ) -> RallyRecordProjection:
        annotation = self.state.annotations[video_id]
        pairing = self.state.pairings.get(video_id)
        commentary_outcome = self.state.commentary_outcomes.get(
            video_id,
            StageOutcome.UNAVAILABLE,
        )
        commentary_reason = self.state.commentary_reasons.get(video_id)
        if commentary_outcome is not StageOutcome.PROCESSED and commentary_reason is None:
            commentary_reason = "commentary pairing was unavailable"
        missing_reasons = self._missing_commentary_reasons(video_id, annotation, pairing)
        return assemble_rally_records(
            manifest=manifest,
            source_dataset=self.config.source_dataset,
            video_id=video_id,
            source_reference=self.state.sources[video_id],
            metadata=self.state.metadata[video_id],
            annotation=annotation.run.result,
            annotation_fps=self.state.metadata[video_id].fps,
            annotation_frame_count=self.state.metadata[video_id].frame_count,
            pairing=pairing,
            chunks=self.state.chunks.get(video_id, []),
            commentary_outcome=commentary_outcome,
            commentary_reason=commentary_reason,
            commentary_missing_reasons=missing_reasons,
            commentary_provenance=self._commentary_provenance(video_id),
            mask_stage_name=self._video_stage("annotation", video_id),
        )

    def _restore_projection(
        self,
        video_id: str,
        path: Path,
        manifest: RunManifest,
    ) -> None:
        self.state.projections[video_id] = load_rally_record_projection(
            path,
            manifest,
            video_id=video_id,
        )

    def _validate_projection(
        self,
        video_id: str,
        path: Path,
        manifest: RunManifest,
    ) -> bool:
        self._restore_projection(video_id, path, manifest)
        return True


def _fixed_candidate(entry: FixedSourceEntry) -> dict[str, object]:
    """Return one candidate-shaped row for the shared reporting contracts."""
    return {
        "video_id": entry.video_id,
        "url": entry.source_url,
        "title": PurePosixPath(entry.ground_truth.annotation_directory).name,
        "channel": "ShuttleSet",
        "duration_s": "",
        "upload_date": "",
        "search_term": "fixed_source_manifest",
        "substream": "match",
        "doubles_suspect": "False",
        "duration_suspect": "False",
        "upload_date_suspect": "False",
        "keep": "True",
        "triage_verdict": "fixed_source",
    }


def _projection_input_manifest(manifest: RunManifest) -> RunManifest:
    """Return the common manifest snapshot captured before projection starts."""
    stages = tuple(
        stage
        for stage in manifest.stages
        if not stage.name.startswith("primitive_projection:")
        and not stage.name.startswith("artifact_index:")
        and stage.name not in {"assembly", "report"}
    )
    return manifest if stages == manifest.stages else replace(manifest, stages=stages)


def _annotation_configuration() -> dict[str, object]:
    base = BaseAnnotatorConfig()
    return {
        "thresholds": dict(base.thresholds._asdict()),
        "dead_mask_mode": base.dead_mask_mode.value,
        "smoothing_mode": base.smoothing_mode.value,
        "overrides_base30": base.overrides_base30,
        "span_open": None if base.span_open is None else base.span_open.value,
        "gap_state_demotion_bound": base.gap_state_demotion_bound,
        "reentry_guard_variant": (
            None if base.reentry_guard_variant is None else base.reentry_guard_variant.value
        ),
        "reentry_guard_buffer": base.reentry_guard_buffer,
        "quiet_start_window": base.quiet_start_window,
        "rejected_grades": sorted(base.rejected_grades),
        "landing_filter": dict(SHIPPED_LANDING_FILTER_OPTIONS._asdict()),
    }
