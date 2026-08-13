"""Coordinator plans for derived TrackNet input, shuttle, and pose stages."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import numpy as np

from bst_x.pipeline.shuttle_extractor import extract_all_shuttles
from dataset_builder._runtime_support import RuntimeState
from dataset_builder.cli import BuilderConfig, SemanticValidator, StageExecution, StagePlan
from dataset_builder.models import InterpreterIdentity, RunManifest, StageOutcome
from dataset_builder.pose_sharding import (
    POSE_SHARD_DECODE_MODE,
    extract_sharded_rtmlib_pose_stage,
)
from dataset_builder.shuttle_evidence import (
    ShuttleEvidenceArtifacts,
    persist_shuttle_evidence,
    shuttle_evidence_artifacts,
)
from dataset_builder.tracknet_input import (
    create_tracknet_input,
    load_tracknet_input,
    tracknet_input_configuration,
    tracknet_input_paths,
    tracknet_input_temporary_path,
    tracknet_proxy_command,
    validate_tracknet_input,
)
from dataset_builder.vision import convert_tracknet_csv_stage, extract_rtmlib_pose_stage
from scraper import config as scraper_config


class VisionPlanRuntime(Protocol):
    """Runtime surface required by TrackNet-input, shuttle, and pose plans."""

    config: BuilderConfig
    state: RuntimeState

    def _active_video_ids(self) -> list[str]: ...
    def _video_dir(self, phase: str, video_id: str) -> Path: ...
    def _reset_stage_dir(self, phase: str, video_id: str | None = None) -> Path: ...
    def _video_stage(self, phase: str, video_id: str) -> str: ...
    def _tracknet(self) -> InterpreterIdentity: ...
    def _pose(self) -> InterpreterIdentity: ...
    def _ffmpeg(self) -> InterpreterIdentity: ...
    def _exclude(self, video_id: str, reason: str) -> None: ...
    def _restore_shuttle(
        self,
        video_id: str,
        artifacts: ShuttleEvidenceArtifacts,
    ) -> None: ...
    def _validate_shuttle(
        self,
        video_id: str,
        artifacts: ShuttleEvidenceArtifacts,
    ) -> bool: ...
    def _restore_pose(self, video_id: str, output_dir: Path) -> None: ...
    def _validate_pose(self, video_id: str, output_dir: Path) -> bool: ...

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
        on_failure: Callable[[str], None] | None = None,
    ) -> StagePlan: ...


def tracknet_input_plans(
    runtime: VisionPlanRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered TrackNet proxy plan per active video."""
    return tuple(_tracknet_input_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _tracknet_input_plan(runtime: VisionPlanRuntime, video_id: str) -> StagePlan:
    source = runtime.state.metadata[video_id]
    output_dir = runtime._video_dir("tracknet_input", video_id)
    proxy_path, _ = tracknet_input_paths(source, output_dir)
    temporary_path = tracknet_input_temporary_path(proxy_path)
    command = tracknet_proxy_command(
        ffmpeg=runtime._ffmpeg().path,
        source_path=source.source_path,
        output_path=temporary_path,
    )

    def execute() -> StageExecution:
        runtime._reset_stage_dir("tracknet_input", video_id)
        tracknet_input = create_tracknet_input(
            source=source,
            output_dir=output_dir,
            ffmpeg=runtime._ffmpeg().path,
        )
        runtime.state.tracknet_inputs[video_id] = tracknet_input
        return StageExecution(
            StageOutcome.PROCESSED,
            tracknet_input.as_mapping(),
            {"frames": tracknet_input.metadata.frame_count},
        )

    def restore() -> None:
        runtime.state.tracknet_inputs[video_id] = load_tracknet_input(
            source=source,
            output_dir=output_dir,
        )

    return runtime._plan(
        name=runtime._video_stage("tracknet_input", video_id),
        dependencies=(runtime._video_stage("metadata", video_id),),
        command=tuple(command),
        configuration=tracknet_input_configuration(),
        interpreter=runtime._ffmpeg(),
        inputs={"source_video": source.source_path},
        execute=execute,
        restore=restore,
        validators={
            "tracknet_input_metadata": lambda _root: validate_tracknet_input(
                source=source,
                output_dir=output_dir,
            ),
        },
        on_failure=lambda reason: runtime._exclude(video_id, reason),
    )


def shuttle_plans(
    runtime: VisionPlanRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered TrackNet inference plan per active video."""
    return tuple(_shuttle_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _shuttle_plan(runtime: VisionPlanRuntime, video_id: str) -> StagePlan:
    canonical = runtime.state.metadata[video_id]
    tracknet_input = runtime.state.tracknet_inputs[video_id]
    proxy = tracknet_input.metadata
    output_dir = runtime._video_dir("shuttle", video_id)
    artifacts = shuttle_evidence_artifacts(
        output_dir,
        input_video=proxy.source_path,
        stride=runtime.config.tracknet_stride,
    )
    weights = {"tracknet": runtime.config.tracknet_model}
    if runtime.config.inpaint_model is not None:
        weights["inpaintnet"] = runtime.config.inpaint_model

    def execute() -> StageExecution:
        runtime._reset_stage_dir("shuttle", video_id)
        extract_all_shuttles(
            tracknet_dir=runtime.config.tracknet_dir,
            clips_dir=scraper_config.VIDEOS_DIR,
            video_paths=[proxy.source_path],
            output_csv_dir=output_dir,
            model_path=runtime.config.tracknet_model,
            inpaintnet_path=runtime.config.inpaint_model,
            tracknet_python=Path(runtime._tracknet().path),
            max_workers=runtime.config.tracknet_workers,
            batch_size=runtime.config.tracknet_batch_size,
            tracknet_stride=runtime.config.tracknet_stride,
            large_video=runtime.config.tracknet_large_video,
            enable_inpainting=runtime.config.inpaint_model is not None,
        )
        shuttle = convert_tracknet_csv_stage(
            artifacts.tracknet_csv,
            video_id=video_id,
            metadata=proxy,
            output_path=artifacts.shuttle_track,
        )
        evidence = persist_shuttle_evidence(
            track=shuttle.track,
            artifacts=artifacts,
            input_video=proxy.source_path,
            input_height=proxy.height,
            frame_count=proxy.frame_count,
            stride=runtime.config.tracknet_stride,
            tracknet_model=runtime.config.tracknet_model,
            inpaint_model=runtime.config.inpaint_model,
        )
        runtime.state.shuttles[video_id] = evidence
        return StageExecution(
            StageOutcome.PROCESSED,
            evidence.artifacts.as_mapping(),
            {
                "frames": canonical.frame_count,
                "inpaint_filled_frames": int(evidence.inpaint_fill_mask.sum()),
                "guard_flagged_frames": int(np.count_nonzero(evidence.guard_codes)),
            },
        )

    return runtime._plan(
        name=runtime._video_stage("shuttle", video_id),
        dependencies=(runtime._video_stage("tracknet_input", video_id),),
        command=(runtime._tracknet().path, "TrackNetV3", os.fspath(proxy.source_path)),
        configuration={
            "stride": runtime.config.tracknet_stride,
            "large_video": runtime.config.tracknet_large_video,
            "workers": runtime.config.tracknet_workers,
            "batch_size": runtime.config.tracknet_batch_size,
            "inpainting": runtime.config.inpaint_model is not None,
            "coordinate_space": "tracknet_input_pixels",
            "tracknet_directory": os.fspath(runtime.config.tracknet_dir.resolve(strict=True)),
        },
        interpreter=runtime._tracknet(),
        model_weights=weights,
        inputs={
            "tracknet_input_video": proxy.source_path,
            **_tracknet_code_inputs(runtime.config.tracknet_dir),
        },
        execute=execute,
        restore=lambda: runtime._restore_shuttle(video_id, artifacts),
        validators={
            "shuttle_evidence_schema": (
                lambda _root: runtime._validate_shuttle(video_id, artifacts)
            ),
        },
        on_failure=lambda reason: runtime._exclude(video_id, reason),
    )


def pose_plans(
    runtime: VisionPlanRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered canonical-video pose plan per active video."""
    return tuple(_pose_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _pose_plan(runtime: VisionPlanRuntime, video_id: str) -> StagePlan:
    metadata = runtime.state.metadata[video_id]
    output_dir = runtime._video_dir("pose", video_id)
    shards = runtime.config.pose_shards
    decode_mode = "sequential" if shards == 1 else POSE_SHARD_DECODE_MODE

    def execute() -> StageExecution:
        runtime._reset_stage_dir("pose", video_id)
        if shards == 1:
            extraction = extract_rtmlib_pose_stage(
                metadata=metadata,
                output_dir=output_dir,
                interpreter=runtime._pose().path,
                device=runtime.config.pose_device,
                n_max=runtime.config.pose_n_max,
            )
        else:
            extraction = extract_sharded_rtmlib_pose_stage(
                metadata=metadata,
                output_dir=output_dir,
                interpreter=runtime._pose().path,
                shards=shards,
                device=runtime.config.pose_device,
                n_max=runtime.config.pose_n_max,
                decode_mode=POSE_SHARD_DECODE_MODE,
            )
        runtime.state.poses[video_id] = extraction.arrays
        return StageExecution(
            StageOutcome.PROCESSED,
            extraction.artifacts.as_mapping(),
            {"frames": metadata.frame_count},
        )

    return runtime._plan(
        name=runtime._video_stage("pose", video_id),
        dependencies=(runtime._video_stage("metadata", video_id),),
        command=(
            runtime._pose().path,
            "-m",
            "dataset_builder.vision" if shards == 1 else "dataset_builder.pose_sharding",
            "_extract-rtmlib-pose" if shards == 1 else "_extract-sharded-rtmlib-pose",
        ),
        configuration={
            "device": runtime.config.pose_device,
            "n_max": runtime.config.pose_n_max,
            "shards": shards,
            "decode_mode": decode_mode,
        },
        interpreter=runtime._pose(),
        inputs={"source_video": metadata.source_path},
        execute=execute,
        restore=lambda: runtime._restore_pose(video_id, output_dir),
        validators={
            "pose_schema": lambda _root: runtime._validate_pose(video_id, output_dir),
        },
        on_failure=lambda reason: runtime._exclude(video_id, reason),
    )


def _tracknet_code_inputs(tracknet_dir: Path) -> dict[str, Path]:
    """Return every Python implementation file under a configured TrackNet tree."""
    root = Path(tracknet_dir).resolve(strict=True)
    return {
        f"tracknet_code.{path.relative_to(root).as_posix()}": path
        for path in sorted(root.rglob("*.py"))
        if path.is_file()
    }
