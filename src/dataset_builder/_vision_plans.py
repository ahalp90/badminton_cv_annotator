"""Coordinator plans for derived TrackNet input, shuttle, and pose stages."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from bst_x.pipeline.shuttle_extractor import extract_all_shuttles
from dataset_builder._runtime_support import _tracknet_code_inputs
from dataset_builder.cli import StageExecution, StagePlan
from dataset_builder.models import RunManifest, StageOutcome
from dataset_builder.pose_sharding import (
    POSE_SHARD_DECODE_MODE,
    extract_sharded_rtmlib_pose_stage,
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

if TYPE_CHECKING:
    from dataset_builder._pipeline_runtime import DefaultPipelineRuntime


def tracknet_input_plans(
    runtime: DefaultPipelineRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered TrackNet proxy plan per active video."""
    return tuple(_tracknet_input_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _tracknet_input_plan(runtime: DefaultPipelineRuntime, video_id: str) -> StagePlan:
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
    runtime: DefaultPipelineRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered TrackNet inference plan per active video."""
    return tuple(_shuttle_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _shuttle_plan(runtime: DefaultPipelineRuntime, video_id: str) -> StagePlan:
    canonical = runtime.state.metadata[video_id]
    tracknet_input = runtime.state.tracknet_inputs[video_id]
    proxy = tracknet_input.metadata
    output_dir = runtime._video_dir("shuttle", video_id)
    csv_path = output_dir / f"{proxy.source_path.stem}_ball.csv"
    track_path = output_dir / "shuttle_track.npy.xz"
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
            csv_path,
            video_id=video_id,
            metadata=proxy,
            output_path=track_path,
        ).require_value()
        runtime.state.tracks[video_id] = shuttle.track
        return StageExecution(
            StageOutcome.PROCESSED,
            {"tracknet_csv": csv_path, "shuttle_track": track_path},
            {"frames": canonical.frame_count},
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
        restore=lambda: runtime._restore_track(video_id, track_path),
        validators={
            "track_schema": lambda _root: runtime._validate_track(video_id, track_path),
        },
        on_failure=lambda reason: runtime._exclude(video_id, reason),
    )


def pose_plans(
    runtime: DefaultPipelineRuntime,
    _manifest: RunManifest,
) -> tuple[StagePlan, ...]:
    """Return one source-ordered canonical-video pose plan per active video."""
    return tuple(_pose_plan(runtime, video_id) for video_id in runtime._active_video_ids())


def _pose_plan(runtime: DefaultPipelineRuntime, video_id: str) -> StagePlan:
    metadata = runtime.state.metadata[video_id]
    output_dir = runtime._video_dir("pose", video_id)
    shards = runtime.config.pose_shards
    decode_mode = "sequential" if shards == 1 else POSE_SHARD_DECODE_MODE

    def execute() -> StageExecution:
        runtime._reset_stage_dir("pose", video_id)
        if shards == 1:
            result = extract_rtmlib_pose_stage(
                metadata=metadata,
                output_dir=output_dir,
                interpreter=runtime._pose().path,
                device=runtime.config.pose_device,
                n_max=runtime.config.pose_n_max,
            )
        else:
            result = extract_sharded_rtmlib_pose_stage(
                metadata=metadata,
                output_dir=output_dir,
                interpreter=runtime._pose().path,
                shards=shards,
                device=runtime.config.pose_device,
                n_max=runtime.config.pose_n_max,
                decode_mode=POSE_SHARD_DECODE_MODE,
            )
        extraction = result.require_value()
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
