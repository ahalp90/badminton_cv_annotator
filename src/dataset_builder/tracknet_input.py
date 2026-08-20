"""Lossless, frame-aligned TrackNet input derived from a canonical video."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from fractions import Fraction
import os
from pathlib import Path
import subprocess

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder.vision import load_json_gz, save_json_gz


TRACKNET_INPUT_WIDTH = 512
TRACKNET_INPUT_HEIGHT = 288
TRACKNET_RESIZE_FILTER = "bicubic"
TRACKNET_PROXY_CODEC = "ffv1"
TRACKNET_PROXY_PIXEL_FORMAT = "yuv420p"
TRACKNET_PROXY_SAMPLE_ASPECT_RATIO = "1/1"
TRACKNET_PROXY_METADATA_FILENAME = "tracknet_input_metadata.json.gz"
TRACKNET_STREAM_IMPLEMENTATION = "ffmpeg-ffv1-nut-pipe/1"
TRACKNET_PROXY_IMPLEMENTATION = "persisted-ffv1-proxy/1"


class TrackNetInputMode(StrEnum):
    """Supported physical boundaries for the logical TrackNet input."""

    EXACT_FFV1_STREAM = "exact_ffv1_stream"
    PERSISTED_FFV1_PROXY = "persisted_ffv1_proxy"


@dataclass(frozen=True)
class TrackNetInput:
    """Validated physical input, persisted metadata, and exact pixel contract."""

    mode: TrackNetInputMode
    video_path: Path
    metadata_path: Path
    metadata: VideoMetadata

    def as_mapping(self) -> dict[str, Path]:
        """Return the run-owned stage artifacts for the selected mode."""
        artifacts = {"tracknet_input_metadata": self.metadata_path}
        if self.mode is TrackNetInputMode.PERSISTED_FFV1_PROXY:
            artifacts["tracknet_input_video"] = self.video_path
        return artifacts


def tracknet_input_paths(source: VideoMetadata, output_dir: Path) -> tuple[Path, Path]:
    """Return the canonical proxy and metadata paths for one source."""
    root = Path(output_dir)
    return (
        root / f"{source.source_path.stem}.avi",
        root / TRACKNET_PROXY_METADATA_FILENAME,
    )


def tracknet_input_temporary_path(proxy_path: Path) -> Path:
    """Return the same-directory AVI path used for atomic publication."""
    proxy = Path(proxy_path)
    return proxy.with_name(f".{proxy.name}.tmp.avi")


def tracknet_proxy_command(
    *,
    ffmpeg: str | Path,
    source_path: Path,
    output_path: Path,
) -> list[str]:
    """Build the exact FFmpeg command for the lossless TrackNet proxy."""
    return [
        os.fspath(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        os.fspath(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-vf",
        (
            f"scale={TRACKNET_INPUT_WIDTH}:{TRACKNET_INPUT_HEIGHT}:"
            f"flags={TRACKNET_RESIZE_FILTER},setsar={TRACKNET_PROXY_SAMPLE_ASPECT_RATIO}"
        ),
        "-c:v",
        TRACKNET_PROXY_CODEC,
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        TRACKNET_PROXY_PIXEL_FORMAT,
        os.fspath(output_path),
    ]


def tracknet_stream_producer_command(
    *,
    ffmpeg: str | Path,
    source_path: Path,
    sample_step: int | None = None,
) -> list[str]:
    """Build the exact FFV1 NUT producer command for one stream pass."""
    if sample_step is not None and (
        isinstance(sample_step, bool) or not isinstance(sample_step, int) or sample_step <= 0
    ):
        raise ValueError(f"sample_step must be a positive integer, got {sample_step!r}")
    filters: list[str] = []
    if sample_step is not None:
        filters.append(f"select=not(mod(n\\,{sample_step}))")
    filters.extend((
        (
            f"scale={TRACKNET_INPUT_WIDTH}:{TRACKNET_INPUT_HEIGHT}:"
            f"flags={TRACKNET_RESIZE_FILTER}"
        ),
        f"setsar={TRACKNET_PROXY_SAMPLE_ASPECT_RATIO}",
    ))
    return [
        os.fspath(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        os.fspath(source_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-vf",
        ",".join(filters),
        "-c:v",
        TRACKNET_PROXY_CODEC,
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        TRACKNET_PROXY_PIXEL_FORMAT,
        "-fps_mode",
        "passthrough",
        "-f",
        "nut",
        "pipe:1",
    ]


def tracknet_stream_decoder_command(*, ffmpeg: str | Path) -> list[str]:
    """Build the decoder command that supplies exact BGR frames to TrackNet."""
    return [
        os.fspath(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]


def create_tracknet_input(
    *,
    source: VideoMetadata,
    output_dir: Path,
    ffmpeg: str | Path,
    mode: TrackNetInputMode = TrackNetInputMode.PERSISTED_FFV1_PROXY,
) -> TrackNetInput:
    """Create the selected physical boundary and its logical metadata."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    proxy_path, metadata_path = tracknet_input_paths(source, root)
    if mode is TrackNetInputMode.EXACT_FFV1_STREAM:
        metadata = replace(
            source,
            width=TRACKNET_INPUT_WIDTH,
            height=TRACKNET_INPUT_HEIGHT,
            sample_aspect_ratio=Fraction(1),
        )
        _validate_tracknet_metadata(source, metadata, expected_path=source.source_path)
        save_json_gz(metadata_path, metadata.to_dict())
        return TrackNetInput(mode, source.source_path, metadata_path, metadata)

    temporary_path = tracknet_input_temporary_path(proxy_path)
    command = tracknet_proxy_command(
        ffmpeg=ffmpeg,
        source_path=source.source_path,
        output_path=temporary_path,
    )
    temporary_path.unlink(missing_ok=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = _subprocess_detail(completed)
            raise RuntimeError(
                f"FFmpeg TrackNet preprocessing exited with status {completed.returncode}: {detail}"
            )
        if not temporary_path.is_file():
            raise RuntimeError("FFmpeg TrackNet preprocessing did not create its output")
        probed = probe_video_metadata(temporary_path)
        _validate_tracknet_metadata(source, probed, expected_path=temporary_path)
        os.replace(temporary_path, proxy_path)
        metadata = replace(probed, source_path=proxy_path.resolve(strict=True))
        save_json_gz(metadata_path, metadata.to_dict())
    finally:
        temporary_path.unlink(missing_ok=True)
    return TrackNetInput(mode, proxy_path, metadata_path, metadata)


def load_tracknet_input(
    *,
    source: VideoMetadata,
    output_dir: Path,
    mode: TrackNetInputMode = TrackNetInputMode.PERSISTED_FFV1_PROXY,
) -> TrackNetInput:
    """Restore a selected input whose artifacts were checked by the manifest."""
    proxy_path, metadata_path = tracknet_input_paths(source, output_dir)
    metadata = VideoMetadata.from_dict(load_json_gz(metadata_path))
    if mode is TrackNetInputMode.EXACT_FFV1_STREAM:
        _validate_tracknet_metadata(source, metadata, expected_path=source.source_path)
        return TrackNetInput(mode, source.source_path, metadata_path, metadata)
    if proxy_path.is_symlink() or not proxy_path.is_file():
        raise FileNotFoundError(f"TrackNet input proxy is not a regular file: {proxy_path}")
    _validate_tracknet_metadata(source, metadata, expected_path=proxy_path)
    return TrackNetInput(mode, proxy_path, metadata_path, metadata)


def validate_tracknet_input(
    *,
    source: VideoMetadata,
    output_dir: Path,
    mode: TrackNetInputMode = TrackNetInputMode.PERSISTED_FFV1_PROXY,
) -> bool:
    """Validate selected input metadata after manifest integrity succeeds."""
    load_tracknet_input(source=source, output_dir=output_dir, mode=mode)
    return True


def tracknet_input_configuration(
    mode: TrackNetInputMode = TrackNetInputMode.PERSISTED_FFV1_PROXY,
) -> dict[str, str | int | bool]:
    """Return the complete effective TrackNet input contract for fingerprinting."""
    return {
        "mode": mode.value,
        "implementation": (
            TRACKNET_STREAM_IMPLEMENTATION
            if mode is TrackNetInputMode.EXACT_FFV1_STREAM
            else TRACKNET_PROXY_IMPLEMENTATION
        ),
        "container": "nut" if mode is TrackNetInputMode.EXACT_FFV1_STREAM else "avi",
        "persisted_video": mode is TrackNetInputMode.PERSISTED_FFV1_PROXY,
        "width": TRACKNET_INPUT_WIDTH,
        "height": TRACKNET_INPUT_HEIGHT,
        "resize_filter": TRACKNET_RESIZE_FILTER,
        "codec": TRACKNET_PROXY_CODEC,
        "codec_level": 3,
        "keyframe_interval": 1,
        "pixel_format": TRACKNET_PROXY_PIXEL_FORMAT,
        "sample_aspect_ratio": TRACKNET_PROXY_SAMPLE_ASPECT_RATIO,
        "audio": False,
        "subtitles": False,
        "data_streams": False,
        "source_metadata": False,
    }


def _validate_tracknet_metadata(
    source: VideoMetadata,
    logical_input: VideoMetadata,
    *,
    expected_path: Path,
) -> None:
    expected = Path(expected_path).resolve(strict=True)
    if logical_input.source_path.resolve(strict=True) != expected:
        raise ValueError("TrackNet input metadata source differs from the selected physical input")
    if logical_input.frame_count != source.frame_count:
        raise ValueError(
            f"TrackNet input frame count {logical_input.frame_count} != "
            f"canonical {source.frame_count}"
        )
    if logical_input.fps != source.fps:
        raise ValueError(f"TrackNet input FPS {logical_input.fps} != canonical {source.fps}")
    if (logical_input.width, logical_input.height) != (
        TRACKNET_INPUT_WIDTH,
        TRACKNET_INPUT_HEIGHT,
    ):
        raise ValueError(
            f"TrackNet input dimensions {(logical_input.width, logical_input.height)} != "
            f"{(TRACKNET_INPUT_WIDTH, TRACKNET_INPUT_HEIGHT)}"
        )
    if logical_input.sample_aspect_ratio != Fraction(1):
        raise ValueError(
            "TrackNet input sample aspect ratio must be square, "
            f"got {logical_input.sample_aspect_ratio}"
        )


def _subprocess_detail(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    if not detail:
        return "no diagnostic output"
    return detail[-2000:]
