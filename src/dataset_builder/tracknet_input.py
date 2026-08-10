"""Lossless, frame-aligned TrackNet input derived from a canonical video."""

from __future__ import annotations

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class TrackNetInput:
    """Validated proxy video, persisted metadata, and exact pixel contract."""

    video_path: Path
    metadata_path: Path
    metadata: VideoMetadata

    def as_mapping(self) -> dict[str, Path]:
        """Return stable manifest names mapped to the two stage artefacts."""
        return {
            "tracknet_input_video": self.video_path,
            "tracknet_input_metadata": self.metadata_path,
        }


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


def create_tracknet_input(
    *,
    source: VideoMetadata,
    output_dir: Path,
    ffmpeg: str | Path,
) -> TrackNetInput:
    """Create, validate, and atomically publish one TrackNet-only proxy."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    proxy_path, metadata_path = tracknet_input_paths(source, root)
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
        _validate_proxy_metadata(source, probed, expected_path=temporary_path)
        os.replace(temporary_path, proxy_path)
        metadata = replace(probed, source_path=proxy_path.resolve(strict=True))
        save_json_gz(metadata_path, metadata.to_dict())
    finally:
        temporary_path.unlink(missing_ok=True)
    return TrackNetInput(proxy_path, metadata_path, metadata)


def load_tracknet_input(*, source: VideoMetadata, output_dir: Path) -> TrackNetInput:
    """Restore a proxy whose bytes were already checked by the manifest."""
    proxy_path, metadata_path = tracknet_input_paths(source, output_dir)
    if proxy_path.is_symlink() or not proxy_path.is_file():
        raise FileNotFoundError(f"TrackNet input proxy is not a regular file: {proxy_path}")
    metadata = VideoMetadata.from_dict(load_json_gz(metadata_path))
    _validate_proxy_metadata(source, metadata, expected_path=proxy_path)
    return TrackNetInput(proxy_path, metadata_path, metadata)


def validate_tracknet_input(*, source: VideoMetadata, output_dir: Path) -> bool:
    """Validate persisted proxy metadata after manifest integrity succeeds."""
    load_tracknet_input(source=source, output_dir=output_dir)
    return True


def tracknet_input_configuration() -> dict[str, str | int | bool]:
    """Return the complete effective proxy configuration for fingerprinting."""
    return {
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


def _validate_proxy_metadata(
    source: VideoMetadata,
    proxy: VideoMetadata,
    *,
    expected_path: Path,
) -> None:
    expected = Path(expected_path).resolve(strict=True)
    if proxy.source_path.resolve(strict=True) != expected:
        raise ValueError("TrackNet input metadata source differs from the proxy video")
    if proxy.frame_count != source.frame_count:
        raise ValueError(
            f"TrackNet input frame count {proxy.frame_count} != canonical {source.frame_count}"
        )
    if proxy.fps != source.fps:
        raise ValueError(f"TrackNet input FPS {proxy.fps} != canonical {source.fps}")
    if (proxy.width, proxy.height) != (TRACKNET_INPUT_WIDTH, TRACKNET_INPUT_HEIGHT):
        raise ValueError(
            f"TrackNet input dimensions {(proxy.width, proxy.height)} != "
            f"{(TRACKNET_INPUT_WIDTH, TRACKNET_INPUT_HEIGHT)}"
        )
    if proxy.sample_aspect_ratio != Fraction(1):
        raise ValueError(
            f"TrackNet input sample aspect ratio must be square, got {proxy.sample_aspect_ratio}"
        )


def _subprocess_detail(completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout).strip()
    if not detail:
        return "no diagnostic output"
    return detail[-2000:]
