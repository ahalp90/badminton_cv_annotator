"""Canonical exact video metadata for frame-aligned annotator pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any


_FULL_METADATA_ENTRIES = (
    "stream=codec_type,nb_frames,nb_read_frames,width,height,r_frame_rate,"
    "avg_frame_rate,start_time,time_base,sample_aspect_ratio:"
    "stream_tags=rotate:stream_side_data:format=start_time:"
    "frame=best_effort_timestamp"
)


@dataclass(frozen=True)
class VideoMetadata:
    """Validated CFR metadata shared by every consumer of a source video."""

    source_path: Path
    fps: Fraction
    frame_count: int
    width: int
    height: int
    sample_aspect_ratio: Fraction = Fraction(1)

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, Path):
            raise ValueError(f"source_path must be a Path: {self.source_path!r}")
        if not self.source_path.is_absolute():
            raise ValueError(f"source_path must be absolute: {self.source_path}")
        if not isinstance(self.fps, Fraction) or self.fps <= 0:
            raise ValueError(f"fps must be positive: {self.fps}")
        for name, value in (
            ("frame_count", self.frame_count),
            ("width", self.width),
            ("height", self.height),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer: {value!r}")
        if not isinstance(self.sample_aspect_ratio, Fraction) or self.sample_aspect_ratio <= 0:
            raise ValueError(f"sample_aspect_ratio must be positive: {self.sample_aspect_ratio}")

    @property
    def path(self) -> Path:
        """Compatibility name used by the validation-overlay renderer."""
        return self.source_path

    @property
    def nb_frames(self) -> int:
        """Compatibility name used by the validation-overlay renderer."""
        return self.frame_count

    def to_dict(self) -> dict[str, str | int]:
        """Return the exact, JSON-compatible persisted metadata contract."""
        return {
            "source_path": str(self.source_path),
            "fps": _fraction_text(self.fps),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "sample_aspect_ratio": _fraction_text(self.sample_aspect_ratio),
        }

    @classmethod
    def from_dict(cls, payload: object) -> VideoMetadata:
        """Validate and restore persisted canonical metadata."""
        if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
            raise ValueError("video metadata must be a JSON object")
        expected = {"source_path", "fps", "frame_count", "width", "height", "sample_aspect_ratio"}
        if set(payload) != expected:
            raise ValueError(
                f"video metadata keys differ: missing={sorted(expected - set(payload))}, "
                f"extra={sorted(set(payload) - expected)}"
            )
        source_path = payload["source_path"]
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"video metadata source_path must be a non-empty string: {source_path!r}")
        return cls(
            source_path=Path(source_path),
            fps=_parse_fraction(payload["fps"], "fps"),
            frame_count=_parse_positive_int(payload["frame_count"], "frame_count"),
            width=_parse_positive_int(payload["width"], "width"),
            height=_parse_positive_int(payload["height"], "height"),
            sample_aspect_ratio=_parse_fraction(payload["sample_aspect_ratio"], "sample_aspect_ratio"),
        )


def probe_video_metadata(video: Path) -> VideoMetadata:
    """Read strict CFR metadata and validate the decoded frame total.

    A container header count is checked when present. The decoded total and
    complete frame timestamp sequence remain required.

    :param video: Source video path.
    :return: Exact source dimensions, frame count and frame rate.
    :raises FileNotFoundError: if ``video`` is not a regular file.
    :raises ValueError: if required ffprobe metadata violates the contract.
    :raises RuntimeError: if ffprobe cannot inspect the file.
    """
    requested_path = Path(video)
    if not requested_path.is_file():
        raise FileNotFoundError(f"video is not a regular file: {requested_path}")
    source_path = requested_path.resolve(strict=True)
    metadata = _run_ffprobe(source_path)
    streams = metadata.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"video has no video stream: {source_path}")
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise ValueError(f"ffprobe returned a malformed video stream: {source_path}")
    format_metadata = metadata.get("format")
    if not isinstance(format_metadata, dict):
        raise ValueError(f"video format metadata is missing: {source_path}")

    counted_frame_count = _parse_positive_int(stream.get("nb_read_frames"), "nb_read_frames")
    header_frame_count = _parse_optional_positive_int(stream.get("nb_frames"), "nb_frames")
    if header_frame_count is not None and header_frame_count != counted_frame_count:
        raise ValueError(
            "video has conflicting frame counts: "
            f"nb_frames={header_frame_count}, nb_read_frames={counted_frame_count}"
        )
    rate = _parse_cfr_rate(stream)

    stream_start = _parse_start_time(stream.get("start_time"), "stream start_time")
    format_start = _parse_start_time(format_metadata.get("start_time"), "format start_time")
    if stream_start != 0 or format_start != 0:
        raise ValueError(
            f"video start_time must be exactly zero: stream={stream_start}, format={format_start}"
        )
    if _has_rotation_metadata(stream):
        raise ValueError(f"video has rotation metadata: {source_path}")
    _validate_frame_timestamps(metadata, stream, rate, counted_frame_count)

    return VideoMetadata(
        source_path=source_path,
        fps=rate,
        frame_count=counted_frame_count,
        width=_parse_positive_int(stream.get("width"), "width"),
        height=_parse_positive_int(stream.get("height"), "height"),
        sample_aspect_ratio=_parse_sample_aspect_ratio(stream.get("sample_aspect_ratio")),
    )


def probe_video_fps(video: Path) -> Fraction:
    """Read only the exact CFR rate for legacy callers that do not need metadata."""
    video = Path(video)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate",
        "-of",
        "json",
        str(video),
    ]
    metadata = _execute_ffprobe(video, command)
    streams = metadata.get("streams")
    if not isinstance(streams, list) or not streams or not isinstance(streams[0], dict):
        raise ValueError(f"ffprobe returned a malformed video stream: {video}")
    return _parse_cfr_rate(streams[0])


def _run_ffprobe(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_frames",
        "-show_entries",
        _FULL_METADATA_ENTRIES,
        "-of",
        "json",
        str(video),
    ]
    return _execute_ffprobe(video, command)


def _execute_ffprobe(video: Path, command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"could not run ffprobe for {video}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"ffprobe failed for {video} with exit status {completed.returncode}: {stderr}"
        )
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ffprobe returned unparseable metadata for {video}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"ffprobe returned malformed metadata for {video}")
    return metadata


def _parse_cfr_rate(stream: dict[str, Any]) -> Fraction:
    rate = _parse_fraction(stream.get("r_frame_rate"), "r_frame_rate")
    average_rate = _parse_fraction(stream.get("avg_frame_rate"), "avg_frame_rate")
    if rate != average_rate:
        raise ValueError(
            "variable frame rate is unsupported: "
            f"r_frame_rate={rate}, avg_frame_rate={average_rate}"
        )
    return rate


def _validate_frame_timestamps(
    metadata: dict[str, Any],
    stream: dict[str, Any],
    fps: Fraction,
    frame_count: int,
) -> None:
    frames = metadata.get("frames")
    if not isinstance(frames, list):
        raise ValueError("video frame timestamps are missing")
    if len(frames) != frame_count:
        raise ValueError(
            f"video has {len(frames)} frame timestamps, expected frame_count={frame_count}"
        )
    time_base = _parse_fraction(stream.get("time_base"), "time_base")
    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"video frame timestamp {frame_index} is malformed")
        timestamp = _parse_int(frame.get("best_effort_timestamp"), "best_effort_timestamp")
        observed_time = timestamp * time_base
        expected_time = Fraction(frame_index, 1) / fps
        if observed_time != expected_time:
            raise ValueError(
                "variable frame rate is unsupported: "
                f"frame {frame_index} timestamp is {observed_time}, expected {expected_time}"
            )


def _parse_fraction(value: object, field_name: str) -> Fraction:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"video metadata {field_name} is missing")
    try:
        result = Fraction(value.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"video metadata {field_name} is unparseable: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"video metadata {field_name} must be positive: {value!r}")
    return result


def _parse_start_time(value: object, field_name: str) -> Fraction:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"video metadata {field_name} is missing")
    try:
        return Fraction(value.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"video metadata {field_name} is unparseable: {value!r}") from exc


def _parse_positive_int(value: object, field_name: str) -> int:
    result = _parse_int(value, field_name)
    if result <= 0:
        raise ValueError(f"video metadata {field_name} must be positive: {value!r}")
    return result


def _parse_optional_positive_int(value: object, field_name: str) -> int | None:
    if value in (None, "N/A"):
        return None
    return _parse_positive_int(value, field_name)


def _parse_int(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"video metadata {field_name} is unparseable: {value!r}")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str) and value.strip():
        try:
            result = int(value.strip(), 10)
        except ValueError as exc:
            raise ValueError(f"video metadata {field_name} is unparseable: {value!r}") from exc
    elif value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"video metadata {field_name} is missing")
    else:
        raise ValueError(f"video metadata {field_name} is unparseable: {value!r}")
    return result


def _parse_sample_aspect_ratio(value: object) -> Fraction:
    """Read a sample aspect ratio, treating ffprobe's unspecified forms as square."""
    if value is None or (isinstance(value, str) and value.strip().upper() in {"", "N/A"}):
        return Fraction(1)
    if not isinstance(value, str):
        raise ValueError(f"video metadata sample_aspect_ratio is unparseable: {value!r}")
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"video metadata sample_aspect_ratio is unparseable: {value!r}")
    try:
        numerator, denominator = (int(part) for part in parts)
        result = Fraction(numerator, denominator)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"video metadata sample_aspect_ratio is unparseable: {value!r}") from exc
    if result == 0:
        return Fraction(1)
    if result < 0:
        raise ValueError(f"video metadata sample_aspect_ratio must be positive: {value!r}")
    return result


def _has_rotation_metadata(stream: dict[str, Any]) -> bool:
    tags = stream.get("tags")
    if isinstance(tags, dict) and "rotate" in tags:
        return True
    side_data = stream.get("side_data_list")
    if not isinstance(side_data, list):
        return False
    for entry in side_data:
        if not isinstance(entry, dict):
            continue
        if "rotation" in entry or "displaymatrix" in entry:
            return True
        side_data_type = entry.get("side_data_type")
        if isinstance(side_data_type, str) and "display matrix" in side_data_type.lower():
            return True
    return False


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"
