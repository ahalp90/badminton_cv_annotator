"""Exact metadata probing and bounded raw-video span decoding."""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Validated metadata needed by the overlay renderer."""

    path: Path
    fps: Fraction
    nb_frames: int
    width: int
    height: int
    # Pixel width over pixel height. 1 for every ordinary file; anything else is
    # anamorphic footage whose coded frame is stored at the wrong shape.
    sample_aspect_ratio: Fraction = Fraction(1)


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
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"video metadata {field_name} is missing")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"video metadata {field_name} is unparseable: {value!r}") from exc
    if result <= 0:
        raise ValueError(f"video metadata {field_name} must be positive: {value!r}")
    return result


def _parse_sample_aspect_ratio(value: object) -> Fraction:
    """Read a sample aspect ratio, treating "unspecified" as square.

    Three spellings all mean unspecified and all become 1:1: the key absent
    entirely, the string "N/A", and ffmpeg's own "0:1". Absent is the common
    case, since a plain libx264 encode with no ``-aspect`` records nothing and
    neither does this tool's output. A malformed or negative ratio still fails,
    because that is a corrupt file rather than a silent one.
    """
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
    except (TypeError, ValueError, ZeroDivisionError) as exc:
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


def _run_ffprobe(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_format", "-of", "json", str(video),
    ]
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


def probe_video(video: Path) -> VideoInfo:
    """Read and validate exact video metadata without decoding frames.

    :param video: Source video path.
    :return: Validated source dimensions, frame count and exact frame rate.
    :raises FileNotFoundError: if ``video`` is not a regular file.
    :raises ValueError: if required metadata violates the CFR contract.
    :raises RuntimeError: if ffprobe cannot inspect the file.
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"video is not a regular file: {video}")
    metadata = _run_ffprobe(video)
    streams = metadata.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"video has no video stream: {video}")
    stream = streams[0]
    if not isinstance(stream, dict) or stream.get("codec_type") != "video":
        raise ValueError(f"ffprobe returned a malformed video stream: {video}")
    formats = metadata.get("format")
    if not isinstance(formats, dict):
        raise ValueError(f"video format metadata is missing: {video}")

    nb_frames = _parse_positive_int(stream.get("nb_frames"), "nb_frames")
    width = _parse_positive_int(stream.get("width"), "width")
    height = _parse_positive_int(stream.get("height"), "height")
    rate = _parse_fraction(stream.get("r_frame_rate"), "r_frame_rate")
    average_rate = _parse_fraction(stream.get("avg_frame_rate"), "avg_frame_rate")
    if rate != average_rate:
        raise ValueError(
            f"video has unequal exact frame rates: r_frame_rate={rate}, avg_frame_rate={average_rate}"
        )

    stream_start = _parse_start_time(stream.get("start_time"), "stream start_time")
    format_start = _parse_start_time(formats.get("start_time"), "format start_time")
    if stream_start != 0 or format_start != 0:
        raise ValueError(
            f"video start_time must be exactly zero: stream={stream_start}, format={format_start}"
        )
    if _has_rotation_metadata(stream):
        raise ValueError(f"video has rotation metadata: {video}")
    # Non-square pixels are carried, not rejected. Marks are fractions of the
    # coded frame, so their placement is unaffected; the ratio only decides the
    # shape the output is written at. See make_render_plan.
    sample_aspect_ratio = _parse_sample_aspect_ratio(stream.get("sample_aspect_ratio"))

    return VideoInfo(video, rate, nb_frames, width, height, sample_aspect_ratio)


def _decode_command(
    video: Path,
    first: int,
    last: int,
    fps: Fraction,
    guard_s: int,
) -> list[str]:
    n_frames = last - first + 1
    seek = max(Fraction(0), Fraction(first) / fps - guard_s)
    duration = Fraction(last - first) / fps + guard_s + 2
    t_first = (Fraction(first) - Fraction(1, 2)) / fps
    t_last = (Fraction(last) + Fraction(1, 2)) / fps
    return [
        "ffmpeg", "-v", "error", "-ss", f"{float(seek):.6f}", "-t", f"{float(duration):.6f}",
        "-copyts", "-i", str(video), "-map", "0:v:0",
        "-vf", f"select='between(t\\,{float(t_first):.6f}\\,{float(t_last):.6f})'",
        "-fps_mode", "passthrough", "-frames:v", str(n_frames), "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]


def _decode_error(command: list[str], returncode: int, stderr: bytes) -> RuntimeError:
    message = stderr.decode("utf-8", errors="replace").strip()
    return RuntimeError(
        f"ffmpeg decode failed with exit status {returncode}: {message}\ncommand: {shlex.join(command)}"
    )


def _iter_decoded_frames(
    command: list[str], n_frames: int, width: int, height: int
) -> Iterator[np.ndarray]:
    frame_bytes = width * height * 3
    expected_bytes = n_frames * frame_bytes
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"could not run ffmpeg decode: {exc}\ncommand: {shlex.join(command)}") from exc
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("ffmpeg pipes were not created")

    buffer = bytearray()
    bytes_read = 0
    try:
        while True:
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            buffer.extend(chunk)
            while len(buffer) >= frame_bytes:
                frame = np.frombuffer(bytes(buffer[:frame_bytes]), dtype=np.uint8)
                del buffer[:frame_bytes]
                yield frame.reshape(height, width, 3).copy()
        returncode = process.wait()
        stderr = process.stderr.read()
        if returncode != 0:
            raise _decode_error(command, returncode, stderr)
        if bytes_read != expected_bytes:
            raise RuntimeError(
                f"ffmpeg decode returned {bytes_read} bytes, expected {expected_bytes} "
                f"({n_frames} frames at {width}x{height}) while reporting exit status "
                f"{returncode}\n"
                f"stderr: {stderr.decode('utf-8', errors='replace').strip()}\n"
                f"command: {shlex.join(command)}"
            )
        if buffer:
            raise RuntimeError(
                f"ffmpeg decode ended with a partial frame tail of {len(buffer)} bytes\n"
                f"command: {shlex.join(command)}"
            )
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        process.stderr.read()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()


def iter_span_frames(
    video: Path,
    first: int,
    last: int,
    fps: Fraction,
    width: int,
    height: int,
    guard_s: int = 2,
) -> Iterator[np.ndarray]:
    """Yield decoded BGR frames for an inclusive source span.

    The caller supplies dimensions when it already has validated metadata. The
    decoder still checks the exact byte count and ffmpeg exit status.
    """
    if first < 0:
        raise ValueError(f"first frame must be non-negative, got {first}")
    if last < first:
        raise ValueError(f"last frame must be at least first frame, got {first}, {last}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if guard_s < 0:
        raise ValueError(f"guard_s must be non-negative, got {guard_s}")
    command = _decode_command(Path(video), first, last, fps, guard_s)
    yield from _iter_decoded_frames(command, last - first + 1, width, height)


def fetch_span(
    video: Path,
    first: int,
    last: int,
    fps: Fraction,
    guard_s: int = 2,
) -> np.ndarray:
    """Decode source frames ``[first, last]`` inclusive as ``(n, h, w, 3)`` BGR.

    Seeks deliberately early and selects the window by absolute source
    timestamp, so the result never depends on ``-ss`` landing on an exact
    frame. A short read is an error even when ffmpeg reports success.
    """
    info = probe_video(Path(video))
    # Probing anyway, so refuse a caller fps that disagrees with the file. A wrong
    # rate still satisfies the byte count when the shifted window happens to hold
    # enough frames, which would return correctly-counted, wrongly-indexed frames.
    if fps != info.fps:
        raise ValueError(f"fps {fps} does not match the probed rate {info.fps} for {video}")
    frames = list(iter_span_frames(video, first, last, fps, info.width, info.height, guard_s))
    expected_count = last - first + 1
    if len(frames) != expected_count:
        raise RuntimeError(
            f"ffmpeg decode returned {len(frames)}, expected {expected_count} "
            f"for source span [{first}, {last}]"
        )
    return np.stack(frames, axis=0)
