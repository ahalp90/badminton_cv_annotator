"""Bounded raw-video span decoding using caller-owned canonical metadata."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np

from annotator.video_metadata import VideoMetadata, probe_video_metadata


# Kept as import-compatible names for existing overlay consumers. The probing
# implementation and canonical field contract live outside this namespace.
VideoInfo = VideoMetadata
probe_video = probe_video_metadata


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
