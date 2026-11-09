"""One long-lived ffmpeg raw-video encoder with atomic-output support hooks."""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import numpy as np


class EncodeError(RuntimeError):
    """Raised when ffmpeg cannot encode the composed frame stream."""


def _encode_command(output: Path, width: int, height: int, fps: Fraction) -> list[str]:
    exact_fps = f"{fps.numerator}/{fps.denominator}"
    return [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-framerate",
        exact_fps,
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def _validate_frame(frame: np.ndarray, width: int, height: int, index: int) -> np.ndarray:
    expected_shape = (height, width, 3)
    if frame.shape != expected_shape:
        raise ValueError(f"composed frame {index} has shape {frame.shape}, expected {expected_shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"composed frame {index} has dtype {frame.dtype}, expected uint8")
    return np.ascontiguousarray(frame)


def encode_frames(
    frames: Iterable[np.ndarray],
    output: Path,
    width: int,
    height: int,
    fps: Fraction,
) -> int:
    """Write an iterable of BGR frames through one libx264 process.

    :return: Number of frames accepted by the encoder process.
    :raises EncodeError: if ffmpeg exits non-zero or the pipe breaks.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"encoder dimensions must be positive, got {width}x{height}")
    if fps <= 0:
        raise ValueError(f"encoder fps must be positive, got {fps}")
    command = _encode_command(Path(output), width, height, fps)
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        raise EncodeError(f"could not run ffmpeg encoder: {exc}\ncommand: {shlex.join(command)}") from exc
    if process.stdin is None or process.stderr is None:
        raise EncodeError("ffmpeg encoder pipes were not created")

    frame_count = 0
    try:
        for frame_index, frame in enumerate(frames):
            checked_frame = _validate_frame(frame, width, height, frame_index)
            try:
                process.stdin.write(checked_frame.tobytes(order="C"))
            except (BrokenPipeError, OSError) as exc:
                process.wait()
                stderr = process.stderr.read()
                raise EncodeError(
                    f"ffmpeg encoder pipe failed with exit status {process.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace').strip()}\n"
                    f"command: {shlex.join(command)}"
                ) from exc
            frame_count += 1
        process.stdin.close()
        returncode = process.wait()
        stderr = process.stderr.read()
        if returncode != 0:
            raise EncodeError(
                f"ffmpeg encoder failed with exit status {returncode}: "
                f"{stderr.decode('utf-8', errors='replace').strip()}\n"
                f"command: {shlex.join(command)}"
            )
        return frame_count
    except BaseException:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stderr.read()
        raise
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.stderr.close()
