"""Exact FFV1 encode/decode streams for the dataset-builder TrackNet lane."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import io
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
from types import FrameType
from typing import BinaryIO

import numpy as np

from dataset_builder.tracknet_input import (
    TRACKNET_INPUT_HEIGHT,
    TRACKNET_INPUT_WIDTH,
    tracknet_stream_decoder_command,
    tracknet_stream_producer_command,
)


TRACKNET_BGR_FRAME_BYTES = TRACKNET_INPUT_WIDTH * TRACKNET_INPUT_HEIGHT * 3
_PROCESS_TIMEOUT_SECONDS = 5.0
_DIAGNOSTIC_LIMIT = 4_000


def sampled_frame_count(frame_count: int, sample_step: int) -> int:
    """Return how many zero-based frames the median sampling rule selects."""
    _positive_integer(frame_count, "frame_count")
    _positive_integer(sample_step, "sample_step")
    return len(range(0, frame_count, sample_step))


def iter_exact_ffv1_frames(
    source: Path,
    *,
    expected_source_frames: int,
    ffmpeg: str | Path,
    sample_step: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield exact proxy-equivalent BGR frames and validate the whole stream."""
    source_path = Path(source)
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(f"TrackNet stream source is not a regular file: {source_path}")
    _positive_integer(expected_source_frames, "expected_source_frames")
    if sample_step is not None:
        _positive_integer(sample_step, "sample_step")
    expected_output_frames = (
        expected_source_frames
        if sample_step is None
        else sampled_frame_count(expected_source_frames, sample_step)
    )
    pipeline = _FFV1Pipeline(
        tracknet_stream_producer_command(
            ffmpeg=ffmpeg,
            source_path=source_path,
            sample_step=sample_step,
        ),
        tracknet_stream_decoder_command(ffmpeg=ffmpeg),
    )
    with _exit_on_cancellation():
        try:
            pipeline.start()
            stdout = pipeline.decoder_stdout()
            for frame_index in range(expected_output_frames):
                payload = _read_exact(stdout, TRACKNET_BGR_FRAME_BYTES)
                if not payload:
                    raise RuntimeError(
                        "exact FFV1 stream was truncated: "
                        f"observed {frame_index} of {expected_output_frames} frames"
                    )
                if len(payload) != TRACKNET_BGR_FRAME_BYTES:
                    raise RuntimeError(
                        "exact FFV1 stream produced a malformed frame: "
                        f"frame {frame_index} has {len(payload)} bytes, "
                        f"expected {TRACKNET_BGR_FRAME_BYTES}"
                    )
                yield np.frombuffer(payload, dtype=np.uint8).reshape(
                    TRACKNET_INPUT_HEIGHT,
                    TRACKNET_INPUT_WIDTH,
                    3,
                )
            trailing = _read_exact(stdout, TRACKNET_BGR_FRAME_BYTES)
            if len(trailing) == TRACKNET_BGR_FRAME_BYTES:
                raise RuntimeError(
                    "exact FFV1 stream produced extra frames: "
                    f"expected {expected_output_frames}"
                )
            if trailing:
                raise RuntimeError(
                    "exact FFV1 stream produced malformed trailing output: "
                    f"{len(trailing)} bytes after {expected_output_frames} frames"
                )
            pipeline.finish()
        except BaseException:
            pipeline.close(terminate=True)
            raise


class _FFV1Pipeline:
    """Own one FFV1 producer and raw-BGR decoder pair."""

    def __init__(self, producer_command: list[str], decoder_command: list[str]) -> None:
        self.producer_command = producer_command
        self.decoder_command = decoder_command
        self.producer: subprocess.Popen[bytes] | None = None
        self.decoder: subprocess.Popen[bytes] | None = None
        self.producer_stderr: BinaryIO | None = None
        self.decoder_stderr: BinaryIO | None = None
        self.closed = False

    def start(self) -> None:
        """Start both FFmpeg processes and connect their binary pipe."""
        self.producer_stderr = tempfile.TemporaryFile(mode="w+b")
        self.decoder_stderr = tempfile.TemporaryFile(mode="w+b")
        try:
            self.producer = subprocess.Popen(
                self.producer_command,
                stdout=subprocess.PIPE,
                stderr=self.producer_stderr,
            )
        except OSError as error:
            self.close(terminate=True)
            raise RuntimeError(f"could not start exact FFV1 producer: {error}") from error
        producer_stdout = self.producer.stdout
        if producer_stdout is None:
            self.close(terminate=True)
            raise RuntimeError("exact FFV1 producer stdout pipe is unavailable")
        try:
            self.decoder = subprocess.Popen(
                self.decoder_command,
                stdin=producer_stdout,
                stdout=subprocess.PIPE,
                stderr=self.decoder_stderr,
            )
        except OSError as error:
            producer_stdout.close()
            self.close(terminate=True)
            raise RuntimeError(f"could not start exact FFV1 decoder: {error}") from error
        producer_stdout.close()
        if self.decoder.stdout is None:
            self.close(terminate=True)
            raise RuntimeError("exact FFV1 decoder stdout pipe is unavailable")

    def decoder_stdout(self) -> BinaryIO:
        """Return the live raw-video pipe after successful startup."""
        if self.decoder is None or self.decoder.stdout is None:
            raise RuntimeError("exact FFV1 decoder is not running")
        return self.decoder.stdout

    def finish(self) -> None:
        """Reap a fully consumed pair and reject either nonzero exit."""
        if self.decoder is None or self.producer is None:
            raise RuntimeError("exact FFV1 pipeline was not started")
        if self.decoder.stdout is not None:
            self.decoder.stdout.close()
        try:
            decoder_returncode = self.decoder.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
            producer_returncode = self.producer.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            self.close(terminate=True)
            raise RuntimeError("exact FFV1 processes did not exit after end of stream") from error
        detail = self._failure_detail()
        self.close(terminate=False)
        if decoder_returncode != 0 or producer_returncode != 0:
            raise RuntimeError(
                "exact FFV1 stream failed: "
                f"producer={producer_returncode} decoder={decoder_returncode}: {detail}"
            )

    def close(self, *, terminate: bool) -> None:
        """Close pipes, terminate live processes when requested, and reap both."""
        if self.closed:
            return
        self.closed = True
        if self.decoder is not None and self.decoder.stdout is not None:
            self.decoder.stdout.close()
        if self.producer is not None and self.producer.stdout is not None:
            self.producer.stdout.close()
        if terminate:
            _terminate_and_reap(self.decoder)
            _terminate_and_reap(self.producer)
        else:
            _reap_if_needed(self.decoder)
            _reap_if_needed(self.producer)
        for stream in (self.decoder_stderr, self.producer_stderr):
            if stream is not None:
                stream.close()

    def _failure_detail(self) -> str:
        details = []
        for name, stream in (
            ("producer", self.producer_stderr),
            ("decoder", self.decoder_stderr),
        ):
            detail = _read_diagnostic(stream)
            if detail:
                details.append(f"{name}: {detail}")
        return "; ".join(details) or "no diagnostic output"


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _terminate_and_reap(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=_PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_PROCESS_TIMEOUT_SECONDS)


def _reap_if_needed(process: subprocess.Popen[bytes] | None) -> None:
    if process is not None and process.poll() is None:
        process.wait(timeout=_PROCESS_TIMEOUT_SECONDS)


def _read_diagnostic(stream: BinaryIO | None) -> str:
    if stream is None or stream.closed:
        return ""
    stream.flush()
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - _DIAGNOSTIC_LIMIT))
    return stream.read().decode(errors="replace").strip()


@contextmanager
def _exit_on_cancellation() -> Iterator[None]:
    """Turn terminal signals into unwinding so active children are reaped."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in handled}

    def cancel(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    try:
        for signum in handled:
            signal.signal(signum, cancel)
        yield
    finally:
        for signum in handled:
            signal.signal(signum, previous[signum])


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
