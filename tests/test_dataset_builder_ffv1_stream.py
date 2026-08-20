"""Exact-pixel and process-lifecycle tests for the FFV1 TrackNet stream."""

from __future__ import annotations

from fractions import Fraction
import io
from pathlib import Path
import shutil
import subprocess
from typing import BinaryIO

import cv2
import numpy as np
import pytest

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder import ffv1_stream
from dataset_builder.ffv1_stream import TRACKNET_BGR_FRAME_BYTES, iter_exact_ffv1_frames
from dataset_builder.tracknet_input import TrackNetInputMode, create_tracknet_input


def _source(tmp_path: Path, frame_count: int = 2) -> VideoMetadata:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"fixture")
    return VideoMetadata(path.resolve(), Fraction(30), frame_count, 640, 480)


def _decoded_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open test video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration tools are unavailable",
)
def test_stream_frames_and_median_samples_match_persisted_proxy_bytes(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x480:rate=30",
            "-frames:v",
            "24",
            "-c:v",
            "mpeg4",
            "-pix_fmt",
            "yuv420p",
            str(source_path),
        ],
        check=True,
    )
    source = probe_video_metadata(source_path)
    proxy = create_tracknet_input(
        source=source,
        output_dir=tmp_path / "proxy",
        ffmpeg="ffmpeg",
        mode=TrackNetInputMode.PERSISTED_FFV1_PROXY,
    )

    persisted = _decoded_frames(proxy.video_path)
    streamed = list(iter_exact_ffv1_frames(
        source.source_path,
        expected_source_frames=source.frame_count,
        ffmpeg="ffmpeg",
    ))
    sampled = list(iter_exact_ffv1_frames(
        source.source_path,
        expected_source_frames=source.frame_count,
        ffmpeg="ffmpeg",
        sample_step=5,
    ))

    assert len(streamed) == len(persisted) == source.frame_count
    assert all(np.array_equal(left, right) for left, right in zip(streamed, persisted))
    assert len(sampled) == len(persisted[::5])
    assert all(np.array_equal(left, right) for left, right in zip(sampled, persisted[::5]))
    np.testing.assert_array_equal(np.median(sampled, axis=0), np.median(persisted[::5], axis=0))


class _FakeProcess:
    _next_pid = 10_000

    def __init__(self, payload: bytes, returncode: int = 0) -> None:
        self.stdout: BinaryIO = io.BytesIO(payload)
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.terminated = False
        self.killed = False
        self.waited = False
        self.pid = self._next_pid
        type(self)._next_pid += 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        self.returncode = self.final_returncode
        return self.final_returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = self.final_returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = self.final_returncode


def _install_processes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decoder_payload: bytes,
    producer_returncode: int = 0,
    decoder_returncode: int = 0,
) -> tuple[_FakeProcess, _FakeProcess]:
    producer = _FakeProcess(b"encoded", producer_returncode)
    decoder = _FakeProcess(decoder_payload, decoder_returncode)
    processes = iter((producer, decoder))
    monkeypatch.setattr(ffv1_stream.subprocess, "Popen", lambda *_args, **_kwargs: next(processes))
    return producer, decoder


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "truncated"),
        (b"x" * 19, "malformed frame"),
        (b"x" * (TRACKNET_BGR_FRAME_BYTES * 2), "extra frames"),
        (b"x" * (TRACKNET_BGR_FRAME_BYTES + 19), "malformed trailing output"),
    ],
)
def test_stream_rejects_bad_frame_boundaries_and_reaps_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    message: str,
) -> None:
    source = _source(tmp_path, frame_count=1)
    producer, decoder = _install_processes(monkeypatch, decoder_payload=payload)

    with pytest.raises(RuntimeError, match=message):
        list(iter_exact_ffv1_frames(
            source.source_path,
            expected_source_frames=1,
            ffmpeg="ffmpeg",
        ))

    assert producer.waited and decoder.waited
    assert producer.terminated and decoder.terminated


@pytest.mark.parametrize(
    ("producer_returncode", "decoder_returncode"),
    [(7, 0), (0, 9)],
)
def test_stream_rejects_nonzero_ffmpeg_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer_returncode: int,
    decoder_returncode: int,
) -> None:
    source = _source(tmp_path, frame_count=1)
    producer, decoder = _install_processes(
        monkeypatch,
        decoder_payload=b"x" * TRACKNET_BGR_FRAME_BYTES,
        producer_returncode=producer_returncode,
        decoder_returncode=decoder_returncode,
    )

    with pytest.raises(
        RuntimeError,
        match=f"producer={producer_returncode} decoder={decoder_returncode}",
    ):
        list(iter_exact_ffv1_frames(
            source.source_path,
            expected_source_frames=1,
            ffmpeg="ffmpeg",
        ))

    assert producer.waited and decoder.waited


@pytest.mark.parametrize("failed_start", ["producer", "decoder"])
def test_stream_startup_failure_reaps_started_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_start: str,
) -> None:
    source = _source(tmp_path, frame_count=1)
    producer = _FakeProcess(b"encoded")
    calls = 0

    def popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        nonlocal calls
        calls += 1
        if failed_start == "producer" or calls == 2:
            raise OSError("missing executable")
        return producer

    monkeypatch.setattr(ffv1_stream.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match=f"could not start exact FFV1 {failed_start}"):
        list(iter_exact_ffv1_frames(
            source.source_path,
            expected_source_frames=1,
            ffmpeg="ffmpeg",
        ))

    if failed_start == "decoder":
        assert producer.terminated and producer.waited


def test_stream_cancellation_during_decoder_startup_reaps_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, frame_count=1)
    producer = _FakeProcess(b"encoded")
    temporary_files: list[BinaryIO] = []
    calls = 0

    def temporary_file(*_args: object, **_kwargs: object) -> BinaryIO:
        stream = io.BytesIO()
        temporary_files.append(stream)
        return stream

    def popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise SystemExit(143)
        return producer

    monkeypatch.setattr(ffv1_stream.tempfile, "TemporaryFile", temporary_file)
    monkeypatch.setattr(ffv1_stream.subprocess, "Popen", popen)

    with pytest.raises(SystemExit) as error:
        list(iter_exact_ffv1_frames(
            source.source_path,
            expected_source_frames=1,
            ffmpeg="ffmpeg",
        ))

    assert error.value.code == 143
    assert producer.terminated and producer.waited
    assert producer.stdout.closed
    assert len(temporary_files) == 2
    assert all(stream.closed for stream in temporary_files)


def test_closing_stream_terminates_and_reaps_both_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, frame_count=2)
    producer, decoder = _install_processes(
        monkeypatch,
        decoder_payload=b"x" * (TRACKNET_BGR_FRAME_BYTES * 2),
    )
    frames = iter_exact_ffv1_frames(
        source.source_path,
        expected_source_frames=2,
        ffmpeg="ffmpeg",
    )

    next(frames)
    frames.close()

    assert producer.terminated and decoder.terminated
    assert producer.waited and decoder.waited
