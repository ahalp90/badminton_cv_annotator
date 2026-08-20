"""Frame, metadata, and failure gates for exact TrackNet inputs."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pandas as pd
import pytest

from annotator.video_metadata import VideoMetadata, probe_video_metadata
from dataset_builder import cli, tracknet_input
from dataset_builder.tracknet_input import (
    TRACKNET_INPUT_HEIGHT,
    TRACKNET_INPUT_WIDTH,
    TrackNetInputMode,
    create_tracknet_input,
    load_tracknet_input,
    tracknet_input_paths,
    tracknet_proxy_command,
    tracknet_stream_decoder_command,
    tracknet_stream_producer_command,
)
from dataset_builder.vision import convert_tracknet_csv_stage, load_npy_xz


def _metadata(tmp_path: Path, *, frame_count: int = 6) -> VideoMetadata:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture source")
    return VideoMetadata(
        source.resolve(),
        Fraction(30),
        frame_count,
        1920,
        1080,
    )


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


def test_tracked_trial_configuration_has_the_bounded_external_scope() -> None:
    config = cli.load_builder_config(cli.REPO_ROOT / "configs/dataset_builder/trial.toml")

    assert config.search_count == 5
    assert config.max_videos == 2
    assert config.tracknet_stride == 8
    assert config.tracknet_input_mode is TrackNetInputMode.EXACT_FFV1_STREAM
    assert config.pose_shards == 8
    assert list(config.search_terms) == ["match"]
    assert len(config.search_terms["match"]) == 1


def test_persisted_proxy_is_an_explicit_selectable_fallback(tmp_path: Path) -> None:
    tracked = cli.REPO_ROOT / "configs/dataset_builder/trial.toml"
    payload = tracked.read_text(encoding="utf-8")
    payload = payload.replace('tracknet_input_mode = "exact_ffv1_stream"', (
        'tracknet_input_mode = "persisted_ffv1_proxy"'
    ))
    payload = payload.replace("tracknet_stride = 8", "tracknet_stride = 1")
    payload = payload.replace("tracknet_large_video = true", "tracknet_large_video = false")
    config_path = tmp_path / "proxy.toml"
    config_path.write_text(payload, encoding="utf-8")

    config = cli.load_builder_config(config_path)

    assert config.tracknet_input_mode is TrackNetInputMode.PERSISTED_FFV1_PROXY
    assert config.tracknet_stride == 1
    assert config.tracknet_large_video is False


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("tracknet_stride = 8", "tracknet_stride = 1", "requires stride 8"),
        (
            "tracknet_large_video = true",
            "tracknet_large_video = false",
            "requires tracknet_large_video=true",
        ),
    ],
)
def test_exact_stream_configuration_rejects_incompatible_tracknet_modes(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    tracked = cli.REPO_ROOT / "configs/dataset_builder/trial.toml"
    config_path = tmp_path / "invalid.toml"
    config_path.write_text(
        tracked.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        cli.load_builder_config(config_path)


def test_proxy_command_is_lossless_bicubic_video_only(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "proxy.avi"

    command = tracknet_proxy_command(
        ffmpeg="/usr/bin/ffmpeg",
        source_path=source,
        output_path=output,
    )

    assert command[0] == "/usr/bin/ffmpeg"
    assert command[command.index("-vf") + 1] == "scale=512:288:flags=bicubic,setsar=1/1"
    assert command[command.index("-c:v") + 1] == "ffv1"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert {"-an", "-sn", "-dn"}.issubset(command)
    assert command[-1] == str(output)


def test_stream_commands_keep_the_ffv1_colour_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"

    producer = tracknet_stream_producer_command(
        ffmpeg="/usr/bin/ffmpeg",
        source_path=source,
        sample_step=17,
    )
    decoder = tracknet_stream_decoder_command(ffmpeg="/usr/bin/ffmpeg")

    assert producer[producer.index("-vf") + 1] == (
        "select=not(mod(n\\,17)),scale=512:288:flags=bicubic,setsar=1/1"
    )
    assert producer[producer.index("-c:v") + 1] == "ffv1"
    assert producer[producer.index("-pix_fmt") + 1] == "yuv420p"
    assert producer[-2:] == ["nut", "pipe:1"]
    assert decoder[decoder.index("-i") + 1] == "pipe:0"
    assert decoder[decoder.index("-pix_fmt") + 1] == "bgr24"
    assert decoder[-1] == "pipe:1"


def test_stream_input_persists_only_logical_metadata(tmp_path: Path) -> None:
    source = _metadata(tmp_path)
    output_dir = tmp_path / "tracknet-input"

    result = create_tracknet_input(
        source=source,
        output_dir=output_dir,
        ffmpeg="ffmpeg",
        mode=TrackNetInputMode.EXACT_FFV1_STREAM,
    )

    proxy_path, metadata_path = tracknet_input_paths(source, output_dir)
    assert result.mode is TrackNetInputMode.EXACT_FFV1_STREAM
    assert result.video_path == source.source_path
    assert result.as_mapping() == {"tracknet_input_metadata": metadata_path}
    assert not proxy_path.exists()
    assert result.metadata.frame_count == source.frame_count
    assert (result.metadata.width, result.metadata.height) == (512, 288)
    assert load_tracknet_input(
        source=source,
        output_dir=output_dir,
        mode=TrackNetInputMode.EXACT_FFV1_STREAM,
    ) == result


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration tools are unavailable",
)
def test_proxy_generation_is_deterministic_and_preserves_exact_timing(tmp_path: Path) -> None:
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

    first = create_tracknet_input(
        source=source,
        output_dir=tmp_path / "first",
        ffmpeg="ffmpeg",
    )
    second = create_tracknet_input(
        source=source,
        output_dir=tmp_path / "second",
        ffmpeg="ffmpeg",
    )

    for result in (first, second):
        assert result.metadata.fps == source.fps
        assert result.metadata.frame_count == source.frame_count
        assert (result.metadata.width, result.metadata.height) == (
            TRACKNET_INPUT_WIDTH,
            TRACKNET_INPUT_HEIGHT,
        )
        assert result.metadata.sample_aspect_ratio == Fraction(1)
        assert load_tracknet_input(source=source, output_dir=result.video_path.parent) == result
    first_frames = _decoded_frames(first.video_path)
    second_frames = _decoded_frames(second.video_path)
    assert len(first_frames) == len(second_frames) == source.frame_count
    assert all(np.array_equal(left, right) for left, right in zip(first_frames, second_frames))
    assert first.video_path.read_bytes() == second.video_path.read_bytes()


def test_ffmpeg_failure_leaves_no_proxy_or_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _metadata(tmp_path)
    output_dir = tmp_path / "tracknet-input"

    def fail(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 9, stdout="", stderr="conversion failed")

    monkeypatch.setattr(tracknet_input.subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="status 9: conversion failed"):
        create_tracknet_input(source=source, output_dir=output_dir, ffmpeg="ffmpeg")

    proxy_path, metadata_path = tracknet_input_paths(source, output_dir)
    assert not proxy_path.exists()
    assert not metadata_path.exists()
    assert not list(output_dir.glob("*.tmp.avi"))


def test_metadata_mismatch_leaves_no_completed_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _metadata(tmp_path)
    output_dir = tmp_path / "tracknet-input"

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        Path(command[-1]).write_bytes(b"proxy")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def mismatched(path: Path) -> VideoMetadata:
        return VideoMetadata(
            Path(path).resolve(),
            source.fps,
            source.frame_count - 1,
            TRACKNET_INPUT_WIDTH,
            TRACKNET_INPUT_HEIGHT,
        )

    monkeypatch.setattr(tracknet_input.subprocess, "run", fake_run)
    monkeypatch.setattr(tracknet_input, "probe_video_metadata", mismatched)

    with pytest.raises(ValueError, match="frame count"):
        create_tracknet_input(source=source, output_dir=output_dir, ffmpeg="ffmpeg")

    proxy_path, metadata_path = tracknet_input_paths(source, output_dir)
    assert not proxy_path.exists()
    assert not metadata_path.exists()


def test_proxy_pixel_coordinates_normalise_to_canonical_proportions(tmp_path: Path) -> None:
    proxy_path = tmp_path / "source.avi"
    proxy_path.write_bytes(b"fixture proxy")
    proxy = VideoMetadata(
        proxy_path.resolve(),
        Fraction(30),
        2,
        TRACKNET_INPUT_WIDTH,
        TRACKNET_INPUT_HEIGHT,
    )
    csv_path = tmp_path / "source_ball.csv"
    pd.DataFrame({
        "Frame": [0, 1],
        "X": [256, 128],
        "Y": [144, 72],
        "Visibility": [1, 1],
    }).to_csv(csv_path, index=False)
    output_path = tmp_path / "shuttle_track.npy.xz"

    result = convert_tracknet_csv_stage(
        csv_path,
        video_id="source-id",
        metadata=proxy,
        output_path=output_path,
    )

    expected = np.array([[0.5, 0.5, 1.0], [0.25, 0.25, 1.0]])
    np.testing.assert_array_equal(result.track, expected)
    np.testing.assert_array_equal(load_npy_xz(output_path), expected)
