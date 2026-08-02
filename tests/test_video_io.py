"""Tests for classifier video metadata."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from classifier_shared.video_io import get_video_info

SYNTH_FPS = 30.0
SYNTH_WIDTH = 320
SYNTH_HEIGHT = 240
SYNTH_N_FRAMES = 10


@pytest.fixture(scope='session')
def synth_video(tmp_path_factory) -> Path:
    """Write a small video with stable metadata."""
    out = tmp_path_factory.mktemp('video') / 'synth.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(out), fourcc, SYNTH_FPS, (SYNTH_WIDTH, SYNTH_HEIGHT),
    )
    frame = np.zeros((SYNTH_HEIGHT, SYNTH_WIDTH, 3), dtype=np.uint8)
    for _ in range(SYNTH_N_FRAMES):
        writer.write(frame)
    writer.release()
    return out


def test_video_metadata_round_trips(synth_video):
    info = get_video_info(synth_video)
    assert info.path == synth_video
    assert info.width == SYNTH_WIDTH
    assert info.height == SYNTH_HEIGHT
    assert info.fps == pytest.approx(SYNTH_FPS, rel=0.01)
    assert info.n_frames == SYNTH_N_FRAMES
    assert info.duration_sec == pytest.approx(SYNTH_N_FRAMES / SYNTH_FPS)


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_video_info(tmp_path / 'no-such-file.mp4')
