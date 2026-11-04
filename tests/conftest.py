"""Shared test builders and generated fixtures."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def annotator_models():
    """A current-runtime fitted bundle for annotation and persistence tests."""
    from sklearn.dummy import DummyClassifier

    from annotator.models import AnnotatorModels
    from annotator.sequence.rally_model import RallySelector

    tree = DummyClassifier(strategy='prior').fit(np.zeros((3, 85)), [0, 0, 1])
    return AnnotatorModels(tree, RallySelector(tree), tree)


@pytest.fixture
def write_doubles_flags():
    """Write complete video, rally, and doubles-flag rows."""
    def write(path: Path, rows: list[tuple[str, str, str]]) -> None:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['video_id', 'rally_id', 'doubles_flag'])
            writer.writerows(rows)

    return write


@pytest.fixture(scope="session")
def validation_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a lossless-enough grayscale MP4 with one distinct fill per frame."""
    width, height, frame_count = 64, 48, 8
    output = tmp_path_factory.mktemp("validation-overlay-video") / "identifiable.mp4"
    frames = [np.full((height, width, 3), index * 20, dtype=np.uint8) for index in range(frame_count)]
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-framerate", "25", "-i", "-",
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-aspect", "4:3", str(output),
    ]
    completed = subprocess.run(
        command,
        input=b"".join(frame.tobytes() for frame in frames),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return output
