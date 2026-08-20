"""Regression tests for TrackNetV3 whole-video iteration boundaries."""

from __future__ import annotations

import importlib.util
from itertools import zip_longest
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from annotator.video_metadata import probe_video_metadata
from dataset_builder.tracknet_input import TrackNetInputMode, create_tracknet_input


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKNET_DIR = REPO_ROOT / "src" / "shared" / "tracknetv3"


class _VideoCapture:
    def __init__(self, module: ModuleType, frame_count: int) -> None:
        self._module = module
        self._frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(frame_count)]
        self._position = 0
        self.released = False

    def get(self, property_id: int) -> float:
        values = {
            self._module.cv2.CAP_PROP_FRAME_COUNT: len(self._frames),
            self._module.cv2.CAP_PROP_FPS: 30,
            self._module.cv2.CAP_PROP_FRAME_WIDTH: 2,
            self._module.cv2.CAP_PROP_FRAME_HEIGHT: 2,
        }
        return float(values.get(property_id, 0))

    def set(self, property_id: int, value: float) -> bool:
        if property_id == self._module.cv2.CAP_PROP_POS_FRAMES:
            self._position = int(value)
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._position >= len(self._frames):
            return False, None
        frame = self._frames[self._position]
        self._position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _load_dataset_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    height: int = 2,
    width: int = 2,
) -> ModuleType:
    general = ModuleType("utils.general")
    general.HEIGHT = height
    general.WIDTH = width
    general.SIGMA = 1
    general.IMG_FORMAT = "png"
    general.get_rally_dirs = lambda *_args, **_kwargs: []
    general.get_match_median = lambda *_args, **_kwargs: None
    utils = ModuleType("utils")
    utils.general = general
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setitem(sys.modules, "utils.general", general)

    path = TRACKNET_DIR / "dataset.py"
    spec = importlib.util.spec_from_file_location("tracknet_video_iterable_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load TrackNet dataset module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("frame_count", "expected_ids"),
    [
        (8, [[*range(8)]]),
        (9, [[*range(8)], [8] * 8]),
    ],
    ids=("exact_multiple", "final_partial_window"),
)
def test_video_iterable_stops_after_real_frames(
    monkeypatch: pytest.MonkeyPatch,
    frame_count: int,
    expected_ids: list[list[int]],
) -> None:
    module = _load_dataset_module(monkeypatch)
    capture = _VideoCapture(module, frame_count)
    monkeypatch.setattr(module.cv2, "VideoCapture", lambda _path: capture)
    dataset = module.Video_IterableDataset(
        "fixture.avi",
        seq_len=8,
        sliding_step=8,
        HEIGHT=2,
        WIDTH=2,
    )

    batches = list(dataset)

    assert [indices[:, 1].tolist() for indices, _frames in batches] == expected_ids
    assert all(frames.shape == (24, 2, 2) for _indices, frames in batches)
    assert capture.released is True


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg integration tools are unavailable",
)
def test_exact_stream_matches_proxy_tracknet_model_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
            "9",
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
    module = _load_dataset_module(monkeypatch, height=288, width=512)
    common = {
        "seq_len": 8,
        "sliding_step": 8,
        "bg_mode": "subtract_concat",
        "HEIGHT": 288,
        "WIDTH": 512,
        "max_sample_num": 4,
    }
    persisted = module.Video_IterableDataset(str(proxy.video_path), **common)
    streamed = module.ExactFFV1StreamDataset(
        str(source.source_path),
        **common,
        expected_frame_count=source.frame_count,
        ffmpeg="ffmpeg",
    )

    np.testing.assert_array_equal(streamed.median, persisted.median)
    sentinel = object()
    for streamed_batch, persisted_batch in zip_longest(streamed, persisted, fillvalue=sentinel):
        assert streamed_batch is not sentinel and persisted_batch is not sentinel
        streamed_indices, streamed_frames = streamed_batch
        persisted_indices, persisted_frames = persisted_batch
        np.testing.assert_array_equal(streamed_indices, persisted_indices)
        np.testing.assert_array_equal(streamed_frames, persisted_frames)
