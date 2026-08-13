"""Regression tests for TrackNetV3 whole-video iteration boundaries."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest


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


def _load_dataset_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    general = ModuleType("utils.general")
    general.HEIGHT = 2
    general.WIDTH = 2
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
