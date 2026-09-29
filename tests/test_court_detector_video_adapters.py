"""The live court detector's line and scene adapters meet their input contracts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from court_detector import line_sources
from court_detector.line_sources import (
    DeepLSDLines,
    LineSource,
    SavedLines,
)
from court_detector.scene_sources import (
    HISTOGRAM_BINS,
    PySceneDetectSource,
    SceneInfo,
    SceneSource,
    _luma_histogram,
)
from experiments.annotator.independent_court import export_lines

REPO = Path(__file__).resolve().parents[1]
FPS = 25
# (start frame, exclusive end frame, grey level of the first frame). The level rises
# by two per frame so each frame is identifiable; the jumps between scenes are cuts.
SCENE_SPANS = ((0, 20, 10), (20, 45, 150), (45, 60, 60))


class FakeDeepLSD:
    """Stands in for the loaded network: blank fields, then fixed fragments in working pixels."""

    def __init__(self, working_lines: np.ndarray) -> None:
        self.working_lines = working_lines  # (fragments, 2, 2) endpoints, as DeepLSD returns them
        self.extractions: list[dict[str, Any]] = []

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]  # (1, 1, height, width) tensor
        return {"df": image[0].clone(), "line_level": image[0].clone()}

    def detect_afm_lines(self, grey: np.ndarray, distance_field: np.ndarray, angle_field: np.ndarray,
                         **options: Any) -> np.ndarray:
        self.extractions.append({"grey": grey, "field_shapes": (distance_field.shape, angle_field.shape), **options})
        return self.working_lines


def deeplsd_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, network: FakeDeepLSD,
                  loads: list[tuple[Path, Path, str]]) -> LineSource:
    def load(source: Path, weights: Path, device: Any) -> FakeDeepLSD:
        loads.append((source, weights, str(device)))
        return network

    monkeypatch.setattr(line_sources, "load_deeplsd", load)
    return DeepLSDLines(tmp_path, tmp_path / "weights.tar", device="cpu")


@pytest.mark.parametrize(
    ("width", "height", "working_width", "working_height"),
    [(1001, 563, 960, 540), (500, 1300, 369, 960), (640, 360, 640, 360)],
)
def test_deeplsd_fragments_return_in_native_pixels_with_separate_axis_factors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, width: int, height: int, working_width: int, working_height: int,
) -> None:
    # Working-image corners map exactly to native corners only when x and y scale separately.
    network = FakeDeepLSD(np.array([[[0.0, 0.0], [working_width, working_height]],
                                    [[working_width, 0.0], [0.0, working_height]]]))
    lines = deeplsd_lines(monkeypatch, tmp_path, network, [])
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    segments = lines.segments(frame, frame_index=7)

    grey = network.extractions[0]["grey"]
    assert grey.shape == (working_height, working_width) and grey.dtype == np.uint8
    assert network.extractions[0]["field_shapes"] == ((working_height, working_width),) * 2
    assert segments.dtype == np.float32
    np.testing.assert_array_equal(segments, [[0, 0, width, height], [width, 0, 0, height]])


def test_deeplsd_loads_once_and_extracts_with_the_saved_export_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    network = FakeDeepLSD(np.array([[[1.0, 2.0], [3.0, 4.0]]]))
    loads: list[tuple[Path, Path, str]] = []
    lines = deeplsd_lines(monkeypatch, tmp_path, network, loads)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    for frame_index in range(3):
        lines.segments(frame, frame_index)

    assert loads == [(tmp_path, tmp_path / "weights.tar", "cpu")]
    for extraction in network.extractions:
        options = {name: extraction[name] for name in ("filtering", "merge", "grad_thresh", "grad_nfa")}
        assert options == {"filtering": "normal", "merge": False, "grad_thresh": 3, "grad_nfa": True}


def test_deeplsd_resize_matches_the_saved_export() -> None:
    frame = np.random.default_rng(0).integers(0, 256, size=(563, 1001, 3), dtype=np.uint8)

    grey, working_size = line_sources._working_grey(frame, max_dimension=960)
    exported_grey, exported_size = export_lines._working_image(frame)

    assert working_size == exported_size
    np.testing.assert_array_equal(grey, exported_grey)


@pytest.mark.parametrize("empty", [np.empty((0, 2, 2)), np.empty(0)])
def test_deeplsd_without_fragments_returns_zero_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, empty: np.ndarray,
) -> None:
    lines = deeplsd_lines(monkeypatch, tmp_path, FakeDeepLSD(empty), [])

    segments = lines.segments(np.zeros((540, 960, 3), dtype=np.uint8), 0)

    assert segments.shape == (0, 4) and segments.dtype == np.float32


def test_cuda_request_fails_rather_than_running_on_cpu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import torch

    loads: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(line_sources, "load_deeplsd", lambda *arguments: loads.append(arguments))

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        DeepLSDLines(tmp_path, tmp_path / "weights.tar")
    assert loads == []


def test_saved_lines_return_each_frame_extract() -> None:
    lines: LineSource = SavedLines({10: [[1, 2, 3, 4]], 20: []})
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    saved = lines.segments(frame, 10)
    empty = lines.segments(frame, 20)

    assert saved.dtype == np.float32
    np.testing.assert_array_equal(saved, [[1, 2, 3, 4]])
    assert empty.shape == (0, 4) and empty.dtype == np.float32
    with pytest.raises(KeyError):
        lines.segments(frame, 30)


@pytest.mark.parametrize("segments", [[[1, 2, 3]], [[1, 2, 3, np.nan]], [[[1, 2], [3, 4]]]])
def test_saved_lines_reject_malformed_extracts(segments: list[Any]) -> None:
    with pytest.raises(ValueError, match="line segments"):
        SavedLines({0: segments})


def write_video(path: Path, frames: list[np.ndarray]) -> Path:
    """Encode near-lossless H.264 with a keyframe every 12 frames, so seeks decode forward."""
    height, width = frames[0].shape[:2]
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", "-g", "12", str(path),
    ]
    completed = subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), capture_output=True,
                               check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", errors="replace"))
    return path


@pytest.fixture(scope="module")
def three_scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    frames = []
    for start_frame, end_frame, first_level in SCENE_SPANS:
        for offset in range(end_frame - start_frame):
            frames.append(np.full((48, 64, 3), first_level + 2 * offset, dtype=np.uint8))
    return write_video(tmp_path_factory.mktemp("scenes") / "three_scenes.mp4", frames)


def decoded_luma_levels(video_path: Path) -> list[int]:
    """The luma level of each uniform frame, read in order without seeking."""
    capture = cv2.VideoCapture(str(video_path))
    levels = []
    while True:
        has_frame, frame = capture.read()
        if not has_frame:
            break
        levels.append(int(cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)[0, 0, 0]))
    capture.release()
    return levels


def test_scenes_start_at_each_cut_and_cover_every_frame(three_scene_video: Path) -> None:
    source: SceneSource = PySceneDetectSource()

    scenes = source.scenes(three_scene_video, expected_frames=60, fps=FPS)

    # Cuts at frames 20 and 45 each start the following scene.
    assert scenes == [SceneInfo(start_frame, end_frame) for start_frame, end_frame, _ in SCENE_SPANS]


def test_video_without_cuts_is_one_scene(tmp_path: Path) -> None:
    frames = [np.full((48, 64, 3), 90, dtype=np.uint8)] * 30
    video_path = write_video(tmp_path / "one_scene.mp4", frames)

    assert PySceneDetectSource().scenes(video_path, expected_frames=30, fps=FPS) == [SceneInfo(0, 30)]


def test_frame_count_mismatch_fails(three_scene_video: Path) -> None:
    with pytest.raises(ValueError, match="expected 61"):
        PySceneDetectSource().scenes(three_scene_video, expected_frames=61, fps=FPS)


def test_scene_histograms_describe_each_middle_frame(three_scene_video: Path) -> None:
    scenes = PySceneDetectSource(histograms=True).scenes(three_scene_video, expected_frames=60, fps=FPS)
    levels = decoded_luma_levels(three_scene_video)

    # Upper middle of the 20-frame scene, then the middles of the 25- and 15-frame scenes.
    assert [scene.middle_frame for scene in scenes] == [10, 32, 52]
    for scene in scenes:
        middle_frame = scene.middle_frame
        # Neighbouring frames differ in level, so a seek to the wrong frame fails here.
        assert levels[middle_frame] not in (levels[middle_frame - 1], levels[middle_frame + 1])
        assert scene.histogram is not None and scene.histogram.shape == (HISTOGRAM_BINS,)
        assert scene.histogram.sum() == pytest.approx(1.0)
        assert scene.histogram[levels[middle_frame]] == pytest.approx(1.0)


def test_luma_histogram_sums_to_one() -> None:
    frame = np.full((48, 64, 3), 40, dtype=np.uint8)
    frame[:, 32:] = 200

    histogram = _luma_histogram(frame)

    assert histogram.dtype == np.float64
    assert histogram[40] == pytest.approx(0.5) and histogram[200] == pytest.approx(0.5)
    assert histogram.sum() == pytest.approx(1.0)


def test_saved_lines_and_scene_source_import_without_optional_packages() -> None:
    code = """
import sys
before = list(sys.path)
import numpy as np
from court_detector.line_sources import SavedLines
from court_detector.scene_sources import PySceneDetectSource
lines = SavedLines({3: [[0.0, 1.0, 2.0, 3.0]]})
assert lines.segments(np.zeros((4, 4, 3), np.uint8), 3).shape == (1, 4)
PySceneDetectSource(histograms=True)
for package in ("torch", "deeplsd", "scenedetect", "experiments", "annotator"):
    loaded = [name for name in sys.modules if name == package or name.startswith(package + ".")]
    assert not loaded, loaded
assert sys.path == before
"""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(REPO), str(REPO / "src")))}
    completed = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=environment, capture_output=True,
                               text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr
