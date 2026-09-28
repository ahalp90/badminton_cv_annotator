"""Where the court detector's scene boundaries come from: a PySceneDetect cut pass or a saved file.

A scene runs from one broadcast cut to the next. The detector keeps its foot
samples inside the target frame's scene. The annotation pipeline and
PySceneDetect load only when a pass runs, so the core detector imports and
runs without them.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

HISTOGRAM_BINS = 256  # one per 8-bit luma level, as in PySceneDetect's HistogramDetector


@dataclass(frozen=True)
class SceneInfo:
    """One scene: the video frames from one cut up to the next, `[start_frame, end_frame)`."""

    start_frame: int  # the scene's first frame, zero-based
    end_frame: int  # exclusive: the next scene's first frame, or the video's frame count
    # (HISTOGRAM_BINS,) luma distribution of the middle frame, summing to one. Similar
    # histograms suggest which scenes to compare first. They never show that two
    # scenes share a court.
    histogram: np.ndarray | None = None

    @property
    def middle_frame(self) -> int:
        """The scene's middle frame, which the detector analyses.

        An even-length scene has two middle frames; this is the upper one.
        """
        return (self.start_frame + self.end_frame) // 2


class SceneSource(Protocol):
    def scenes(self, video_path: Path, expected_frames: int, fps: float) -> list[SceneInfo]:
        """Every scene in frame order, covering `[0, expected_frames)` without gaps or overlaps."""
        ...


def _luma_histogram(frame: np.ndarray) -> np.ndarray:
    """(HISTOGRAM_BINS,) float64 share of a BGR frame's pixels at each luma level."""
    from scenedetect import HistogramDetector

    # normalize=True scales to unit length (cv2.normalize's L2 default), not the
    # unit sum its docstring states, so divide by the pixel count instead. OpenCV 4
    # returns a (bins, 1) column; the reshape accepts either layout.
    counts = HistogramDetector.calculate_histogram(frame, bins=HISTOGRAM_BINS, normalize=False)
    counts = counts.astype(np.float64).reshape(-1)
    return counts / counts.sum()


def _frame_histograms(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    """One luma histogram per requested frame, reading each frame after a seek.

    Each seek decodes from the nearest earlier keyframe. The cost therefore
    grows with the number of requested frames, not with the video's length.
    """
    from scenedetect import open_video

    video = open_video(str(video_path))
    histograms = []
    for frame_index in frame_indices:
        # PySceneDetect's seek leaves this 0-based frame as the next one read.
        video.seek(frame_index)
        frame = video.read()
        # read() returns False, not a frame, when decoding fails.
        if not isinstance(frame, np.ndarray):
            raise OSError(f"{video_path.name}: could not read frame {frame_index} after seeking to it")
        histograms.append(_luma_histogram(frame))
    return histograms


class PySceneDetectSource:
    """Scenes from the annotation pipeline's cut pass, optionally with one luma histogram each.

    Cuts use the composition mask's settings: ContentDetector threshold 27 and a
    minimum scene length of half a second, rounded to frames (13 at 25 fps).
    Histograms add one seek and one frame read per scene, at `SceneInfo.middle_frame`.
    """

    def __init__(self, histograms: bool = False) -> None:
        self.histograms = histograms

    def scenes(self, video_path: Path, expected_frames: int, fps: float) -> list[SceneInfo]:
        """Every scene in frame order, covering `[0, expected_frames)` without gaps or overlaps.

        :param video_path: the video whose frame indices the detector uses.
        :param expected_frames: frame count of that timeline. The container and
            the decoded video must both match it.
        :param fps: the video's frame rate, which sets the minimum scene length.
        :return: one `SceneInfo` per scene, including the opening and closing
            scenes. A video without cuts is one scene.
        """
        from annotator.composition_mask import detect_cuts
        from annotator.config import COMPOSITION_CONTENT_THRESHOLD
        from annotator.fps_constants import scale_for_fps

        min_scene_len = scale_for_fps(fps).composition_min_scene_len
        # detect_cuts raises unless both frame counts equal expected_frames. Each cut
        # is the first frame of a new scene, in ascending order, and so the exclusive
        # end of the scene before it.
        cut_frames = detect_cuts(video_path, expected_frames, COMPOSITION_CONTENT_THRESHOLD, min_scene_len).tolist()
        start_frames = [0, *cut_frames]
        end_frames = [*cut_frames, expected_frames]
        scenes = [SceneInfo(start_frame, end_frame) for start_frame, end_frame in zip(start_frames, end_frames)]
        if not self.histograms:
            return scenes

        histograms = _frame_histograms(video_path, [scene.middle_frame for scene in scenes])
        return [replace(scene, histogram=histogram) for scene, histogram in zip(scenes, histograms, strict=True)]


class SavedScenes:
    """Scenes from a gzipped JSON list of `[start_frame, end_frame]` pairs, with exclusive ends.

    The file holds no histograms. The caller checks that the scenes cover the video.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def scenes(self, video_path: Path, expected_frames: int, fps: float) -> list[SceneInfo]:
        with gzip.open(self.path, 'rt') as stream:
            spans = json.load(stream)
        if not isinstance(spans, list):
            raise ValueError(f'{self.path}: expected a list of scene ranges')  # noqa: TRY004 — invalid file contents
        scenes = []
        for span in spans:
            if not (isinstance(span, list) and len(span) == 2 and all(type(frame) is int for frame in span)):
                raise ValueError(f'{self.path}: {span!r} must be a [start_frame, end_frame] pair of integers')
            scenes.append(SceneInfo(*span))
        return scenes
