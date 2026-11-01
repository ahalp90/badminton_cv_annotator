"""Evaluate the player-supported three-frame court detector at labelled anchors."""

from __future__ import annotations

import argparse
import json
import lzma
import time
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from experiments.court_detector.benchmark.manual_courts import read_json
from experiments.court_detector.benchmark.sam3.benchmark_rallies import write_json

if TYPE_CHECKING:
    from court_detector.inputs import FrameReader, PeopleSource, PersonSample
    from court_detector.run_video import CourtTools
    from court_detector.video_inputs import VideoFrames


@dataclass(frozen=True)
class VideoSource:
    path: Path
    frame_offset: int = 0
    pose_directory: Path | None = None
    pose_cache_directory: Path | None = None


def video_source(value: str | dict) -> VideoSource:
    if isinstance(value, str):
        return VideoSource(Path(value))
    directory = value.get("pose_directory")
    cache_directory = value.get("pose_cache_directory")
    return VideoSource(Path(value["path"]), value.get("frame_offset", 0),
                       None if directory is None else Path(directory),
                       None if cache_directory is None else Path(cache_directory))


class BenchmarkFrames:
    """Read source context at the resolution of the labelled benchmark images."""

    def __init__(self, source: VideoFrames, size: tuple[int, int]) -> None:
        from court_detector.video_inputs import FRAME_CACHE_SIZE

        self.source = source
        self.source_size = source.size
        self.size = size
        self.fps = source.fps
        self.frame_count = source.frame_count
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = FRAME_CACHE_SIZE

    def read(self, frame_indices: Sequence[int]) -> list[np.ndarray]:
        missing = sorted(set(frame_indices) - self.cache.keys())
        for index, frame in zip(missing, self.source.read(missing), strict=True):
            if self.source_size != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            frame.flags.writeable = False
            self.cache[index] = frame
        result = [self.cache[index] for index in frame_indices]
        for index in frame_indices:
            self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return result

    def close(self) -> None:
        self.cache.clear()
        self.source.close()


class ScaledPeople:
    """Scale saved native-coordinate poses to the benchmark images."""

    def __init__(self, source: PeopleSource, scale: np.ndarray) -> None:
        self.source = source
        self.scale = scale

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        from court_detector.inputs import PersonSample

        samples = self.source.samples(frame_indices)
        return [PersonSample(index, sample.boxes_px * np.tile(self.scale, 2), sample.keypoints_px * self.scale)
                for index, sample in zip(frame_indices, samples, strict=True)]


class FilePosePeople:
    """Reuse individual saved pose frames and infer only uncovered samples."""

    def __init__(self, directory: Path, fallback: PeopleSource, scale: np.ndarray) -> None:
        self.directory = directory
        self.fallback = fallback
        self.scale = scale

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        from court_detector.inputs import PersonSample

        found = {}
        missing = []
        for index in frame_indices:
            path = self.directory / f"frame_{index:09d}.npy.xz"
            if not path.exists():
                missing.append(index)
                continue
            with lzma.open(path, "rb") as stream:
                packed = np.load(stream, allow_pickle=False)
            if packed.ndim != 2 or packed.shape[1] != 38:
                raise ValueError(f"saved pose frame {index} must contain four box and 34 keypoint coordinates")
            boxes = packed[:, :4] * np.tile(self.scale, 2)
            keypoints = packed[:, 4:].reshape(-1, 17, 2) * self.scale
            found[index] = PersonSample(index, boxes, keypoints)
        found.update((sample.frame_index, sample) for sample in self.fallback.samples(missing))
        return [found[index] for index in frame_indices]


class SparsePeople:
    """Persist only requested pose samples, including overlapping-window reuse."""

    def __init__(self, source: PeopleSource, path: Path) -> None:
        from court_detector.inputs import PersonSample

        self.source = source
        self.path = path
        self.cache: dict[int, PersonSample] = {}
        self.requested: set[int] = set()
        if path.exists():
            for index, value in read_json(path).items():
                frame_index = int(index)
                boxes = np.asarray(value["boxes"], dtype=np.float32).reshape(-1, 4)
                keypoints = np.asarray(value["keypoints"], dtype=np.float32).reshape(-1, 17, 2)
                self.cache[frame_index] = PersonSample(frame_index, boxes, keypoints)

    def samples(self, frame_indices: Sequence[int]) -> list[PersonSample]:
        self.requested.update(frame_indices)
        missing = sorted(set(frame_indices) - self.cache.keys())
        if missing:
            samples = self.source.samples(missing)
            if [sample.frame_index for sample in samples] != missing:
                raise ValueError("pose source did not return the requested frames")
            self.cache.update((sample.frame_index, sample) for sample in samples)
            stored = {str(index): {"boxes": sample.boxes_px.tolist(), "keypoints": sample.keypoints_px.tolist()}
                      for index, sample in self.cache.items()}
            write_json(self.path, stored)
        return [self.cache[index] for index in frame_indices]


def evaluate_anchor(key: str, anchor: int, frame_count: int, frames: FrameReader,
                    people: SparsePeople, tools: CourtTools) -> dict:
    from court_detector import feet
    from court_detector.detect import CourtFitError
    from court_detector.geometry import normalise_output_corners
    from court_detector.run_video import endpoint_views, frame_view
    from court_detector.scene_sources import SceneInfo

    started = time.monotonic()
    people.requested.clear()
    try:
        window = feet.window_frames(anchor, frames.fps, 0, frame_count)
    except ValueError as error:
        return {"corners": None, "status": "scene_too_short_for_feet", "source_frame_index": anchor,
                "court_frames": [], "pose_window": [], "pose_frames": [],
                "no_court_reason": str(error), "composition": None,
                "seconds": time.monotonic() - started}
    scene = SceneInfo(0, frame_count)
    decoded = dict(zip(window, frames.read(window), strict=True))
    view = frame_view(key, decoded[anchor], anchor, scene, tools.lines, people)
    endpoint_indices = [window[0], window[-1]]
    # At a video edge, keep three distinct images within the unchanged pose window.
    if anchor in endpoint_indices:
        endpoint_indices[endpoint_indices.index(anchor)] = window[len(window) // 2]
    endpoints = partial(endpoint_views, decoded, endpoint_indices, key, scene, tools.lines, people)
    try:
        result = tools.detector.detect(view, people, frames, endpoint_views=endpoints)
        corners = result.corners_native_px
        record = {"corners": None if corners is None else normalise_output_corners(corners).tolist(),
                  "status": "no_court" if corners is None else "court", "composition": result.composition,
                  "no_court_reason": result.no_court_reason, "chosen_key": result.chosen_key}
    except CourtFitError as error:
        record = {"corners": None, "status": "fit_failed", "composition": None, "error": str(error)}
    record.update(source_frame_index=anchor, court_frames=[anchor, *endpoint_indices],
                  pose_window=window, pose_frames=sorted(people.requested),
                  seconds=time.monotonic() - started)
    return record


def run(output: Path, video_map: Path, deeplsd_source: Path, deeplsd_weights: Path,
        workers: int = 4, device: str = "cuda") -> None:
    inputs = {row["image_key"]: row for row in read_json(output / "inputs.json.gz")}
    destination = output / "ours_robust.json.gz"
    records = read_json(destination) if destination.exists() else {}
    pending = {key: row for key, row in inputs.items() if key not in records}
    if not pending:
        return
    sources = {video: video_source(value) for video, value in json.loads(video_map.read_text()).items()}
    grouped = {}
    for key, row in pending.items():
        grouped.setdefault(row["video_id"], []).append((key, row))

    from court_detector.detect import Switches
    from court_detector.run_video import load_court_tools
    from court_detector.video_inputs import PoseArrays, RtmlibPeople, VideoFrames

    switches = Switches(require_people=True, workers=workers, timing=True)
    tools = load_court_tools(switches, deeplsd_source=deeplsd_source, deeplsd_weights=deeplsd_weights,
                            device=device, live_pose=any(sources[video].pose_directory is None for video in grouped))
    poses = output / "sparse_poses"
    poses.mkdir(exist_ok=True)
    with tools.detector:
        for video, requests in grouped.items():
            source = sources[video]
            frames = BenchmarkFrames(VideoFrames(source.path), tuple(requests[0][1]["image_size"]))
            try:
                pose_source: PeopleSource
                scale = np.asarray(frames.size) / frames.source_size
                if source.pose_directory is None:
                    assert tools.pose_extractor is not None
                    pose_source = RtmlibPeople(frames, tools.pose_extractor)
                    if source.pose_cache_directory is not None:
                        pose_source = FilePosePeople(source.pose_cache_directory, pose_source, scale)
                else:
                    pose_source = ScaledPeople(PoseArrays.from_directory(source.pose_directory), scale)
                # Cached samples use source frame indices and benchmark-image pixels.
                width, height = frames.size
                cache = poses / f"{video}_{width}x{height}.json.gz"
                people = SparsePeople(pose_source, cache)
                for key, row in requests:
                    if tuple(row["image_size"]) != frames.size:
                        raise ValueError(f"{video}: source resolution differs from the labelled images")
                    anchor = row["frame_index"] + source.frame_offset
                    records[key] = evaluate_anchor(key, anchor, frames.frame_count, frames, people, tools)
                    write_json(destination, records)
                    print(f"ours_robust: {len(records)}/{len(inputs)} {key} {records[key]['status']}", flush=True)
            finally:
                frames.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-map", type=Path, required=True)
    parser.add_argument("--deeplsd-source", type=Path, required=True)
    parser.add_argument("--deeplsd-weights", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.out, args.video_map, args.deeplsd_source, args.deeplsd_weights,
        args.workers, args.device)


if __name__ == "__main__":
    main()
