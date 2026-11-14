"""Dense original-frame targets and interval-safe reusable RGB readers."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from experiments.annotator.video_models_benchmark.data import read_json, target_labels

IGNORE = -100


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("Frame intervals must be non-empty and half-open")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def dense_targets(video: dict, mode: str) -> np.ndarray:
    """Scheme A: positives +/-1, ignore +/-2; unknown hitters stay ignored."""
    labels = np.full(video["frame_count"], 3 if mode == "discovery" else IGNORE, dtype=np.int64)
    for start, end in merge_intervals([[span["start_frame"], span["end_frame"]] for span in video["spans"]]):
        labels[start:end] = 0
    # Apply ignored bands before positives, including where padded spans overlap.
    for span in video["spans"]:
        local = target_labels(span, "A")
        view = labels[span["start_frame"]:span["end_frame"]]
        view[local == IGNORE] = IGNORE
    for span in video["spans"]:
        local = target_labels(span, "A")
        view = labels[span["start_frame"]:span["end_frame"]]
        positive = local > 0
        view[positive] = np.maximum(view[positive], local[positive])
    return labels


def overlap_starts(start: int, end: int, length: int) -> list[int]:
    """Cover short clips and tails once or more, with a half-clip stride."""
    if length < 2 or end <= start:
        raise ValueError("Need a non-empty interval and clip length >=2")
    if end - start < length:
        return [start - (length - (end - start)) // 2]
    last = end - length
    starts = list(range(start, last + 1, length // 2))
    if starts[-1] != last:
        starts.append(last)
    return starts


class FrameReader:
    """Keep one JPEG ZIP reader open per video in each persistent worker."""
    def __init__(self, cache: Path, video: dict) -> None:
        self.directory = cache / video["item_id"]
        self.metadata = read_json(self.directory / "frames.json.gz")
        self.intervals = self.metadata["intervals"]
        if merge_intervals(self.intervals) != self.intervals:
            raise ValueError("Cache intervals must be sorted, merged and disjoint")
        self.archive: ZipFile | None = None
        if (self.metadata.get("storage") != "jpeg_zip" or self.metadata.get("geometry") != "height224-aspect"
                or self.metadata.get("jpeg_quality") != 95 or self.metadata.get("jpeg_subsampling") != 2):
            raise ValueError("Prepare height224 aspect-preserving JPEG95/420 frames with the E2E preparation command")

    def containing_interval(self, start: int, end: int) -> tuple[int, int]:
        for left, right in self.intervals:
            if left <= start and end <= right:
                return left, right
        raise ValueError(f"{self.directory.name}: requested scope [{start}, {end}) crosses a missing-frame gap")

    def read(self, start: int, end: int) -> np.ndarray:
        self.containing_interval(start, end)
        if self.archive is None:
            self.archive = ZipFile(self.directory / "frames.zip")
        frames = []
        for frame in range(start, end):
            with Image.open(BytesIO(self.archive.read(f"{frame:09d}.jpg"))) as image:
                frames.append(np.asarray(image.convert("RGB")))
        return np.stack(frames)

    def __getstate__(self) -> dict:
        return {**self.__dict__, "archive": None}


class DenseClips(Dataset):
    """Sample RNG depends only on seed and absolute index, independent of workers."""
    def __init__(self, cache: Path, role: str, mode: str, length: int, *, seed: int = 0,
                 frames_per_cycle: int = 500_000, cycles: int = 50, training: bool = False) -> None:
        self.videos = [video for video in read_json(cache / "index.json.gz")["videos"] if video["role"] == role]
        if not self.videos:
            raise ValueError(f"No {role} videos in cache")
        self.readers = [FrameReader(cache, video) for video in self.videos]
        self.targets = [dense_targets(video, mode) for video in self.videos]
        self.length, self.seed, self.training = length, seed, training
        self.mode = mode
        self.frames_per_cycle = frames_per_cycle
        self.samples_per_cycle = (frames_per_cycle + length - 1) // length
        self.cycles = cycles
        self.scopes: list[tuple[int, int, int]] = []
        self.windows: list[tuple[int, int, int, int]] = []
        for video_index, (video, reader) in enumerate(zip(self.videos, self.readers, strict=True)):
            intervals = ([[0, video["frame_count"]]] if mode == "discovery" else
                         [[span["start_frame"], span["end_frame"]] for span in video["spans"]])
            for start, end in intervals:
                reader.containing_interval(start, end)
                self.scopes.append((video_index, start, end))
                for clip_start in overlap_starts(start, end, length):
                    self.windows.append((video_index, clip_start, start, end))
        self.cumulative = np.cumsum([end - start for _, start, end in self.scopes])
        if not len(self.cumulative):
            raise ValueError("No target scopes")

    def __len__(self) -> int:
        return self.samples_per_cycle * self.cycles if self.training else len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, int]:
        if self.training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
            draw = rng.integers(int(self.cumulative[-1]))
            scope_index = int(np.searchsorted(self.cumulative, draw, side="right"))
            video_index, scope_start, scope_end = self.scopes[scope_index]
            if self.mode == "supplied":
                centre = int(rng.integers(scope_start, scope_end))
                start = centre - self.length // 2
            else:
                start = int(rng.integers(scope_start, max(scope_start + 1, scope_end - self.length + 1)))
            positions = min(self.length, self.frames_per_cycle - (index % self.samples_per_cycle) * self.length)
        else:
            video_index, start, scope_start, scope_end = self.windows[index]
            positions = self.length
        reader = self.readers[video_index]
        real_start = max(start, scope_start)
        real_end = min(start + self.length, scope_end)
        real_frames = reader.read(real_start, real_end)
        if real_frames.shape[1] != 224 or real_frames.shape[2] < 224:
            raise ValueError("Expected aspect-preserving frames of height224 and width>=224")
        crop_left = (int(rng.integers(real_frames.shape[2] - 224 + 1)) if self.training
                     else (real_frames.shape[2] - 224) // 2)
        frames = np.zeros((self.length, 224, 224, 3), dtype=np.uint8)
        offset = real_start - start
        frames[offset:offset + len(real_frames)] = real_frames[:, :, crop_left:crop_left + 224]
        ids = np.arange(start, start + self.length)
        eligible = (ids >= real_start) & (ids < real_end)
        labels = np.full(self.length, IGNORE, dtype=np.int64)
        labels[eligible] = self.targets[video_index][ids[eligible]]
        labels[positions:] = IGNORE
        ids[~eligible] = -1
        return (torch.from_numpy(frames).permute(3, 0, 1, 2), torch.from_numpy(labels),
                video_index, torch.from_numpy(ids), positions)


def cycle_batches(samples_per_cycle: int, batch_size: int, cycles: int, *, start_cycle: int = 0,
                  start_batch: int = 0) -> list[list[int]]:
    """Keep final partial batches inside their cycle, including on resume."""
    batches = []
    for cycle in range(start_cycle, cycles):
        first_batch = start_batch if cycle == start_cycle else 0
        for offset in range(first_batch * batch_size, samples_per_cycle, batch_size):
            begin = cycle * samples_per_cycle + offset
            end = cycle * samples_per_cycle + min(offset + batch_size, samples_per_cycle)
            batches.append(list(range(begin, end)))
    return batches
