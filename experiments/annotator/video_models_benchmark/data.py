"""Reusable full-view RGB frames and centre-labelled clips."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.annotator.monotrack_benchmark.hitnet_retrain.build_training_set import (
    load_videos,
)

IMAGE_SIZE = 256
CLIP_OFFSETS = np.arange(-8, 8)
SIDE_CLASSES = {"Bot": 1, "Top": 2}
SPLIT_PATH = Path(__file__).resolve().parents[1] / "monotrack_benchmark/hitnet_retrain/shuttleset_development_split.json.gz"


def read_json(path: Path) -> Any:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with gzip.open(temporary, "wt") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)


def development_videos(spans: Path) -> list[dict[str, Any]]:
    videos = load_videos(spans, SPLIT_PATH)
    if any(video["fps"] > 30 for video in videos):
        raise ValueError("Frame mapping above 30 fps has not been defined for this pilot")
    return videos


def context_intervals(video: dict) -> list[list[int]]:
    """Merge overlapping clip context so each source frame is stored once."""
    intervals: list[list[int]] = []
    for span in video["spans"]:
        start = max(0, span["start_frame"] + int(CLIP_OFFSETS[0]))
        end = min(video["frame_count"], span["end_frame"] + int(CLIP_OFFSETS[-1]))
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return intervals


def target_labels(span: dict, scheme: str) -> np.ndarray:
    """Positive labels override ignored centres; TOP wins overlapping positives."""
    start, end = span["start_frame"], span["end_frame"]
    centres = np.arange(start, end)
    labels = np.zeros(end - start, dtype=np.int64)
    if scheme not in ("A", "B"):
        raise ValueError(f"Unknown target scheme: {scheme}")
    radius = 1 if scheme == "A" else 2
    for frame in span["label_frames"]:
        labels[np.abs(centres - frame) <= 2] = -100
    for frame, side in zip(span["label_frames"], span["label_sides"], strict=True):
        # Unknown test hitters retain their ignored band for frame loss.
        if side is None:
            continue
        positive = np.abs(centres - frame) <= radius
        labels[positive] = np.maximum(labels[positive], SIDE_CLASSES[side])
    return labels


class ClipDataset(Dataset):
    """Read uint8 clips from shared memory-mapped frame files.

    Rows are in video/rally/frame order; only training removes ignored centres.
    Workers open mappings lazily, sharing the operating system's file cache.
    """

    def __init__(self, cache: Path, role: str, scheme: str, *, training: bool) -> None:
        self.cache = cache
        self.videos = [video for video in read_json(cache / "index.json.gz")["videos"] if video["role"] == role]
        self.maps: dict[int, np.memmap] = {}
        self.frame_ids = []
        rows, labels = [], []
        for video_index, video in enumerate(self.videos):
            metadata = read_json(cache / video["item_id"] / "frames.json.gz")
            if metadata["image_size"] != IMAGE_SIZE or metadata["intervals"] != context_intervals(video):
                raise ValueError(f"{video['item_id']}: cache does not cover the requested clips")
            frame_ids = np.concatenate([np.arange(start, end) for start, end in metadata["intervals"]])
            self.frame_ids.append(frame_ids)
            for rally_index, span in enumerate(video["spans"]):
                targets = target_labels(span, scheme)
                centres = np.arange(span["start_frame"], span["end_frame"])
                eligible = targets != -100 if training else np.ones(len(centres), dtype=bool)
                rows.append(np.column_stack((
                    np.full(eligible.sum(), video_index), np.full(eligible.sum(), rally_index), centres[eligible],
                )))
                labels.append(targets[eligible])
        self.rows = np.concatenate(rows).astype(np.int32)
        self.labels = np.concatenate(labels)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        video_index, _rally_index, centre = self.rows[index]
        video = self.videos[video_index]
        if video_index not in self.maps:
            self.maps[video_index] = np.memmap(
                self.cache / video["item_id"] / "frames.rgb", mode="r", dtype=np.uint8,
                shape=(len(self.frame_ids[video_index]), IMAGE_SIZE, IMAGE_SIZE, 3),
            )
        frames = np.clip(centre + CLIP_OFFSETS, 0, video["frame_count"] - 1)
        positions = np.searchsorted(self.frame_ids[video_index], frames)
        # Advanced indexing copies the 16 frames out of the read-only shared mapping.
        clip = torch.from_numpy(self.maps[video_index][positions]).permute(3, 0, 1, 2)
        return clip, int(self.labels[index])
