"""Run one disposable PySceneDetect cut configuration over one video."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

from scenedetect import AdaptiveDetector, ContentDetector, HashDetector, HistogramDetector
from scenedetect import SceneManager, open_video
from scenedetect.detectors.content_detector import FlashFilter

from annotator.fps_constants import scale_for_fps


def detector_for(config: str, min_scene_len: int):
    """Build the named detector with the study's frame-scaled scene length."""
    edge_weights = ContentDetector.Components(1.0, 1.0, 1.0, 1.0)
    if config == "content27":
        return ContentDetector(threshold=27, min_scene_len=min_scene_len)
    if config == "content22":
        return ContentDetector(threshold=22, min_scene_len=min_scene_len)
    if config == "content32":
        return ContentDetector(threshold=32, min_scene_len=min_scene_len)
    if config == "content27sup":
        return ContentDetector(
            threshold=27,
            min_scene_len=min_scene_len,
            filter_mode=FlashFilter.Mode.SUPPRESS,
        )
    if config == "content27edge":
        return ContentDetector(threshold=27, min_scene_len=min_scene_len, weights=edge_weights)
    if config == "content40edge":
        return ContentDetector(threshold=40, min_scene_len=min_scene_len, weights=edge_weights)
    if config == "adaptive3":
        return AdaptiveDetector(min_scene_len=min_scene_len)
    if config == "adaptive2":
        return AdaptiveDetector(adaptive_threshold=2.0, min_scene_len=min_scene_len)
    if config == "hash":
        return HashDetector(min_scene_len=min_scene_len)
    if config == "hist":
        return HistogramDetector(min_scene_len=min_scene_len)
    raise ValueError(f"unknown detector config: {config}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", type=Path, required=True, help="output cut list, written gzipped (.csv.gz)")
    args = parser.parse_args()

    video = open_video(str(args.video))
    n_frames = video.duration.frame_num
    min_scene_len = scale_for_fps(args.fps).composition_min_scene_len
    manager = SceneManager()
    manager.add_detector(detector_for(args.config, min_scene_len))
    n_read = manager.detect_scenes(video, show_progress=False)
    assert n_frames == n_read, f"{args.video}: duration has {n_frames} frames, read {n_read}"

    cut_frames = [scene[0].frame_num for scene in manager.get_scene_list()[1:]]
    with gzip.open(args.out, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["cut_frame", "n_frames"])
        writer.writeheader()
        for cut_frame in cut_frames:
            writer.writerow({"cut_frame": cut_frame, "n_frames": n_frames})


if __name__ == "__main__":
    main()
