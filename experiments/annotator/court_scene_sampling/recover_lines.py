"""Recover DeepLSD evidence for accepted saved fits, without searching for courts."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

from court_detector.line_sources import DeepLSDLines, LineSource
from court_detector.video_inputs import VideoFrames

from .sampling import read_json


def accepted_frames(video: dict[str, Any]) -> dict[int, set[str]]:
    """Group accepted view IDs by source frame so all methods share one extract."""
    by_frame: dict[int, set[str]] = {}
    for scene in video["scenes"] + video["later_scenes"]:
        for outcome in scene.get("methods", {}).values():
            for frame in outcome.get("frames", []):
                if frame["status"] == "court":
                    by_frame.setdefault(frame["frame_index"], set()).add(frame["view_id"])
    return by_frame


def recover(results: dict[str, Any], lines: LineSource, frames_dir: Path | None = None) -> dict[str, Any]:
    """Extract native-pixel lines once per accepted source frame.

    Optional PNGs allow later outlines to be drawn locally without copying videos.
    Only line inference and decoding run here; player evidence is unnecessary.
    """
    started = perf_counter()
    cache: dict[str, Any] = {"schema": "court-scene-lines/1", "frames": {}}
    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=False)
    for video in results["videos"]:
        by_frame = accepted_frames(video)
        if not by_frame:
            continue
        with VideoFrames(Path(video["video"])) as frames:
            for frame_index, view_ids in sorted(by_frame.items()):
                frame, = frames.read([frame_index])
                height, width = frame.shape[:2]
                if [width, height] != video["native_size"]:
                    raise ValueError(f"{video['video_id']}: decoded frame size differs from saved results")
                entry = {"native_size": [width, height], "frame_index": frame_index,
                         "segments_native_px": lines.segments(frame, frame_index).tolist()}
                for view_id in sorted(view_ids):
                    if view_id in cache["frames"]:
                        raise ValueError(f"Duplicate view ID: {view_id}")
                    cache["frames"][view_id] = entry
                    if frames_dir is not None:
                        if Path(view_id).name != view_id:
                            raise ValueError(f"View ID is not a plain file name: {view_id}")
                        if not cv2.imwrite(str(frames_dir / f"{view_id}.png"), frame):
                            raise OSError(f"Could not write frame {view_id}")
                print(f"{video['video_id']} frame {frame_index}: {len(entry['segments_native_px'])} lines", flush=True)
    cache["recovery_seconds"] = perf_counter() - started
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new .json.gz line cache")
    parser.add_argument("--frames-dir", type=Path, help="new directory for optional original PNGs")
    parser.add_argument("--deeplsd-source", type=Path, required=True)
    parser.add_argument("--deeplsd-weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("--output must end in .json.gz")
    cv2.setNumThreads(1)
    started = perf_counter()
    lines = DeepLSDLines(args.deeplsd_source, args.deeplsd_weights, device=args.device)
    setup_seconds = perf_counter() - started
    cache = recover(read_json(args.results), lines, args.frames_dir)
    cache["model_setup_seconds"] = setup_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "xt") as stream:
        json.dump(cache, stream, allow_nan=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
