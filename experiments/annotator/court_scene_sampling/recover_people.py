"""Recover same-frame person boxes for occlusion-aware scoring of saved court fits."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .sampling import read_json


def recover(lines_cache: dict[str, Any], frames_dir: Path,
            detect_boxes: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
    """Measure person boxes on the same PNGs used to recover each frame's lines."""
    started = perf_counter()
    cache: dict[str, Any] = {"schema": "court-scene-people/1", "frames": {}}
    for view_id, entry in lines_cache["frames"].items():
        if Path(view_id).name != view_id:
            raise ValueError(f"View ID is not a plain file name: {view_id}")
        path = frames_dir / f"{view_id}.png"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        height, width = frame.shape[:2]
        if [width, height] != entry["native_size"]:
            raise ValueError(f"{view_id}: PNG size differs from the line cache")
        boxes = detect_boxes(frame)
        cache["frames"][view_id] = {"native_size": [width, height], "boxes_native_px": boxes.tolist()}
        print(f"{view_id}: {len(boxes)} person boxes", flush=True)
    cache["recovery_seconds"] = perf_counter() - started
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lines-cache", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new .json.gz people cache")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffixes[-2:] != [".json", ".gz"]:
        raise ValueError("--output must end in .json.gz")
    from shared.rtmlib_pose import RtmlibPoseExtractor

    cv2.setNumThreads(1)
    started = perf_counter()
    extractor = RtmlibPoseExtractor(device=args.device)
    setup_seconds = perf_counter() - started
    cache = recover(read_json(args.lines_cache), args.frames_dir,
                    lambda frame: extractor.detect_frame(frame).bboxes)
    cache["model_setup_seconds"] = setup_seconds
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "xt") as stream:
        json.dump(cache, stream, allow_nan=False, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
