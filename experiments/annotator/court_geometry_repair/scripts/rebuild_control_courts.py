"""Compare historical and current CourtKeyNet scene-court outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TextIO

import cv2
import numpy as np

REFERENCE_SIZE = (1280, 720)


@contextmanager
def _open_csv(path: Path) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", newline="", encoding="utf-8") as source:
            yield source
    else:
        with path.open(newline="", encoding="utf-8") as source:
            yield source


def json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


def load_ground_truth(path: Path, video_id: int) -> np.ndarray:
    with _open_csv(path) as source:
        rows = [row for row in csv.DictReader(source) if int(row["id"]) == video_id]
    if len(rows) != 1:
        raise ValueError(
            f"video id {video_id}: expected one homography row, got {len(rows)}"
        )
    row = rows[0]
    return np.asarray(
        [
            [row["upleft_x"], row["upleft_y"]],
            [row["upright_x"], row["upright_y"]],
            [row["downright_x"], row["downright_y"]],
            [row["downleft_x"], row["downleft_y"]],
        ],
        dtype=np.float32,
    )


def load_pad_scenes(path: Path, video_id: int) -> list[tuple[int, int]]:
    with _open_csv(path) as source:
        rows = [row for row in csv.DictReader(source) if row["mode"] == "pad"]
    if not rows or any(int(row["video_id"]) != video_id for row in rows):
        raise ValueError(f"{path}: expected pad scenes for video {video_id}")
    return [(int(row["scene_start"]), int(row["scene_end"])) for row in rows]


def load_baseline(path: Path) -> ModuleType:
    name = "courtkeynet._baseline_court_corners"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load baseline module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_scene(capture: cv2.VideoCapture, samples: list[int]) -> list[np.ndarray]:
    frames = []
    for sample in samples:
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, sample):
            raise OSError(f"could not seek source frame {sample}")
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"could not read source frame {sample}")
        frames.append(cv2.resize(frame, REFERENCE_SIZE, interpolation=cv2.INTER_AREA))
    return frames


def corner_error(quad: Any, ground_truth: np.ndarray) -> list[float] | None:
    if quad is None:
        return None
    return np.linalg.norm(
        np.asarray(quad.corners_px, dtype=np.float64) - ground_truth, axis=1
    ).tolist()


def draw_quad(frame: np.ndarray, corners: Any, colour: tuple[int, int, int]) -> None:
    if corners is None:
        return
    points = np.rint(np.asarray(corners)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [points], True, colour, 3)


def draw_legend(frame: np.ndarray) -> None:
    for y, label, colour in (
        (32, "baseline", (255, 128, 0)),
        (62, "current", (0, 128, 255)),
        (92, "GT", (255, 0, 255)),
    ):
        cv2.putText(frame, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)


def run(args: argparse.Namespace) -> None:
    from courtkeynet import court_corners as current
    from courtkeynet.wrapper import CourtKeyNetDetector

    scenes = load_pad_scenes(args.scenes_csv, args.video_id)
    ground_truth = load_ground_truth(args.homography_csv, args.video_id)
    baseline = load_baseline(args.baseline_module)
    args.output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise OSError(f"could not open source video {args.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if min(width, height, frame_count) <= 0:
        raise ValueError(
            "source container has incomplete width, height or frame-count metadata"
        )
    detector = CourtKeyNetDetector(device=args.device, resize_mode="pad")
    records = []
    try:
        for scene_index, (start, end) in enumerate(scenes):
            samples = sorted(set(np.linspace(start, end - 1, 10).astype(int).tolist()))
            if len(samples) != 10 or max(samples) >= frame_count:
                raise ValueError(f"scene {scene_index}: invalid samples {samples}")
            frames = read_scene(capture, samples)
            detections = detector.detect_batch(frames)
            old_quad = baseline.pick_scene_corners(frames, detections)
            new_quad = current.pick_scene_corners(frames, detections)
            records.append(
                {
                    "scene_index": scene_index,
                    "interval": [start, end],
                    "sampled_frame_indices": samples,
                    "raw_nn": [asdict(detection) for detection in detections],
                    "baseline_quad": old_quad,
                    "current_quad": new_quad,
                    "baseline_error_refpx": corner_error(old_quad, ground_truth),
                    "current_error_refpx": corner_error(new_quad, ground_truth),
                }
            )
            old_source = old_quad.source if old_quad is not None else "none"
            new_source = new_quad.source if new_quad is not None else "none"
            print(
                f"scene {scene_index:03d} [{start}, {end}): baseline={old_source} current={new_source}",
                flush=True,
            )
            if scene_index == 0:
                middle = frames[len(frames) // 2]
                if not cv2.imwrite(str(args.output / "scene_0000_middle.png"), middle):
                    raise OSError("could not write middle frame")
                overlay = middle.copy()
                draw_quad(
                    overlay,
                    old_quad.corners_px if old_quad is not None else None,
                    (255, 128, 0),
                )
                draw_quad(
                    overlay,
                    new_quad.corners_px if new_quad is not None else None,
                    (0, 128, 255),
                )
                draw_quad(overlay, ground_truth, (255, 0, 255))
                draw_legend(overlay)
                if not cv2.imwrite(
                    str(args.output / "scene_0000_overlay.png"), overlay
                ):
                    raise OSError("could not write overlay")
    finally:
        capture.release()
    metadata = {
        "video_id": args.video_id,
        "source": str(args.video),
        "source_width": width,
        "source_height": height,
        "source_fps": fps,
        "source_frame_count": frame_count,
        "resize": {
            "width": REFERENCE_SIZE[0],
            "height": REFERENCE_SIZE[1],
            "interpolation": "INTER_AREA",
        },
        "detector_resize_mode": "pad",
        "device": args.device,
        "scene_count": len(records),
        "scenes_csv": str(args.scenes_csv),
        "homography_csv": str(args.homography_csv),
        "baseline_module": str(args.baseline_module),
    }
    with gzip.open(
        args.output / "comparison.json.gz", "wt", encoding="utf-8"
    ) as destination:
        json.dump(
            {
                "metadata": metadata,
                "ground_truth_corners_px": ground_truth,
                "scene_records": records,
            },
            destination,
            default=json_value,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--scenes-csv", type=Path, required=True)
    parser.add_argument(
        "--homography-csv",
        type=Path,
        required=True,
        help="ShuttleSet homography CSV or .csv.gz with one row for this video",
    )
    parser.add_argument(
        "--baseline-module",
        type=Path,
        required=True,
        help="a111181 court_corners.py materialised with git show",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
