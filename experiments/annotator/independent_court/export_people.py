"""Export bounded RTMDet person detections from independent-court videos.

The exporter decodes a window in frame order, running the detector only for the
scheduled samples. Anchor frames are saved as native-resolution images for the
line exporter and are never silently added to the person-sample denominator.
The rtmlib model is imported only when the command is run, so ``--help`` and
the testable export function do not require the inference environment.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

DEFAULT_MODEL_BASENAME = "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip"
MAX_MEDIAN_FRAMES = 15


def _portable_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise ValueError(f"{field} must contain only letters, numbers, '_' or '-'")
    return value


def _portable_relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{field} must use portable '/' separators")
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute() or ":" in value:
        raise ValueError(f"{field} must be relative")
    parts = posix_path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{field} must not contain '.', '..', or empty path components")
    return "/".join(parts)


def _validate_window(window: Any) -> dict[str, Any]:
    if not isinstance(window, dict):
        raise TypeError("each manifest window must be an object")
    required = ("id", "video", "start_frame", "end_frame", "anchor_frames")
    missing = [key for key in required if key not in window]
    if missing:
        raise ValueError(f"window is missing required fields: {', '.join(missing)}")
    window_id = _portable_identifier(window["id"], "window id")
    video = _portable_relative_path(window["video"], "window video")
    start_frame, end_frame = window["start_frame"], window["end_frame"]
    if isinstance(start_frame, bool) or not isinstance(start_frame, int) or start_frame < 0:
        raise ValueError(f"{window_id}: start_frame must be a non-negative integer")
    if isinstance(end_frame, bool) or not isinstance(end_frame, int) or end_frame <= start_frame:
        raise ValueError(f"{window_id}: end_frame must be an integer after start_frame")
    anchors = window["anchor_frames"]
    if not isinstance(anchors, list):
        raise TypeError(f"{window_id}: anchor_frames must be a list")
    validated_anchors: list[int] = []
    for anchor in anchors:
        if isinstance(anchor, bool) or not isinstance(anchor, int):
            raise TypeError(f"{window_id}: anchor frame indices must be integers")
        if not start_frame <= anchor < end_frame:
            raise ValueError(f"{window_id}: anchor frame {anchor} is outside the window")
        validated_anchors.append(anchor)
    if len(set(validated_anchors)) != len(validated_anchors):
        raise ValueError(f"{window_id}: anchor_frames must not contain duplicates")
    return {
        "id": window_id,
        "video": video,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "anchor_frames": validated_anchors,
    }


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("windows"), list):
        raise TypeError(f"{path}: manifest must contain a windows list")
    windows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for window in manifest["windows"]:
        validated = _validate_window(window)
        case_key = validated["id"].casefold()
        if case_key in seen_ids:
            raise ValueError(f"duplicate window id: {validated['id']}")
        seen_ids.add(case_key)
        windows.append(validated)
    return windows


def _scheduled_indices(start_frame: int, end_frame: int, source_fps: float, sample_fps: float) -> list[int]:
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and positive")
    step = source_fps / sample_fps
    values = np.arange(start_frame, end_frame, step, dtype=np.float64)
    indices = np.rint(values).astype(np.int64)
    indices = np.unique(indices[(indices >= start_frame) & (indices < end_frame)])
    if len(indices) == 0:
        raise RuntimeError("sampling produced no frame indices")
    return [int(index) for index in indices]


def _known_frame_count(cap: cv2.VideoCapture, video_path: Path) -> int:
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(frame_count) or frame_count <= 0 or not frame_count.is_integer():
        raise ValueError(f"{video_path}: video has an unknown decode frame count")
    return int(frame_count)


def _source_fps(cap: cv2.VideoCapture, video_path: Path) -> float:
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"{video_path}: video has an unknown source frame rate")
    return source_fps


def _write_png(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image) or not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"could not write image: {path}")


def _write_gzip_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as target:
        json.dump(value, target, allow_nan=False)
    os.replace(temporary, path)


def _model_basename(detector: Any) -> str:
    return PurePosixPath(getattr(detector, "model_basename", DEFAULT_MODEL_BASENAME)).name


def _normalise_detections(result: Any, score_min: float) -> tuple[list[list[float]], list[float]]:
    try:
        boxes, scores = result
    except (TypeError, ValueError) as error:
        raise ValueError("detector must return (boxes, scores)") from error
    box_array = np.asarray(boxes, dtype=np.float64)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if box_array.size == 0:
        box_array = np.empty((0, 4), dtype=np.float64)
    elif box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError(f"detector returned unexpected boxes shape {box_array.shape}")
    if box_array.shape[0] != score_array.shape[0]:
        raise ValueError("detector returned incompatible boxes and scores")
    if not np.isfinite(box_array).all() or not np.isfinite(score_array).all():
        raise ValueError("detector returned non-finite boxes or scores")
    keep = score_array > score_min
    return box_array[keep].tolist(), score_array[keep].tolist()


def _video_path(video_root: Path, relative_video: str) -> Path:
    root = video_root.resolve()
    candidate = (root / Path(relative_video)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"video path escapes --video-root: {relative_video}") from error
    if not candidate.is_file():
        raise FileNotFoundError(f"video does not exist: {relative_video}")
    return candidate


def export_window(
    window: dict[str, Any],
    video_root: Path | str,
    output: Path | str,
    detector: Callable[[np.ndarray], Any],
    sample_fps: float,
    score_min: float,
) -> dict[str, Any]:
    """Export one validated manifest window and return its JSON record.

    :param window: A manifest window with a relative video path and frame bounds.
    :param video_root: Root directory containing the manifest's relative videos.
    :param output: Export directory, which receives the record and images.
    :param detector: Callable accepting native BGR pixels and returning boxes and scores.
    :param sample_fps: Requested detector sampling rate.
    :param score_min: Strict lower confidence cut applied to detector output.
    :return: The record written to ``<output>/<window id>.json.gz``.
    """
    validated = _validate_window(window)
    if not math.isfinite(score_min) or score_min < 0:
        raise ValueError("score_min must be finite and non-negative")
    video_root_path = Path(video_root)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    image_root = output_path / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    video_path = _video_path(video_root_path, validated["video"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video: {validated['video']}")
    try:
        frame_count = _known_frame_count(cap, video_path)
        source_fps = _source_fps(cap, video_path)
        start_frame = validated["start_frame"]
        end_frame = validated["end_frame"]
        if end_frame > frame_count:
            raise ValueError(
                f"{validated['id']}: window [{start_frame}, {end_frame}) exceeds {frame_count} frames"
            )
        scheduled = _scheduled_indices(start_frame, end_frame, source_fps, sample_fps)
        scheduled_set = set(scheduled)
        anchor_set = set(validated["anchor_frames"])
        median_count = min(MAX_MEDIAN_FRAMES, len(scheduled))
        median_positions = np.linspace(0, len(scheduled) - 1, num=median_count, dtype=int)
        median_set = {scheduled[int(position)] for position in median_positions}
        median_frames: list[np.ndarray] = []
        samples: list[dict[str, Any]] = []
        anchor_images: list[dict[str, Any]] = []
        dimensions: tuple[int, int] | None = None

        target_indices = scheduled_set | anchor_set
        for frame_index in range(end_frame):
            if frame_index in target_indices:
                ok, frame = cap.read()
            else:
                ok = cap.grab()
                frame = None
            if not ok:
                raise OSError(f"{validated['id']}: could not decode frame {frame_index}")
            if frame_index not in target_indices:
                continue
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"{validated['id']}: decoded frame {frame_index} has invalid pixels")
            frame_height, frame_width = frame.shape[:2]
            if dimensions is None:
                dimensions = (frame_width, frame_height)
            elif dimensions != (frame_width, frame_height):
                raise ValueError(f"{validated['id']}: video dimensions changed within the window")
            if frame_index in anchor_set:
                image_name = f"{validated['id']}_frame_{frame_index:08d}.png"
                image_path = image_root / image_name
                _write_png(image_path, frame)
                anchor_images.append(
                    {"frame_index": frame_index, "image": f"images/{image_name}"}
                )
            if frame_index in scheduled_set:
                boxes, scores = _normalise_detections(detector(frame), score_min)
                samples.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": frame_index / source_fps,
                        "bboxes": boxes,
                        "scores": scores,
                    }
                )
                if frame_index in median_set:
                    median_frames.append(frame.copy())
        if dimensions is None or len(samples) != len(scheduled) or len(median_frames) != len(median_set):
            raise RuntimeError(f"{validated['id']}: decoded target frames were incomplete")
        median_image = np.median(np.stack(median_frames, axis=0), axis=0).astype(np.uint8)
        median_name = f"{validated['id']}_median.png"
        _write_png(image_root / median_name, median_image)
    finally:
        cap.release()

    record: dict[str, Any] = {
        "id": validated["id"],
        "video": Path(validated["video"]).name,
        "dimensions": {"width": dimensions[0], "height": dimensions[1]},
        "source_fps": source_fps,
        "start_frame": validated["start_frame"],
        "end_frame": validated["end_frame"],
        "sample_fps": sample_fps,
        "export_score_min": score_min,
        "model_basename": _model_basename(detector),
        "samples": samples,
        "anchor_images": anchor_images,
        "median_image": f"images/{median_name}",
    }
    _write_gzip_json(output_path / f"{validated['id']}.json.gz", record)
    return record


def _build_detector(device: str, score_min: float) -> Any:
    # Keep this import inside the command path: importing this adapter loads rtmlib.
    from preparing_data.rtmlib_pose import DET_INPUT_SIZE, DET_URL, RTMDetScored

    detector = RTMDetScored(DET_URL, model_input_size=DET_INPUT_SIZE, device=device)
    if device.startswith("cuda") and "CUDAExecutionProvider" not in detector.session.get_providers():
        raise RuntimeError("CUDA was requested but ONNX Runtime did not activate its CUDA provider")
    detector.score_thr = score_min
    detector.model_basename = PurePosixPath(DET_URL).name
    return detector


def _write_manifests(output: Path, records: list[dict[str, Any]]) -> None:
    manifest = {
        "windows": [{"id": record["id"], "record": f"{record['id']}.json.gz"} for record in records]
    }
    cases: list[dict[str, str]] = []
    for record in records:
        window_id = record["id"]
        cases.append(
            {
                "id": f"{window_id}_median",
                "image": record["median_image"],
                "reference_status": "unlabelled",
            }
        )
        for anchor in record["anchor_images"]:
            cases.append(
                {
                    "id": f"{window_id}_frame_{anchor['frame_index']}",
                    "image": anchor["image"],
                    "reference_status": "unlabelled",
                }
            )
    _write_gzip_json(output / "manifest.json.gz", manifest)
    _write_gzip_json(output / "line_manifest.json.gz", {"cases": cases})


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-fps", type=_positive_float, default=10.0)
    parser.add_argument("--score-min", type=_non_negative_float, default=0.2)
    arguments = parser.parse_args(argv)

    if not arguments.manifest.is_file():
        raise FileNotFoundError(f"--manifest must name a file: {arguments.manifest}")
    if not arguments.video_root.is_dir():
        raise FileNotFoundError(f"--video-root must name a directory: {arguments.video_root}")
    windows = _load_manifest(arguments.manifest)
    # Validate every path before loading the GPU model, so a malformed batch fails early.
    for window in windows:
        _video_path(arguments.video_root, window["video"])
    arguments.output.mkdir(parents=True, exist_ok=True)
    detector = _build_detector(arguments.device, arguments.score_min) if windows else None
    records: list[dict[str, Any]] = []
    for window in windows:
        if detector is None:
            raise RuntimeError("internal error: detector was not initialised")
        records.append(
            export_window(
                window,
                arguments.video_root,
                arguments.output,
                detector,
                arguments.sample_fps,
                arguments.score_min,
            )
        )
    _write_manifests(arguments.output, records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
