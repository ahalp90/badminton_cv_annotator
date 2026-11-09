"""Prepare and score the hand-labelled courts, including provisional video midpoints."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.court_detector.benchmark.sam3.benchmark_rallies import (
    score_corners,
    summarise,
    write_json,
)
from experiments.court_detector.benchmark.sam3.merge_pieces import (
    fit_quadrilateral,
    largest_group,
)

ANNOTATIONS = ("hand_corners.csv", "2026-09-08/hand_corners.csv.gz", "2026-09-09/hand_corners.csv")
MOVING_CAMERA = "C1jR4vZmrkI"
SOURCES = ("ours", "monotrack", "sam3")


def read_json(path: Path) -> Any:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def requested_frames(capture: cv2.VideoCapture, indices: set[int], sequential: bool) -> dict[int, np.ndarray]:
    """Read decoded-frame indices; short downloaded clips contain discarded preroll."""
    images = {}
    if sequential:
        for index in range(max(indices) + 1):
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"cannot decode clip frame {index}")
            if index in indices:
                images[index] = frame
    else:
        for index in sorted(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise OSError(f"cannot decode video frame {index}")
            images[index] = frame
    return images


def prepare(labels: Path, video_map: Path, output: Path, include_moving_midpoint: bool) -> None:
    paths = json.loads(video_map.read_text())
    clips = read_json(labels / "additional_examples" / "manifest.json.gz")["cases"]
    grouped = {}
    for name in ANNOTATIONS:
        path = labels / name
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for row in csv.DictReader(stream):
                video = Path(row["video"]).stem
                grouped.setdefault(video, {}).setdefault(int(row["frame"]), []).append(row)
    frames = output / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    records = []
    for video, annotated in grouped.items():
        capture = cv2.VideoCapture(paths[video])
        if not capture.isOpened():
            raise OSError(f"cannot open {video}")
        try:
            size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
            clip_count = next((clip["decoded_sample_frame_count"] for clip in clips
                               if video.startswith(clip["video_id"])), None)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if clip_count is None else clip_count
            references = []
            requests = []
            for index, rows in sorted(annotated.items()):
                rows = sorted(rows, key=lambda row: int(row["corner_idx"]))
                if [int(row["corner_idx"]) for row in rows] != [0, 1, 2, 3]:
                    raise ValueError(f"{video}:{index} must have four labelled corners")
                reference = np.array([[float(row["x_norm"]), float(row["y_norm"])] for row in rows]) * size
                references.append(reference)
                requests.append((index, "labelled_frame", reference))
            if video != MOVING_CAMERA or include_moving_midpoint:
                midpoint = count // 2
                labelled = {index: reference for index, _, reference in requests}
                if midpoint in labelled:
                    requests.append((midpoint, "labelled_midpoint", labelled[midpoint]))
                else:
                    requests.append((midpoint, "video_mean_midpoint", np.mean(references, axis=0)))
            images = requested_frames(capture, {request[0] for request in requests}, clip_count is not None)
            for index, frame in images.items():
                if not cv2.imwrite(str(frames / f"{video}_f{index}.png"), frame):
                    raise OSError(f"cannot save {video}:{index}")
            for index, reference_type, reference in requests:
                # A labelled frame can also be the midpoint; keep one image for both.
                key = f"{video}_f{index}"
                records.append({"video_id": video, "frame_index": index, "image_key": key,
                                "image_size": size, "reference_type": reference_type,
                                "reference_corners": reference.tolist(), "annotation_frames": sorted(annotated),
                                "moving_camera": video == MOVING_CAMERA})
        finally:
            capture.release()
    write_json(output / "inputs.json.gz", records)
    print(f"Prepared {len(records)} comparisons from {len(grouped)} videos", flush=True)


def unique_images(output: Path) -> dict[str, Path]:
    records = read_json(output / "inputs.json.gz")
    return {row["image_key"]: output / "frames" / f"{row['image_key']}.png" for row in records}


def save_predictions(output: Path, source: str, records: dict) -> None:
    write_json(output / f"{source}.json.gz", records)
    print(f"{source}: {len(records)} images complete", flush=True)


def pending_images(output: Path, source: str, reuse: Path | None = None) -> tuple[dict, dict[str, Path]]:
    images = unique_images(output)
    records = {} if reuse is None else read_json(reuse)
    checkpoint = output / f"{source}.json.gz"
    if checkpoint.exists():
        records.update(read_json(checkpoint))
    records = {key: record for key, record in records.items() if key in images}
    return records, {key: path for key, path in images.items() if key not in records}


def run_sam(output: Path) -> None:
    records, pending = pending_images(output, "sam3")
    if not pending:
        return
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for key, path in pending.items():
            with Image.open(path) as image:
                state = processor.set_image(image.convert("RGB"))
            detected = processor.set_text_prompt(state=state, prompt="court markings")
            scores = detected["scores"].float().cpu().numpy()
            masks = detected["masks"][:, 0].cpu().numpy()[scores > 0.5]
            corners = fit_quadrilateral(largest_group(masks))
            records[key] = {"corners": None if corners is None else corners.tolist(),
                            "status": "no_court" if corners is None else "court"}
            save_predictions(output, "sam3", records)


def run_ours(output: Path, deeplsd_source: Path, deeplsd_weights: Path) -> None:
    records, pending = pending_images(output, "ours")
    if not pending:
        return
    from court_detector.detect import CourtFitError, Switches
    from court_detector.run_image import detect_image, load_image_tools, read_image

    switches = Switches(workers=4, timing=True, require_people=False, full_no_people_search=True)
    tools = load_image_tools(switches, deeplsd_source, deeplsd_weights)
    with tools.detector:
        for key, path in pending.items():
            started = time.monotonic()
            try:
                result = detect_image(read_image(path), tools, image_id=key, source=path.name)
                record = {"corners": result["corners_native_px"], "status": result["status"]}
            except CourtFitError as error:
                record = {"corners": None, "status": "fit_failed", "error": str(error)}
            record["seconds"] = time.monotonic() - started
            records[key] = record
            save_predictions(output, "ours", records)


def monotrack_image(binary: Path, path: Path, raw: Path) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run([str(binary), str(path), str(raw)], capture_output=True, text=True, check=False)
        if result.returncode == 3:
            record = {"corners": None, "status": "fit_failed", "error": result.stderr.strip()}
        elif result.returncode != 0:
            raise RuntimeError(f"MonoTrack {path.stem} exited {result.returncode}: {result.stderr}")
        else:
            points = np.loadtxt(raw, delimiter=";")
            if points.shape != (6, 2) or not np.isfinite(points).all():
                raise ValueError(f"MonoTrack {path.stem} must return six finite landmarks")
            record = {"corners": points[[0, 3, 2, 1]].tolist(), "status": "court"}
        record["seconds"] = time.monotonic() - started
        return record
    finally:
        raw.unlink(missing_ok=True)


def run_monotrack(output: Path, binary: Path, reuse: Path | None = None, workers: int = 4) -> None:
    records, pending = pending_images(output, "monotrack", reuse)
    save_predictions(output, "monotrack", records)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(monotrack_image, binary, path, output / f"{key}_landmarks.txt"): key
                   for key, path in pending.items()}
        for future in as_completed(futures):
            records[futures[future]] = future.result()
            save_predictions(output, "monotrack", records)


def frame_summary(values: list[dict]) -> dict:
    statistics = summarise(values)
    statistics["frames"] = statistics.pop("rallies")
    statistics["frames_without_visible_reference_corners"] = statistics.pop(
        "rallies_without_visible_reference_corners")
    statistics["iou_all_frames_missing_zero"] = statistics.pop("iou_all_rallies_missing_zero")
    return statistics


def score(output: Path, ours_source: str = "ours", result_prefix: str = "comparison") -> None:
    inputs = read_json(output / "inputs.json.gz")
    predictions = {source: read_json(output / f"{ours_source if source == 'ours' else source}.json.gz")
                   for source in SOURCES}
    expected = {row["image_key"] for row in inputs}
    if any(set(rows) != expected for rows in predictions.values()):
        raise ValueError("each detector must report every prepared image")
    records = []
    for row in inputs:
        reference = np.asarray(row["reference_corners"], np.float32)
        detectors = {}
        for source in SOURCES:
            prediction = predictions[source][row["image_key"]]
            points = prediction["corners"]
            corners = None if points is None else cv2.convexHull(np.asarray(points, np.float32)).reshape(4, 2)
            detectors[source] = {
                "corners": None if corners is None else corners.tolist(), "status": prediction["status"],
                **score_corners(corners, reference, row["image_size"]),
            }
            if source == "ours" and ours_source == "ours_robust":
                composition = prediction.get("composition")
                detectors[source]["sampling"] = {
                    "source_frame_index": prediction["source_frame_index"],
                    "court_frames": prediction["court_frames"],
                    "pose_window": prediction["pose_window"], "pose_frames": prediction["pose_frames"],
                    "selected_fit": None if composition is None else composition["court"],
                }
        records.append({**row, "our_detector_mode": "image_only" if ours_source == "ours" else "scene_robust",
                        "detectors": detectors})
    summary = {"per_video": {}}
    groups = {"labelled_frame": [row for row in records if row["reference_type"] == "labelled_frame"],
              "midpoint": [row for row in records if row["reference_type"] != "labelled_frame"]}
    for group, selected in groups.items():
        summary[group] = {}
        summary["per_video"][group] = {}
        for source in SOURCES:
            values = [{"video_id": row["video_id"], **row["detectors"][source]} for row in selected]
            summary[group][source] = frame_summary(values)
            summary["per_video"][group][source] = {
                video: frame_summary([row for row in values if row["video_id"] == video])
                for video in sorted({row["video_id"] for row in values})
            }
    summary["midpoint_reference_counts"] = {
        kind: sum(row["reference_type"] == kind for row in records)
        for kind in ("labelled_midpoint", "video_mean_midpoint")
    }
    summary["settings"] = {
        "ours": ("image mode, full line search, no people" if ours_source == "ours" else
                 "required people; three-frame scene-robust composition and best individual selection"),
        "corner_error_reporting": "1080p-equivalent pixels: native error * 1080 / image height; aspect ratio preserved",
        "monotrack": "released detector stages through the existing image adapter",
        "sam3": "court markings; scores > 0.5; enclosing approxPolyN fit at native resolution",
    }
    write_json(output / f"{result_prefix}_frames.json.gz", records)
    write_json(output / f"{result_prefix}_summary.json.gz", summary)
    print(json.dumps(summary, indent=2), flush=True)


def render_examples(output: Path, keys: list[str], reference_type: str, result_prefix: str = "comparison") -> None:
    from experiments.court_detector.benchmark.render_examples import render

    records = read_json(output / f"{result_prefix}_frames.json.gz")
    selected = {row["image_key"]: row for row in records if row["reference_type"] == reference_type}
    destination = output / ("examples" if result_prefix == "comparison" else f"{result_prefix}_examples")
    destination.mkdir(exist_ok=True)
    for key in keys:
        frame = cv2.imread(str(output / "frames" / f"{key}.png"))
        if frame is None:
            raise OSError(f"cannot read {key}")
        if not cv2.imwrite(str(destination / f"{key}.jpg"), render(selected[key], frame),
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"cannot render {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", *SOURCES, "score", "render"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--video-map", type=Path, help="JSON mapping label video stems to local source paths")
    parser.add_argument("--include-moving-midpoint", action="store_true")
    parser.add_argument("--deeplsd-source", type=Path)
    parser.add_argument("--deeplsd-weights", type=Path)
    parser.add_argument("--monotrack-binary", type=Path)
    parser.add_argument("--reuse-monotrack", type=Path, help="existing predictions for the same image keys")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keys", nargs="+")
    parser.add_argument("--ours-source", choices=("ours", "ours_robust"), default="ours")
    parser.add_argument("--result-prefix", default="comparison")
    parser.add_argument("--reference-type", choices=("labelled_frame", "labelled_midpoint", "video_mean_midpoint"),
                        default="labelled_frame")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.labels, args.video_map, args.out, args.include_moving_midpoint)
    elif args.stage == "ours":
        run_ours(args.out, args.deeplsd_source, args.deeplsd_weights)
    elif args.stage == "monotrack":
        run_monotrack(args.out, args.monotrack_binary, args.reuse_monotrack, args.workers)
    elif args.stage == "sam3":
        run_sam(args.out)
    elif args.stage == "render":
        render_examples(args.out, args.keys, args.reference_type, args.result_prefix)
    else:
        score(args.out, args.ours_source, args.result_prefix)


if __name__ == "__main__":
    main()
