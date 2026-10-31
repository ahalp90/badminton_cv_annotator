"""Run the fixed SAM 3 court method on the MonoTrack manifest's middle frames."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from pathlib import Path

import cv2
import numpy as np

from experiments.court_detector.benchmark.sam3.merge_pieces import (
    fit_quadrilateral,
    largest_group,
)

REFERENCE_COLUMNS = ("upleft", "upright", "downleft", "downright")
REFERENCE_SIZE = np.array([1280.0, 720.0])


def write_json(path: Path, value: object) -> None:
    """Replace a compressed result atomically so status readers see complete files."""
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False)
    temporary.replace(path)


def reference_corners(row: dict, size: list[int]) -> np.ndarray:
    """Scale the official points and put them in cyclic convex-hull order."""
    points = np.array([[float(row[f"{name}_x"]), float(row[f"{name}_y"])] for name in REFERENCE_COLUMNS])
    points *= np.array(size) / REFERENCE_SIZE
    hull = cv2.convexHull(points.astype(np.float32)).reshape(-1, 2)
    if len(hull) != 4:
        raise ValueError("reference corners must form a convex quadrilateral")
    return hull


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return float(abs(np.dot(points[:, 0], np.roll(points[:, 1], -1))
                     - np.dot(points[:, 1], np.roll(points[:, 0], -1))) / 2)


def clip_convex_polygon(subject: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Clip a convex polygon in float64, including coincident image-edge segments."""
    points = np.asarray(subject, dtype=np.float64)
    boundary = np.asarray(boundary, dtype=np.float64)
    signed_area = (np.dot(boundary[:, 0], np.roll(boundary[:, 1], -1))
                   - np.dot(boundary[:, 1], np.roll(boundary[:, 0], -1)))
    orientation = 1 if signed_area >= 0 else -1
    for start, end in zip(boundary, np.roll(boundary, -1, axis=0), strict=True):
        if not len(points):
            break
        edge = end - start
        distances = orientation * (edge[0] * (points[:, 1] - start[1])
                                   - edge[1] * (points[:, 0] - start[0]))
        clipped = []
        previous, previous_distance = points[-1], distances[-1]
        for current, distance in zip(points, distances, strict=True):
            # Reconstructing an on-edge vertex can create a tiny, reversed clipping edge.
            if previous_distance * distance < 0:
                fraction = previous_distance / (previous_distance - distance)
                clipped.append(previous + fraction * (current - previous))
            if distance >= 0:
                clipped.append(current)
            previous, previous_distance = current, distance
        points = np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
    return points


def clipped_iou(predicted: np.ndarray, reference: np.ndarray, size: list[int]) -> float:
    """Measure polygon area within the image rectangle, using continuous coordinates."""
    width, height = size
    rectangle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    reference_clip = clip_convex_polygon(reference, rectangle)
    predicted_clip = clip_convex_polygon(predicted, rectangle)
    reference_area = polygon_area(reference_clip)
    predicted_area = polygon_area(predicted_clip)
    if reference_area <= 0:
        raise ValueError("reference court has no visible area")
    if predicted_area <= 0:
        return 0.0
    overlap = polygon_area(clip_convex_polygon(predicted_clip, reference_clip))
    return float(np.clip(overlap / (predicted_area + reference_area - overlap), 0, 1))


def score_corners(predicted: np.ndarray | None, reference: np.ndarray, size: list[int]) -> dict:
    """Match cyclic walks using only true reference vertices within the image."""
    visible = ((reference >= 0) & (reference < np.array(size))).all(axis=1)
    result = {"visible_reference_corners": int(visible.sum()), "corner_error_px": None, "corner_error_1080p_px": None, "iou": 0.0}
    if predicted is None:
        return result
    result["iou"] = clipped_iou(predicted, reference, size)
    if visible.any():
        errors = []
        for direction in (predicted, predicted[::-1]):
            for shift in range(4):
                matched = np.roll(direction, shift, axis=0)
                errors.append(float(np.linalg.norm(matched[visible] - reference[visible], axis=1).mean()))
        result["corner_error_px"] = min(errors)
        result["corner_error_1080p_px"] = result["corner_error_px"] * 1080 / size[1]
    return result


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"count": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)),
            "p10": float(np.quantile(values, 0.1)), "p90": float(np.quantile(values, 0.9))}


def summarise(records: list[dict]) -> dict:
    present = [record for record in records if record["corners"] is not None]
    errors = [record["corner_error_px"] for record in records if record["corner_error_px"] is not None]
    normalised_errors = [record["corner_error_1080p_px"] for record in records
                         if record["corner_error_1080p_px"] is not None]
    return {
        "rallies": len(records), "videos": len({record["video_id"] for record in records}),
        "predictions": len(present), "missing_predictions": len(records) - len(present),
        "rallies_without_visible_reference_corners": sum(
            record["visible_reference_corners"] == 0 for record in records),
        "corner_error_px_present": distribution(errors),
        "corner_error_1080p_px_present": distribution(normalised_errors),
        "iou_all_rallies_missing_zero": distribution([record["iou"] for record in records]),
        "iou_present": distribution([record["iou"] for record in present]),
    }


def evaluate_video(item: dict, reference_row: dict, video_root: Path, processor: object,
                   records: list[dict], output: Path, total: int, started: float) -> None:
    from PIL import Image

    video_path = video_root / item["source_filename"]
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {item['video_id']}")
    try:
        size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        if size != item["resolution"] or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != item["frame_count"]:
            raise ValueError(f"video {item['video_id']}: dimensions or frame count differ from manifest")
        reference = reference_corners(reference_row, size)
        for rally_number, span in enumerate(item["spans"], 1):
            midpoint = (span["start_frame"] + span["end_frame"]) // 2
            capture.set(cv2.CAP_PROP_POS_FRAMES, midpoint)
            ok, frame = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != midpoint + 1:
                raise OSError(f"video {item['video_id']}: cannot decode frame {midpoint}")
            state = processor.set_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            detected = processor.set_text_prompt(state=state, prompt="court markings")
            scores = detected["scores"].float().cpu().numpy()
            masks = detected["masks"][:, 0].cpu().numpy()[scores > 0.5]
            mask = largest_group(masks)
            corners = fit_quadrilateral(mask)
            record = {
                "video_id": item["video_id"], "rally": rally_number, "rally_id": span["rally_id"],
                "start_frame": span["start_frame"], "end_frame": span["end_frame"],
                "midpoint_frame": midpoint, "image_size": size, "reference_corners": reference.tolist(),
                "corners": None if corners is None else corners.tolist(),
                "instances": int((scores > 0.5).sum()), "retained_pixels": int(mask.sum()),
                **score_corners(corners, reference, size),
            }
            records.append(record)
            elapsed = time.monotonic() - started
            status = {"state": "running", "completed": len(records), "total": total,
                      "video_id": item["video_id"], "rally": rally_number, "elapsed_seconds": elapsed,
                      "estimated_remaining_seconds": elapsed / len(records) * (total - len(records))}
            write_json(output / "status.json.gz", status)
            print(f"{len(records)}/{total} video={item['video_id']} rally={rally_number} "
                  f"court={corners is not None} elapsed={elapsed:.1f}s", flush=True)
    finally:
        capture.release()


def run(args: argparse.Namespace) -> None:
    import sam3
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    with gzip.open(args.manifest, "rt") as stream:
        items = json.load(stream)["items"]
    with gzip.open(args.references, "rt") as stream:
        references = {int(row["id"]): row for row in csv.DictReader(stream)}
    for item in items:
        number, name = item["directory"].split(" ", 1)
        if int(number) != item["video_id"] or name != references[item["video_id"]]["video"]:
            raise ValueError(f"reference identity differs for video {item['video_id']}")
        if not (args.videos / item["source_filename"]).is_file():
            raise FileNotFoundError(f"missing source video {item['video_id']}")
    total = sum(len(item["spans"]) for item in items)
    started = time.monotonic()
    write_json(args.out / "status.json.gz", {"state": "loading_model", "completed": 0, "total": total})
    processor = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.5)
    records = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for item in items:
            evaluate_video(item, references[item["video_id"]], args.videos, processor, records,
                           args.out, total, started)
            write_json(args.out / "rallies.json.gz", records)
    summary = summarise(records)
    summary["settings"] = {"prompt": "court markings", "confidence_strictly_above": 0.5,
                           "fit": "cv2.approxPolyN(points, 4), default arguments, native resolution",
                           "midpoint": "(start_frame + end_frame) // 2; end_frame exclusive"}
    summary["environment"] = {"opencv": cv2.__version__, "numpy": np.__version__,
                              "torch": torch.__version__, "sam3": getattr(sam3, "__version__", "unknown"),
                              "gpu": torch.cuda.get_device_name(), "autocast": "bfloat16"}
    summary["elapsed_seconds"] = time.monotonic() - started
    write_json(args.out / "summary.json.gz", summary)
    write_json(args.out / "status.json.gz", {"state": "complete", "completed": len(records), "total": total,
                                           "elapsed_seconds": summary["elapsed_seconds"]})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True, help="official homography.csv.gz")
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "rallies.json.gz").exists():
        raise FileExistsError("output already contains predictions; choose a fresh output directory")
    try:
        run(args)
    except Exception as error:
        write_json(args.out / "status.json.gz", {"state": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
