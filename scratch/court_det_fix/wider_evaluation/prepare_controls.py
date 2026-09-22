"""Prepare the recorded broadcast controls for the wider evaluation runner."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

CONTROL_RELATIVE = Path("experiments/annotator/independent_court/recorded/controls.json.gz")
LINE_RELATIVE = Path("experiments/annotator/independent_court/recorded/neural_lines/deeplsd_md_default.json.gz")
PEOPLE_RELATIVE = Path("evidence/independent_proposals/development/people.json.gz")
IMAGE_ROOT_RELATIVE = Path("evidence/independent_proposals/development/inputs/controls")
PERSON_SCORE_CUTOFF = 0.3

def read_gzip(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)

def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def dimension_pair(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError(f"{label}: dimensions must contain width and height only")
    width, height = value["width"], value["height"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in (width, height)):
        raise ValueError(f"{label}: dimensions must be positive integers")
    return width, height

def finite_number(value: Any, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")

def validate_segments(record: dict[str, Any], case_id: str) -> list[list[float]]:
    segments = record.get("segments_px")
    if not isinstance(segments, list):
        raise TypeError(f"{case_id}: line segments must be a list")
    for index, segment in enumerate(segments):
        if not isinstance(segment, list) or len(segment) != 4:
            raise ValueError(f"{case_id}: line segment {index} must contain four values")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in segment):
            raise ValueError(f"{case_id}: line segment {index} contains a non-finite value")
    return segments

def validate_people(record: dict[str, Any], case_id: str) -> tuple[list[list[float]], list[float]]:
    samples = record.get("samples")
    if not isinstance(samples, list) or len(samples) != 1:
        raise ValueError(f"{case_id}: expected exactly one person sample")
    sample = samples[0]
    if not isinstance(sample, dict) or set(sample) != {"bboxes", "scores"}:
        raise ValueError(f"{case_id}: person sample must contain bboxes and scores only")
    boxes, scores = sample["bboxes"], sample["scores"]
    if not isinstance(boxes, list) or not isinstance(scores, list) or len(boxes) != len(scores):
        raise ValueError(f"{case_id}: person boxes and scores must be matching lists")
    for index, box in enumerate(boxes):
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"{case_id}: person box {index} must contain four values")
        for coordinate in box:
            finite_number(coordinate, f"{case_id}: person box {index}")
        finite_number(scores[index], f"{case_id}: person score {index}")
        if scores[index] <= PERSON_SCORE_CUTOFF:
            raise ValueError(f"{case_id}: person score {index} is not above the 0.3 cutoff")
    return boxes, scores

def prepare(root: Path) -> dict[str, Any]:
    repo = root.parents[1]
    controls_path, lines_path = repo / CONTROL_RELATIVE, repo / LINE_RELATIVE
    people_path, image_root = root / PEOPLE_RELATIVE, root / IMAGE_ROOT_RELATIVE
    controls, lines_payload, people_payload = (
        read_gzip(controls_path), read_gzip(lines_path), read_gzip(people_path)
    )
    metadata = controls.get("inputs", {}).get("cases")
    line_rows, people_rows = lines_payload.get("cases"), people_payload.get("records")
    if not all(isinstance(rows, list) for rows in (metadata, line_rows, people_rows)):
        raise ValueError("controls, lines and people inputs must contain lists")
    line_by_id = {row["id"]: row for row in line_rows}
    people_by_id = {row["id"]: row for row in people_rows}
    if len(line_by_id) != len(line_rows) or len(people_by_id) != len(people_rows):
        raise ValueError("line or people cache contains duplicate IDs")
    cases, review, images, empty = [], [], {}, []
    label_counts: Counter[str] = Counter()
    for item in metadata:
        case_id = item["id"]
        if case_id in images or case_id not in line_by_id or case_id not in people_by_id:
            raise ValueError(f"{case_id}: missing or duplicate control cache record")
        image_name = item["image"]
        image_path = (image_root / image_name).resolve()
        try:
            image_path.relative_to(image_root.resolve())
        except ValueError as error:
            raise ValueError(f"{case_id}: image path escapes the controls root") from error
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image_md5 = md5(image_path)
        line, people = line_by_id[case_id], people_by_id[case_id]
        if line.get("image_file_md5", "").lower() != image_md5:
            raise ValueError(f"{case_id}: line cache image MD5 does not match raw PNG")
        dimensions = dimension_pair(line.get("dimensions"), f"{case_id}: lines")
        if dimension_pair(people.get("dimensions"), f"{case_id}: people") != dimensions:
            raise ValueError(f"{case_id}: line and people dimensions differ")
        segments = validate_segments(line, case_id)
        boxes, scores = validate_people(people, case_id)
        feet = [[[(box[0] + box[2]) / 2, box[3]] for box in boxes]]
        while len(feet[0]) < 2:
            feet[0].append(None)
        relative_image = image_path.relative_to(root).as_posix()
        cases.append({
            "id": case_id, "dimensions": {"width": dimensions[0], "height": dimensions[1]},
            "segments_px": segments, "bbox_px": boxes, "bbox_scores": scores,
            "all_feet_px": feet, "image": relative_image, "frame_index": item["frame_index"],
            "provenance": {"people_source": people["source"], "people_model": people_payload["detector_model"],
                           "people_count": len(boxes), "people_sample_count": 1,
                           "person_score_cutoff": PERSON_SCORE_CUTOFF},
        })
        review.append({key: item[key] for key in ("id", "reference_status", "video_id", "frame_index", "image", "image_kind")})
        images[case_id] = image_md5
        label_counts[item["reference_status"]] += 1
        if not boxes:
            empty.append(case_id)
    if len(cases) != 24 or set(images) != {item["id"] for item in metadata}:
        raise ValueError(f"expected 24 unique control cases, found {len(cases)}")
    return {
        "schema": "wider-controls-inputs/1", "input_md5s": {
            str(CONTROL_RELATIVE): md5(controls_path), str(LINE_RELATIVE): md5(lines_path),
            str(PEOPLE_RELATIVE): md5(people_path), "images": images,
        }, "cases": cases, "review": review,
        "settings": {"line_variant": lines_payload["variant"], "person_score_cutoff": PERSON_SCORE_CUTOFF,
                     "footpoint": "bbox bottom-centre", "image_root": str(IMAGE_ROOT_RELATIVE)},
        "summary": {"case_count": len(cases), "label_counts": dict(label_counts),
                    "empty_detection_cases": empty, "empty_detection_count": len(empty),
                    "nonempty_detection_count": len(cases) - len(empty)},
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
