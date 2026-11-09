"""Score stored per-rally courts and SAM 3 with the same visible-court metrics."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.monotrack_benchmark.writeup_scripts.court_metrics import (
    per_rally_courts,
)
from experiments.court_detector.benchmark.sam3.benchmark_rallies import (
    score_corners,
    summarise,
    write_json,
)

SOURCES = ("ours", "monotrack", "sam3")


def compare(sam_records: list[dict], inputs: Path) -> tuple[list[dict], dict]:
    with gzip.open(inputs / "manifest.json.gz", "rt") as stream:
        items = json.load(stream)["items"]
    sam_by_key = {(record["video_id"], record["rally_id"]): record for record in sam_records}
    expected = {(item["video_id"], span["rally_id"]) for item in items for span in item["spans"]}
    if set(sam_by_key) != expected or len(sam_records) != len(expected):
        raise ValueError("SAM results must contain every manifest rally exactly once")
    records = []
    for item in items:
        stored = {}
        for source in SOURCES[:2]:
            path = inputs / source / item["item_id"] / "court_evidence.json.gz"
            stored[source] = per_rally_courts(path, item["spans"])
        for index, span in enumerate(item["spans"]):
            sam = sam_by_key[item["video_id"], span["rally_id"]]
            if (sam["start_frame"], sam["end_frame"]) != (span["start_frame"], span["end_frame"]):
                raise ValueError("SAM result and manifest rally bounds differ")
            record = {key: sam[key] for key in ("video_id", "rally", "rally_id", "start_frame", "end_frame",
                                               "midpoint_frame", "image_size", "reference_corners")}
            reference = np.asarray(sam["reference_corners"], dtype=np.float32)
            predictions = {}
            for source in SOURCES:
                points = sam["corners"] if source == "sam3" else stored[source][index]
                # The hull restores cyclic order, including one crossed MonoTrack
                # outline. All stored predictions have four distinct hull vertices.
                corners = None if points is None else cv2.convexHull(np.asarray(points, np.float32)).reshape(4, 2)
                predictions[source] = {
                    "corners": None if corners is None else corners.tolist(),
                    **score_corners(corners, reference, sam["image_size"]),
                }
            record["detectors"] = predictions
            records.append(record)
    summary = {"rallies": len(records), "videos": len(items), "detectors": {}, "per_video": {}}
    common = [record for record in records
              if all(record["detectors"][source]["corners"] is not None for source in SOURCES)]
    summary["all_three_present"] = {"rallies": len(common), "detectors": {}}
    for source in SOURCES:
        flattened = [{"video_id": record["video_id"], **record["detectors"][source]} for record in records]
        values = summarise(flattened)
        for threshold in (5, 15):
            within = sum(row["corner_error_px"] is not None and row["corner_error_px"] <= threshold
                         for row in flattened)
            values[f"within_{threshold}px"] = {"rallies": within, "fraction_all_rallies": within / len(records)}
        summary["detectors"][source] = values
        summary["all_three_present"]["detectors"][source] = summarise(
            [{"video_id": record["video_id"], **record["detectors"][source]} for record in common])
        summary["per_video"][source] = {
            str(item["video_id"]): summarise([row for row in flattened if row["video_id"] == item["video_id"]])
            for item in items
        }
    return records, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam-results", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True, help="committed MonoTrack benchmark inputs directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.sam_results, "rt") as stream:
        records, summary = compare(json.load(stream), args.inputs)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "comparison_rallies.json.gz", records)
    write_json(args.out / "comparison_summary.json.gz", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "per_video"}, indent=2))


if __name__ == "__main__":
    main()
