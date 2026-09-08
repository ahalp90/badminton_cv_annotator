"""Measure centre-line junction evidence without changing frozen candidate rankings."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import junction_observations as junctions
from .assignment import prepare_observations
from .detector import CORNER_COURT_M
from .run_assignment import read_replay


def run_case(case: dict, frozen: dict) -> dict:
    started = perf_counter()
    width, height = case["dimensions"]["width"], case["dimensions"]["height"]
    size = tuple(frozen["working_size"])
    scale = np.asarray([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    observations = prepare_observations(segments, size)
    boxes = np.asarray(case["bbox_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    entries = []
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        homography = cv2.getPerspectiveTransform(
            CORNER_COURT_M, (np.asarray(source["corners_px"]) / scale).astype(np.float32),
        )
        entries.append({"id": source["id"], **junctions.measure(homography, observations, boxes, size)})
    return {"id": case["id"], "entries": entries, "elapsed_seconds": perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stripes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, _ = read_replay(args.replay)
    saved = json.loads(gzip.decompress(args.stripes.read_bytes()))
    cases = {case["id"]: case for case in inputs["cases"]}
    records = []
    for source in saved["records"]:
        record = run_case(cases[source["id"]], source)
        records.append(record)
        print(f"{record['id']}: {len(record['entries'])} eligible geometries, "
              f"{record['elapsed_seconds']:.2f}s", flush=True)
    output = {"schema": "frozen-junction-diagnostic/1", "diagnostic_only": True, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
