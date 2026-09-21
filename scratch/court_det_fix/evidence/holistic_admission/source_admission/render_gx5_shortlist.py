"""Render post-hoc GX5 shortlist candidates over the frozen frame."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np


COLOURS = {
    "family_map_mean_directions": (255, 128, 0),
    "family_map_minimum_directions": (180, 0, 180),
    "union_map_mean_directions": (0, 180, 255),
    "union_map_minimum_directions": (255, 220, 0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = cv2.imread(str(args.frame))
    if frame is None:
        raise FileNotFoundError(args.frame)
    with gzip.open(args.automatic, "rt") as stream:
        record = json.load(stream)
    for variant_name, colour in COLOURS.items():
        candidate = record["variants"][variant_name]["caps"]["256"]["proposals"][0]
        corners = np.rint(np.asarray(candidate["corners_native"], dtype=float)).astype(int)
        cv2.polylines(frame, [corners.reshape(-1, 1, 2)], True, colour, 4, cv2.LINE_AA)
        anchor = tuple(corners[0] + np.array([8, -8]))
        cv2.putText(frame, variant_name, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), frame):
        raise OSError(args.output)


if __name__ == "__main__":
    main()
