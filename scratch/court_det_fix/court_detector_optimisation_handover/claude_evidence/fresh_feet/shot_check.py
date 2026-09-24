"""Per window sample: mean grey difference from the anchor frame, on 64x36 thumbnails.

A window that crosses a shot change shows a jump. Usage: shot_check.py PEOPLE_DIR VIDEO_KEY VIDEO_PATH
"""
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

people_dir, video_key, video_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
records = []
for path in sorted(people_dir.glob("*.json.gz")):
    with gzip.open(path, "rt") as stream:
        record = json.load(stream)
    if record["video"] == video_key:
        records.append((record["case_id"], record["anchor"], [sample["frame_index"] for sample in record["samples"]]))
wanted = {frame for _, anchor, frames in records for frame in [anchor, *frames]}
capture, thumbnails = cv2.VideoCapture(video_path), {}
for frame_index in range(max(wanted) + 1):
    capture.grab()
    if frame_index in wanted:
        grey = cv2.cvtColor(capture.retrieve()[1], cv2.COLOR_BGR2GRAY)
        thumbnails[frame_index] = cv2.resize(grey, (64, 36), interpolation=cv2.INTER_AREA).astype(np.int16)
for case_id, anchor, frames in records:
    differences = [round(float(np.abs(thumbnails[frame] - thumbnails[anchor]).mean()), 1) for frame in frames]
    print(json.dumps({"case_id": case_id, "anchor": anchor, "max": max(differences), "differences": differences}))
