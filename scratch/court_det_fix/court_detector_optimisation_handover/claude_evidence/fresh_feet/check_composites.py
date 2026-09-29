"""Rebuild each ShuttleSet composite as a grey pixel median and compare with the frozen view."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

CHECKOUT = os.environ["CHECKOUT"] + "/scratch/court_det_fix/"
SOURCES = os.environ["SHUTTLESET_SOURCES"] + "/"
VIDEOS = {"sset03": SOURCES + "3 Kento_MOMOTA_CHOU_Tien_Chen_KOREA_OPEN_2019_Final.mp4",
          "sset21": SOURCES + "21 An_Se_Young_Ratchanok_Intanon_YONEX_Thailand_Open_2021_QuarterFinals.mp4"}
views = [view for view in json.loads(Path(sys.argv[1]).read_text()) if view["composite"]]
for video_key, path in VIDEOS.items():
    wanted = {frame for view in views if view["video"] == video_key for frame in view["image_frames"]}
    wanted |= {frame + 1 for frame in wanted}
    capture, grey = cv2.VideoCapture(path), {}
    for frame_index in range(max(wanted) + 1):
        capture.grab()
        if frame_index in wanted:
            frame = cv2.resize(capture.retrieve()[1], (960, 540), interpolation=cv2.INTER_AREA)
            grey[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
    for view in (view for view in views if view["video"] == video_key):
        frozen = cv2.imread(CHECKOUT + view["frozen_image"])
        channels_equal = bool((frozen[..., 0] == frozen[..., 1]).all() and (frozen[..., 1] == frozen[..., 2]).all())
        frozen_grey = frozen[..., 0].astype(np.int16)
        median = np.median(np.stack([grey[frame] for frame in view["image_frames"]]), axis=0)
        shifted = np.median(np.stack([grey[frame + 1] for frame in view["image_frames"]]), axis=0)
        print(json.dumps({"case_id": view["case_id"], "frozen_is_grey": channels_equal,
                          "median_mad": round(float(np.abs(median - frozen_grey).mean()), 3),
                          "median_max": int(np.abs(median - frozen_grey).max()),
                          "median_of_next_frames_mad": round(float(np.abs(shifted - frozen_grey).mean()), 3)}))
