"""Decode named video frames to JPEGs. Usage: python extract_frames.py REQUESTS OUT_DIR"""
import json
import sys
from pathlib import Path

import cv2

requests, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for request in requests:
    capture = cv2.VideoCapture(request['video'])
    capture.set(cv2.CAP_PROP_POS_FRAMES, request['frame'])
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"{request['video']}: frame {request['frame']} did not decode")
    cv2.imwrite(str(out / request['name']), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(request['name'], image.shape)
