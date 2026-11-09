"""Draw a raw midpoint and three detector overlays for selected benchmark rallies."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.annotator.monotrack_benchmark.writeup_scripts.render_gallery import (
    OutlineStyle,
    dashed_outline,
)

STYLES = {
    "ours": OutlineStyle("Our detector", (255, 0, 255), 2, (18, 8)),
    "monotrack": OutlineStyle("MonoTrack", (0, 255, 255), 2, (7, 7)),
    "sam3": OutlineStyle("SAM 3", (0, 40, 255), 2, (30, 5)),
}
REFERENCE = OutlineStyle("Main-camera reference", (255, 255, 255), 1, (4, 4))


def panel(frame: np.ndarray, title: str, prediction: dict | None, reference: np.ndarray | None,
          style: OutlineStyle | None) -> np.ndarray:
    scale = min(960 / frame.shape[1], 540 / frame.shape[0])
    width, height = round(frame.shape[1] * scale), round(frame.shape[0] * scale)
    left, top = (960 - width) // 2, (540 - height) // 2
    offset = np.array([left, top])
    image = np.full((540, 960, 3), 28, np.uint8)
    image[top:top + height, left:left + width] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if prediction is not None and style is not None:
        if reference is not None:
            dashed_outline(image, reference * scale + offset, REFERENCE)
        if prediction["corners"] is not None:
            dashed_outline(image, np.asarray(prediction["corners"]) * scale + offset, style)
    band = np.full((60, 960, 3), 28, np.uint8)
    colour = (255, 255, 255) if style is None else style.colour
    cv2.putText(band, title, (14, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
    return np.vstack([band, image])


def render(record: dict, frame: np.ndarray, show_reference: bool = True) -> np.ndarray:
    reference = np.asarray(record["reference_corners"]) if show_reference else None
    if "rally" in record:
        title = f"Video {record['video_id']}, rally {record['rally']} - frame {record['midpoint_frame']}"
    else:
        title = f"{record['video_id']} - frame {record['frame_index']}"
    panels = [panel(frame, title, None, reference, None)]
    for source, style in STYLES.items():
        prediction = record["detectors"][source]
        error = prediction["corner_error_1080p_px"]
        score = "no court" if prediction["corners"] is None else ""
        if show_reference and error is not None:
            score = f"{error:.1f} px (1080p); IoU {prediction['iou']:.3f}"
        label = style.label
        if source == "ours" and "our_detector_mode" in record:
            label += " (image only)" if record["our_detector_mode"] == "image_only" else " (3 frames)"
        title = f"{label}: {score}" if score else label
        panels.append(panel(frame, title, prediction, reference, style))
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True, help="video:rally identifiers, e.g. 17:47")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hide-reference", action="store_true", help="omit reference and scores for an unmatched view")
    args = parser.parse_args()
    with gzip.open(args.results, "rt") as stream:
        records = {(row["video_id"], row["rally"]): row for row in json.load(stream)}
    with gzip.open(args.manifest, "rt") as stream:
        items = {item["video_id"]: item for item in json.load(stream)["items"]}
    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.keys:
        video, rally = map(int, key.split(":"))
        record = records[video, rally]
        capture = cv2.VideoCapture(str(args.videos / items[video]["source_filename"]))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, record["midpoint_frame"])
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"cannot read {key}")
        finally:
            capture.release()
        image = render(record, frame, show_reference=not args.hide_reference)
        path = args.out / f"video{video:02d}_rally{rally:03d}.jpg"
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"cannot write {path}")


if __name__ == "__main__":
    main()
