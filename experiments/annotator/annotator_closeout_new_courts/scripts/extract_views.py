"""Render a context sheet and a hit-timing burst around each sampled source frame.

The context sheet repeats the earlier nine stills at half-second steps over ±2 s.
The burst adds nine frames at three-frame steps (30 fps clock) around the label,
which is close enough to see whether a racket meets the shuttle at that time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FRAME_WIDTH = 640
FRAME_HEIGHT = 360
LABEL_HEIGHT = 32
CONTEXT_SECONDS = (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
BURST_FRAMES_BASE30 = (-12, -9, -6, -3, 0, 3, 6, 9, 12)


def read_frame(capture: cv2.VideoCapture, frame: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    success, pixels = capture.read()
    assert success, frame
    return pixels


def write_sheet(capture: cv2.VideoCapture, frames: list[int], fps: float, sample_id: str, path: Path) -> None:
    sheet = np.full((3 * (FRAME_HEIGHT + LABEL_HEIGHT), 3 * FRAME_WIDTH, 3), 255, dtype=np.uint8)
    for index, frame in enumerate(frames):
        top = (index // 3) * (FRAME_HEIGHT + LABEL_HEIGHT)
        left = (index % 3) * FRAME_WIDTH
        sheet[top:top + FRAME_HEIGHT, left:left + FRAME_WIDTH] = cv2.resize(
            read_frame(capture, frame), (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA,
        )
        label = f"{sample_id} | frame {frame} | {frame / fps:.2f} s"
        cv2.putText(sheet, label, (left + 8, top + FRAME_HEIGHT + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def run(sample: Path, sources: Path, output: Path) -> None:
    rows = pd.read_csv(sample)
    output.mkdir(parents=True, exist_ok=True)
    for video, group in rows.groupby("video", sort=False):
        paths = list(sources.glob(f"{int(video):02d} *.mp4"))
        assert len(paths) == 1, (video, paths)
        capture = cv2.VideoCapture(str(paths[0]))
        assert capture.isOpened(), paths[0]
        fps = capture.get(cv2.CAP_PROP_FPS)
        last_frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        for row in group.itertuples(index=False):
            assert abs(fps - row.fps) < 0.01, (fps, row.fps)
            context = [min(last_frame, max(0, round(row.source_frame + offset * fps))) for offset in CONTEXT_SECONDS]
            burst = [min(last_frame, max(0, round(row.source_frame + offset * fps / 30)))
                     for offset in BURST_FRAMES_BASE30]
            write_sheet(capture, context, fps, row.sample_id, output / f"{row.sample_id}_context.jpg")
            write_sheet(capture, burst, fps, row.sample_id, output / f"{row.sample_id}_burst.jpg")
            assert cv2.imwrite(str(output / f"{row.sample_id}_centre.jpg"), read_frame(capture, row.source_frame),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(row.sample_id, "complete", flush=True)
        capture.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.sample, arguments.sources, arguments.output)
