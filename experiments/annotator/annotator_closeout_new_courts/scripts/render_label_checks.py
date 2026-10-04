"""Render frame-exact stills around each label-check query frame, with the shuttle track drawn on.

Frames come from the validation overlay's span decoder, which selects frames by timestamp.
OpenCV seeking can land on a neighbouring frame, so the script can also check which exact
frame each earlier OpenCV still shows.
"""

from __future__ import annotations

import argparse
import json
import lzma
import subprocess
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from annotator.validation_overlay.core.decode import iter_span_frames

STILL_WIDTH = 640
STILL_HEIGHT = 360
CAPTION_HEIGHT = 32
CONTEXT_SECONDS = (-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)
BURST_FRAMES_BASE30 = (-12, -9, -6, -3, 0, 3, 6, 9, 12)
TRACK_COLOUR = (240, 16, 255)
SEEK_CHECK_FRAMES = 3


def video_format(path: Path) -> tuple[Fraction, int, int]:
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=r_frame_rate,width,height", "-of", "json", str(path)]
    stream = json.loads(subprocess.run(command, check=True, capture_output=True).stdout)["streams"][0]
    return Fraction(stream["r_frame_rate"]), int(stream["width"]), int(stream["height"])


def source_video(sources: Path, video: str) -> Path:
    videos = list(sources.glob(f"{int(video):02d} *.mp4"))
    assert len(videos) == 1, (video, videos)
    return videos[0]


def load_track(extracted: Path, video_path: Path) -> np.ndarray:
    with lzma.open(extracted / video_path.stem / "shuttle_track.npy.xz") as source:
        return np.load(source)


def decode_span(path: Path, first: int, last: int) -> dict[int, np.ndarray]:
    fps, width, height = video_format(path)
    frames = iter_span_frames(path, first, last, fps, width, height)
    return dict(zip(range(first, last + 1), frames, strict=True))


def draw_track(image: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Box the tracked shuttle; frames the tracker marked untracked stay unmarked."""
    marked = image.copy()
    x, y, visibility = position
    if visibility == 1.0:
        height, width = image.shape[:2]
        half_edge = round(25 * width / 1920)
        line_width = max(2, round(5 * width / 1920))
        centre_x, centre_y = round(x * width), round(y * height)
        cv2.rectangle(marked, (centre_x - half_edge, centre_y - half_edge),
                      (centre_x + half_edge, centre_y + half_edge), TRACK_COLOUR, line_width)
    return marked


def write_sheet(frames: dict[int, np.ndarray], indices: list[int], track: np.ndarray, case_id: str, query: int,
                path: Path) -> None:
    sheet = np.full((3 * (STILL_HEIGHT + CAPTION_HEIGHT), 3 * STILL_WIDTH, 3), 255, dtype=np.uint8)
    for position, frame in enumerate(indices):
        top = (position // 3) * (STILL_HEIGHT + CAPTION_HEIGHT)
        left = (position % 3) * STILL_WIDTH
        still = cv2.resize(draw_track(frames[frame], track[frame]), (STILL_WIDTH, STILL_HEIGHT),
                           interpolation=cv2.INTER_AREA)
        sheet[top:top + STILL_HEIGHT, left:left + STILL_WIDTH] = still
        offset = frame - query
        caption = f"{case_id} | frame {frame} | query {offset:+d}" if offset else f"{case_id} | frame {frame} | QUERY"
        cv2.putText(sheet, caption, (left + 8, top + STILL_HEIGHT + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 0), 1, cv2.LINE_AA)
    assert cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def render_cases(cases: pd.DataFrame, sources: Path, extracted: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for row in cases.itertuples(index=False):
        path = source_video(sources, row.video)
        track = load_track(extracted, path)
        fps = float(video_format(path)[0])
        assert abs(fps - row.fps) < 0.01, (fps, row.fps)
        context = [row.query_frame + round(offset * fps) for offset in CONTEXT_SECONDS]
        burst = [row.query_frame + round(offset * fps / 30) for offset in BURST_FRAMES_BASE30]
        frames = decode_span(path, min(context), max(context))
        write_sheet(frames, context, track, row.case_id, row.query_frame, output / f"{row.case_id}_context.jpg")
        write_sheet(frames, burst, track, row.case_id, row.query_frame, output / f"{row.case_id}_burst.jpg")
        assert cv2.imwrite(str(output / f"{row.case_id}_centre.jpg"), frames[row.query_frame],
                           [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(row.case_id, "complete", flush=True)


def check_saved_stills(sample: pd.DataFrame, sources: Path, saved: Path) -> None:
    """Report which exact frame each earlier OpenCV centre still shows."""
    for row in sample.itertuples(index=False):
        path = source_video(sources, row.video)
        frames = decode_span(path, row.source_frame - SEEK_CHECK_FRAMES, row.source_frame + SEEK_CHECK_FRAMES)
        still = cv2.imread(str(saved / f"{row.sample_id}_centre.jpg")).astype(np.int16)
        differences = {frame - row.source_frame: float(np.abs(pixels.astype(np.int16) - still).mean())
                       for frame, pixels in frames.items()}
        closest = min(differences, key=differences.__getitem__)
        print(row.sample_id, "closest offset", closest, {offset: round(value, 2) for offset, value in differences.items()},
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--extracted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saved-sample", type=Path, help="earlier miss sample whose OpenCV stills to check")
    parser.add_argument("--saved-stills", type=Path, help="folder holding the earlier centre stills")
    arguments = parser.parse_args()
    if arguments.saved_sample is not None:
        check_saved_stills(pd.read_csv(arguments.saved_sample, dtype={"video": str}), arguments.sources,
                           arguments.saved_stills)
    render_cases(pd.read_csv(arguments.cases, dtype={"video": str}), arguments.sources, arguments.extracted,
                 arguments.output)
