"""Detect people and poses in one 3 s window at 10 fps around each D17 view's analysed frame.

Throwaway evaluation script for follow-up items 10 and 11. Every view gets 31 samples at
anchor + round(k * fps / 10), shifted as a block to stay inside the video; the anchor is
always one of them. The anchor is the analysed frame, or the middle frame of a ShuttleSet
composite. Frames are decoded in order from frame 0, so indices count decoded frames.

It also checks that the decoded frames are the frozen images: each source frame, and its
neighbours, is compared with the frozen PNG, and each composite is rebuilt as the pixel
median of its three frames.

Usage: extract_window_people.py VIEWS_JSON VIDEO_KEY VIDEO_PATH CHECKOUT_ROOT OUTPUT_DIR
"""

import gzip
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SAMPLES = 31
SAMPLE_FPS = 10
PERSON_SCORE_CUTOFF = 0.2  # as in the frozen GX, amateur and broadcast people


def sample_frames(anchor: int, fps: float, frame_count: int) -> list[int]:
    step = fps / SAMPLE_FPS
    lowest_k = -int(anchor // step)
    highest_k = int((frame_count - 1 - anchor) // step)
    first_k = min(max(-(SAMPLES // 2), lowest_k), highest_k - (SAMPLES - 1))
    frames = [anchor + round(k * step) for k in range(first_k, first_k + SAMPLES)]
    if frames[0] < 0 or frames[-1] >= frame_count or anchor not in frames:
        raise ValueError(f"window {frames[0]}..{frames[-1]} does not fit {frame_count} frames around {anchor}")
    return frames


def check_frames(view: dict) -> list[int]:
    """Frames whose pixels are compared with the frozen image."""
    if view["composite"]:
        return view["image_frames"]
    anchor = view["anchor"]
    return [frame for frame in (anchor - 1, anchor, anchor + 1) if frame >= 0]


def to_pack_size(frame: np.ndarray, pack_size: list[int]) -> np.ndarray:
    if [frame.shape[1], frame.shape[0]] == pack_size:
        return frame
    return cv2.resize(frame, tuple(pack_size), interpolation=cv2.INTER_AREA)


def mean_abs_difference(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.abs(first.astype(np.int16) - second.astype(np.int16)).mean())


class VideoEnded(Exception):
    """Decoding stopped before a needed frame; carries how many frames did decode."""

    def __init__(self, frame_count: int) -> None:
        super().__init__(f"video ended after {frame_count} frames")
        self.frame_count = frame_count


def plan_windows(views: list[dict], fps: float, frame_count: int) -> tuple[dict[str, list[int]], set[int]]:
    windows = {view["case_id"]: sample_frames(view["anchor"], fps, frame_count) for view in views}
    return windows, set().union(*windows.values())


def decode(video_path: str, detect_at: set[int], keep_pixels_at: set[int], extractor) -> tuple[dict, dict]:
    """Detections at the sampled frames and pixels at the check frames, decoding from frame 0."""
    capture = cv2.VideoCapture(video_path)
    detections, pixels = {}, {}
    for frame_index in range(max(detect_at | keep_pixels_at) + 1):
        if not capture.grab():
            raise VideoEnded(frame_index)
        if frame_index not in detect_at and frame_index not in keep_pixels_at:
            continue
        ok, frame = capture.retrieve()
        if not ok:
            raise RuntimeError(f"could not retrieve decoded frame {frame_index}")
        if frame_index in keep_pixels_at:
            pixels[frame_index] = frame
        if frame_index in detect_at:
            found = extractor.detect_frame(frame)
            detections[frame_index] = {"bboxes": found.bboxes.tolist(), "scores": found.bbox_scores.tolist(),
                                       "keypoints": found.keypoints.tolist(),
                                       "keypoint_scores": found.kp_scores.tolist()}
    return detections, pixels


def main() -> None:
    views_path, video_key, video_path, checkout_root, output_dir = sys.argv[1:]
    checkout_root, output_dir = Path(checkout_root), Path(output_dir)
    sys.path[:0] = [str(checkout_root / "src"), str(checkout_root / "src/bst_x")]
    from preparing_data.rtmlib_pose import (
        RtmlibPoseExtractor,  # pyrefly: ignore[missing-import]
    )

    views = [view for view in json.loads(Path(views_path).read_text()) if view["video"] == video_key]
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    windows, detect_at = plan_windows(views, fps, frame_count)
    keep_pixels_at = set().union(*(check_frames(view) for view in views))
    extractor = RtmlibPoseExtractor("cuda", det_score_thr=PERSON_SCORE_CUTOFF)

    started = time.perf_counter()
    try:
        detections, pixels = decode(video_path, detect_at, keep_pixels_at, extractor)
    except VideoEnded as ended:
        # The container's frame count can overstate what decodes: the short letterboxed clip
        # reports 230 frames and 91 decode. Refit the windows to the frames that exist.
        frame_count = ended.frame_count
        windows, detect_at = plan_windows(views, fps, frame_count)
        detections, pixels = decode(video_path, detect_at, keep_pixels_at, extractor)
    decode_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    for view in views:
        frozen = cv2.imread(str(checkout_root / "scratch/court_det_fix" / view["frozen_image"]))
        if frozen is None:
            raise FileNotFoundError(view["frozen_image"])
        if [frozen.shape[1], frozen.shape[0]] != view["pack_size"]:
            raise ValueError(f"{view['case_id']}: frozen image is {frozen.shape[:2]}, pack says {view['pack_size']}")
        if view["composite"]:
            resized = [to_pack_size(pixels[frame], view["pack_size"]) for frame in view["image_frames"]]
            median = np.median(np.stack(resized), axis=0).astype(np.uint8)
            pixel_check = {"median_of_image_frames": mean_abs_difference(median, frozen),
                           **{str(frame): mean_abs_difference(image, frozen)
                              for frame, image in zip(view["image_frames"], resized, strict=True)}}
        else:
            pixel_check = {str(frame): mean_abs_difference(to_pack_size(pixels[frame], view["pack_size"]), frozen)
                           for frame in check_frames(view)}
        record = {"case_id": view["case_id"], "video": video_key, "video_path": video_path, "fps": fps,
                  "frame_count": frame_count, "video_size": video_size, "pack_size": view["pack_size"],
                  "anchor": view["anchor"], "image_frames": view["image_frames"], "composite": view["composite"],
                  "person_score_cutoff": PERSON_SCORE_CUTOFF, "pixel_mean_abs_difference": pixel_check,
                  "samples": [{"frame_index": frame, **detections[frame]} for frame in windows[view["case_id"]]]}
        with gzip.open(output_dir / f"{view['case_id']}.json.gz", "wt") as stream:
            json.dump(record, stream)
        print(json.dumps({"case_id": view["case_id"], "window": [windows[view["case_id"]][0], windows[view["case_id"]][-1]],
                          "people_per_sample": [len(detections[frame]["scores"]) for frame in windows[view["case_id"]]],
                          "pixel_mean_abs_difference": pixel_check}))
    print(json.dumps({"video": video_key, "fps": fps, "frame_count": frame_count, "video_size": video_size,
                      "decode_and_detect_s": decode_seconds}))


if __name__ == "__main__":
    main()
