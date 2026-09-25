"""Standing players' feet from a 3 s window of people detections around the analysed frame.

The rules match the fresh-feet evidence scripts in
``court_detector_optimisation_handover/claude_evidence/fresh_feet/``: the window from
``extract_window_people.sample_frames``, the grey differences from ``shot_check.py``, and the
"standing" feet from ``build_feet_variants.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from scratch.court_det_fix.court_detector.inputs import (
    FrameReader,
    PeopleSource,
    PersonSample,
    ViewInputs,
)

SAMPLES = 31
SAMPLE_FPS = 10
THUMBNAIL_SIZE = (64, 36)
# In-shot samples of the 20 court views differ from their anchor by at most 5.1 grey levels;
# the dissolve in control frame 1 and the passer-by at the lens in am3 frame 0 exceed 9.
SAME_SHOT_GREY_LEVELS = 8.0
# The seated-person rule from bst_x preparing_data.heuristics.base (sticky_anchor's rule).
# Copied because importing it loads pandas and BST-X's pipeline config;
# tests/test_court_detector_feet.py checks the copy agrees with the original.
SITTING_THRESHOLD = -0.3
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14


class FeetWindow(NamedTuple):
    """How one view's feet were gathered."""

    frames: list[int]  # the sampled frames, anchor included
    grey_differences: list[float] | None  # one per sampled frame; None with scene consistency off
    kept_frames: list[int]  # the sampled frames in the anchor's shot
    all_feet_px: list[list]  # one row per kept frame: [x, y] or None, padded to one width


def window_frames(anchor: int, fps: float, first_frame: int, last_frame: int) -> list[int]:
    """31 frames at 10 fps around the anchor, shifted as a block to stay inside the scene.

    :param first_frame: The scene's first frame.
    :param last_frame: The scene's last frame, inclusive.
    """
    step = fps / SAMPLE_FPS
    lowest_k = -int((anchor - first_frame) // step)
    highest_k = int((last_frame - anchor) // step)
    first_k = min(max(-(SAMPLES // 2), lowest_k), highest_k - (SAMPLES - 1))
    frames = [anchor + round(k * step) for k in range(first_k, first_k + SAMPLES)]
    if frames[0] < first_frame or frames[-1] > last_frame or anchor not in frames:
        raise ValueError(f"window {frames[0]}..{frames[-1]} does not fit scene {first_frame}..{last_frame} "
                         f"around {anchor}")
    return frames


def grey_thumbnail(frame: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grey, THUMBNAIL_SIZE, interpolation=cv2.INTER_AREA).astype(np.int16)


def grey_differences(anchor_frame: np.ndarray, frames: list[np.ndarray]) -> list[float]:
    """Mean grey-level difference of each frame from the anchor on thumbnails, rounded to 0.1.

    The rounding is part of the rule: the same-shot threshold compares rounded values.
    """
    anchor = grey_thumbnail(anchor_frame)
    return [round(float(np.abs(grey_thumbnail(frame) - anchor).mean()), 1) for frame in frames]


def same_shot_run(differences: list[float], anchor_position: int) -> range:
    """The unbroken run of sample positions around the anchor that stay in the anchor's shot."""
    first = anchor_position
    while first > 0 and differences[first - 1] <= SAME_SHOT_GREY_LEVELS:
        first -= 1
    last = anchor_position
    while last < len(differences) - 1 and differences[last + 1] <= SAME_SHOT_GREY_LEVELS:
        last += 1
    return range(first, last + 1)


def is_sitting(keypoints: np.ndarray) -> np.ndarray:
    """True for each pose whose knees sit off the body axis; (people, 17, 2) -> (people,).

    Projects the knee offset from the hips onto the hip-to-shoulder axis. A standing or
    airborne player's ratio is about -0.7 to -0.9; a seated person's is near 0.
    """
    shoulders = (keypoints[:, SHOULDER_L] + keypoints[:, SHOULDER_R]) / 2
    hips = (keypoints[:, HIP_L] + keypoints[:, HIP_R]) / 2
    knees = (keypoints[:, KNEE_L] + keypoints[:, KNEE_R]) / 2
    body_up = shoulders - hips
    knee_offset = knees - hips
    torso_length_squared = (body_up * body_up).sum(axis=1)
    degenerate = torso_length_squared < 1e-6
    ratio = (knee_offset * body_up).sum(axis=1) / np.where(degenerate, 1.0, torso_length_squared)
    return (ratio > SITTING_THRESHOLD) & ~degenerate


def standing_feet(samples: list[PersonSample], scale: np.ndarray, frame_size: tuple[int, int]) -> list[list]:
    """all_feet_px rows, one per sample: the standing people's feet in frame pixels.

    A foot is the bottom centre of a person box. A foot outside the frame is None, and rows
    are padded with None to one width of at least two slots, as in the frozen packs.

    :param scale: Frame pixels per FrameReader pixel, (x, y).
    :param frame_size: (width, height) of the analysed frame.
    """
    width, height = frame_size
    rows = []
    for sample in samples:
        x1, _y1, x2, y2 = sample.boxes_px.T
        feet = np.column_stack(((x1 + x2) / 2, y2)) * scale
        standing = ~is_sitting(sample.keypoints_px)
        row = []
        for foot_x, foot_y in feet[standing]:
            on_image = 0 <= foot_x < width and 0 <= foot_y < height
            row.append([float(foot_x), float(foot_y)] if on_image else None)
        rows.append(row)
    slots = max(2, *(len(row) for row in rows))
    return [row + [None] * (slots - len(row)) for row in rows]


def window_feet(view: ViewInputs, people: PeopleSource, frames: FrameReader,
                enforce_scene_consistency: bool) -> FeetWindow:
    """Gather the standing feet around the view's frame.

    :param enforce_scene_consistency: Keep only the samples in the anchor's shot, judged by
        grey thumbnails. Off keeps the whole window.
    """
    window = window_frames(view.frame_index, frames.fps, *view.scene_frames)
    anchor_position = window.index(view.frame_index)
    if enforce_scene_consistency:
        images = frames.read(window)
        differences = grey_differences(images[anchor_position], images)
        kept = [window[position] for position in same_shot_run(differences, anchor_position)]
    else:
        differences, kept = None, window
    samples = people.samples(kept)
    returned = [sample.frame_index for sample in samples]
    if returned != kept:
        raise ValueError(f"{view.view_id}: people source returned frames {returned}, asked for {kept}")
    frame_height, frame_width = view.frame.shape[:2]
    scale = np.asarray((frame_width, frame_height)) / np.asarray(frames.size)
    return FeetWindow(window, differences, kept, standing_feet(samples, scale, (frame_width, frame_height)))
