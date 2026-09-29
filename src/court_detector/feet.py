"""Standing players' feet from a 3 s window of people detections around the analysed frame.

Sample within the anchor's shot, then restore brief crouches on tracks whose
observations are mostly standing. The player-position checks still run on every
kept sample.
"""

from __future__ import annotations

from collections import Counter
from typing import NamedTuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .inputs import FrameReader, PeopleSource, PersonSample, ViewInputs

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
MAX_TRACK_STEP_HEIGHTS = 1.0
UNMATCHABLE_STEP = 1e6


class FeetWindow(NamedTuple):
    """How one view's feet were gathered."""

    frames: list[int]  # the sampled frames, anchor included
    grey_differences: list[float] | None  # one per sampled frame; None with scene consistency off
    kept_frames: list[int]  # the sampled frames in the anchor's shot
    all_feet_px: list[list]  # one row per kept frame: [x, y] or None, padded to one width


def window_frames(anchor: int, fps: float, start_frame: int, end_frame: int) -> list[int]:
    """31 frames at 10 fps around the anchor, shifted as a block to stay inside the scene.

    :param start_frame: The scene's first frame.
    :param end_frame: The frame after the scene's last frame (exclusive end).
    """
    step = fps / SAMPLE_FPS
    # Measure to the last frame itself. Flooring the distance to end_frame instead can
    # put the final sample on end_frame, outside the scene.
    last_frame = end_frame - 1
    lowest_k = -int((anchor - start_frame) // step)
    highest_k = int((last_frame - anchor) // step)
    first_k = min(max(-(SAMPLES // 2), lowest_k), highest_k - (SAMPLES - 1))
    frames = [anchor + round(k * step) for k in range(first_k, first_k + SAMPLES)]
    if frames[0] < start_frame or frames[-1] >= end_frame or anchor not in frames:
        raise ValueError(f"window {frames[0]}..{frames[-1]} does not fit scene [{start_frame}, {end_frame}) "
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


def track_feet(all_feet: list[np.ndarray], heights: list[np.ndarray]) -> list[np.ndarray]:
    """Link consecutive detections by foot distance in mean body heights.

    A missing detection or a move beyond one body height ends a track. This
    short-window matcher can swap identities when people cross.
    """
    track_ids = []
    next_id = 0
    for sample_index, (sample_feet, sample_heights) in enumerate(zip(all_feet, heights, strict=True)):
        ids = np.full(len(sample_feet), -1, dtype=int)
        if sample_index and len(sample_feet) and len(all_feet[sample_index - 1]):
            previous_feet, previous_heights = all_feet[sample_index - 1], heights[sample_index - 1]
            distance = np.linalg.norm(previous_feet[:, None] - sample_feet[None], axis=2)
            steps = distance / ((previous_heights[:, None] + sample_heights[None]) / 2)
            # A finite cost permits an assignment even when a row has no usable match.
            costs = np.where(steps <= MAX_TRACK_STEP_HEIGHTS, steps, UNMATCHABLE_STEP)
            previous_rows, rows = linear_sum_assignment(costs)
            for previous_row, row in zip(previous_rows, rows, strict=True):
                if costs[previous_row, row] < UNMATCHABLE_STEP:
                    ids[row] = track_ids[-1][previous_row]
        for row in np.flatnonzero(ids < 0):
            ids[row] = next_id
            next_id += 1
        track_ids.append(ids)
    return track_ids


def standing_feet(samples: list[PersonSample], scale: np.ndarray, frame_size: tuple[int, int]) -> list[list]:
    """Feet of standing people, including brief crouches on mostly-standing tracks.

    Keep every standing observation. Restore a sitting observation only when a
    strict majority of its track is standing; ties keep the per-sample decision.

    A foot is the bottom centre of a person box. A foot outside the frame is None, and rows
    are padded with None to one width of at least two slots, as in the frozen packs.

    :param scale: Frame pixels per FrameReader pixel, (x, y).
    :param frame_size: (width, height) of the analysed frame.
    """
    all_feet, heights, standing = [], [], []
    for sample in samples:
        x1, y1, x2, y2 = sample.boxes_px.T
        all_feet.append(np.column_stack(((x1 + x2) / 2, y2)))
        heights.append(y2 - y1)
        standing.append(~is_sitting(sample.keypoints_px))
    tracks = track_feet(all_feet, heights)
    observations, standing_observations = Counter(), Counter()
    for ids, sample_standing in zip(tracks, standing, strict=True):
        observations.update(ids)
        standing_observations.update(ids[sample_standing])

    width, height = frame_size
    rows = []
    for sample_feet, ids, sample_standing in zip(all_feet, tracks, standing, strict=True):
        majority_standing = np.asarray([2 * standing_observations[track] > observations[track] for track in ids],
                                      dtype=bool)
        keep = sample_standing | majority_standing
        row = []
        for foot_x, foot_y in sample_feet[keep] * scale:
            on_image = 0 <= foot_x < width and 0 <= foot_y < height
            row.append([float(foot_x), float(foot_y)] if on_image else None)
        rows.append(row)
    slots = max(2, *(len(row) for row in rows))
    return [row + [None] * (slots - len(row)) for row in rows]


def can_satisfy_player_requirement(all_feet_px: list[list]) -> bool:
    """Enough standing people to possibly pass the required court checks.

    Each sample needs one person, and at least half need two. Their projected
    court positions are checked later. standing_feet already removes non-finite
    and off-image positions, so each non-None entry is a usable observation.
    """
    counts = [sum(foot is not None for foot in frame) for frame in all_feet_px]
    return min(counts, default=0) >= 1 and sum(count >= 2 for count in counts) * 2 >= len(counts)


def window_feet(view: ViewInputs, people: PeopleSource | None, frames: FrameReader,
                enforce_scene_consistency: bool, allow_short_window: bool = False) -> FeetWindow:
    """Gather the standing feet around the view's frame.

    :param enforce_scene_consistency: Keep only the samples in the anchor's shot, judged by
        grey thumbnails. Off keeps the whole window.
    """
    if people is None:
        return FeetWindow([], None, [], [])
    try:
        window = window_frames(view.frame_index, frames.fps, *view.scene_frames)
    except ValueError:
        if not allow_short_window:
            raise
        window = [view.frame_index]
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
