"""Production formulas for the frozen v1 features.

Issue #22 defines the formulas. Issue #104 kept posture variability, rally
timestamps, and linear-interpolation provenance. Issue #18 moved those parts
here from the benchmark-only prototype so the dataset export and the ShuttleSet
benchmark share one implementation. Issue #138 moved recovery_at_opponent_contacts
and movement_inefficiency here too, once human ShuttleSet contacts made them
reliable enough to export.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
from fractions import Fraction
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.court_evidence import detected_court_info
from annotator.fps_constants import ScalingKind
from annotator.point_winner import project_pixels_to_court
from annotator.rally.evidence import build_sticky_result, tracker_segments
from dataset_builder.vision import CourtVision, PoseArrays
from shared.court import HOMOGRAPHY_RESOLUTION


EYE_INDICES = (1, 2)
HIP_INDICES = (11, 12)
ANKLE_INDICES = (15, 16)
COURT_SIDES = ("top", "bottom")
# Issue #32 fixed the clip offsets: 2 s of lead-in before the first contact and
# 3 s of tail after the last, so a clip keeps the serve setup and the point ending.
CLIP_LEAD_SECONDS = 2
CLIP_TAIL_SECONDS = 3
# Issue #138: the away-from-centre recovery window is +/- 5 base-30 frames around
# a contact, and each side recovers toward its own half-court centre.
RECOVERY_HALF_WINDOW_BASE30 = 5
HALF_CENTRES = np.array(((0.5, 0.25), (0.5, 0.75)), dtype=float)


class InterpolationType(IntEnum):
    """Issue #22's compact per-frame interpolation provenance values."""

    OBSERVED = 0
    LINEAR = 1
    BACKWARD_EXTRAPOLATED = 2


class PlayerFeatureInputs(NamedTuple):
    """Frame-aligned player signals derived through production primitives.

    Every per-slot axis is ordered ``[top, bottom]`` like the sticky picker.
    """

    posture: np.ndarray
    court_positions: np.ndarray
    posture_interpolation: np.ndarray
    position_interpolation: np.ndarray
    tracker_segments: tuple[tuple[int, int], ...]


class PlayerRallyFeatures(NamedTuple):
    """The frozen v1 per-player, per-rally feature values."""

    court_side: str
    posture_frames_valid: int
    posture_frames_linear: int
    posture_mad: float | None
    position_frames_valid: int
    position_frames_linear: int


def derive_player_feature_inputs(
    track: np.ndarray,
    pose: PoseArrays,
    court: CourtVision,
    video_id: str,
) -> PlayerFeatureInputs:
    """Build issue #22 player signals through the production sticky-player path."""
    inputs = court.evidence.inputs
    if inputs is None:
        raise ValueError("court evidence has no operational inputs")
    if len(track) != len(pose.kps) or len(track) != len(court.evidence.court_present):
        raise ValueError("shuttle, pose, and court frame populations differ")
    segments = tracker_segments(
        inputs.homography_rows.to_dict("records"),
        court.evidence.court_present,
        len(track),
    )
    sticky = build_sticky_result(
        track,
        segments,
        pose.bboxes,
        pose.scores,
        pose.kps,
        pose.ndet,
        video_id,
        inputs.gate_court_info,
        inputs.gate_resolution_table,
        inputs.resolution,
    )
    selected_keypoints = select_sticky_keypoints(pose.kps, sticky.picks)
    raw_posture = posture_signal(selected_keypoints)
    posture_values, posture_interpolation = interpolate_internal_gaps(
        raw_posture[:, :, np.newaxis], segments
    )
    image_positions = sticky.ankle_pos * np.asarray(inputs.resolution, dtype=float)
    court_positions = project_positions_by_scene(
        image_positions,
        inputs.homography_rows,
        inputs.resolution,
    )
    interpolated, position_interpolation = interpolate_internal_gaps(
        court_positions, segments
    )
    return PlayerFeatureInputs(
        posture_values[:, :, 0],
        interpolated,
        posture_interpolation,
        position_interpolation,
        tuple(segments),
    )


def select_sticky_keypoints(pose_kps: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Select the two sticky-player pose rows, leaving unavailable rows as NaN."""
    pose = np.asarray(pose_kps, dtype=float)
    selected = np.asarray(picks)
    if pose.ndim != 4 or pose.shape[2:] != (17, 2):
        raise ValueError("pose_kps must have shape (frames, detections, 17, 2)")
    if selected.shape != (len(pose), 2):
        raise ValueError("picks must have shape (frames, 2)")
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("picks must contain integer detection slots")
    if np.any(selected < -1) or np.any(selected >= pose.shape[1]):
        raise ValueError("picks contain an invalid detection slot")

    result = np.full((len(pose), 2, 17, 2), np.nan, dtype=float)
    for slot in range(2):
        valid = selected[:, slot] >= 0
        frames = np.flatnonzero(valid)
        result[frames, slot] = pose[frames, selected[frames, slot]]
    return result


def posture_signal(player_keypoints: np.ndarray) -> np.ndarray:
    """Return issue #22's eye-to-ankle height divided by Euclidean hip width."""
    keypoints = np.asarray(player_keypoints, dtype=float)
    if keypoints.ndim != 4 or keypoints.shape[1:] != (2, 17, 2):
        raise ValueError("player_keypoints must have shape (frames, 2, 17, 2)")

    eyes_y = np.mean(keypoints[:, :, EYE_INDICES, 1], axis=2)
    ankles_y = np.mean(keypoints[:, :, ANKLE_INDICES, 1], axis=2)
    hip_delta = (
        keypoints[:, :, HIP_INDICES[0], :]
        - keypoints[:, :, HIP_INDICES[1], :]
    )
    hip_width = np.linalg.norm(hip_delta, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        posture = np.abs(eyes_y - ankles_y) / hip_width
    posture[~np.isfinite(posture) | (hip_width <= 0.0)] = np.nan
    return posture


def median_absolute_deviation(values: np.ndarray) -> float | None:
    """Return the issue #22 median absolute deviation over finite samples."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return None
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def recovery_at_opponent_contacts(
    court_positions: np.ndarray,
    contact_frames: Sequence[int],
    striker_slots: Sequence[int],
    fps: float,
    *,
    frame_range: tuple[int, int] | None = None,
) -> list[dict[str, int | float | None]]:
    """Measure the other player's mean distance from half-centre around contacts."""
    positions = np.asarray(court_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError("court_positions must have shape (frames, 2, 2)")
    if len(contact_frames) != len(striker_slots):
        raise ValueError("contact_frames and striker_slots must have equal length")
    half_window = int(
        ScalingKind.FRAME_COUNT.scale(RECOVERY_HALF_WINDOW_BASE30, fps)
    )
    range_start, range_end = (0, len(positions)) if frame_range is None else frame_range
    if not 0 <= range_start < range_end <= len(positions):
        raise ValueError("recovery frame range is invalid")
    rows: list[dict[str, int | float | None]] = []
    for frame, striker_slot in zip(contact_frames, striker_slots, strict=True):
        if not 0 <= frame < len(positions):
            raise ValueError("contact frame is outside the frame population")
        if striker_slot not in (0, 1):
            raise ValueError("striker slots must be 0 or 1")
        if not range_start <= frame < range_end:
            raise ValueError("contact frame is outside the recovery frame range")
        measured_slot = 1 - striker_slot
        start = max(range_start, frame - half_window)
        end = min(range_end, frame + half_window + 1)
        samples = positions[start:end, measured_slot]
        valid = np.isfinite(samples).all(axis=1)
        distances = np.linalg.norm(
            samples[valid] - HALF_CENTRES[measured_slot], axis=1
        )
        rows.append(
            {
                "contact_frame": frame,
                "measured_slot": measured_slot,
                "window_start": start,
                "window_end": end,
                "valid_frames": int(valid.sum()),
                "mean_distance": float(np.mean(distances)) if len(distances) else None,
            }
        )
    return rows


def movement_inefficiency(
    court_positions: np.ndarray, contact_frames: Sequence[int]
) -> np.ndarray:
    """Return path length minus straight displacement for each contact interval."""
    positions = np.asarray(court_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError("court_positions must have shape (frames, 2, 2)")
    contacts = np.asarray(contact_frames, dtype=int)
    if len(contacts) and (
        contacts.min() < 0
        or contacts.max() >= len(positions)
        or np.any(np.diff(contacts) <= 0)
    ):
        raise ValueError("contact frames must be strictly increasing and in range")
    result = np.full((max(0, len(contacts) - 1), 2), np.nan, dtype=float)
    for interval, (start, end) in enumerate(zip(contacts, contacts[1:])):
        for slot in range(2):
            path = positions[start : end + 1, slot]
            if not np.isfinite(path).all():
                continue
            path_length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            displacement = float(np.linalg.norm(path[-1] - path[0]))
            result[interval, slot] = path_length - displacement
    return result


def project_positions_by_scene(
    image_positions: np.ndarray,
    homography_rows: pd.DataFrame,
    resolution: tuple[float, float],
) -> np.ndarray:
    """Project frame-aligned player positions through each production scene row."""
    positions = np.asarray(image_positions, dtype=float)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError("image_positions must have shape (frames, slots, 2)")
    width, height = map(float, resolution)
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0:
        raise ValueError("resolution must contain positive finite values")

    projected = np.full_like(positions, np.nan, dtype=float)
    occupied = np.zeros(len(positions), dtype=bool)
    for row in homography_rows.to_dict("records"):
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if not 0 <= start <= end <= len(positions):
            raise ValueError("homography row is outside the frame population")
        if occupied[start:end].any():
            raise ValueError("homography rows overlap")
        occupied[start:end] = True
        ref_corners = scene_ref_corners(row, (width, height))
        court_info = detected_court_info(ref_corners)
        for slot in range(positions.shape[1]):
            points = positions[start:end, slot]
            valid = np.isfinite(points).all(axis=1)
            if not valid.any():
                continue
            frames = np.flatnonzero(valid) + start
            projected[frames, slot] = project_pixels_to_court(
                points[valid].T, (width, height), court_info
            ).T
    return projected


def scene_ref_corners(
    row: Mapping[str, object], resolution: tuple[float, float]
) -> np.ndarray:
    """Scale one scene row's native corner quad to the homography reference frame."""
    width, height = map(float, resolution)
    if not math.isfinite(width) or not math.isfinite(height) or min(width, height) <= 0:
        raise ValueError("resolution must contain positive finite values")
    native = np.array(
        [
            [row["upleft_x"], row["upleft_y"]],
            [row["upright_x"], row["upright_y"]],
            [row["downright_x"], row["downright_y"]],
            [row["downleft_x"], row["downleft_y"]],
        ],
        dtype=float,
    )
    if not np.isfinite(native).all():
        raise ValueError("scene corners must be finite")
    return native * np.array(
        [HOMOGRAPHY_RESOLUTION[0] / width, HOMOGRAPHY_RESOLUTION[1] / height]
    )


def interpolate_internal_gaps(
    values: np.ndarray, segments: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill only gaps bounded by observations inside one scene segment."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 3 or result.shape[2] < 1:
        raise ValueError("values must have shape (frames, slots, channels)")
    provenance = np.full(result.shape[:2], InterpolationType.OBSERVED, dtype=np.int8)
    occupied = np.zeros(len(result), dtype=bool)
    for start, end in segments:
        if not 0 <= start <= end <= len(result):
            raise ValueError("interpolation segment is outside the frame population")
        if occupied[start:end].any():
            raise ValueError("interpolation segments overlap")
        occupied[start:end] = True
        frame_axis = np.arange(start, end)
        for slot in range(result.shape[1]):
            block = result[start:end, slot]
            valid = np.isfinite(block).all(axis=1)
            if valid.sum() < 2:
                continue
            first = int(np.flatnonzero(valid)[0])
            last = int(np.flatnonzero(valid)[-1])
            fill = ~valid
            fill[:first] = False
            fill[last + 1 :] = False
            for coordinate in range(result.shape[2]):
                block[fill, coordinate] = np.interp(
                    frame_axis[fill], frame_axis[valid], block[valid, coordinate]
                )
            provenance[start:end, slot][fill] = InterpolationType.LINEAR
    return result, provenance


def rally_timestamps(start_frame: int, end_frame: int, fps: float) -> dict[str, object]:
    """Return one half-open rally frame range and its exact second range."""
    validate_frame_range(start_frame, end_frame)
    validate_fps(fps)
    return {
        "frame_range": [start_frame, end_frame],
        "second_range": [start_frame / fps, end_frame / fps],
        "fps": fps,
    }


def clip_frames(
    start_frame: int, end_frame: int, fps: Fraction, frame_count: int
) -> tuple[int, int]:
    """Return the issue #32 clip range around one rally, clamped to the video."""
    validate_frame_range(start_frame, end_frame)
    if frame_count < end_frame:
        raise ValueError(f"frame_count {frame_count} does not cover end_frame {end_frame}")
    return (
        max(0, start_frame - round(CLIP_LEAD_SECONDS * fps)),
        min(frame_count, end_frame + round(CLIP_TAIL_SECONDS * fps)),
    )


def player_rally_features(
    inputs: PlayerFeatureInputs, start_frame: int, end_frame: int
) -> tuple[PlayerRallyFeatures, PlayerRallyFeatures]:
    """Summarise one half-open rally for the top and bottom court sides."""
    validate_frame_range(start_frame, end_frame)
    posture = np.asarray(inputs.posture, dtype=float)
    positions = np.asarray(inputs.court_positions, dtype=float)
    if posture.ndim != 2 or posture.shape[1] != 2:
        raise ValueError("posture must have shape (frames, 2)")
    if positions.shape != (len(posture), 2, 2):
        raise ValueError("court_positions must have shape (frames, 2, 2)")
    if inputs.posture_interpolation.shape != posture.shape:
        raise ValueError("posture_interpolation must have shape (frames, 2)")
    if inputs.position_interpolation.shape != posture.shape:
        raise ValueError("position_interpolation must have shape (frames, 2)")
    if end_frame > len(posture):
        raise ValueError("rally range exceeds the feature arrays")

    rows = []
    for slot, side in enumerate(COURT_SIDES):
        posture_values = posture[start_frame:end_frame, slot]
        position_values = positions[start_frame:end_frame, slot]
        rows.append(
            PlayerRallyFeatures(
                court_side=side,
                posture_frames_valid=int(np.isfinite(posture_values).sum()),
                posture_frames_linear=_linear_count(
                    inputs.posture_interpolation[start_frame:end_frame, slot]
                ),
                posture_mad=median_absolute_deviation(posture_values),
                position_frames_valid=int(np.isfinite(position_values).all(axis=1).sum()),
                position_frames_linear=_linear_count(
                    inputs.position_interpolation[start_frame:end_frame, slot]
                ),
            )
        )
    return rows[0], rows[1]


def validate_frame_range(start_frame: int, end_frame: int) -> None:
    """Reject frame ranges that are negative, empty, or not half-open."""
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("frame range must be non-negative and half-open")


def validate_fps(fps: float) -> None:
    """Reject frame rates that cannot convert frames to seconds."""
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")


def _linear_count(provenance: np.ndarray) -> int:
    return int(np.count_nonzero(provenance == InterpolationType.LINEAR))
