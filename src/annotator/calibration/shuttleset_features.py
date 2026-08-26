"""Benchmark-only prototypes for the trial features defined in issue #22."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import IntEnum
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.court_evidence import detected_court_info
from annotator.fps_constants import BASE_FPS, ScalingKind
from annotator.point_winner import project_pixels_to_court
from annotator.rally.evidence import build_sticky_result, tracker_segments
from annotator.shuttle_track import validate_shuttle_track
from dataset_builder.vision import CourtVision, PoseArrays
from shared.court import HOMOGRAPHY_RESOLUTION


EYE_INDICES = (1, 2)
HIP_INDICES = (11, 12)
ANKLE_INDICES = (15, 16)
RECOVERY_HALF_WINDOW_BASE30 = 5
HALF_CENTRES = np.array(((0.5, 0.25), (0.5, 0.75)), dtype=float)


class InterpolationType(IntEnum):
    """Issue #22's compact interpolation provenance values."""

    LINEAR = 1
    BACKWARD_EXTRAPOLATED = 2


class PlayerFeatureInputs(NamedTuple):
    """Frame-aligned signals derived through existing production primitives."""

    posture: np.ndarray
    court_positions: np.ndarray
    posture_interpolation: np.ndarray
    position_interpolation: np.ndarray
    tracker_segments: tuple[tuple[int, int], ...]


class ShuttleFeatureInputs(NamedTuple):
    """Masked, projected shuttle positions and their exclusion populations."""

    court_positions: np.ndarray
    invisible: np.ndarray
    guard_rejected: np.ndarray


def derive_shuttle_feature_inputs(
    track: np.ndarray,
    guard_codes: np.ndarray,
    rejected_grades: frozenset[int],
    homography_rows: pd.DataFrame,
    resolution: tuple[float, float],
) -> ShuttleFeatureInputs:
    """Mask production-ineligible shuttle frames before court projection."""
    validate_shuttle_track(track, len(track))
    codes = np.asarray(guard_codes)
    if codes.shape != (len(track),) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("guard_codes must be an integer vector matching the track")
    if not isinstance(rejected_grades, frozenset) or not rejected_grades.issubset(
        {1, 2, 3}
    ):
        raise ValueError("rejected_grades must be a frozenset subset of {1, 2, 3}")
    invisible = track[:, 2] != 1
    guard_rejected = np.isin(codes, tuple(sorted(rejected_grades)))
    image_positions = track[:, :2, np.newaxis].transpose(0, 2, 1).astype(float)
    image_positions *= np.asarray(resolution, dtype=float)
    image_positions[invisible | guard_rejected, 0] = np.nan
    projected = project_positions_by_scene(
        image_positions, homography_rows, resolution
    )[:, 0]
    return ShuttleFeatureInputs(projected, invisible, guard_rejected)


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
        ref_corners = _scene_ref_corners(row, (width, height))
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


def coordinate_error_summary(
    predicted: np.ndarray, ground_truth: np.ndarray
) -> dict[str, int | float | None]:
    """Summarise paired court-coordinate errors with exact exclusions."""
    prediction = np.asarray(predicted, dtype=float)
    truth = np.asarray(ground_truth, dtype=float)
    if prediction.shape != truth.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("coordinate pairs must share shape (observations, 2)")
    prediction_valid = np.isfinite(prediction).all(axis=1)
    truth_valid = np.isfinite(truth).all(axis=1)
    eligible = prediction_valid & truth_valid
    errors = np.linalg.norm(prediction[eligible] - truth[eligible], axis=1)
    return {
        "population": len(prediction),
        "eligible": int(eligible.sum()),
        "excluded_prediction": int((~prediction_valid & truth_valid).sum()),
        "excluded_ground_truth": int((~truth_valid).sum()),
        "mean_error": None if not len(errors) else float(np.mean(errors)),
        "median_error": None if not len(errors) else float(np.median(errors)),
        "p90_error": None if not len(errors) else float(np.percentile(errors, 90)),
    }


def court_corner_error_rows(
    homography_rows: pd.DataFrame,
    ground_truth_corners_refpx: np.ndarray,
    resolution: tuple[float, float],
) -> list[dict[str, object]]:
    """Compare every accepted production scene quad with ShuttleSet's static quad."""
    truth = np.asarray(ground_truth_corners_refpx, dtype=float)
    if truth.shape != (4, 2) or not np.isfinite(truth).all():
        raise ValueError("ground-truth corners must have finite shape (4, 2)")
    rows = []
    for raw_row in homography_rows.to_dict("records"):
        corners = _scene_ref_corners(raw_row, resolution)
        errors = np.linalg.norm(corners - truth, axis=1)
        rows.append(
            {
                "start_frame": int(raw_row["start_frame"]),
                "end_frame": int(raw_row["end_frame"]),
                "corners": 4,
                "corner_errors_px": [float(error) for error in errors],
                "mean_error_px": float(np.mean(errors)),
                "median_error_px": float(np.median(errors)),
                "max_error_px": float(np.max(errors)),
            }
        )
    return rows


def score_contact_coordinates(
    ground_truth: pd.DataFrame,
    shuttle: ShuttleFeatureInputs,
    player_court_positions: np.ndarray,
    static_court_info: dict[str, object],
) -> dict[str, object]:
    """Score shuttle and both player positions at authoritative GT contact frames."""
    required = {
        "frame_num",
        "player_side",
        "hit_x",
        "hit_y",
        "player_location_x",
        "player_location_y",
        "opponent_location_x",
        "opponent_location_y",
    }
    missing = required.difference(ground_truth.columns)
    if missing:
        raise ValueError(f"contact ground truth is missing columns: {sorted(missing)}")
    shuttle_positions = np.asarray(shuttle.court_positions, dtype=float)
    players = np.asarray(player_court_positions, dtype=float)
    if shuttle_positions.ndim != 2 or shuttle_positions.shape[1] != 2:
        raise ValueError("shuttle positions must have shape (frames, 2)")
    if players.shape != (len(shuttle_positions), 2, 2):
        raise ValueError("player positions must have shape (frames, 2, 2)")
    frames = ground_truth["frame_num"].to_numpy(dtype=int)
    if len(frames) and (
        frames.min() < 0 or frames.max() >= len(shuttle_positions)
    ):
        raise ValueError("ground-truth contact frame is outside the frame population")

    gt_shuttle = _project_gt_columns(ground_truth, "hit", static_court_info)
    gt_striker = _project_gt_columns(
        ground_truth, "player_location", static_court_info
    )
    gt_opponent = _project_gt_columns(
        ground_truth, "opponent_location", static_court_info
    )
    resolved_slots = [_optional_side_slot(value) for value in ground_truth["player_side"]]
    attribution_available = np.array(
        [slot is not None for slot in resolved_slots], dtype=bool
    )
    striker_slots = np.array(
        [0 if slot is None else slot for slot in resolved_slots], dtype=int
    )
    opponent_slots = 1 - striker_slots
    predicted_shuttle = shuttle_positions[frames]
    predicted_striker = players[frames, striker_slots]
    predicted_opponent = players[frames, opponent_slots]
    predicted_striker[~attribution_available] = np.nan
    predicted_opponent[~attribution_available] = np.nan
    gt_striker[~attribution_available] = np.nan
    gt_opponent[~attribution_available] = np.nan

    pairs = {
        "shuttle": (predicted_shuttle, gt_shuttle),
        "striker": (predicted_striker, gt_striker),
        "opponent": (predicted_opponent, gt_opponent),
    }
    summaries = {
        name: coordinate_error_summary(prediction, truth)
        for name, (prediction, truth) in pairs.items()
    }
    rows = []
    for index, frame in enumerate(frames):
        row: dict[str, object] = {
            "frame": int(frame),
            "striker_slot": (
                int(striker_slots[index]) if attribution_available[index] else None
            ),
            "attribution_available": bool(attribution_available[index]),
        }
        for name, (prediction, truth) in pairs.items():
            row[f"{name}_error"] = _paired_error(prediction[index], truth[index])
        rows.append(row)
    return {
        "units": "normalized doubles-court Euclidean distance",
        "matching": "exact authoritative GT contact frame",
        "unusable_attribution": int((~attribution_available).sum()),
        "summary": summaries,
        "rows": rows,
    }


def interpolate_internal_gaps(
    values: np.ndarray, segments: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill only gaps bounded by observations inside one scene segment."""
    result = np.asarray(values, dtype=float).copy()
    if result.ndim != 3 or result.shape[2] < 1:
        raise ValueError("values must have shape (frames, slots, channels)")
    provenance = np.zeros(result.shape[:2], dtype=np.int8)
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


def rally_timestamps(start_frame: int, end_frame: int, fps: float) -> dict[str, object]:
    """Return one half-open rally frame range and its exact second range."""
    _validate_frame_range(start_frame, end_frame)
    _validate_fps(fps)
    return {
        "frame_range": [start_frame, end_frame],
        "second_range": [start_frame / fps, end_frame / fps],
        "fps": fps,
    }


def rally_duration_base30(
    start_frame: int,
    final_contact_frame: int,
    fps: float,
    *,
    end_offset_base30: int,
) -> float:
    """Apply the caller-selected issue #22 offset and return base-30 frames."""
    _validate_fps(fps)
    if start_frame < 0 or final_contact_frame < start_frame:
        raise ValueError("rally contact range is invalid")
    if (
        isinstance(end_offset_base30, bool)
        or not isinstance(end_offset_base30, int)
        or end_offset_base30 < 0
    ):
        raise ValueError("end_offset_base30 must be a non-negative integer")
    offset = (
        0
        if end_offset_base30 == 0
        else int(ScalingKind.FRAME_COUNT.scale(end_offset_base30, fps))
    )
    return (final_contact_frame + offset - start_frame) * BASE_FPS / fps


def serve_speed_proxy(
    shuttle: ShuttleFeatureInputs,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> float | None:
    """Calculate displacement/time after the caller selects the serve endpoint."""
    positions = np.asarray(shuttle.court_positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("shuttle positions must have shape (frames, 2)")
    _validate_fps(fps)
    if not 0 <= start_frame < end_frame < len(positions):
        raise ValueError("serve frame range is invalid")
    endpoints = positions[[start_frame, end_frame]]
    if not np.isfinite(endpoints).all():
        return None
    displacement = float(np.linalg.norm(endpoints[1] - endpoints[0]))
    return displacement / ((end_frame - start_frame) / fps)


def least_squares_trend(values: np.ndarray) -> float | None:
    """Return the raw least-squares slope over finite observations."""
    observations = np.asarray(values, dtype=float)
    valid = np.isfinite(observations)
    if valid.sum() < 2:
        return None
    x = np.flatnonzero(valid).astype(float)
    y = observations[valid]
    x -= x.mean()
    denominator = float(np.dot(x, x))
    return None if denominator == 0.0 else float(np.dot(x, y - y.mean()) / denominator)


def tanh_degradation(slope: float, *, temperature: float) -> float:
    """Normalise a trend after the caller supplies issue #22's unresolved temperature."""
    if not math.isfinite(slope):
        raise ValueError("slope must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    return math.tanh(slope / temperature)


def evaluate_rally_features(
    record: Mapping[str, object],
    posture: np.ndarray,
    court_positions: np.ndarray,
    posture_interpolation: np.ndarray,
    position_interpolation: np.ndarray,
    fps: float,
    *,
    end_offset_base30: int | None = None,
) -> dict[str, object]:
    """Evaluate the defined feature parts for one validated rally record."""
    rally = _object(record.get("rally"), "rally")
    contacts = _object(record.get("contacts"), "contacts")
    outcomes = _object(record.get("outcomes"), "outcomes")
    start = _integer(rally.get("start_frame"), "start_frame")
    end = _integer(rally.get("end_frame"), "end_frame")
    _validate_frame_range(start, end)
    if end > len(posture) or end > len(court_positions):
        raise ValueError("rally range exceeds feature arrays")
    if posture_interpolation.shape != court_positions.shape[:2]:
        raise ValueError("posture_interpolation must have shape (frames, 2)")
    if position_interpolation.shape != court_positions.shape[:2]:
        raise ValueError("position_interpolation must have shape (frames, 2)")

    accepted = _list(contacts.get("accepted"), "accepted contacts")
    contact_frames = [
        _integer(_object(row, "accepted contact").get("contact_frame"), "contact_frame")
        for row in accepted
    ]
    if contact_frames != sorted(set(contact_frames)):
        raise ValueError("accepted contacts must be unique and ordered")
    if any(not start <= frame < end for frame in contact_frames):
        raise ValueError("accepted contact lies outside its rally")
    stroke_count = _integer(contacts.get("stroke_count"), "stroke_count")
    if stroke_count != len(contact_frames):
        raise ValueError("stroke_count differs from accepted contacts")

    server = outcomes.get("server_prediction")
    server_slot = {"Top": 0, "Bot": 1}.get(server) if isinstance(server, str) else None
    striker_slots = (
        None
        if server_slot is None
        else [(server_slot + index) % 2 for index in range(len(contact_frames))]
    )
    recovery = (
        []
        if striker_slots is None
        else recovery_at_opponent_contacts(
            court_positions,
            contact_frames,
            striker_slots,
            fps,
            frame_range=(start, end),
        )
    )
    movement = movement_inefficiency(court_positions, contact_frames)
    movement_rows = [
        {
            "start_contact_frame": contact_frames[index],
            "end_contact_frame": contact_frames[index + 1],
            "top": _finite_or_none(values[0]),
            "bottom": _finite_or_none(values[1]),
        }
        for index, values in enumerate(movement)
    ]
    posture_rows = []
    for slot, name in enumerate(("top", "bottom")):
        values = posture[start:end, slot]
        posture_rows.append(
            {
                "slot": name,
                "valid_frames": int(np.isfinite(values).sum()),
                "mad": median_absolute_deviation(values),
            }
        )
    duration = (
        None
        if end_offset_base30 is None or not contact_frames
        else rally_duration_base30(
            start,
            contact_frames[-1],
            fps,
            end_offset_base30=end_offset_base30,
        )
    )
    return {
        "rally_id": _integer(rally.get("rally_id"), "rally_id"),
        "timestamps": rally_timestamps(start, end, fps),
        "shots_per_rally": stroke_count,
        "rally_duration_base30": duration,
        "posture": posture_rows,
        "recovery": {
            "attribution_available": striker_slots is not None,
            "population": len(contact_frames),
            "observations": recovery,
            "median_top": _median_recovery(recovery, 0),
            "median_bottom": _median_recovery(recovery, 1),
        },
        "movement_inefficiency": movement_rows,
        "interpolation": {
            "posture_linear_top_frames": int(
                np.count_nonzero(
                    posture_interpolation[start:end, 0] == InterpolationType.LINEAR
                )
            ),
            "posture_linear_bottom_frames": int(
                np.count_nonzero(
                    posture_interpolation[start:end, 1] == InterpolationType.LINEAR
                )
            ),
            "position_linear_top_frames": int(
                np.count_nonzero(
                    position_interpolation[start:end, 0] == InterpolationType.LINEAR
                )
            ),
            "position_linear_bottom_frames": int(
                np.count_nonzero(
                    position_interpolation[start:end, 1] == InterpolationType.LINEAR
                )
            ),
            "backward_extrapolated_frames": 0,
        },
    }


def feature_population(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Count exact eligible feature observations in saved per-rally rows."""
    posture_total = posture_eligible = 0
    recovery_total = recovery_eligible = 0
    movement_total = movement_eligible = 0
    duration_eligible = 0
    for row in rows:
        duration_eligible += int(row.get("rally_duration_base30") is not None)
        for posture in _list(row.get("posture"), "posture rows"):
            posture_total += 1
            posture_eligible += int(_object(posture, "posture row").get("mad") is not None)
        recovery = _object(row.get("recovery"), "recovery")
        recovery_total += _integer(
            recovery.get("population"), "recovery population"
        )
        for observation in _list(recovery.get("observations"), "recovery observations"):
            recovery_eligible += int(
                _object(observation, "recovery observation").get("mean_distance")
                is not None
            )
        for interval in _list(row.get("movement_inefficiency"), "movement rows"):
            movement = _object(interval, "movement row")
            for name in ("top", "bottom"):
                movement_total += 1
                movement_eligible += int(movement.get(name) is not None)
    return {
        "rallies": len(rows),
        "duration_eligible": duration_eligible,
        "posture_total": posture_total,
        "posture_eligible": posture_eligible,
        "recovery_total": recovery_total,
        "recovery_eligible": recovery_eligible,
        "movement_total": movement_total,
        "movement_eligible": movement_eligible,
    }


def _validate_frame_range(start_frame: int, end_frame: int) -> None:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("frame range must be non-negative and half-open")


def _validate_fps(fps: float) -> None:
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")


def _median_recovery(
    rows: Sequence[Mapping[str, int | float | None]], slot: int
) -> float | None:
    values = [
        float(row["mean_distance"])
        for row in rows
        if row["measured_slot"] == slot and row["mean_distance"] is not None
    ]
    return None if not values else float(np.median(values))


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _scene_ref_corners(
    row: Mapping[str, object], resolution: tuple[float, float]
) -> np.ndarray:
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


def _project_gt_columns(
    rows: pd.DataFrame, prefix: str, court_info: dict[str, object]
) -> np.ndarray:
    pixels = rows[[f"{prefix}_x", f"{prefix}_y"]].to_numpy(dtype=float)
    result = np.full_like(pixels, np.nan, dtype=float)
    valid = np.isfinite(pixels).all(axis=1)
    if valid.any():
        result[valid] = project_pixels_to_court(
            pixels[valid].T,
            tuple(map(float, HOMOGRAPHY_RESOLUTION)),
            court_info,
        ).T
    return result


def _side_slot(value: object) -> int:
    if value == "Top":
        return 0
    if value in ("Bot", "Bottom"):
        return 1
    raise ValueError(f"unsupported player side: {value!r}")


def _optional_side_slot(value: object) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return _side_slot(value)


def _paired_error(predicted: np.ndarray, ground_truth: np.ndarray) -> float | None:
    if not np.isfinite(predicted).all() or not np.isfinite(ground_truth).all():
        return None
    return float(np.linalg.norm(predicted - ground_truth))


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
