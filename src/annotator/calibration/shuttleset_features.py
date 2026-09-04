"""Benchmark-only prototypes for the trial features defined in issue #22.

Issue #18 moved the formulas issue #104 kept into ``dataset_builder.features``;
issue #138 moved ``recovery_at_opponent_contacts`` and ``movement_inefficiency``
there too, once human ShuttleSet contacts made them reliable enough to export.
This module re-exports all of them and keeps the prototypes that never shipped
to production.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from annotator.fps_constants import BASE_FPS, ScalingKind
from annotator.point_winner import project_pixels_to_court
from annotator.shuttle_track import validate_shuttle_track
from dataset_builder.features import (
    ANKLE_INDICES,
    EYE_INDICES,
    HALF_CENTRES,
    HIP_INDICES,
    RECOVERY_HALF_WINDOW_BASE30,
    InterpolationType,
    PlayerFeatureInputs,
    derive_player_feature_inputs,
    interpolate_internal_gaps,
    median_absolute_deviation,
    movement_inefficiency,
    posture_signal,
    project_positions_by_scene,
    rally_timestamps,
    recovery_at_opponent_contacts,
    scene_ref_corners,
    select_sticky_keypoints,
    validate_fps,
    validate_frame_range,
)
from shared.court import HOMOGRAPHY_RESOLUTION

# Re-exported for annotator.calibration.shuttleset22_features,
# annotator.calibration.shuttleset_benchmark, and tests/test_shuttleset_features.py.
# recovery_at_opponent_contacts and movement_inefficiency are also used directly
# below, by evaluate_rally_features; the rest are unused by this module's own
# prototypes.
__all__ = (
    "ANKLE_INDICES",
    "EYE_INDICES",
    "HALF_CENTRES",
    "HIP_INDICES",
    "RECOVERY_HALF_WINDOW_BASE30",
    "PlayerFeatureInputs",
    "derive_player_feature_inputs",
    "interpolate_internal_gaps",
    "movement_inefficiency",
    "posture_signal",
    "recovery_at_opponent_contacts",
    "select_sticky_keypoints",
)


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
        corners = scene_ref_corners(raw_row, resolution)
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


def rally_duration_base30(
    start_frame: int,
    final_contact_frame: int,
    fps: float,
    *,
    end_offset_base30: int,
) -> float:
    """Apply the caller-selected issue #22 offset and return base-30 frames."""
    validate_fps(fps)
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
    validate_fps(fps)
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
    validate_frame_range(start, end)
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
