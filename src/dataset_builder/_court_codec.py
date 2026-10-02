"""Strict decoder for persisted dataset-builder court provenance."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from annotator.court_evidence import CourtSceneRecord, SceneStatus


T = TypeVar("T")


def load_court_provenance(
    scene_payload: object,
    *,
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> tuple[CourtSceneRecord, ...]:
    """Decode and cross-check the per-scene court records."""
    if not isinstance(scene_payload, list):
        raise ValueError("court scene_records must be a list")
    records = tuple(
        _scene_record(item, f"scene_records[{index}]")
        for index, item in enumerate(scene_payload)
    )
    _validate_scene_records(records, raw_cuts, video_id)
    return records


def _scene_record(payload: object, name: str) -> CourtSceneRecord:
    from annotator.court_evidence import CourtSceneRecord

    record = _object(payload, name)
    expected = {field.name for field in fields(CourtSceneRecord)}
    if set(record) != expected:
        raise ValueError(f"{name} fields differ from CourtSceneRecord")
    return CourtSceneRecord(
        video_id=_video_id(record["video_id"], f"{name}.video_id"),
        case_id=_string(record["case_id"], f"{name}.case_id"),
        parent=_string(record["parent"], f"{name}.parent"),
        scene_index=_integer(record["scene_index"], f"{name}.scene_index"),
        start_frame=_integer(record["start_frame"], f"{name}.start_frame"),
        end_frame=_integer(record["end_frame"], f"{name}.end_frame"),
        status=_status(record["status"], f"{name}.status"),
        analysed_frame=_optional(record["analysed_frame"], _integer, f"{name}.analysed_frame"),
        corners_native_px=_optional_array(record["corners_native_px"], f"{name}.corners_native_px", (4, 2)),
        no_court_reason=_optional_string(record["no_court_reason"], f"{name}.no_court_reason"),
        reused_from=_optional_string(record["reused_from"], f"{name}.reused_from"),
        error=_optional_string(record["error"], f"{name}.error"),
        exactly_two_count=_integer(record["exactly_two_count"], f"{name}.exactly_two_count"),
        exactly_two_fraction=_finite(record["exactly_two_fraction"], f"{name}.exactly_two_fraction"),
        scene_valid=_boolean(record["scene_valid"], f"{name}.scene_valid"),
    )


def _status(payload: object, name: str) -> SceneStatus:
    from annotator.court_evidence import SceneStatus

    try:
        return SceneStatus(_string(payload, name))
    except ValueError as error:
        raise ValueError(f"{name} {payload!r} is not a court scene status") from error


def _validate_scene_records(
    records: Sequence[CourtSceneRecord],
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> None:
    from annotator.court_evidence import SceneStatus

    if len(records) != len(raw_cuts):
        raise ValueError("court scene record count differs from raw cuts")
    for index, (record, interval) in enumerate(zip(records, raw_cuts)):
        if record.video_id != video_id or not isinstance(record.video_id, str):
            raise ValueError("court scene video_id differs from the exact requested string")
        if record.scene_index != index or (record.start_frame, record.end_frame) != interval:
            raise ValueError("court scene ordering or interval differs from raw cuts")
        if record.analysed_frame is not None and not record.start_frame <= record.analysed_frame < record.end_frame:
            raise ValueError("court analysed frame lies outside its scene interval")
        if (record.corners_native_px is not None) != (record.status is SceneStatus.COURT):
            raise ValueError("court scene corners must be present exactly when its status is court")
        if record.scene_valid and record.status is not SceneStatus.COURT:
            raise ValueError("court scene without a court cannot be valid")
        if (record.error is not None) != (record.status is SceneStatus.DETECTION_FAILED):
            raise ValueError("court scene error must be present exactly when detection failed")
        duration = record.end_frame - record.start_frame
        if not 0 <= record.exactly_two_count <= duration:
            raise ValueError("court exactly-two count lies outside its scene")
        expected_fraction = record.exactly_two_count / duration
        if not math.isclose(record.exactly_two_fraction, expected_fraction):
            raise ValueError("court exactly-two fraction differs from its count")


def _object(payload: object, name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be an object with string keys")
    return payload


def _video_id(payload: object, name: str) -> int | str:
    if isinstance(payload, bool) or not isinstance(payload, (int, str)):
        raise ValueError(f"{name} must be an integer or string")
    return payload


def _string(payload: object, name: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be a non-empty string")
    return payload


def _integer(payload: object, name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise ValueError(f"{name} must be an integer")
    return payload


def _boolean(payload: object, name: str) -> bool:
    if not isinstance(payload, bool):
        raise ValueError(f"{name} must be a boolean")
    return payload


def _finite(payload: object, name: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(payload)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _optional(payload: object, loader: Callable[[object, str], T], name: str) -> T | None:
    return None if payload is None else loader(payload, name)


def _optional_string(payload: object, name: str) -> str | None:
    return _optional(payload, _string, name)


def _array(payload: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        values = np.asarray(payload, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers") from error
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return values


def _optional_array(payload: object, name: str, shape: tuple[int, ...]) -> np.ndarray | None:
    return None if payload is None else _array(payload, name, shape)
