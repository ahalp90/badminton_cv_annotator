"""Read saved court-scene results and check their video and frame ranges."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from annotator.courts.prepare_courts import CourtSceneRecord, SceneStatus


def load_court_records(
    scene_payload: object,
    *,
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> tuple[CourtSceneRecord, ...]:
    """Decode and cross-check the per-scene court records."""
    if not isinstance(scene_payload, list):
        raise TypeError("court scene_records must be a list")
    records = tuple(
        _scene_record(item, f"scene_records[{index}]")
        for index, item in enumerate(scene_payload)
    )
    _validate_scene_records(records, raw_cuts, video_id)
    return records


def _scene_record(payload: object, name: str) -> CourtSceneRecord:
    from annotator.courts.prepare_courts import CourtSceneRecord

    record = _object(payload, name)
    expected = {field.name for field in fields(CourtSceneRecord)}
    if set(record) != expected:
        raise ValueError(f"{name} fields differ from CourtSceneRecord")
    return CourtSceneRecord(
        video_id=_video_id(record["video_id"], f"{name}.video_id"),
        scene_index=_integer(record["scene_index"], f"{name}.scene_index"),
        start_frame=_integer(record["start_frame"], f"{name}.start_frame"),
        end_frame=_integer(record["end_frame"], f"{name}.end_frame"),
        status=_status(record["status"], f"{name}.status"),
        analysed_frame=_optional(record["analysed_frame"], _integer, f"{name}.analysed_frame"),
        corners_native_px=_optional_array(record["corners_native_px"], f"{name}.corners_native_px", (4, 2)),
        no_court_reason=_optional_string(record["no_court_reason"], f"{name}.no_court_reason"),
        reused_from=_optional_string(record["reused_from"], f"{name}.reused_from"),
        error=_optional_string(record["error"], f"{name}.error"),
    )


def _status(payload: object, name: str) -> SceneStatus:
    from annotator.courts.prepare_courts import SceneStatus

    try:
        return SceneStatus(_string(payload, name))
    except ValueError as error:
        raise ValueError(f"{name} {payload!r} is not a court scene status") from error


def _validate_scene_records(
    records: Sequence[CourtSceneRecord],
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> None:
    from annotator.courts.prepare_courts import SceneStatus

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
        if (record.error is not None) != (record.status is SceneStatus.DETECTION_FAILED):
            raise ValueError("court scene error must be present exactly when detection failed")


def _object(payload: object, name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be an object with string keys")
    return payload


def _video_id(payload: object, name: str) -> int | str:
    if isinstance(payload, bool) or not isinstance(payload, (int, str)):
        raise TypeError(f"{name} must be an integer or string")
    return payload


def _string(payload: object, name: str) -> str:
    if not isinstance(payload, str) or not payload:
        raise ValueError(f"{name} must be a non-empty string")
    return payload


def _integer(payload: object, name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise TypeError(f"{name} must be an integer")
    return payload


def _optional[T](payload: object, loader: Callable[[object, str], T], name: str) -> T | None:
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
