"""Strict decoder for persisted dataset-builder court provenance."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields
import math
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from annotator.court_evidence import CourtSceneRecord
    from courtkeynet.court_corners import ConsensusRepair, FallbackDiagnostics


T = TypeVar("T")


def load_court_provenance(
    scene_payload: object,
    consensus_payload: object,
    *,
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> tuple[tuple[CourtSceneRecord, ...], ConsensusRepair | None]:
    """Decode and cross-check scene records and their consensus arrays."""
    if not isinstance(scene_payload, list):
        raise ValueError("court scene_records must be a list")
    records = tuple(
        _scene_record(item, f"scene_records[{index}]")
        for index, item in enumerate(scene_payload)
    )
    _validate_scene_records(records, raw_cuts, video_id)
    consensus = _consensus(consensus_payload, records)
    return records, consensus


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
        sampled_frame_indices=_integer_tuple(
            record["sampled_frame_indices"],
            f"{name}.sampled_frame_indices",
        ),
        raw_corners_px=_optional_array(record["raw_corners_px"], f"{name}.raw_corners_px", (4, 2)),
        raw_source=_optional_string(record["raw_source"], f"{name}.raw_source"),
        raw_peaks=_optional_array(record["raw_peaks"], f"{name}.raw_peaks", (4,)),
        raw_corner_source=_optional_strings(
            record["raw_corner_source"],
            f"{name}.raw_corner_source",
            4,
        ),
        fallback_diagnostics=_fallback_diagnostics(
            record["fallback_diagnostics"],
            f"{name}.fallback_diagnostics",
        ),
        exactly_two_count=_integer(record["exactly_two_count"], f"{name}.exactly_two_count"),
        exactly_two_fraction=_finite(record["exactly_two_fraction"], f"{name}.exactly_two_fraction"),
        scene_valid=_boolean(record["scene_valid"], f"{name}.scene_valid"),
        consensus_distance_px=_optional(record["consensus_distance_px"], _finite, f"{name}.consensus_distance_px"),
        consensus_flag=_optional(record["consensus_flag"], _boolean, f"{name}.consensus_flag"),
        active_corners_native_px=_optional_array(
            record["active_corners_native_px"],
            f"{name}.active_corners_native_px",
            (4, 2),
        ),
    )


def _fallback_diagnostics(payload: object, name: str) -> FallbackDiagnostics | None:
    from courtkeynet.court_corners import FallbackDiagnostics

    if payload is None:
        return None
    record = _object(payload, name)
    expected = {field.name for field in fields(FallbackDiagnostics)}
    if set(record) != expected:
        raise ValueError(f"{name} fields differ from FallbackDiagnostics")
    return FallbackDiagnostics(
        reproj_line_px=_finite(record["reproj_line_px"], f"{name}.reproj_line_px"),
        reproj_anchor_px=_finite(record["reproj_anchor_px"], f"{name}.reproj_anchor_px"),
        gate_line_frac=_finite(record["gate_line_frac"], f"{name}.gate_line_frac"),
        gate_anchor_frac=_finite(record["gate_anchor_frac"], f"{name}.gate_anchor_frac"),
        n_lines_used=_integer(record["n_lines_used"], f"{name}.n_lines_used"),
        n_correspondences=_integer(record["n_correspondences"], f"{name}.n_correspondences"),
        max_sagitta_px=_finite(record["max_sagitta_px"], f"{name}.max_sagitta_px"),
    )


def _consensus(
    payload: object,
    records: Sequence[CourtSceneRecord],
) -> ConsensusRepair | None:
    from courtkeynet.court_corners import ConsensusRepair

    if payload is None:
        if any(
            record.consensus_distance_px is not None or record.consensus_flag is not None
            for record in records
        ):
            raise ValueError("court scene consensus fields require consensus evidence")
        return None
    record = _object(payload, "consensus")
    expected = {field.name for field in fields(ConsensusRepair)}
    if set(record) != expected:
        raise ValueError("consensus fields differ from ConsensusRepair")
    accepted = [item for item in records if item.consensus_distance_px is not None]
    count = len(accepted)
    if count == 0:
        raise ValueError("court consensus requires at least one accepted scene")
    if any(item.scene_valid != (item.consensus_distance_px is not None) for item in records):
        raise ValueError("court scene validity differs from consensus membership")
    result = ConsensusRepair(
        consensus_quad=_array(record["consensus_quad"], "consensus.consensus_quad", (4, 2)),
        distances_px=_array(record["distances_px"], "consensus.distances_px", (count,)),
        flagged=_bool_array(record["flagged"], "consensus.flagged", (count,)),
        repaired_quads=_array(record["repaired_quads"], "consensus.repaired_quads", (count, 4, 2)),
    )
    for index, scene in enumerate(accepted):
        if scene.consensus_flag is None or scene.active_corners_native_px is None:
            raise ValueError("accepted court consensus records require flags and active corners")
        if scene.consensus_flag != bool(result.flagged[index]):
            raise ValueError("court scene consensus flag differs from consensus array")
        if scene.consensus_distance_px != float(result.distances_px[index]):
            raise ValueError("court scene consensus distance differs from consensus array")
        if not np.array_equal(scene.active_corners_native_px, result.repaired_quads[index]):
            raise ValueError("court scene active corners differ from repaired consensus quad")
    return result


def _validate_scene_records(
    records: Sequence[CourtSceneRecord],
    raw_cuts: Sequence[tuple[int, int]],
    video_id: str,
) -> None:
    if len(records) != len(raw_cuts):
        raise ValueError("court scene record count differs from raw cuts")
    for index, (record, interval) in enumerate(zip(records, raw_cuts)):
        if record.video_id != video_id or not isinstance(record.video_id, str):
            raise ValueError("court scene video_id differs from the exact requested string")
        if record.scene_index != index or (record.start_frame, record.end_frame) != interval:
            raise ValueError("court scene ordering or interval differs from raw cuts")
        if any(frame < record.start_frame or frame >= record.end_frame for frame in record.sampled_frame_indices):
            raise ValueError("court sampled frame lies outside its scene interval")
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


def _integer_tuple(payload: object, name: str) -> tuple[int, ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return tuple(_integer(item, f"{name}[]") for item in payload)


def _optional_strings(payload: object, name: str, length: int) -> tuple[str, ...] | None:
    if payload is None:
        return None
    if not isinstance(payload, list) or len(payload) != length:
        raise ValueError(f"{name} must contain {length} strings")
    return tuple(_string(item, f"{name}[]") for item in payload)


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


def _bool_array(payload: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(payload)
    if values.shape != shape or values.dtype != np.dtype(bool):
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    return values
