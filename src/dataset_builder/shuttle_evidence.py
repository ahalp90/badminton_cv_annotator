"""Validated TrackNet provenance and derived shuttle-guard evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import numpy as np

from annotator.inpaint_guard import code_counts, grade_track
from annotator.shuttle_track import validate_shuttle_track
from dataset_builder.vision import (
    load_json_gz,
    load_npy_xz,
    save_json_gz,
    save_npy_xz,
)


INPAINT_SIDECAR_SCHEMA = "inpaint_fill_mask/1"
SHUTTLE_GUARD_SCHEMA = "shuttle-guard/0.1"
GUARD_CODES_FILENAME = "shuttle_guard_codes.npy.xz"
GUARD_DIAGNOSTICS_FILENAME = "shuttle_guard_diagnostics.json.gz"

_SIDECAR_REQUIRED_FIELDS = {
    "schema",
    "index_space",
    "inpaint_status",
    "n_rows",
    "eval_mode",
    "stride",
    "th_h_px",
    "tracknet_ckpt",
    "inpaintnet_ckpt",
    "input_video",
    "extracted_utc",
    "inpaint_selected",
}
_SIDECAR_PROVENANCE_FIELDS = {"dataset", "video_id", "title", "url", "fps"}


@dataclass(frozen=True)
class ShuttleEvidenceArtifacts:
    """Manifest-owned shuttle outputs and their provenance."""

    tracknet_csv: Path
    inpaint_sidecar: Path
    shuttle_track: Path
    guard_codes: Path
    guard_diagnostics: Path

    def as_mapping(self) -> dict[str, Path]:
        """Return stable manifest names mapped to their paths."""
        return {
            "tracknet_csv": self.tracknet_csv,
            "inpaint_sidecar": self.inpaint_sidecar,
            "shuttle_track": self.shuttle_track,
            "shuttle_guard_codes": self.guard_codes,
            "shuttle_guard_diagnostics": self.guard_diagnostics,
        }


@dataclass(frozen=True)
class ShuttleEvidence:
    """Frame-aligned shuttle values, fill provenance, and guard grades."""

    track: np.ndarray
    inpaint_fill_mask: np.ndarray
    guard_codes: np.ndarray
    guard_diagnostics: Mapping[str, object]
    artifacts: ShuttleEvidenceArtifacts

    def __post_init__(self) -> None:
        frame_count = len(self.track)
        validate_shuttle_track(self.track, frame_count)
        _validate_fill_mask(self.inpaint_fill_mask, frame_count)
        _validate_guard_codes(self.guard_codes, frame_count)
        object.__setattr__(self, "track", np.ascontiguousarray(self.track).copy())
        object.__setattr__(
            self,
            "inpaint_fill_mask",
            np.ascontiguousarray(self.inpaint_fill_mask).copy(),
        )
        object.__setattr__(self, "guard_codes", np.ascontiguousarray(self.guard_codes).copy())
        object.__setattr__(self, "guard_diagnostics", dict(self.guard_diagnostics))


def shuttle_evidence_artifacts(
    output_dir: Path,
    *,
    input_video: Path,
    stride: int,
) -> ShuttleEvidenceArtifacts:
    """Return the expected artifact paths for one shuttle stage."""
    root = Path(output_dir)
    sidecar = root / f"{input_video.stem}_stride{stride}_inpaint_mask.json.gz"
    return ShuttleEvidenceArtifacts(
        tracknet_csv=root / f"{input_video.stem}_ball.csv",
        inpaint_sidecar=sidecar,
        shuttle_track=root / "shuttle_track.npy.xz",
        guard_codes=root / GUARD_CODES_FILENAME,
        guard_diagnostics=root / GUARD_DIAGNOSTICS_FILENAME,
    )


def persist_shuttle_evidence(
    *,
    track: np.ndarray,
    artifacts: ShuttleEvidenceArtifacts,
    input_video: Path,
    input_height: int,
    frame_count: int,
    stride: int,
    tracknet_model: Path,
    inpaint_model: Path | None,
) -> ShuttleEvidence:
    """Validate producer provenance, derive guard grades, and persist them."""
    validate_shuttle_track(track, frame_count)
    fill_mask = load_inpaint_fill_mask(
        artifacts.inpaint_sidecar,
        input_video=input_video,
        input_height=input_height,
        frame_count=frame_count,
        stride=stride,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
    )
    guard_codes, guard_info = grade_track(track)
    guard_payload = _guard_payload(frame_count, guard_codes, guard_info)
    save_npy_xz(artifacts.guard_codes, guard_codes)
    save_json_gz(artifacts.guard_diagnostics, guard_payload)
    return ShuttleEvidence(track, fill_mask, guard_codes, guard_payload, artifacts)


def load_shuttle_evidence(
    *,
    artifacts: ShuttleEvidenceArtifacts,
    input_video: Path,
    input_height: int,
    frame_count: int,
    stride: int,
    tracknet_model: Path,
    inpaint_model: Path | None,
) -> ShuttleEvidence:
    """Load and semantically revalidate persisted shuttle evidence."""
    track = load_npy_xz(artifacts.shuttle_track)
    validate_shuttle_track(track, frame_count)
    fill_mask = load_inpaint_fill_mask(
        artifacts.inpaint_sidecar,
        input_video=input_video,
        input_height=input_height,
        frame_count=frame_count,
        stride=stride,
        tracknet_model=tracknet_model,
        inpaint_model=inpaint_model,
    )
    guard_codes = load_npy_xz(artifacts.guard_codes)
    _validate_guard_codes(guard_codes, frame_count)
    expected_codes, guard_info = grade_track(track)
    if not np.array_equal(guard_codes, expected_codes):
        raise ValueError("persisted shuttle guard codes differ from the final track")
    expected_payload = _guard_payload(frame_count, expected_codes, guard_info)
    guard_payload = load_json_gz(artifacts.guard_diagnostics)
    if guard_payload != expected_payload:
        raise ValueError("persisted shuttle guard diagnostics differ from the final track")
    return ShuttleEvidence(track, fill_mask, guard_codes, guard_payload, artifacts)


def load_inpaint_fill_mask(
    path: Path,
    *,
    input_video: Path,
    input_height: int,
    frame_count: int,
    stride: int,
    tracknet_model: Path,
    inpaint_model: Path | None,
) -> np.ndarray:
    """Load a strict producer sidecar as a frame-aligned boolean mask."""
    payload = load_json_gz(path)
    fields = set(payload)
    missing = _SIDECAR_REQUIRED_FIELDS - fields
    unexpected = fields - _SIDECAR_REQUIRED_FIELDS - _SIDECAR_PROVENANCE_FIELDS
    if missing or unexpected:
        raise ValueError(
            f"inpaint sidecar fields differ: missing={sorted(missing)} "
            f"unexpected={sorted(unexpected)}"
        )
    expected_mode = "weight" if stride == 1 else "nonoverlap"
    expected_status = "disabled" if inpaint_model is None else "applied"
    expected_values = {
        "schema": INPAINT_SIDECAR_SCHEMA,
        "index_space": "frame",
        "inpaint_status": expected_status,
        "n_rows": frame_count,
        "eval_mode": expected_mode,
        "stride": stride,
        "tracknet_ckpt": tracknet_model.name,
        "inpaintnet_ckpt": None if inpaint_model is None else inpaint_model.name,
        "input_video": input_video.name,
    }
    for field, expected in expected_values.items():
        if payload[field] != expected:
            raise ValueError(
                f"inpaint sidecar {field} is {payload[field]!r}, expected {expected!r}"
            )
    threshold = payload["th_h_px"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or float(threshold) != float(input_height) * 0.05
    ):
        raise ValueError("inpaint sidecar th_h_px differs from the TrackNet input height")
    _validate_extracted_utc(payload["extracted_utc"])
    mask = _mask_from_spans(payload["inpaint_selected"], frame_count)
    if expected_status == "disabled" and mask.any():
        raise ValueError("disabled inpaint sidecar must not select frames")
    return mask


def _mask_from_spans(payload: object, frame_count: int) -> np.ndarray:
    if not isinstance(payload, list):
        raise ValueError("inpaint_selected must be a list of half-open spans")
    mask = np.zeros(frame_count, dtype=bool)
    previous_stop = 0
    for span in payload:
        if (
            not isinstance(span, list)
            or len(span) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in span)
        ):
            raise ValueError(f"invalid inpaint_selected span: {span!r}")
        start, stop = span
        if start < previous_stop or start < 0 or stop <= start or stop > frame_count:
            raise ValueError(f"unsorted or out-of-bounds inpaint_selected span: {span!r}")
        mask[start:stop] = True
        previous_stop = stop
    return mask


def _validate_extracted_utc(payload: object) -> None:
    if not isinstance(payload, str) or not payload.endswith("Z"):
        raise ValueError("inpaint sidecar extracted_utc must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(payload.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError("inpaint sidecar extracted_utc must be a UTC timestamp") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("inpaint sidecar extracted_utc must be a UTC timestamp")


def _validate_fill_mask(mask: np.ndarray, frame_count: int) -> None:
    values = np.asarray(mask)
    if values.shape != (frame_count,) or values.dtype != np.dtype(bool):
        raise ValueError(
            f"inpaint fill mask must have shape {(frame_count,)} and boolean dtype"
        )


def _validate_guard_codes(codes: np.ndarray, frame_count: int) -> None:
    values = np.asarray(codes)
    if values.shape != (frame_count,) or values.dtype != np.dtype(np.uint8):
        raise ValueError(
            f"shuttle guard codes must have shape {(frame_count,)} and uint8 dtype"
        )
    if not np.isin(values, (0, 1, 2, 3)).all():
        raise ValueError("shuttle guard codes must be in {0, 1, 2, 3}")


def _guard_payload(
    frame_count: int,
    guard_codes: np.ndarray,
    guard_info: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "schema": SHUTTLE_GUARD_SCHEMA,
        "frame_count": frame_count,
        "counts_per_code": {
            str(code): count for code, count in code_counts(guard_codes).items()
        },
        "detector": _json_ready(guard_info),
    }


def _json_ready(value: object) -> object:
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("shuttle guard diagnostics must contain finite values")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"shuttle guard diagnostics contain unsupported {type(value).__name__}")
