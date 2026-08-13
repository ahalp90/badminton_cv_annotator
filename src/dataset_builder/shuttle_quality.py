"""Stage-level quality summary for production shuttle evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SHUTTLE_QUALITY_SCHEMA = "shuttle-quality/0.1"


@dataclass(frozen=True)
class ShuttleQualitySummary:
    """Compact relationship between fill provenance and guard grades."""

    frame_count: int
    inpaint_filled_frames: int
    inpaint_visible_filled_frames: int
    guard_counts_per_code: tuple[int, int, int, int]
    filled_counts_per_code: tuple[int, int, int, int]
    rejected_grades: tuple[int, ...]
    guard_rejected_frames: int
    filled_guard_rejected_frames: int

    def __post_init__(self) -> None:
        scalar_counts = (
            self.frame_count,
            self.inpaint_filled_frames,
            self.inpaint_visible_filled_frames,
            self.guard_rejected_frames,
            self.filled_guard_rejected_frames,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in scalar_counts):
            raise ValueError("shuttle quality scalar counts must be integers")
        if any(value < 0 for value in scalar_counts):
            raise ValueError("shuttle quality scalar counts must be non-negative")
        _validate_code_counts(self.guard_counts_per_code, "guard_counts_per_code")
        _validate_code_counts(self.filled_counts_per_code, "filled_counts_per_code")
        if sum(self.guard_counts_per_code) != self.frame_count:
            raise ValueError("guard counts must sum to frame_count")
        if sum(self.filled_counts_per_code) != self.inpaint_filled_frames:
            raise ValueError("filled guard counts must sum to inpaint_filled_frames")
        if any(
            filled > guard
            for filled, guard in zip(self.filled_counts_per_code, self.guard_counts_per_code)
        ):
            raise ValueError("filled guard counts cannot exceed total guard counts")
        if self.inpaint_visible_filled_frames > self.inpaint_filled_frames:
            raise ValueError("visible filled frames cannot exceed all filled frames")
        if self.rejected_grades != tuple(sorted(set(self.rejected_grades))):
            raise ValueError("rejected grades must be sorted and unique")
        if any(grade not in {1, 2, 3} for grade in self.rejected_grades):
            raise ValueError("rejected grades must be a subset of {1, 2, 3}")
        expected_rejected = sum(self.guard_counts_per_code[grade] for grade in self.rejected_grades)
        if self.guard_rejected_frames != expected_rejected:
            raise ValueError("guard_rejected_frames differs from rejected grades")
        expected_filled_rejected = sum(
            self.filled_counts_per_code[grade] for grade in self.rejected_grades
        )
        if self.filled_guard_rejected_frames != expected_filled_rejected:
            raise ValueError("filled_guard_rejected_frames differs from rejected grades")

    def to_payload(self) -> dict[str, object]:
        """Return the versioned persisted representation."""
        return {
            "schema": SHUTTLE_QUALITY_SCHEMA,
            "frame_count": self.frame_count,
            "inpaint_filled_frames": self.inpaint_filled_frames,
            "inpaint_visible_filled_frames": self.inpaint_visible_filled_frames,
            "guard_counts_per_code": list(self.guard_counts_per_code),
            "filled_counts_per_code": list(self.filled_counts_per_code),
            "rejected_grades": list(self.rejected_grades),
            "guard_rejected_frames": self.guard_rejected_frames,
            "filled_guard_rejected_frames": self.filled_guard_rejected_frames,
        }

    @classmethod
    def from_payload(cls, payload: object) -> ShuttleQualitySummary:
        """Load an exact versioned persisted representation."""
        if not isinstance(payload, dict):
            raise ValueError("shuttle quality payload must be an object")
        expected_fields = {
            "schema",
            "frame_count",
            "inpaint_filled_frames",
            "inpaint_visible_filled_frames",
            "guard_counts_per_code",
            "filled_counts_per_code",
            "rejected_grades",
            "guard_rejected_frames",
            "filled_guard_rejected_frames",
        }
        if set(payload) != expected_fields:
            raise ValueError("shuttle quality payload fields differ")
        if payload["schema"] != SHUTTLE_QUALITY_SCHEMA:
            raise ValueError("shuttle quality schema differs")
        return cls(
            frame_count=_integer(payload["frame_count"], "frame_count"),
            inpaint_filled_frames=_integer(
                payload["inpaint_filled_frames"],
                "inpaint_filled_frames",
            ),
            inpaint_visible_filled_frames=_integer(
                payload["inpaint_visible_filled_frames"],
                "inpaint_visible_filled_frames",
            ),
            guard_counts_per_code=_code_counts(
                payload["guard_counts_per_code"],
                "guard_counts_per_code",
            ),
            filled_counts_per_code=_code_counts(
                payload["filled_counts_per_code"],
                "filled_counts_per_code",
            ),
            rejected_grades=tuple(
                _integer(value, "rejected grade")
                for value in _list(payload["rejected_grades"], "rejected_grades")
            ),
            guard_rejected_frames=_integer(
                payload["guard_rejected_frames"],
                "guard_rejected_frames",
            ),
            filled_guard_rejected_frames=_integer(
                payload["filled_guard_rejected_frames"],
                "filled_guard_rejected_frames",
            ),
        )


def summarize_shuttle_quality(
    track: np.ndarray,
    inpaint_fill_mask: np.ndarray,
    guard_codes: np.ndarray,
    rejected_grades: frozenset[int],
) -> ShuttleQualitySummary:
    """Summarize source fill provenance without using it as rejection evidence."""
    track_values = np.asarray(track)
    fill_values = np.asarray(inpaint_fill_mask)
    code_values = np.asarray(guard_codes)
    if track_values.ndim != 2 or track_values.shape[1] != 3:
        raise ValueError("shuttle quality track must have shape (frames, 3)")
    frame_count = len(track_values)
    if fill_values.shape != (frame_count,) or fill_values.dtype != np.dtype(bool):
        raise ValueError("inpaint fill mask must be frame-aligned and boolean")
    if code_values.shape != (frame_count,) or code_values.dtype != np.dtype(np.uint8):
        raise ValueError("guard codes must be frame-aligned uint8 values")
    if not np.isin(code_values, (0, 1, 2, 3)).all():
        raise ValueError("guard codes must be in {0, 1, 2, 3}")
    rejected = tuple(sorted(rejected_grades))
    guard_counts = tuple(
        int(np.count_nonzero(code_values == code)) for code in range(4)
    )
    filled_counts = tuple(
        int(np.count_nonzero(fill_values & (code_values == code))) for code in range(4)
    )
    return ShuttleQualitySummary(
        frame_count=frame_count,
        inpaint_filled_frames=int(np.count_nonzero(fill_values)),
        inpaint_visible_filled_frames=int(
            np.count_nonzero(fill_values & (track_values[:, 2] == 1.0))
        ),
        guard_counts_per_code=guard_counts,
        filled_counts_per_code=filled_counts,
        rejected_grades=rejected,
        guard_rejected_frames=sum(guard_counts[grade] for grade in rejected),
        filled_guard_rejected_frames=sum(filled_counts[grade] for grade in rejected),
    )


def _validate_code_counts(values: tuple[int, ...], name: str) -> None:
    if len(values) != 4:
        raise ValueError(f"{name} must contain four code counts")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError(f"{name} must contain non-negative integers")


def _code_counts(payload: object, name: str) -> tuple[int, int, int, int]:
    values = _list(payload, name)
    if len(values) != 4:
        raise ValueError(f"{name} must contain four values")
    return (
        _integer(values[0], name),
        _integer(values[1], name),
        _integer(values[2], name),
        _integer(values[3], name),
    )


def _list(payload: object, name: str) -> list[object]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    return payload


def _integer(payload: object, name: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int):
        raise ValueError(f"{name} must be an integer")
    return payload
