"""CSV segment validation and pure ordered frame-plan construction."""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from pathlib import Path


LOGGER = logging.getLogger(__name__)
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")


class SpanState(StrEnum):
    """The context state shown in the core HUD."""

    LEAD_IN = "LEAD-IN"
    TARGET = "TARGET"
    LEAD_OUT = "LEAD-OUT"


@dataclass(frozen=True)
class Segment:
    """One inclusive target span in source-frame coordinates."""

    start_frame: int
    end_frame: int
    label: str | None = None
    csv_row: int | None = None


@dataclass(frozen=True)
class PlannedFrame:
    """One real source frame and its HUD state."""

    source_idx: int
    state: SpanState
    segment_index: int
    segment_label: str | None
    show_segment_label: bool


@dataclass(frozen=True)
class SegmentPlan:
    """Expanded, clipped source frames for one segment."""

    segment: Segment
    requested_first: int
    requested_last: int
    effective_first: int
    effective_last: int
    frames: tuple[PlannedFrame, ...]

    @property
    def source_indices(self) -> tuple[int, ...]:
        """Return this segment's source indices in output order."""
        return tuple(frame.source_idx for frame in self.frames)


@dataclass(frozen=True)
class SpacerPlan:
    """A run of synthetic black frames between two segments."""

    count: int


PlanPart = SegmentPlan | SpacerPlan


@dataclass(frozen=True)
class TimelinePlan:
    """Pure ordered plan of segment frames and inter-segment spacers."""

    nb_frames: int
    fps: Fraction
    parts: tuple[PlanPart, ...]

    @property
    def frames(self) -> tuple[PlannedFrame | None, ...]:
        """Flatten real frames and spacers, using ``None`` for spacers."""
        flattened: list[PlannedFrame | None] = []
        for part in self.parts:
            if isinstance(part, SpacerPlan):
                flattened.extend([None] * part.count)
            else:
                flattened.extend(part.frames)
        return tuple(flattened)

    @property
    def ordered_source_indices(self) -> tuple[int | None, ...]:
        """Return source indices in output order, with ``None`` for spacers."""
        return tuple(frame.source_idx if frame is not None else None for frame in self.frames)

    @property
    def distinct_source_indices(self) -> frozenset[int]:
        """Return source indices needed by the plan."""
        return frozenset(index for index in self.ordered_source_indices if index is not None)

    @property
    def output_frame_count(self) -> int:
        """Return the number of real and synthetic output frames."""
        return len(self.frames)


def _frame_count(seconds: float | Fraction, fps: Fraction) -> int:
    if isinstance(seconds, Fraction):
        exact_seconds = seconds
    else:
        exact_seconds = Fraction(str(seconds))
    count = int(exact_seconds * fps)
    if count < 0:
        raise ValueError(f"duration must be non-negative, got {seconds}")
    return count


def _parse_frame_bound(value: object, field_name: str, csv_row: int) -> int:
    if value is None or not str(value).strip():
        raise ValueError(f"CSV row {csv_row}: blank {field_name} bound")
    text = str(value).strip()
    if not _INTEGER_RE.fullmatch(text):
        raise ValueError(f"CSV row {csv_row}: {field_name} is not a base-10 integer: {value!r}")
    return int(text, 10)


def read_segments(
    csv_path: Path,
    nb_frames: int,
    start_col: str = "start_frame",
    end_col: str = "end_frame",
    label_col: str = "label",
) -> tuple[Segment, ...]:
    """Read and validate inclusive segments from a UTF-8-with-BOM CSV.

    :param csv_path: Segment CSV path.
    :param nb_frames: Validated source frame count.
    :param start_col: Header containing inclusive start indices.
    :param end_col: Header containing inclusive end indices.
    :param label_col: Optional header containing labels.
    :return: Segments in CSV row order.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"segments CSV is not a regular file: {csv_path}")
    if not start_col or not end_col or not label_col:
        raise ValueError("segment column names must be non-empty")

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError("segments CSV is missing a header")
            if len(set(fieldnames)) != len(fieldnames):
                raise ValueError("segments CSV has duplicate headers")
            if start_col not in fieldnames:
                raise ValueError(f"segments CSV is missing start column {start_col!r}")
            if end_col not in fieldnames:
                raise ValueError(f"segments CSV is missing end column {end_col!r}")

            segments: list[Segment] = []
            for row in reader:
                csv_row = reader.line_num
                start = _parse_frame_bound(row.get(start_col), start_col, csv_row)
                end = _parse_frame_bound(row.get(end_col), end_col, csv_row)
                if end < start:
                    raise ValueError(f"CSV row {csv_row}: end frame {end} is before start frame {start}")
                if start < 0 or end >= nb_frames:
                    raise ValueError(
                        f"CSV row {csv_row}: span [{start}, {end}] is outside [0, {nb_frames})"
                    )
                raw_label = row.get(label_col) if label_col in fieldnames else None
                label = raw_label.strip() if raw_label is not None and raw_label.strip() else None
                segments.append(Segment(start, end, label, csv_row))
    except csv.Error as exc:
        raise ValueError(f"could not parse segments CSV {csv_path}: {exc}") from exc

    if not segments:
        raise ValueError(f"segments CSV has a header but no rows: {csv_path}")
    return tuple(segments)


def build_timeline(
    segments: Iterable[Segment],
    nb_frames: int,
    fps: Fraction,
    lead_in: float | Fraction = 2.5,
    lead_out: float | Fraction = 2.5,
    spacer: float | Fraction = 1.0,
) -> TimelinePlan:
    """Build a pure ordered plan from validated segments and timing settings.

    Lead context clips to the source file edges. Target spans remain unchanged
    and have already been range-validated by ``read_segments`` or the caller.
    """
    if nb_frames <= 0:
        raise ValueError(f"nb_frames must be positive, got {nb_frames}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    segment_entries = tuple(segments)
    if not segment_entries:
        raise ValueError("at least one segment is required")

    lead_in_frames = _frame_count(lead_in, fps)
    lead_out_frames = _frame_count(lead_out, fps)
    spacer_frames = _frame_count(spacer, fps)
    parts: list[PlanPart] = []
    for segment_index, segment in enumerate(segment_entries):
        if segment.start_frame < 0 or segment.end_frame >= nb_frames or segment.end_frame < segment.start_frame:
            raise ValueError(f"invalid segment span: {segment}")
        requested_first = segment.start_frame - lead_in_frames
        requested_last = segment.end_frame + lead_out_frames
        effective_first = max(0, requested_first)
        effective_last = min(nb_frames - 1, requested_last)
        if requested_first != effective_first or requested_last != effective_last:
            row_name = f"CSV row {segment.csv_row}" if segment.csv_row is not None else "segment"
            LOGGER.warning(
                "%s lead context clipped: requested [%d, %d], effective [%d, %d]",
                row_name,
                requested_first,
                requested_last,
                effective_first,
                effective_last,
            )

        segment_frames: list[PlannedFrame] = []
        target_frame_count = segment.end_frame - segment.start_frame + 1
        first_target_label = int(fps)
        for source_idx in range(effective_first, effective_last + 1):
            if source_idx < segment.start_frame:
                state = SpanState.LEAD_IN
                show_segment_label = segment.label is not None
            elif source_idx <= segment.end_frame:
                state = SpanState.TARGET
                show_segment_label = (
                    segment.label is not None
                    and source_idx - segment.start_frame < min(first_target_label, target_frame_count)
                )
            else:
                state = SpanState.LEAD_OUT
                show_segment_label = False
            segment_frames.append(
                PlannedFrame(
                    source_idx,
                    state,
                    segment_index,
                    segment.label,
                    show_segment_label,
                )
            )
        parts.append(
            SegmentPlan(
                segment,
                requested_first,
                requested_last,
                effective_first,
                effective_last,
                tuple(segment_frames),
            )
        )
        if segment_index < len(segment_entries) - 1:
            parts.append(SpacerPlan(spacer_frames))
    return TimelinePlan(nb_frames, fps, tuple(parts))
