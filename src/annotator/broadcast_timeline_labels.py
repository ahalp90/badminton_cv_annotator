"""Truth contract for manually reviewed broadcast-timeline intervals."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
import gzip
import io
import math
from pathlib import Path
from typing import TextIO


LABEL_CSV_HEADER = (
    "video_id",
    "fps",
    "frame_count",
    "start_frame",
    "end_frame",
    "truth",
    "note",
)


class SceneTruth(StrEnum):
    """Allowed human scene classes."""

    LIVE = "live"
    LIVE_NON_STANDARD = "live-non-standard"
    REPLAY = "replay"
    CUTAWAY = "cutaway"
    OTHER = "other"


@dataclass(frozen=True)
class VideoMetadata:
    """Source identity repeated on each label row."""

    video_id: str
    fps: float
    frame_count: int

    def __post_init__(self) -> None:
        if not self.video_id:
            raise ValueError("video_id must not be empty")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be finite and positive, got {self.fps!r}")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int) or self.frame_count <= 0:
            raise ValueError(f"frame_count must be a positive integer, got {self.frame_count!r}")


@dataclass(frozen=True)
class LabelInterval:
    """One zero-based, half-open human scene interval."""

    video_id: str
    fps: float
    frame_count: int
    start_frame: int
    end_frame: int
    truth: SceneTruth
    note: str = ""

    @property
    def metadata(self) -> VideoMetadata:
        return VideoMetadata(self.video_id, self.fps, self.frame_count)


def make_interval(
    metadata: VideoMetadata,
    start_frame: int,
    end_frame: int,
    truth: SceneTruth,
    note: str = "",
) -> LabelInterval:
    """Build one interval using a shared source identity."""
    return LabelInterval(
        video_id=metadata.video_id,
        fps=metadata.fps,
        frame_count=metadata.frame_count,
        start_frame=start_frame,
        end_frame=end_frame,
        truth=truth,
        note=note,
    )


def _validate_interval(interval: LabelInterval, row_number: int) -> None:
    interval.metadata
    for field_name, value in (("start_frame", interval.start_frame), ("end_frame", interval.end_frame)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"row {row_number}: {field_name} must be an integer, got {value!r}")
    if not 0 <= interval.start_frame < interval.end_frame <= interval.frame_count:
        raise ValueError(
            f"row {row_number}: interval [{interval.start_frame}, {interval.end_frame}) "
            f"is outside [0, {interval.frame_count})"
        )
    if not isinstance(interval.truth, SceneTruth):
        raise ValueError(f"row {row_number}: invalid truth {interval.truth!r}")
    if not isinstance(interval.note, str):
        raise ValueError(f"row {row_number}: note must be text")


def validate_intervals(
    intervals: Sequence[LabelInterval],
    *,
    expected_metadata: VideoMetadata | None = None,
) -> VideoMetadata:
    """Validate row values, shared metadata, ordering, and non-overlap."""
    if not intervals:
        if expected_metadata is None:
            raise ValueError("empty intervals require expected_metadata")
        return expected_metadata

    metadata = expected_metadata or intervals[0].metadata
    previous: LabelInterval | None = None
    for row_number, interval in enumerate(intervals, start=2):
        _validate_interval(interval, row_number)
        if interval.metadata != metadata:
            raise ValueError(
                f"row {row_number}: metadata {interval.metadata!r} does not match {metadata!r}"
            )
        if previous is not None:
            if interval.start_frame < previous.start_frame:
                raise ValueError(f"row {row_number}: intervals are not ordered by start_frame")
            if interval.start_frame < previous.end_frame:
                raise ValueError(
                    f"row {row_number}: interval starts at {interval.start_frame} before "
                    f"the previous interval ends at {previous.end_frame}"
                )
        previous = interval
    return metadata


def validate_partition(
    intervals: Sequence[LabelInterval],
    *,
    covered_start: int = 0,
    covered_end: int | None = None,
    expected_metadata: VideoMetadata | None = None,
) -> VideoMetadata:
    """Require intervals to cover one declared range without gaps."""
    metadata = validate_intervals(intervals, expected_metadata=expected_metadata)
    end = metadata.frame_count if covered_end is None else covered_end
    if not 0 <= covered_start < end <= metadata.frame_count:
        raise ValueError(
            f"covered range [{covered_start}, {end}) is outside [0, {metadata.frame_count})"
        )
    if not intervals:
        raise ValueError(f"no intervals cover [{covered_start}, {end})")
    if intervals[0].start_frame != covered_start:
        raise ValueError(
            f"partition starts at {intervals[0].start_frame}, expected {covered_start}"
        )
    for previous, current in zip(intervals, intervals[1:]):
        if current.start_frame != previous.end_frame:
            raise ValueError(f"gap [{previous.end_frame}, {current.start_frame})")
    if intervals[-1].end_frame != end:
        raise ValueError(f"partition ends at {intervals[-1].end_frame}, expected {end}")
    return metadata


def interval_index_at(intervals: Sequence[LabelInterval], frame: int) -> int | None:
    """Return the interval containing ``frame``."""
    for index, interval in enumerate(intervals):
        if interval.start_frame <= frame < interval.end_frame:
            return index
        if interval.start_frame > frame:
            break
    return None


def replace_interval(
    intervals: Sequence[LabelInterval],
    index: int,
    *,
    truth: SceneTruth | None = None,
    note: str | None = None,
) -> list[LabelInterval]:
    """Replace the editable fields of one selected interval."""
    if not 0 <= index < len(intervals):
        raise IndexError(f"interval index {index} is outside 0..{len(intervals) - 1}")
    updated = list(intervals)
    updated[index] = replace(
        updated[index],
        truth=updated[index].truth if truth is None else truth,
        note=updated[index].note if note is None else note,
    )
    validate_intervals(updated)
    return updated


def _parse_int(value: str | None, field_name: str, row_number: int) -> int:
    try:
        return int(value or "")
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be an integer, got {value!r}") from exc


def _parse_float(value: str | None, field_name: str, row_number: int) -> float:
    try:
        return float(value or "")
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be numeric, got {value!r}") from exc


@contextmanager
def _read_text(path: Path) -> Iterator[TextIO]:
    if path.name.endswith(".csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            yield handle
        return
    with path.open(encoding="utf-8", newline="") as handle:
        yield handle


def read_label_csv(path: Path) -> list[LabelInterval]:
    """Read and validate a plain or gzip-compressed label CSV."""
    path = Path(path)
    with _read_text(path) as handle:
        reader = csv.DictReader(handle)
        found_header = tuple(reader.fieldnames or ())
        if found_header != LABEL_CSV_HEADER:
            raise ValueError(f"{path} has columns {found_header}, expected {LABEL_CSV_HEADER}")
        intervals: list[LabelInterval] = []
        for row_number, row in enumerate(reader, start=2):
            truth_value = str(row.get("truth", ""))
            try:
                truth = SceneTruth(truth_value)
            except ValueError as exc:
                raise ValueError(f"row {row_number}: invalid truth {truth_value!r}") from exc
            intervals.append(LabelInterval(
                video_id=str(row.get("video_id", "")),
                fps=_parse_float(row.get("fps"), "fps", row_number),
                frame_count=_parse_int(row.get("frame_count"), "frame_count", row_number),
                start_frame=_parse_int(row.get("start_frame"), "start_frame", row_number),
                end_frame=_parse_int(row.get("end_frame"), "end_frame", row_number),
                truth=truth,
                note=str(row.get("note", "")),
            ))
    if intervals:
        validate_intervals(intervals)
    return intervals


@contextmanager
def _write_text(path: Path, *, compressed: bool) -> Iterator[TextIO]:
    if compressed:
        with path.open("wb") as raw_handle:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
                with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                    yield text_handle
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        yield handle


def write_label_csv(
    path: Path,
    intervals: Sequence[LabelInterval],
    metadata: VideoMetadata,
) -> None:
    """Atomically write intervals and verify their round trip."""
    path = Path(path)
    if not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
        raise ValueError(f"label path must end in .csv or .csv.gz: {path}")
    validate_intervals(intervals, expected_metadata=metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith(".csv.gz"):
        temporary_path = path.with_name(f".{path.name[:-7]}.tmp.csv.gz")
    else:
        temporary_path = path.with_name(f".{path.stem}.tmp.csv")
    try:
        with _write_text(temporary_path, compressed=path.name.endswith(".csv.gz")) as handle:
            writer = csv.DictWriter(handle, fieldnames=LABEL_CSV_HEADER)
            writer.writeheader()
            for interval in intervals:
                writer.writerow({
                    "video_id": interval.video_id,
                    "fps": repr(interval.fps),
                    "frame_count": interval.frame_count,
                    "start_frame": interval.start_frame,
                    "end_frame": interval.end_frame,
                    "truth": interval.truth.value,
                    "note": interval.note,
                })
        reloaded = read_label_csv(temporary_path)
        if reloaded != list(intervals):
            raise RuntimeError(f"label CSV round trip changed values: {path}")
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
