"""Dependency-light frame selection shared by multiscale builders and runners."""

from __future__ import annotations

from itertools import pairwise

from .multiscale_schema import Segment


def required_storyboard_frames(
    segments: tuple[Segment, ...],
    start_frame: int,
    end_frame: int,
    target_start_frame: int,
    target_end_frame: int,
) -> set[int]:
    """Return every cut side and target edge that a storyboard must retain."""
    required = {start_frame, end_frame - 1, target_start_frame, target_end_frame - 1}
    if target_start_frame > start_frame:
        required.add(target_start_frame - 1)
    if target_end_frame < end_frame:
        required.add(target_end_frame)
    for segment in segments:
        required.add(segment.source_start_frame)
        required.add(segment.source_end_frame - 1)
    return required


def storyboard_source_frames(
    segments: tuple[Segment, ...],
    start_frame: int,
    end_frame: int,
    target_start_frame: int,
    target_end_frame: int,
    max_frames: int,
) -> tuple[int, ...] | None:
    """Keep every required frame, then fill the largest source-time gaps."""
    required = required_storyboard_frames(
        segments,
        start_frame,
        end_frame,
        target_start_frame,
        target_end_frame,
    )
    if len(required) > max_frames or end_frame - start_frame < max_frames:
        return None
    selected = set(required)
    while len(selected) < max_frames:
        ordered = sorted(selected)
        gaps = []
        for left, right in pairwise(ordered):
            available = right - left - 1
            if available:
                gaps.append((available, left, right))
        if not gaps:
            return None
        _, left, right = max(gaps, key=lambda gap: (gap[0], -gap[1]))
        selected.add((left + right) // 2)
    return tuple(sorted(selected))


def segment_for_frame(
    segments: tuple[Segment, ...],
    source_frame: int,
) -> Segment:
    """Find the cut-bounded segment containing one source-global frame."""
    for segment in segments:
        if segment.source_start_frame <= source_frame < segment.source_end_frame:
            return segment
    raise ValueError(f"source frame {source_frame} has no segment")


__all__ = [
    "required_storyboard_frames",
    "segment_for_frame",
    "storyboard_source_frames",
]
