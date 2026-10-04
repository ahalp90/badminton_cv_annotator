"""Contact events, rally contact sequences and the candidate scores behind them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from annotator.fps_constants import ScalingKind
from annotator.outcomes.point_winner import Half


@dataclass(frozen=True)
class ContactEvent:
    """One predicted contact and the court half of its raw striker guess."""

    frame: int
    probability: float
    side: Half | None


@dataclass(frozen=True)
class ContactSequence:
    """One half-open rally interval and the contacts it currently owns."""

    span_id: int
    start_frame: int
    end_frame: int
    events: tuple[ContactEvent, ...]


class CandidateScores(NamedTuple):
    """Every scored contact candidate in one video, in frame order.

    ``kept`` marks the candidates the contact detector selected. Those rows are
    the full-stream contact events, including any outside a rally interval.
    """

    frames: np.ndarray  # (candidate,) int
    interval_ids: np.ndarray  # (candidate,) index into the search intervals
    probabilities: np.ndarray  # (candidate,) float
    kept: np.ndarray  # (candidate,) bool


def scale_frames(frames_at_30_fps: int, fps: float) -> int:
    """Scale a frame count defined at 30 fps with the repository's half-up rule."""
    return int(ScalingKind.FRAME_COUNT.scale(float(frames_at_30_fps), float(fps)))


def events_between(events: Sequence[ContactEvent], start_frame: int, end_frame: int) -> tuple[ContactEvent, ...]:
    """Return the events inside one half-open frame interval, keeping their order."""
    return tuple(event for event in events if start_frame <= event.frame < end_frame)


def initial_sequences(
    spans: Sequence[tuple[int, int]], events: Sequence[ContactEvent],
) -> tuple[ContactSequence, ...]:
    """Number chronological half-open rally spans and give each its contacts.

    Later stages rely on the spans being ordered and disjoint: serve edits may
    move a start back only as far as the previous span's end.

    :param spans: Heuristic ``(start_frame, end_frame)`` rally intervals.
    :param events: Full-stream contact events in frame order.
    :return: One contact sequence per span, numbered by position.
    """
    sequences = []
    previous_end = -1
    for span_id, (start_frame, end_frame) in enumerate(spans):
        if not previous_end <= start_frame < end_frame:
            raise ValueError(f"span {span_id}: rally spans must be ordered, disjoint and non-empty")
        owned = events_between(events, start_frame, end_frame)
        sequences.append(ContactSequence(span_id, start_frame, end_frame, owned))
        previous_end = end_frame
    return tuple(sequences)
