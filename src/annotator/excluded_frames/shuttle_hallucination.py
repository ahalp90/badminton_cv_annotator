"""Flag unusually frequent repeats of shuttle positions.

Distinguish moving patterns, stationary positions, surrounding frames and
matching positions elsewhere. Stationary repeats are less conclusive than
moving ones: a real shuttle can rest at one pixel.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from enum import IntEnum
from typing import Any, Self

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_WINDOW = 16
DEFAULT_HALO_FRAMES = 3

MIN_OCCURRENCES_TO_COMPARE = 2
MIN_OCCURRENCES_TO_FLAG = 30
MIN_OCCURRENCE_COUNT_RATIO = 10.0


class ShuttleGrade(IntEnum):
    """Quality of a shuttle position, stored as one integer per video frame."""

    UNFLAGGED = 0, 'unflagged'
    REPEATED_MOVING_PATTERN = 1, 'repeated moving pattern'
    REPEATED_STATIONARY_POSITION = 2, 'repeated stationary position'
    REPEATED_PATTERN_HALO = 3, 'halo around repeated pattern'
    MATCHES_ANY_PATTERN_POS = 4, 'matches any position in a repeated pattern'

    label: str

    def __new__(cls, code: int, label: str) -> Self:
        grade = int.__new__(cls, code)
        grade._value_ = code
        grade.label = label
        return grade

    def __str__(self) -> str:
        return self.label

    @classmethod
    def rejectable(cls) -> frozenset[ShuttleGrade]:
        return frozenset(grade for grade in cls if grade is not cls.UNFLAGGED)


def grade_track(
    track: np.ndarray,
    window: int = DEFAULT_WINDOW,
    *,
    halo_frames: int = DEFAULT_HALO_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign shuttle-quality grades from unusually frequent position patterns.

    Return one uint8 grade per frame and a detection report for the dataset builder.
    The report gives pattern counts, the occurrence threshold and any reason
    detection was unavailable. (0, 0) positions remain UNFLAGGED.

    :param track: Full video track; the first two columns are exact x and y values.
    :param window: Positive number of consecutive positions in a repeated pattern.
    :param halo_frames: Non-negative number of extra frames flagged around repeated runs.
    """
    if len(track) < window:
        reason = f"track length {len(track)} is shorter than window {window}"
        log.warning("repeated-pattern detection unavailable: %s", reason)
        return (
            np.zeros(len(track), dtype=np.uint8),
            unavailable_detection_report(window, halo_frames, reason),
        )

    moving_patterns, stationary_patterns, info = select_repeated_patterns(track, window, halo_frames)
    codes = np.zeros(len(track), dtype=np.uint8)
    if not moving_patterns and not stationary_patterns:
        return codes, info

    require_patterns_in_both_halves(moving_patterns, stationary_patterns, window, len(track))
    repeated_moving_frames = mark_pattern_frames(len(track), moving_patterns, window)
    repeated_stationary_frames = mark_pattern_frames(len(track), stationary_patterns, window)

    repeated_frames = repeated_moving_frames | repeated_stationary_frames
    repeated_pattern_halo = np.zeros(len(track), dtype=bool)
    edges = np.diff(np.concatenate(([False], repeated_frames, [False])).astype(np.int8))
    for start in np.flatnonzero(edges == 1):
        repeated_pattern_halo[max(0, int(start) - halo_frames):int(start)] = True
    for stop in np.flatnonzero(edges == -1):
        stop = int(stop)
        repeated_pattern_halo[stop:min(len(track), stop + halo_frames)] = True

    positions: set[tuple[Any, Any]] = set()
    for key in (*moving_patterns, *stationary_patterns):
        for point in np.frombuffer(key, dtype=track.dtype).reshape(window, 2):
            positions.add((point[0], point[1]))
    matches_any_pattern_pos = np.zeros(len(track), dtype=bool)
    for pos_x, pos_y in positions:
        matches_any_pattern_pos |= (track[:, 0] == pos_x) & (track[:, 1] == pos_y)

    # Exact matches take precedence over the halo; repeated runs keep their own grades.
    codes[repeated_pattern_halo & ~repeated_frames] = ShuttleGrade.REPEATED_PATTERN_HALO
    codes[matches_any_pattern_pos & ~repeated_frames] = ShuttleGrade.MATCHES_ANY_PATTERN_POS
    codes[repeated_stationary_frames] = ShuttleGrade.REPEATED_STATIONARY_POSITION
    codes[repeated_moving_frames] = ShuttleGrade.REPEATED_MOVING_PATTERN
    codes[(track[:, 0] == 0) & (track[:, 1] == 0)] = ShuttleGrade.UNFLAGGED
    return codes, info


def mark_pattern_frames(track_length: int, groups: dict[bytes, list[int]], window: int) -> np.ndarray:
    """Mark every frame covered by the selected repeated windows."""
    covered = np.zeros(track_length, dtype=bool)
    for starts in groups.values():
        for start in starts:
            covered[start:start + window] = True
    return covered


# Find repeated position patterns

def select_repeated_patterns(
    track: np.ndarray,
    window: int,
    halo_frames: int,
) -> tuple[dict[bytes, list[int]], dict[bytes, list[int]], dict[str, Any]]:
    """Select unusually frequent position patterns, split into moving and stationary.

    Require enough separate occurrences and a clear gap from less frequent
    patterns. Return empty groups with a reason if that evidence is too weak.
    """
    starts_by_pattern, occurrences = find_pattern_occurrences(track, window)
    counts = sorted({count for count in occurrences.values() if count >= MIN_OCCURRENCES_TO_COMPARE}, reverse=True)
    if len(counts) < 2:
        threshold, count_ratio = (counts[0] if counts else 1), 1.0
    else:
        # Separate the unusually frequent patterns at the largest jump in counts.
        count_ratio, threshold = max(
            (counts[index] / counts[index + 1], counts[index]) for index in range(len(counts) - 1)
        )
    reason = None
    if len(counts) < 2:
        reason = f"fewer than 2 distinct candidate counts ({counts})"
    elif threshold < MIN_OCCURRENCES_TO_FLAG:
        reason = f"derived threshold {threshold} is below {MIN_OCCURRENCES_TO_FLAG} occurrences"
    elif count_ratio < MIN_OCCURRENCE_COUNT_RATIO:
        reason = f"occurrence count ratio {count_ratio:.6g} is below {MIN_OCCURRENCE_COUNT_RATIO:g}"
    if reason is not None:
        log.warning("repeated-pattern detection unavailable: %s", reason)
        return {}, {}, unavailable_detection_report(window, halo_frames, reason, threshold, count_ratio)

    moving_patterns: dict[bytes, list[int]] = {}
    stationary_patterns: dict[bytes, list[int]] = {}
    for key, count in occurrences.items():
        if count < threshold:
            continue
        points = np.frombuffer(key, dtype=track.dtype).reshape(window, 2)
        moves = np.ptp(points[:, 0]) > 0 or np.ptp(points[:, 1]) > 0
        (moving_patterns if moves else stationary_patterns)[key] = starts_by_pattern[key]
    info = {
        "window": window,
        "halo_frames": halo_frames,
        "threshold": threshold,
        "count_ratio": count_ratio,
        "moving_pattern_count": len(moving_patterns),
        "stationary_pattern_count": len(stationary_patterns),
        "unavailable_reason": None,
    }
    return moving_patterns, stationary_patterns, info


def find_pattern_occurrences(track: np.ndarray, window: int) -> tuple[dict[bytes, list[int]], dict[bytes, int]]:
    """Find exactly repeated position windows and count separate occurrences.

    Starts no more than twice the window length apart count as one occurrence.
    Skip windows containing a (0, 0) position. Return all start frames and the
    separate-occurrence counts, both keyed by the window's exact coordinate bytes.
    """
    n_frames = len(track)
    blank = (track[:, 0] == 0) & (track[:, 1] == 0)
    blank_cumulative = np.concatenate(([0], np.cumsum(blank, dtype=np.int64)))
    coordinates = np.ascontiguousarray(track[:, :2])

    starts_by_pattern: defaultdict[bytes, list[int]] = defaultdict(list)
    for start in range(n_frames - window + 1):
        if blank_cumulative[start + window] - blank_cumulative[start] > 0:
            continue
        key = coordinates[start:start + window].tobytes()
        starts_by_pattern[key].append(start)

    occurrences: dict[bytes, int] = {}
    merge_gap = 2 * window
    for key, starts in starts_by_pattern.items():
        occurrence_count = 1
        previous_start = starts[0]
        for start in starts[1:]:
            if start - previous_start > merge_gap:
                occurrence_count += 1
            previous_start = start
        occurrences[key] = occurrence_count
    return dict(starts_by_pattern), occurrences


def require_patterns_in_both_halves(
    moving_patterns: dict[bytes, list[int]], stationary_patterns: dict[bytes, list[int]],
    window: int, n_frames: int,
) -> None:
    """Require each selected pattern to occur in both halves of the video."""
    midpoint = n_frames // 2
    halves = ((0, midpoint), (midpoint, n_frames))
    for kind, patterns in (("moving", moving_patterns), ("stationary", stationary_patterns)):
        for starts in patterns.values():
            for half_start, half_end in halves:
                if not any(start < half_end and start + window > half_start for start in starts):
                    raise ValueError(f"{kind} pattern must occur in both halves of the video")


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def code_counts(codes: np.ndarray) -> dict[int, int]:
    """Count frames in each shuttle grade."""
    return {int(grade): int(np.count_nonzero(codes == grade)) for grade in ShuttleGrade}


# Detection report when no patterns could be accepted

def unavailable_detection_report(
    window: int,
    halo_frames: int,
    reason: str,
    threshold: int | None = None,
    count_ratio: float | None = None,
) -> dict[str, Any]:
    return {
        "window": window,
        "halo_frames": halo_frames,
        "threshold": threshold,
        "count_ratio": count_ratio,
        "moving_pattern_count": 0,
        "stationary_pattern_count": 0,
        "unavailable_reason": reason,
    }
