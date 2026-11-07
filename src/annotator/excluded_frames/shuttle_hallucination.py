"""Flag repeated shuttle positions that suggest fabricated detections.

Repeated moving patterns are marked fabricated. Repeated flat patterns are
suspect: a real shuttle can rest at one pixel. Nearby frames and other
occurrences of the same positions get a degraded grade.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np

from annotator.types import ShuttleGrade

log = logging.getLogger(__name__)

DETECTOR_VERSION = 4
DEFAULT_WINDOW = 16
DEFAULT_HALO_FRAMES = 3

CODE_NAMES = {
    ShuttleGrade.NO_FLAG: "no flag",
    ShuttleGrade.FABRICATED: "fabricated (proof)",
    ShuttleGrade.SUSPECT_FLAT: "suspect flat",
    ShuttleGrade.NEAR_REPEATED_PATTERN: "near repeated pattern",
    ShuttleGrade.MATCHES_REPEATED_POSITION: "matches repeated position",
}

_MIN_CANDIDATE_EPISODES = 2
_MIN_ACCEPTED_EPISODES = 30
_MIN_ACCEPTED_MARGIN = 10.0
_CACHE: dict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]] = {}


def grade_track(
    track: np.ndarray,
    window: int = DEFAULT_WINDOW,
    *,
    halo_frames: int = DEFAULT_HALO_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-frame shuttle grades and diagnostics, reusing cached results.

    :param track: Full video track; columns 0 and 1 are exact x and y values.
    :param window: Number of consecutive positions in a repeated pattern.
    :param halo_frames: Extra frames marked degraded around repeated runs.
    :return: (codes, info): per-frame grades and a report of pattern counts,
        thresholds and any reason detection was unavailable. Both are copies
        of the cached result.
    """
    _validate_track(track, window)
    _validate_halo_frames(halo_frames)
    window = int(window)
    halo_frames = int(halo_frames)
    key = _cache_key(track, window, halo_frames)
    cached = _CACHE.get(key)
    if cached is not None:
        cached_codes, cached_info = cached
        return cached_codes.copy(), deepcopy(cached_info)
    codes, info = build_mask(track, window, halo_frames=halo_frames)
    _CACHE[key] = (codes.copy(), deepcopy(info))
    return codes, info


def build_mask(
    track: np.ndarray,
    window: int = DEFAULT_WINDOW,
    *,
    halo_frames: int = DEFAULT_HALO_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign shuttle-quality grades from unusually frequent position patterns.

    Output: one uint8 code per original video frame. Moving repeats are
    FABRICATED, flat repeats SUSPECT_FLAT. Adjacent frames are NEAR_REPEATED_PATTERN;
    exact position matches elsewhere are MATCHES_REPEATED_POSITION.
    (0, 0) positions remain NO_FLAG.
    Return (codes, info). ``info`` reports pattern counts, thresholds and any
    reason detection was unavailable. The dataset builder saves it with the grades.
    """
    _validate_track(track, window)
    _validate_halo_frames(halo_frames)
    window = int(window)
    halo_frames = int(halo_frames)
    if len(track) < window:
        reason = f"track length {len(track)} is shorter than window {window}"
        log.warning("inpaint fabrication guard unavailable: %s", reason)
        return (
            np.zeros(len(track), dtype=np.uint8),
            _empty_info(window, halo_frames, reason),
        )

    varying, flat, info = candidate_attractors(
        track,
        window,
        halo_frames,
    )
    codes = np.zeros(len(track), dtype=np.uint8)
    if not varying and not flat:
        info["counts_per_code"] = code_counts(codes)
        return codes, info

    presence = _validate_presence(varying, flat, window, len(track))
    info["presence_validation"] = presence
    proven = _cover(len(track), varying, window)
    suspect = _cover(len(track), flat, window)

    core = proven | suspect
    near_repeated_pattern = np.zeros(len(track), dtype=bool)
    edges = np.diff(np.concatenate(([False], core, [False])).astype(np.int8))
    for start in np.flatnonzero(edges == 1):
        near_repeated_pattern[max(0, int(start) - halo_frames):int(start)] = True
    for stop in np.flatnonzero(edges == -1):
        stop = int(stop)
        near_repeated_pattern[stop:min(len(track), stop + halo_frames)] = True

    positions: set[tuple[Any, Any]] = set()
    for key in (*varying, *flat):
        for point in np.frombuffer(key, dtype=track.dtype).reshape(window, 2):
            positions.add((point[0], point[1]))
    matches_repeated_position = np.zeros(len(track), dtype=bool)
    for pos_x, pos_y in positions:
        matches_repeated_position |= (track[:, 0] == pos_x) & (track[:, 1] == pos_y)

    # Exact matches take precedence over the halo; repeated runs keep their own grades.
    codes[near_repeated_pattern & ~core] = ShuttleGrade.NEAR_REPEATED_PATTERN
    codes[matches_repeated_position & ~core] = ShuttleGrade.MATCHES_REPEATED_POSITION
    codes[suspect] = ShuttleGrade.SUSPECT_FLAT
    codes[proven] = ShuttleGrade.FABRICATED
    codes[(track[:, 0] == 0) & (track[:, 1] == 0)] = ShuttleGrade.NO_FLAG
    info["counts_per_code"] = code_counts(codes)
    return codes, info


# Find repeated position patterns

def candidate_attractors(
    track: np.ndarray,
    window: int,
    halo_frames: int,
) -> tuple[dict[bytes, list[int]], dict[bytes, list[int]], dict[str, Any]]:
    """Select unusually frequent position patterns, split into moving and flat.

    Require enough separate occurrences and a clear gap from less frequent
    patterns. Return empty groups with a reason if that evidence is too weak.
    """
    starts_by_pattern, episodes = pattern_episodes(track, window)
    threshold, margin = adaptive_threshold(episodes)
    candidate_counts = {count for count in episodes.values() if count >= _MIN_CANDIDATE_EPISODES}
    reason = None
    if len(candidate_counts) < 2:
        reason = f"fewer than 2 distinct candidate counts ({sorted(candidate_counts, reverse=True)})"
    elif threshold < _MIN_ACCEPTED_EPISODES:
        reason = f"derived threshold {threshold} is below {_MIN_ACCEPTED_EPISODES} episodes"
    elif margin < _MIN_ACCEPTED_MARGIN:
        reason = f"derived margin {margin:.6g} is below {_MIN_ACCEPTED_MARGIN:g}"
    if reason is not None:
        log.warning("inpaint fabrication guard unavailable: %s", reason)
        return {}, {}, _empty_info(window, halo_frames, reason, threshold, margin)

    varying: dict[bytes, list[int]] = {}
    flat: dict[bytes, list[int]] = {}
    for key, count in episodes.items():
        if count < threshold:
            continue
        points = np.frombuffer(key, dtype=track.dtype).reshape(window, 2)
        moves = np.ptp(points[:, 0]) > 0 or np.ptp(points[:, 1]) > 0
        (varying if moves else flat)[key] = starts_by_pattern[key]
    info = {
        "detector_version": DETECTOR_VERSION,
        "window": window,
        "halo_frames": halo_frames,
        "threshold": threshold,
        "margin": margin,
        "n_varying": len(varying),
        "n_flat": len(flat),
        "presence_validation": "pending",
        "unavailable_reason": None,
        "counts_per_code": {grade: 0 for grade in ShuttleGrade},
    }
    return varying, flat, info


def pattern_episodes(track: np.ndarray, window: int = DEFAULT_WINDOW) -> tuple[dict[bytes, list[int]], dict[bytes, int]]:
    """Find exactly repeated position windows and count separate occurrences.

    Starts no more than twice the window length apart count as one episode.
    Skip windows containing a (0, 0) position. Return starts and episode counts,
    both keyed by the window's exact coordinate bytes.
    """
    _validate_track(track, window)
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

    episodes: dict[bytes, int] = {}
    merge_gap = 2 * window
    for key, starts in starts_by_pattern.items():
        episode_count = 1
        previous_start = starts[0]
        for start in starts[1:]:
            if start - previous_start > merge_gap:
                episode_count += 1
            previous_start = start
        episodes[key] = episode_count
    return dict(starts_by_pattern), episodes


def adaptive_threshold(episodes: dict[bytes, int], floor: int = _MIN_CANDIDATE_EPISODES) -> tuple[int, float]:
    """Find the biggest ratio between neighbouring distinct episode counts.

    Ignore counts below floor. Return the higher count at that gap and the
    ratio. With fewer than two distinct counts, use ratio 1.
    """
    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
        raise ValueError("floor must be a positive integer")
    counts = sorted({count for count in episodes.values() if count >= floor}, reverse=True)
    if len(counts) < 2:
        return (counts[0] if counts else 1), 1.0
    ratios = [(counts[index] / counts[index + 1], counts[index]) for index in range(len(counts) - 1)]
    margin, threshold = max(ratios)
    return threshold, margin


def _validate_presence(
    varying: dict[bytes, list[int]], flat: dict[bytes, list[int]],
    window: int, n_frames: int,
) -> dict[str, int | bool]:
    """Require each selected pattern to occur in both halves of the video."""
    midpoint = n_frames // 2
    halves = ((0, midpoint), (midpoint, n_frames))
    for kind, groups in (("varying", varying), ("flat", flat)):
        for starts in groups.values():
            present = []
            for half_start, half_end in halves:
                present.append(any(
                    start < half_end and start + window > half_start
                    for start in starts
                ))
            if not all(present):
                raise ValueError(
                    f"{kind} attractor is absent from a validation half: "
                    f"first={present[0]} second={present[1]}"
                )
    return {
        "passed": True,
        "varying_first_half": sum(
            any(start < midpoint and start + window > 0 for start in starts)
            for starts in varying.values()
        ),
        "varying_second_half": sum(
            any(start < n_frames and start + window > midpoint for start in starts)
            for starts in varying.values()
        ),
        "flat_first_half": sum(
            any(start < midpoint and start + window > 0 for start in starts)
            for starts in flat.values()
        ),
        "flat_second_half": sum(
            any(start < n_frames and start + window > midpoint for start in starts)
            for starts in flat.values()
        ),
    }


def _cover(track_length: int, groups: dict[bytes, list[int]], window: int) -> np.ndarray:
    """Mark every frame covered by the selected repeated windows."""
    covered = np.zeros(track_length, dtype=bool)
    for starts in groups.values():
        for start in starts:
            covered[start:start + window] = True
    return covered


def code_counts(codes: np.ndarray) -> dict[int, int]:
    """Count frames in each shuttle grade."""
    if codes.ndim != 1 or codes.dtype != np.uint8:
        raise ValueError("codes must be a one-dimensional uint8 array")
    return {grade: int(np.count_nonzero(codes == grade)) for grade in ShuttleGrade}


# Input checks

def _validate_track(track: np.ndarray, window: int) -> None:
    if not isinstance(track, np.ndarray):
        raise TypeError("track must be a numpy array")
    if track.ndim != 2 or track.shape[1] < 2:
        raise ValueError("track must have shape (n_frames, at least 2)")
    if (np.issubdtype(track.dtype, np.bool_) or
            not np.issubdtype(track.dtype, np.number) or
            np.issubdtype(track.dtype, np.complexfloating)):
        raise ValueError("track must have a real numeric dtype")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer")


def _validate_halo_frames(halo_frames: int) -> None:
    if (
        isinstance(halo_frames, bool)
        or not isinstance(halo_frames, (int, np.integer))
        or halo_frames < 0
    ):
        raise ValueError("halo_frames must be a non-negative integer")


# Detection report when no patterns could be accepted

def _empty_info(
    window: int,
    halo_frames: int,
    reason: str,
    threshold: int | None = None,
    margin: float | None = None,
) -> dict[str, Any]:
    return {
        "detector_version": DETECTOR_VERSION,
        "window": window,
        "halo_frames": halo_frames,
        "threshold": threshold,
        "margin": margin,
        "n_varying": 0,
        "n_flat": 0,
        "presence_validation": f"unavailable: {reason}",
        "unavailable_reason": reason,
        "counts_per_code": {grade: 0 for grade in ShuttleGrade},
    }


# Reuse grades and their report for the same track and settings

def _cache_key(track: np.ndarray, window: int, halo_frames: int) -> tuple[Any, ...]:
    canonical_bytes = np.ascontiguousarray(track).tobytes(order="C")
    digest = hashlib.sha256(canonical_bytes).hexdigest()
    return (
        DETECTOR_VERSION,
        int(window),
        int(halo_frames),
        track.dtype.str,
        track.shape,
        digest,
    )


def clear_cache() -> None:
    """Clear the in-memory detector cache, primarily for isolated tests."""
    _CACHE.clear()
