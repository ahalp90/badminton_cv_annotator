"""Synthetic tests for the inpaint fabrication grade detector."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from annotator.inpaint_guard import (
    DEFAULT_HALO_FRAMES,
    DEGRADED,
    FABRICATED,
    NO_FLAG,
    SUSPECT_FLAT,
    clear_cache,
    grade_track,
)


def _track_with_recurrence(
    *, varying_episodes: int = 100, varying_step: int | None = None,
    flat_starts: tuple[int, ...] | None = None, length: int = 14000,
) -> np.ndarray:
    """Build a mostly unique track with separated exact recurrence episodes."""
    track = np.column_stack((np.arange(length, dtype=float), np.arange(length, dtype=float) * 0.37 + 1.0))
    varying = np.column_stack((np.arange(16, dtype=float) + 100.0, np.arange(16, dtype=float) * 2.0 + 200.0))
    if varying_step is None:
        varying_starts = [
            20 + episode * 100 for episode in range(min(varying_episodes, 50))
        ] + [
            7200 + (episode - 50) * 100 for episode in range(50, varying_episodes)
        ]
    else:
        varying_starts = [20 + episode * varying_step for episode in range(varying_episodes)]
    for start in varying_starts:
        track[start:start + 16] = varying
    if flat_starts is None:
        flat_starts = tuple([5500 + episode * 50 for episode in range(25)] +
                            [12500 + episode * 50 for episode in range(25)])
    for start in flat_starts:
        track[start:start + 16] = (900.0, 901.0)
    for start in (6800, 6840, 6880):
        track[start:start + 16] = (1200.0, 1300.0)
    return np.column_stack((track, np.ones(length)))


def test_grade_track_marks_varying_flat_and_degraded_frames() -> None:
    track = _track_with_recurrence()
    codes, info = grade_track(track)

    assert codes.dtype == np.uint8
    assert len(codes) == len(track)
    assert info["threshold"] == 50
    assert info["margin"] == pytest.approx(50 / 3)
    assert info["n_varying"] == 1
    assert info["n_flat"] == 1
    assert np.count_nonzero(codes == FABRICATED) > 0
    assert np.count_nonzero(codes == SUSPECT_FLAT) > 0
    assert np.count_nonzero(codes == DEGRADED) > 0


def test_grade_track_uses_three_absolute_halo_frames() -> None:
    track = _track_with_recurrence()

    codes, info = grade_track(track)

    assert DEFAULT_HALO_FRAMES == 3
    assert info["halo_frames"] == DEFAULT_HALO_FRAMES
    np.testing.assert_array_equal(codes[17:20], np.full(3, DEGRADED, dtype=np.uint8))
    np.testing.assert_array_equal(codes[20:36], np.full(16, FABRICATED, dtype=np.uint8))
    np.testing.assert_array_equal(codes[36:39], np.full(3, DEGRADED, dtype=np.uint8))
    assert codes[16] == NO_FLAG
    assert codes[39] == NO_FLAG


def test_grade_track_clips_halo_at_track_start() -> None:
    track = _track_with_recurrence()
    pattern = track[20:36, :2].copy()
    original_indices = np.arange(20, 36, dtype=float)
    track[20:36, :2] = np.column_stack(
        (original_indices, original_indices * 0.37 + 1.0),
    )
    track[:16, :2] = pattern

    codes, _info = grade_track(track)

    np.testing.assert_array_equal(codes[:16], np.full(16, FABRICATED, dtype=np.uint8))
    np.testing.assert_array_equal(codes[16:19], np.full(3, DEGRADED, dtype=np.uint8))
    assert codes[19] == NO_FLAG


def test_grade_track_clips_halo_at_track_end() -> None:
    track = _track_with_recurrence()
    pattern = track[20:36, :2].copy()
    original_indices = np.arange(20, 36, dtype=float)
    track[20:36, :2] = np.column_stack(
        (original_indices, original_indices * 0.37 + 1.0),
    )
    track[-16:, :2] = pattern

    codes, _info = grade_track(track)

    np.testing.assert_array_equal(codes[-16:], np.full(16, FABRICATED, dtype=np.uint8))
    np.testing.assert_array_equal(codes[-19:-16], np.full(3, DEGRADED, dtype=np.uint8))
    assert codes[-20] == NO_FLAG


def test_grade_track_keeps_distant_exact_attractor_hits_degraded() -> None:
    track = _track_with_recurrence()
    track[60, :2] = track[20, :2]

    codes, _info = grade_track(track)

    assert codes[60] == DEGRADED


def test_grade_track_halo_is_part_of_cache_and_diagnostics_identity() -> None:
    track = _track_with_recurrence()

    narrow_codes, narrow_info = grade_track(track, halo_frames=1)
    broad_codes, broad_info = grade_track(track, halo_frames=4)

    assert narrow_info["halo_frames"] == 1
    assert broad_info["halo_frames"] == 4
    assert narrow_codes[18] == NO_FLAG
    assert broad_codes[18] == DEGRADED


def test_grade_track_cached_diagnostics_are_independent() -> None:
    clear_cache()
    track = _track_with_recurrence()
    _codes, info = grade_track(track)
    counts = info["counts_per_code"]
    assert isinstance(counts, dict)
    counts[NO_FLAG] = -1

    _cached_codes, cached_info = grade_track(track)

    cached_counts = cached_info["counts_per_code"]
    assert isinstance(cached_counts, dict)
    assert cached_counts[NO_FLAG] >= 0


@pytest.mark.parametrize("halo_frames", [-1, 1.5, True])
def test_grade_track_rejects_invalid_halo_frames(halo_frames: Any) -> None:
    with pytest.raises(ValueError, match="halo_frames"):
        grade_track(_track_with_recurrence(), halo_frames=halo_frames)


def test_grade_track_refuses_a_weak_threshold() -> None:
    track = _track_with_recurrence(varying_episodes=20)
    codes, info = grade_track(track)

    assert np.all(codes == NO_FLAG)
    assert "below 30 episodes" in info["unavailable_reason"]


def test_grade_track_refuses_a_weak_margin_even_when_threshold_is_high() -> None:
    track = _track_with_recurrence(
        varying_episodes=50,
        flat_starts=(5500, 6000, 6500, 7000, 7500, 8000),
    )
    codes, info = grade_track(track)

    assert np.all(codes == NO_FLAG)
    assert info["threshold"] >= 30
    assert info["margin"] < 10.0
    assert "margin" in info["unavailable_reason"]


def test_grade_track_refuses_fewer_than_two_distinct_candidate_counts() -> None:
    length = 14000
    track = np.column_stack((np.arange(length, dtype=float), np.arange(length, dtype=float) * 0.37))
    varying = np.column_stack((np.arange(16, dtype=float) + 100.0, np.arange(16, dtype=float) + 200.0))
    for start in [20 + episode * 100 for episode in range(25)] + [7200 + episode * 100 for episode in range(25)]:
        track[start:start + 16] = varying
    track = np.column_stack((track, np.ones(length)))

    codes, info = grade_track(track)

    assert np.all(codes == NO_FLAG)
    assert "fewer than 2 distinct candidate counts" in info["unavailable_reason"]


def test_grade_track_presence_validation_raises() -> None:
    track = _track_with_recurrence(
        varying_episodes=35, varying_step=40, flat_starts=(2000, 2500), length=3000,
    )
    with pytest.raises(ValueError, match="validation half"):
        grade_track(track)


def test_grade_track_disables_a_short_track() -> None:
    track = np.ones((15, 3), dtype=np.float64)
    codes, info = grade_track(track)

    assert np.array_equal(codes, np.zeros(15, dtype=np.uint8))
    assert "shorter than window" in info["unavailable_reason"]


def test_grade_track_preserves_stored_zero_frames_as_code_zero() -> None:
    track = _track_with_recurrence()
    track[18:20, :2] = 0.0
    track_before = track.tobytes()
    codes, _info = grade_track(track)

    assert np.all(codes[18:20] == NO_FLAG)
    assert track.tobytes() == track_before
