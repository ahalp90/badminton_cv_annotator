"""Synthetic tests for the inpaint fabrication grade detector."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pytest

from annotator.excluded_frames.shuttle_hallucination import (
    DEFAULT_HALO_FRAMES,
    ShuttleGrade,
    code_counts,
    grade_track,
)


@pytest.mark.parametrize(('code', 'label'), [
    (0, 'unflagged'),
    (1, 'repeated moving pattern'),
    (2, 'repeated stationary position'),
    (3, 'halo around repeated pattern'),
    (4, 'matches any position in a repeated pattern'),
])
def test_shuttle_grade_preserves_codes_labels_and_pickle_roundtrip(code: int, label: str) -> None:
    grade = ShuttleGrade(code)
    assert int(grade) == code
    assert str(grade) == label
    assert pickle.loads(pickle.dumps(grade)) is grade


def test_shuttle_grades_preserve_array_storage_and_numeric_report_keys() -> None:
    codes = np.asarray(tuple(ShuttleGrade), dtype=np.uint8)
    np.testing.assert_array_equal(codes, np.arange(5, dtype=np.uint8))
    assert ShuttleGrade.rejectable() == frozenset({1, 2, 3, 4})
    assert json.loads(json.dumps(code_counts(codes))) == {str(code): 1 for code in range(5)}
    assert [str(code) for code in code_counts(codes)] == ['0', '1', '2', '3', '4']


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


def test_grade_track_marks_moving_stationary_and_nearby_frames() -> None:
    track = _track_with_recurrence()
    codes, info = grade_track(track)

    assert codes.dtype == np.uint8
    assert len(codes) == len(track)
    assert info["threshold"] == 50
    assert info["count_ratio"] == pytest.approx(50 / 3)
    assert info["moving_pattern_count"] == 1
    assert info["stationary_pattern_count"] == 1
    assert np.count_nonzero(codes == ShuttleGrade.REPEATED_MOVING_PATTERN) > 0
    assert np.count_nonzero(codes == ShuttleGrade.REPEATED_STATIONARY_POSITION) > 0
    assert np.count_nonzero(codes == ShuttleGrade.REPEATED_PATTERN_HALO) > 0


def test_grade_track_uses_three_absolute_halo_frames() -> None:
    track = _track_with_recurrence()

    codes, info = grade_track(track)

    assert DEFAULT_HALO_FRAMES == 3
    assert info["halo_frames"] == DEFAULT_HALO_FRAMES
    np.testing.assert_array_equal(codes[17:20], np.full(3, ShuttleGrade.REPEATED_PATTERN_HALO, dtype=np.uint8))
    np.testing.assert_array_equal(codes[20:36], np.full(16, ShuttleGrade.REPEATED_MOVING_PATTERN, dtype=np.uint8))
    np.testing.assert_array_equal(codes[36:39], np.full(3, ShuttleGrade.REPEATED_PATTERN_HALO, dtype=np.uint8))
    assert codes[16] == ShuttleGrade.UNFLAGGED
    assert codes[39] == ShuttleGrade.UNFLAGGED


def test_grade_track_clips_halo_at_track_start() -> None:
    track = _track_with_recurrence()
    pattern = track[20:36, :2].copy()
    original_indices = np.arange(20, 36, dtype=float)
    track[20:36, :2] = np.column_stack(
        (original_indices, original_indices * 0.37 + 1.0),
    )
    track[:16, :2] = pattern

    codes, _info = grade_track(track)

    np.testing.assert_array_equal(codes[:16], np.full(16, ShuttleGrade.REPEATED_MOVING_PATTERN, dtype=np.uint8))
    np.testing.assert_array_equal(codes[16:19], np.full(3, ShuttleGrade.REPEATED_PATTERN_HALO, dtype=np.uint8))
    assert codes[19] == ShuttleGrade.UNFLAGGED


def test_grade_track_clips_halo_at_track_end() -> None:
    track = _track_with_recurrence()
    pattern = track[20:36, :2].copy()
    original_indices = np.arange(20, 36, dtype=float)
    track[20:36, :2] = np.column_stack(
        (original_indices, original_indices * 0.37 + 1.0),
    )
    track[-16:, :2] = pattern

    codes, _info = grade_track(track)

    np.testing.assert_array_equal(codes[-16:], np.full(16, ShuttleGrade.REPEATED_MOVING_PATTERN, dtype=np.uint8))
    np.testing.assert_array_equal(codes[-19:-16], np.full(3, ShuttleGrade.REPEATED_PATTERN_HALO, dtype=np.uint8))
    assert codes[-20] == ShuttleGrade.UNFLAGGED


def test_grade_track_marks_distant_exact_position_matches() -> None:
    track = _track_with_recurrence()
    track[60, :2] = track[20, :2]

    codes, _info = grade_track(track)

    assert codes[60] == ShuttleGrade.MATCHES_ANY_PATTERN_POS


def test_position_matches_take_precedence_over_halo_for_both_pattern_kinds() -> None:
    track = _track_with_recurrence()
    track[19, :2] = track[5500, :2]
    track[5499, :2] = track[20, :2]
    track[60, :2] = track[20, :2]
    track[65, :2] = track[5500, :2]

    codes, _info = grade_track(track)

    for frame in (19, 5499, 60, 65):
        assert codes[frame] == ShuttleGrade.MATCHES_ANY_PATTERN_POS
    assert codes[18] == ShuttleGrade.REPEATED_PATTERN_HALO
    assert codes[5498] == ShuttleGrade.REPEATED_PATTERN_HALO
    assert codes[20] == ShuttleGrade.REPEATED_MOVING_PATTERN
    assert codes[5500] == ShuttleGrade.REPEATED_STATIONARY_POSITION


def test_halo_width_changes_nearby_frame_grades() -> None:
    track = _track_with_recurrence()

    narrow_codes, narrow_info = grade_track(track, halo_frames=1)
    broad_codes, broad_info = grade_track(track, halo_frames=4)

    assert narrow_info["halo_frames"] == 1
    assert broad_info["halo_frames"] == 4
    assert narrow_codes[18] == ShuttleGrade.UNFLAGGED
    assert broad_codes[18] == ShuttleGrade.REPEATED_PATTERN_HALO



def test_grade_track_refuses_a_weak_threshold() -> None:
    track = _track_with_recurrence(varying_episodes=20)
    codes, info = grade_track(track)

    assert np.all(codes == ShuttleGrade.UNFLAGGED)
    assert "below 30 occurrences" in info["unavailable_reason"]


def test_grade_track_refuses_similar_occurrence_counts() -> None:
    track = _track_with_recurrence(
        varying_episodes=50,
        flat_starts=(5500, 6000, 6500, 7000, 7500, 8000),
    )
    codes, info = grade_track(track)

    assert np.all(codes == ShuttleGrade.UNFLAGGED)
    assert info["threshold"] >= 30
    assert info["count_ratio"] < 10.0
    assert "count ratio" in info["unavailable_reason"]


def test_grade_track_refuses_fewer_than_two_distinct_candidate_counts() -> None:
    length = 14000
    track = np.column_stack((np.arange(length, dtype=float), np.arange(length, dtype=float) * 0.37))
    varying = np.column_stack((np.arange(16, dtype=float) + 100.0, np.arange(16, dtype=float) + 200.0))
    for start in [20 + episode * 100 for episode in range(25)] + [7200 + episode * 100 for episode in range(25)]:
        track[start:start + 16] = varying
    track = np.column_stack((track, np.ones(length)))

    codes, info = grade_track(track)

    assert np.all(codes == ShuttleGrade.UNFLAGGED)
    assert "fewer than 2 distinct candidate counts" in info["unavailable_reason"]


def test_pattern_missing_from_half_raises() -> None:
    track = _track_with_recurrence(
        varying_episodes=35, varying_step=40, flat_starts=(2000, 2500), length=3000,
    )
    with pytest.raises(ValueError, match="both halves"):
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

    assert np.all(codes[18:20] == ShuttleGrade.UNFLAGGED)
    assert track.tobytes() == track_before
