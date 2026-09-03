from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from dataset_builder.features import (
    COURT_SIDES,
    InterpolationType,
    PlayerFeatureInputs,
    clip_frames,
    interpolate_internal_gaps,
    median_absolute_deviation,
    player_rally_features,
    posture_signal,
    rally_timestamps,
    select_sticky_keypoints,
)


def test_posture_signal_formula_and_nan_cases() -> None:
    pose = np.zeros((1, 3, 17, 2), dtype=float)
    pose[0, 2, (1, 2), 1] = 2.0
    pose[0, 2, (15, 16), 1] = 10.0
    pose[0, 2, 11] = (2.0, 6.0)
    pose[0, 2, 12] = (6.0, 6.0)

    selected = select_sticky_keypoints(pose, np.array([[2, -1]], dtype=int))
    posture = posture_signal(selected)

    assert posture[0, 0] == 2.0
    assert np.isnan(posture[0, 1])  # slot 1 has no sticky pick, so its row is all NaN

    zero_hip_width = selected.copy()
    zero_hip_width[0, 0, 12] = zero_hip_width[0, 0, 11]
    assert np.isnan(posture_signal(zero_hip_width)[0, 0])


def test_median_absolute_deviation_formula_and_all_nan() -> None:
    assert median_absolute_deviation(np.array([1.0, 2.0, 9.0])) == 1.0
    assert median_absolute_deviation(np.full(3, np.nan)) is None


def test_interpolate_internal_gaps_fills_only_internal_gaps() -> None:
    values = np.full((6, 2, 2), np.nan)
    values[1, 0] = (0.0, 2.0)
    values[4, 0] = (2.0, 8.0)
    values[:, 1] = np.arange(12).reshape(6, 2)  # fully observed, untouched slot

    filled, provenance = interpolate_internal_gaps(values, [(0, 6)])

    assert np.isnan(filled[0, 0]).all()  # before the first observation: not extrapolated
    assert np.isnan(filled[5, 0]).all()  # after the last observation: not extrapolated
    np.testing.assert_allclose(filled[2, 0], [2.0 / 3.0, 4.0])
    np.testing.assert_allclose(filled[3, 0], [4.0 / 3.0, 6.0])
    assert list(provenance[:, 0]) == [
        InterpolationType.OBSERVED,
        InterpolationType.OBSERVED,
        InterpolationType.LINEAR,
        InterpolationType.LINEAR,
        InterpolationType.OBSERVED,
        InterpolationType.OBSERVED,
    ]
    assert (provenance[:, 1] == InterpolationType.OBSERVED).all()
    np.testing.assert_array_equal(filled[:, 1], values[:, 1])

    with pytest.raises(ValueError, match="overlap"):
        interpolate_internal_gaps(values, [(0, 4), (2, 6)])


def test_rally_timestamps_and_its_validation() -> None:
    assert rally_timestamps(30, 90, 30.0) == {
        "frame_range": [30, 90],
        "second_range": [1.0, 3.0],
        "fps": 30.0,
    }
    with pytest.raises(ValueError, match="half-open"):
        rally_timestamps(90, 30, 30.0)
    with pytest.raises(ValueError, match="positive and finite"):
        rally_timestamps(30, 90, 0.0)


def test_clip_frames_rounds_to_whole_frames_and_clamps_to_the_video() -> None:
    # 2 s and 3 s at 29.97 fps are 59.94 and 89.91 frames, rounded to 60 and 90.
    assert clip_frames(1000, 2000, Fraction(30000, 1001), 10_000) == (940, 2090)
    # A short video clamps the lead-in at 0 and the tail at frame_count.
    assert clip_frames(10, 95, Fraction(25), 100) == (0, 100)

    with pytest.raises(ValueError, match="half-open"):
        clip_frames(90, 30, Fraction(25), 100)
    with pytest.raises(ValueError, match="does not cover end_frame"):
        clip_frames(10, 95, Fraction(25), 90)


def test_player_rally_features_both_sides_and_out_of_range() -> None:
    posture = np.full((6, 2), np.nan)
    posture[:, 0] = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]  # slot 0 (top): 5 valid frames
    # slot 1 (bottom) stays all-NaN, covering the zero-valid-frames case

    posture_interpolation = np.zeros((6, 2), dtype=np.int8)
    posture_interpolation[2, 0] = InterpolationType.LINEAR  # top's frame 2 was filled

    court_positions = np.full((6, 2, 2), np.nan)
    court_positions[0:4, 0] = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    court_positions[:, 1] = [0.0, 0.0]  # bottom fully valid across the rally

    position_interpolation = np.zeros((6, 2), dtype=np.int8)
    position_interpolation[2, 0] = InterpolationType.LINEAR  # top's frame 2 was filled

    inputs = PlayerFeatureInputs(
        posture=posture,
        court_positions=court_positions,
        posture_interpolation=posture_interpolation,
        position_interpolation=position_interpolation,
        tracker_segments=((0, 6),),
    )

    top, bottom = player_rally_features(inputs, 0, 6)

    assert top.court_side == COURT_SIDES[0]
    assert top.posture_frames_valid == 5
    assert top.posture_frames_linear == 1
    assert top.posture_mad == 1.0
    assert top.position_frames_valid == 4
    assert top.position_frames_linear == 1

    assert bottom.court_side == COURT_SIDES[1]
    assert bottom.posture_frames_valid == 0
    assert bottom.posture_frames_linear == 0
    assert bottom.posture_mad is None
    assert bottom.position_frames_valid == 6
    assert bottom.position_frames_linear == 0

    with pytest.raises(ValueError, match="exceeds the feature arrays"):
        player_rally_features(inputs, 0, 7)
