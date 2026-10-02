"""Focused tests for the image-space independent-court tracker."""

from __future__ import annotations

import pytest

from experiments.annotator.independent_court.temporal import (
    Track,
    pair_presence,
    summarise_track,
    track_people,
)


def _sample(timestamp: float, boxes: list[list[float]], scores: list[float] | None = None) -> dict:
    return {
        "timestamp_seconds": timestamp,
        "bboxes": boxes,
        "scores": [0.9] * len(boxes) if scores is None else scores,
    }


def _box(foot_x: float, height: float = 20.0) -> list[float]:
    return [foot_x - 2, 100 - height, foot_x + 2, 100]


def test_nearest_footpoint_association_is_unique_when_detection_order_changes() -> None:
    samples = [
        _sample(0.0, [_box(20), _box(40)]),
        _sample(0.1, [_box(38), _box(22)]),
    ]

    tracks = track_people(samples)

    assert [(track.sample_indices, track.detection_indices, track.feet_px) for track in tracks] == [
        ([0, 1], [0, 1], [[20.0, 100.0], [22.0, 100.0]]),
        ([0, 1], [1, 0], [[40.0, 100.0], [38.0, 100.0]]),
    ]


def test_missing_observation_is_preserved_as_a_gap() -> None:
    samples = [_sample(0.0, [_box(20)]), _sample(0.1, []), _sample(0.2, [_box(22)])]

    tracks = track_people(samples)

    assert len(tracks) == 1
    assert tracks[0].sample_indices == [0, 2]
    assert summarise_track(tracks[0], 3)["observed_fraction"] == pytest.approx(2 / 3)


def test_track_summary_reports_path_displacement_and_normalised_path() -> None:
    track = Track(
        4,
        [0, 1, 2],
        [2, 0, 1],
        [[0.0, 0.0], [3.0, 4.0], [0.0, 8.0]],
        [10.0, 20.0, 10.0],
    )

    summary = summarise_track(track, 4)

    assert summary == {
        "id": 4,
        "observations": 3,
        "observed_fraction": 0.75,
        "path_length_px": 10.0,
        "displacement_px": 8.0,
        "median_foot_px": [0.0, 4.0],
        "median_height_px": 10.0,
        "path_length_heights": 1.0,
    }


def test_timestamp_gap_splits_track() -> None:
    samples = [_sample(0.0, [_box(20)]), _sample(0.6, [_box(21)])]

    tracks = track_people(samples)

    assert [track.sample_indices for track in tracks] == [[0], [1]]


def test_large_footpoint_jump_starts_new_track() -> None:
    samples = [_sample(0.0, [_box(20)]), _sample(0.1, [_box(50)])]

    tracks = track_people(samples, max_step_heights=1.0)

    assert [track.sample_indices for track in tracks] == [[0], [1]]


def test_score_cut_is_strictly_greater_than_score_min() -> None:
    samples = [_sample(0.0, [_box(20), _box(80)], [0.2, 0.200001])]

    tracks = track_people(samples)

    assert len(tracks) == 1
    assert tracks[0].detection_indices == [1]


def test_literal_pair_rule_passes_at_exactly_half_with_no_zero_frames() -> None:
    first = Track(0, [0, 1], [0, 0], [[0, 0]] * 2, [20.0] * 2)
    second = Track(1, [0, 1, 2, 3], [1, 1, 1, 1], [[1, 0]] * 4, [20.0] * 4)

    presence = pair_presence(first, second, 4)

    assert presence == {
        "two_fraction": 0.5,
        "one_or_more_fraction": 1.0,
        "zero_frames": 0,
        "passes_literal": True,
    }


def test_pair_presence_uses_all_samples_for_missing_frames_and_boundary() -> None:
    first = Track(0, [0, 1], [0, 0], [[0, 0], [1, 0]], [20.0, 20.0])
    second = Track(1, [1, 2], [1, 1], [[2, 0], [3, 0]], [20.0, 20.0])

    presence = pair_presence(first, second, 4)

    assert presence["two_fraction"] == 0.25
    assert presence["one_or_more_fraction"] == 0.75
    assert presence["zero_frames"] == 1
    assert presence["passes_literal"] is False


@pytest.mark.parametrize(
    "samples",
    [
        [_sample(1.0, []), _sample(1.0, [])],
        [{"timestamp_seconds": 0.0, "bboxes": [[0, 0, 1]], "scores": [0.9]}],
        [{"timestamp_seconds": 0.0, "bboxes": [[0, 0, 1, 1]], "scores": []}],
    ],
)
def test_malformed_sample_shapes_and_timestamp_order_fail_at_boundary(samples: list[dict]) -> None:
    with pytest.raises(ValueError):
        track_people(samples)
