from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest

from annotator.calibration.shuttleset_features import (
    InterpolationType,
    ShuttleFeatureInputs,
    coordinate_error_summary,
    court_corner_error_rows,
    derive_shuttle_feature_inputs,
    evaluate_rally_features,
    feature_population,
    interpolate_internal_gaps,
    least_squares_trend,
    median_absolute_deviation,
    movement_inefficiency,
    posture_signal,
    project_positions_by_scene,
    rally_duration_base30,
    rally_timestamps,
    recovery_at_opponent_contacts,
    select_sticky_keypoints,
    serve_speed_proxy,
    score_contact_coordinates,
    tanh_degradation,
)
from annotator.court_evidence import detected_court_info


def test_sticky_selection_and_posture_formula() -> None:
    pose = np.zeros((1, 3, 17, 2), dtype=float)
    pose[0, 2, (1, 2), 1] = 2.0
    pose[0, 2, (15, 16), 1] = 10.0
    pose[0, 2, 11] = (2.0, 6.0)
    pose[0, 2, 12] = (6.0, 6.0)

    selected = select_sticky_keypoints(pose, np.array([[2, -1]], dtype=int))
    posture = posture_signal(selected)

    assert posture[0, 0] == 2.0
    assert np.isnan(posture[0, 1])
    assert median_absolute_deviation(np.array([1.0, 2.0, 9.0])) == 1.0


def test_scene_projection_uses_scene_quad_without_clipping() -> None:
    rows = pd.DataFrame(
        [
            {
                "start_frame": 0,
                "end_frame": 2,
                "upleft_x": 0.0,
                "upleft_y": 0.0,
                "upright_x": 100.0,
                "upright_y": 0.0,
                "downleft_x": 0.0,
                "downleft_y": 50.0,
                "downright_x": 100.0,
                "downright_y": 50.0,
            }
        ]
    )
    positions = np.array(
        [
            [[50.0, 12.5], [50.0, 37.5]],
            [[-10.0, 12.5], [110.0, 37.5]],
        ]
    )

    projected = project_positions_by_scene(positions, rows, (100.0, 50.0))

    np.testing.assert_allclose(projected[0], [[0.5, 0.25], [0.5, 0.75]])
    assert projected[1, 0, 0] < 0.0
    assert projected[1, 1, 0] > 1.0

    corner_rows = court_corner_error_rows(
        rows,
        np.array(((0.0, 0.0), (1280.0, 0.0), (1280.0, 720.0), (0.0, 720.0))),
        (100.0, 50.0),
    )
    assert corner_rows[0]["corner_errors_px"] == [0.0, 0.0, 0.0, 0.0]
    assert corner_rows[0]["max_error_px"] == 0.0


def test_coordinate_error_summary_reports_denominators_and_exclusions() -> None:
    predicted = np.array(((0.0, 0.0), (np.nan, np.nan), (3.0, 4.0)))
    truth = np.array(((0.0, 1.0), (1.0, 1.0), (np.nan, np.nan)))

    summary = coordinate_error_summary(predicted, truth)

    assert summary == {
        "population": 3,
        "eligible": 1,
        "excluded_prediction": 1,
        "excluded_ground_truth": 1,
        "mean_error": 1.0,
        "median_error": 1.0,
        "p90_error": 1.0,
    }


def test_contact_coordinate_scoring_uses_exact_frames_and_player_sides() -> None:
    ground_truth = pd.DataFrame(
        {
            "frame_num": [1],
            "player_side": ["Top"],
            "hit_x": [640.0],
            "hit_y": [360.0],
            "player_location_x": [640.0],
            "player_location_y": [180.0],
            "opponent_location_x": [640.0],
            "opponent_location_y": [540.0],
        }
    )
    shuttle = np.full((3, 2), np.nan)
    shuttle[1] = (0.5, 0.5)
    players = np.full((3, 2, 2), np.nan)
    players[1, 0] = (0.5, 0.25)
    players[1, 1] = (0.5, 0.75)
    court_info = detected_court_info(
        np.array(((0.0, 0.0), (1280.0, 0.0), (1280.0, 720.0), (0.0, 720.0)))
    )

    shuttle_inputs = ShuttleFeatureInputs(
        shuttle, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)
    )
    scored = score_contact_coordinates(
        ground_truth, shuttle_inputs, players, court_info
    )

    assert scored["units"] == "normalized doubles-court Euclidean distance"
    assert scored["matching"] == "exact authoritative GT contact frame"
    assert scored["unusable_attribution"] == 0
    summary = scored["summary"]
    assert isinstance(summary, dict)
    assert summary["shuttle"]["eligible"] == 1
    assert summary["striker"]["mean_error"] == 0.0
    assert summary["opponent"]["mean_error"] == 0.0


def test_contact_coordinate_scoring_keeps_shuttle_when_attribution_is_missing() -> None:
    ground_truth = pd.DataFrame(
        {
            "frame_num": [1],
            "player_side": [None],
            "hit_x": [640.0],
            "hit_y": [360.0],
            "player_location_x": [640.0],
            "player_location_y": [180.0],
            "opponent_location_x": [640.0],
            "opponent_location_y": [540.0],
        }
    )
    positions = np.full((3, 2), np.nan)
    positions[1] = (0.5, 0.5)
    players = np.zeros((3, 2, 2), dtype=float)
    court_info = detected_court_info(
        np.array(((0.0, 0.0), (1280.0, 0.0), (1280.0, 720.0), (0.0, 720.0)))
    )

    scored = score_contact_coordinates(
        ground_truth,
        ShuttleFeatureInputs(
            positions, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)
        ),
        players,
        court_info,
    )

    assert scored["unusable_attribution"] == 1
    assert scored["summary"]["shuttle"]["eligible"] == 1
    assert scored["summary"]["striker"]["excluded_ground_truth"] == 1


def test_internal_interpolation_does_not_extrapolate() -> None:
    values = np.full((5, 1, 2), np.nan)
    values[1, 0] = (0.0, 2.0)
    values[3, 0] = (2.0, 4.0)

    filled, provenance = interpolate_internal_gaps(values, [(0, 5)])

    assert np.isnan(filled[0, 0]).all()
    np.testing.assert_allclose(filled[2, 0], [1.0, 3.0])
    assert provenance[2, 0] == InterpolationType.LINEAR
    assert np.isnan(filled[4, 0]).all()


def test_shuttle_projection_excludes_invisible_and_guard_rejected_frames() -> None:
    track = np.array(
        (
            (0.5, 0.5, 1.0),
            (0.6, 0.5, 0.0),
            (0.7, 0.5, 1.0),
        ),
        dtype=float,
    )
    rows = pd.DataFrame(
        [
            {
                "start_frame": 0,
                "end_frame": 3,
                "upleft_x": 0.0,
                "upleft_y": 0.0,
                "upright_x": 100.0,
                "upright_y": 0.0,
                "downleft_x": 0.0,
                "downleft_y": 50.0,
                "downright_x": 100.0,
                "downright_y": 50.0,
            }
        ]
    )

    shuttle = derive_shuttle_feature_inputs(
        track,
        np.array((0, 0, 1), dtype=np.int8),
        frozenset({1, 2, 3}),
        rows,
        (100.0, 50.0),
    )

    np.testing.assert_allclose(shuttle.court_positions[0], (0.5, 0.5))
    assert np.isnan(shuttle.court_positions[1]).all()
    assert np.isnan(shuttle.court_positions[2]).all()
    assert serve_speed_proxy(shuttle, 0, 1, 30.0) is None


def test_recovery_and_movement_use_court_normalised_positions() -> None:
    positions = np.zeros((11, 2, 2), dtype=float)
    positions[:, 0] = (0.5, 0.25)
    positions[:, 1] = (0.5, 1.0)

    recovery = recovery_at_opponent_contacts(positions, [5], [0], 30.0)

    assert recovery == [
        {
            "contact_frame": 5,
            "measured_slot": 1,
            "window_start": 0,
            "window_end": 11,
            "valid_frames": 11,
            "mean_distance": 0.25,
        }
    ]

    path = np.zeros((3, 2, 2), dtype=float)
    path[:, 0] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0))
    inefficiency = movement_inefficiency(path, [0, 2])
    assert math.isclose(inefficiency[0, 0], 2.0 - math.sqrt(2.0))
    assert inefficiency[0, 1] == 0.0


def test_time_speed_and_degradation_require_explicit_policy_inputs() -> None:
    assert rally_timestamps(30, 90, 30.0) == {
        "frame_range": [30, 90],
        "second_range": [1.0, 3.0],
        "fps": 30.0,
    }
    assert rally_duration_base30(
        0, 120, 60.0, end_offset_base30=10
    ) == 70.0
    with pytest.raises(ValueError, match="non-negative integer"):
        rally_duration_base30(
            0, 120, 60.0, end_offset_base30=cast(int, 0.5)
        )

    positions = np.array(((0.0, 0.0), (0.0, 0.0), (3.0, 4.0)))
    shuttle = ShuttleFeatureInputs(
        positions, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool)
    )
    assert serve_speed_proxy(shuttle, 0, 2, 2.0) == 5.0

    slope = least_squares_trend(np.array([1.0, np.nan, 3.0]))
    assert slope == 1.0
    assert tanh_degradation(slope, temperature=2.0) == math.tanh(0.5)


def test_rally_evaluator_reports_values_and_exact_populations() -> None:
    record = {
        "rally": {"rally_id": 0, "start_frame": 0, "end_frame": 5},
        "contacts": {
            "accepted": [
                {"contact_frame": 1},
                {"contact_frame": 3},
            ],
            "stroke_count": 2,
        },
        "outcomes": {"server_prediction": "Top"},
    }
    posture = np.column_stack((np.arange(5, dtype=float), np.arange(5, dtype=float)))
    positions = np.zeros((5, 2, 2), dtype=float)
    positions[:, 0] = (0.5, 0.25)
    positions[:, 1] = (0.5, 0.75)
    posture_interpolation = np.zeros((5, 2), dtype=np.int8)
    position_interpolation = np.zeros((5, 2), dtype=np.int8)
    position_interpolation[2, 0] = InterpolationType.LINEAR

    row = evaluate_rally_features(
        record,
        posture,
        positions,
        posture_interpolation,
        position_interpolation,
        30.0,
        end_offset_base30=0,
    )

    assert row["shots_per_rally"] == 2
    assert row["rally_duration_base30"] == 3.0
    assert row["posture"] == [
        {"slot": "top", "valid_frames": 5, "mad": 1.0},
        {"slot": "bottom", "valid_frames": 5, "mad": 1.0},
    ]
    recovery = row["recovery"]
    interpolation_counts = row["interpolation"]
    assert isinstance(recovery, dict)
    assert isinstance(interpolation_counts, dict)
    assert recovery["population"] == 2
    assert recovery["median_top"] == 0.0
    assert recovery["median_bottom"] == 0.0
    observations = recovery["observations"]
    assert isinstance(observations, list)
    assert all(observation["window_end"] <= 5 for observation in observations)
    assert interpolation_counts["position_linear_top_frames"] == 1
    assert feature_population([row]) == {
        "rallies": 1,
        "duration_eligible": 1,
        "posture_total": 2,
        "posture_eligible": 2,
        "recovery_total": 2,
        "recovery_eligible": 2,
        "movement_total": 2,
        "movement_eligible": 2,
    }
