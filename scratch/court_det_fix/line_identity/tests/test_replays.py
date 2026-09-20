"""Synthetic checks of the replay helpers: the chart decomposition, the step masks and the product search."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axis_replay import (
    courts_from,
    ideal_parameters,
    nearest_over_product,
    step_masks,
    sweep,
)
from paint_profiles import (
    FLANK_WINDOWS_WORKING_PX,
    features,
    profile_offsets,
)

from shared import (
    ALL_CASE_IDS,
    ALL_CASES,
    CASE_IDS,
    CASES,
    LABELS,
    PACK_OF,
    REGRESSION_CASES,
    corner_errors,
)


def random_basis(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    basis = generator.normal(size=(3, 3))
    basis[2] = [0.01, 0.02, 1.]
    return basis


def test_ideal_parameters_round_trip():
    basis = random_basis(0)
    scales_offsets = np.array([[2.5, -0.3], [1.7, 0.4]])
    homography = basis @ np.array([[2.5, 0., -0.3], [0., 1.7, 0.4], [0., 0., 1.]]) * 3.
    recovered, leak = ideal_parameters(basis, homography)
    np.testing.assert_allclose(recovered, scales_offsets, atol=1e-12)
    assert leak < 1e-12
    corners = courts_from(basis, recovered[:1], recovered[1:])
    reference = courts_from(basis, scales_offsets[:1], scales_offsets[1:])
    np.testing.assert_allclose(corners, reference, atol=1e-9)


def test_step_masks_follow_the_matching_order():
    matches = SimpleNamespace(parameters=np.zeros((6, 2)), supported=np.array([3, 2, 5, 3, 4, 3]),
                              player_compatible=np.array([True, True, False, True, True, True]),
                              distinct=np.array([2, 4, 0, 5]))
    masks = step_masks(matches, cap=3)
    assert masks['enumerated'].tolist() == [True] * 6
    assert masks['supported'].tolist() == [True, False, True, True, True, True]
    assert masks['supported_and_players'].tolist() == [True, False, False, True, True, True]
    assert masks['distinct'].tolist() == [True, False, True, False, True, True]
    assert masks['kept'].tolist() == [True, False, True, False, True, False]


def test_nearest_over_product_equals_brute_force():
    basis = random_basis(1)
    generator = np.random.default_rng(2)
    horizontal = generator.normal(size=(7, 2)) + [3., 0.]
    vertical = generator.normal(size=(5, 2)) + [3., 0.]
    control = courts_from(basis, horizontal[3:4], vertical[1:2])[0] + 0.5
    best = nearest_over_product(basis, horizontal, vertical, control)
    brute = min(float(corner_errors(courts_from(basis, horizontal[i:i + 1], vertical[j:j + 1]), control)[0])
                for i in range(7) for j in range(5))
    assert abs(best - brute) < 1e-9
    errors = sweep(basis, 0, horizontal, vertical[1], control)
    assert errors.shape == (7,) and abs(errors[3] - float(corner_errors(control[None] - 0.5, control)[0])) < 1e-9


def synthetic_profile(brightness_across: np.ndarray, saturation_value: float) -> np.ndarray:
    samples = 12
    brightness = np.tile(brightness_across, (1, samples, 1)).astype(float)
    saturation = np.full_like(brightness, saturation_value)
    return np.stack((brightness, saturation))


def test_features_pass_a_ridge_and_fail_a_step():
    offsets = profile_offsets()
    flank_near = FLANK_WINDOWS_WORKING_PX[0][0]
    ridge = np.where(np.abs(offsets) <= 1.5, 230., 150.)
    step = np.where(offsets >= 0., 230., 150.)
    plateau = np.where(offsets >= -1., 230., 150.)  # bright on one side out past every flank window
    measured = features(synthetic_profile(np.stack((ridge, step, plateau))[:, None, :], 30.))
    contrast = measured[:, 0]
    assert contrast[0] > 70, contrast
    assert contrast[1] <= 0, contrast
    assert contrast[2] <= 0, contrast
    assert measured[0, 1] == 30.
    assert flank_near > 1.5


def test_control_vanishing_points_and_angles_recover_a_known_homography():
    import cv2
    from filter_replay import (
        angles_to_control,
        control_vanishing_points,
        fragment_masks,
    )

    from experiments.annotator.independent_court import detector
    size = (960, 540)
    homography = np.array([[80., 10., 200.], [5., 60., 100.], [0.001, 0.02, 1.]])
    corners, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    control_directions = control_vanishing_points(corners[0], size)
    # The homography's first two columns are the vanishing points; feed them back as selected directions.
    points = homography[:, :2].T
    angles = angles_to_control(points, control_directions, size)
    assert angles.shape == (2,) and angles.max() < 1e-4, angles
    # A box that covers the first fragment's midpoint and nothing else.
    frame = np.full((40, 60, 3), 120, dtype=np.uint8)
    cv2.line(frame, (10, 20), (50, 20), (255, 255, 255), 2)
    source = {'segments_px': [[10., 20., 50., 20.], [10., 5., 50., 5.]], 'bbox_px': [[20., 15., 40., 25.]]}
    masks = fragment_masks(source, frame, np.array([1., 1.]))
    assert masks['person'].tolist() == [False, True]
    assert masks['paint'].tolist() == [True, False]
    assert masks['baseline'].all()


def test_unused_cases_are_explicit_without_changing_regression_defaults():
    assert CASES == REGRESSION_CASES
    assert CASE_IDS == tuple(case_id for case_id, _, _ in REGRESSION_CASES)
    assert len(ALL_CASES) == 27
    assert len(ALL_CASE_IDS) == 27
    assert 'gxBQ_window_00_frame_689' in ALL_CASE_IDS
    assert PACK_OF['gxBQ_window_00_frame_689'] == 'gx'
    assert LABELS['gxBQ_window_00_frame_689'] == 'gxBQ_window_00_frame_689'
