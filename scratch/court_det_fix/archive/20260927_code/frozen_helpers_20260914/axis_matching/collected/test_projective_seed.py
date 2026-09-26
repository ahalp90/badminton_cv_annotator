"""Synthetic boundaries for homogeneous direction and court-spacing matching."""

import numpy as np
import pytest
from projective_seed import Settings, basis_for, combine, match_axis

from experiments.annotator.independent_court import assignment, detector


@pytest.mark.parametrize('homography', [
    np.array([[60., 0., 100.], [0., 25., 50.], [0., 0., 1.]]),
    np.array([[70., 12., 120.], [3., 40., 40.], [.01, .04, 1.]]),
    np.array([[-60., 0., 650.], [0., 25., 50.], [0., 0., 1.]]),
    np.array([[10., 45., 70.], [55., 3., 50.], [.01, .02, 1.]]),
    np.array([[75., 8., -35.], [2., 65., 35.], [.005, .12, 1.]]),
])
def test_retains_court_with_missing_marking_duplicate_edges_and_clutter(homography: np.ndarray) -> None:
    size = (960, 540)
    segments, _ = detector.project(homography[None], detector.SEGMENTS_M)
    raw = segments.reshape(-1, 4)
    # One far service observation is absent; a duplicate response is not another identity.
    raw = np.delete(raw, 7, axis=0)
    clutter = np.array([[10., 430., 420., 435.], [80., 30., 100., 450.], [840., 50., 800., 510.]])
    observations = assignment.prepare_observations(np.concatenate((raw, raw[:2] + .3, clutter)), size)
    settings = Settings()
    best_error = np.inf
    truth, _ = detector.project(homography[None], detector.CORNER_COURT_M)
    for points in (homography[:, :2].T, homography[:, [1, 0]].T):
        basis, details = basis_for(points, size, settings)
        assert details['status'] == 'valid' and basis is not None
        horizontal = match_axis(basis, 0, detector.X_COORDS, observations, size, settings)
        vertical = match_axis(basis, 1, detector.Y_COORDS, observations, size, settings)
        proposals, _ = combine(basis, horizontal, vertical)
        if len(proposals):
            corners, _ = detector.project(proposals, detector.CORNER_COURT_M)
            errors = np.linalg.norm(corners - truth, axis=2).max(axis=1)
            best_error = min(best_error, float(errors.min()))
    assert best_error < 1., best_error


def test_rejects_degenerate_directions_and_two_line_pattern() -> None:
    settings = Settings()
    basis, details = basis_for(np.array([[1., 0., 0.], [1., 0., 0.]]), (960, 540), settings)
    assert basis is None and details['status'] == 'degenerate_basis'
    observations = assignment.prepare_observations(np.array([[100, 40, 100, 450], [500, 40, 500, 450]]), (960, 540))
    valid, _ = basis_for(np.array([[1., 0., 0.], [0., 1., 0.]]), (960, 540), settings)
    assert valid is not None
    matched = match_axis(valid, 0, detector.X_COORDS, observations, (960, 540), settings)
    assert matched.diagnostics['enumerated'] > 0
    assert len(matched.retained) == 0


def test_horizon_crossing_is_recorded() -> None:
    from projective_seed import offsets

    # This invertible chart places its projective horizon on image x=200.
    rectification = np.array([[1., 0., 0.], [0., 1., 0.], [.005, 0., -1.]])
    basis = np.linalg.inv(rectification)
    segments = np.array([[100., 150., 300., 150.], [100., 250., 300., 250.], [250., 50., 300., 60.]])
    observations = assignment.prepare_observations(segments, (960, 540))
    _, _, _, details = offsets(basis, 1, observations, (960, 540), Settings(angle_deg=90.))
    assert len(details['horizon_excluded_group_ids']) == 2


def test_axis_player_rule_is_necessary_for_joint_rule() -> None:
    from projective_seed import necessary_players

    random = np.random.default_rng(20260914)
    rectified = np.array([[[1., 2.], [2., 8.]], [[1.5, 2.5], [2.5, 8.5]]])
    parameters = random.uniform(-15, 15, (1000, 2, 2))
    parameters[:, :, 0] += .0001
    court = (rectified[None] - parameters[:, None, None, :, 1]) / parameters[:, None, None, :, 0]
    court /= detector.CORNER_COURT_M.max(axis=0)
    inside = (court >= -.15).all(axis=3) & (court <= 1.15).all(axis=3)
    far, near = inside & (court[..., 1] < .5), inside & (court[..., 1] >= .5)
    joint = inside.any(axis=2).all(axis=1) & ((far.any(axis=2) & near.any(axis=2)).mean(axis=1) >= .5)
    x_pass = necessary_players(parameters[:, 0], rectified[..., 0], 0, float(detector.X_COORDS.max()))
    y_pass = necessary_players(parameters[:, 1], rectified[..., 1], 1, float(detector.Y_COORDS.max()))
    assert joint.any()
    assert np.all((x_pass & y_pass)[joint])
    assert not (x_pass & y_pass).all()
