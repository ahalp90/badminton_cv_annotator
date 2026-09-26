"""Synthetic boundaries for homogeneous direction and court-spacing matching."""

import numpy as np
import pytest
from projective_seed import Settings, basis_for, combine, corner_errors, match_axis

from experiments.annotator.independent_court import assignment, detector


def test_corner_error_allows_180_relabelling_but_preserves_geometry_error() -> None:
    court = np.array([[90., 40.], [170., 45.], [200., 400.], [30., 410.]])
    rotated = court[[2, 3, 0, 1]]
    assert float(corner_errors(rotated, court)) == 0.
    moved = rotated + np.array([3., 4.])
    assert float(corner_errors(moved, court)) == 5.
    assert float(corner_errors(court[[1, 0, 3, 2]], court)) > 0.


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


def test_finite_ranking_distinguishes_shift_alias_and_preserves_rotation() -> None:
    from run_given import canonicalise, finite_scores

    size = (960, 540)
    homography = np.array([[60., 0., 100.], [0., 25., 50.], [0., 0., 1.]])
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    observations = assignment.prepare_observations(projected.reshape(-1, 4), size)
    basis, _ = basis_for(homography[:, :2].T, size, Settings())
    assert basis is not None
    axes = (match_axis(basis, 0, detector.X_COORDS, observations, size, Settings()),
            match_axis(basis, 1, detector.Y_COORDS, observations, size, Settings()))
    shift = np.eye(3)
    shift[1, 2] = float(detector.Y_COORDS[2] - detector.Y_COORDS[1])
    rotation = np.array([[-1., 0., detector.X_COORDS.max()],
                         [0., -1., detector.Y_COORDS.max()], [0., 0., 1.]])
    candidates, _ = canonicalise(np.array([homography, homography @ shift, homography @ rotation]))
    scores = finite_scores(candidates, observations, axes, size)
    assert scores[0] > scores[1] + .2
    np.testing.assert_allclose(scores[0], scores[2], atol=1e-12)


@pytest.mark.parametrize('narrow', [True, False])
def test_existing_ridge_measurement_distinguishes_stripe_and_step(narrow: bool) -> None:
    from inspect_appearance import ridge_mask

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, 47:53 if narrow else 100] = 200
    segments = np.array([[50., 10., 50., 90.], [15., 10., 15., 90.], [50., 10., 50., 90.]])
    np.testing.assert_array_equal(ridge_mask(frame, segments), [narrow, False, narrow])
