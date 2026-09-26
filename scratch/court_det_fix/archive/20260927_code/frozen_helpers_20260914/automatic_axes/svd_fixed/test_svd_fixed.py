"""Synthetic checks for fixed-membership fitting without real frame computation."""

import numpy as np
from diagnose_directions import control_fit
from run_svd_fixed import fit_groups, svd_direction

from experiments.annotator.independent_court import detector


def assert_same_point(actual: np.ndarray, expected: np.ndarray) -> None:
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    assert abs(actual @ expected) > 1 - 1e-12


def test_two_lines_keep_the_full_nullspace() -> None:
    point, record = svd_direction(np.array([[1., 0., -2.], [0., 1., 3.]]))
    assert_same_point(point, np.array([2., -3., 1.]))
    assert record['singular_values'][2] == 0.
    assert record['normalised_nullspace_gap'] > 0.


def test_parallel_lines_retain_a_point_at_infinity() -> None:
    lines = np.array([[0., 1., -2.], [0., 1., 3.], [0., 1., 5.]])
    point, record = svd_direction(lines)
    assert_same_point(point, np.array([1., 0., 0.]))
    assert record['algebraic_rms'] < 1e-12


def test_arbitrary_line_scales_do_not_change_the_fit() -> None:
    lines = np.array([[1., 0., -2.], [0., 1., 3.], [1., 1., .8], [1., -1., -5.2]])
    original, record = svd_direction(lines)
    scaled, scaled_record = svd_direction(lines * np.array([100., -.01, 3., -7.])[:, None])
    assert_same_point(original, scaled)
    np.testing.assert_allclose(record['singular_values'], scaled_record['singular_values'], rtol=1e-12)


def test_fixed_membership_excludes_an_unrelated_line() -> None:
    lines = np.array([[1., 0., -2.], [0., 1., 3.], [1., 1., 1.], [1., 0., 80.]])
    masks = np.array([[True, True, True, False]])
    original_masks = masks.copy()
    points, records = fit_groups(lines, masks)
    assert_same_point(points[0], np.array([2., -3., 1.]))
    np.testing.assert_array_equal(masks, original_masks)
    assert records[0]['support_line_ids'] == [0, 1, 2]


def test_normalised_refits_recover_working_geometry() -> None:
    homography = np.array([[30., 5., 100.], [2., 20., 50.], [.04, .02, 1.]])
    transform = np.array([[1000., 0., 480.], [0., 1000., 270.], [0., 0., 1.]])
    normals = np.array([[1., 0.], [0., 1.], [1., 1.]])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    working_points = homography[:, :2].T
    line_groups = []
    for point in working_points:
        offsets = -(normals @ (point[:2] / point[2]))
        line_groups.append(np.column_stack((normals, offsets)))
    lines = np.concatenate(line_groups) @ transform
    masks = np.array([[True, True, True, False, False, False],
                      [False, False, False, True, True, True]])
    refit, _ = fit_groups(lines, masks)
    refit_working = refit @ transform.T
    for actual, expected in zip(refit_working, working_points, strict=True):
        assert_same_point(actual, expected)
    court_homogeneous = np.column_stack((detector.CORNER_COURT_M, np.ones(4)))
    corners = court_homogeneous @ homography.T
    corners = corners[:, :2] / corners[:, 2:]
    fitted = control_fit(refit_working, corners)
    assert fitted['converged']
    assert fitted['max_corner_working_px'] < 1e-6
