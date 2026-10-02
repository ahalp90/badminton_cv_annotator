"""Synthetic geometry and original-rectangle compatibility checks."""

import json
from dataclasses import replace

import cv2
import numpy as np
from run_score import corner_summary
from vp_pruning import (
    Settings,
    angular_residuals,
    estimate,
    normalisation,
    rectangle_population,
    retain_pencils,
    select,
)

from experiments.annotator.independent_court import detector


def grid() -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    offsets = (10, 50, 90)
    families = (np.array([[1., 0, -offset] for offset in offsets]),
                np.array([[0., 1, -offset] for offset in offsets]))
    segments = np.array([[offset, 0, offset, 100] for offset in offsets]
                        + [[0, offset, 100, offset] for offset in offsets], dtype=float)
    return families, segments


def test_finite_infinite_and_coordinate_change() -> None:
    lines = np.array([[1., 0, -20], [0, 1, -30], [1, 1, -50]])
    point = np.array([[20., 30, 1]])
    np.testing.assert_allclose(angular_residuals(lines, point), 0, atol=1e-6)
    infinite = np.array([[1., 0, 0]])
    np.testing.assert_allclose(angular_residuals(lines[1:2], infinite), 0, atol=1e-6)
    # The measurement point transforms with the image centre under resizing.
    transform = normalisation((100, 100))
    centred_lines = lines @ transform
    centred_point = point @ np.linalg.inv(transform).T
    np.testing.assert_allclose(angular_residuals(centred_lines, centred_point), 0, atol=1e-6)
    np.testing.assert_allclose(angular_residuals(centred_lines, -centred_point), 0, atol=1e-6)
    assert angular_residuals(np.array([[1., 0, 0]]), np.array([[0., 0, 1]]))[0, 0] == 90


def test_estimation_parallel_lines_and_duplicate_edges() -> None:
    _, segments = grid()
    points, details = estimate(segments, (100, 100), Settings())
    transform = normalisation((100, 100))
    families, _ = grid()
    residuals = [angular_residuals(lines @ transform, points @ np.linalg.inv(transform).T) for lines in families]
    assert all(np.any(np.max(angles, axis=1) < 1e-5) for angles in residuals)
    _, repeated = estimate(np.repeat(segments, 3, axis=0), (100, 100), Settings())
    assert repeated['merged_direction_count'] == details['merged_direction_count'] == 6
    assert max(repeated['support_counts']) == max(details['support_counts'])


def test_original_rectangle_equality_and_exchanged_roles() -> None:
    families, _ = grid()
    points = np.array([[0., 1, 0], [1, 0, 0]])
    settings = detector.DEFAULT_SETTINGS
    quads, _, ids = rectangle_population(families, (100, 100))
    original = detector._image_rectangles(*families, (100, 100), settings)
    reconstructed = np.asarray([cv2.getPerspectiveTransform(detector.UNIT_CORNERS, quad.astype(np.float32))
                                for quad in quads])
    np.testing.assert_array_equal(original, reconstructed)
    result, details = select(families, points, (100, 100), settings, Settings())
    other, swapped = select(families, points[::-1], (100, 100), settings, Settings())
    assert sorted(details['retained_pair_product_ids']) == ids.tolist()
    for matrices, record in ((result, details), (other, swapped)):
        order = np.argsort(record['retained_pair_product_ids'])
        np.testing.assert_array_equal(matrices[order], original)


def test_estimate_finite_pencils_with_missing_lines_and_clutter() -> None:
    families, segments = grid()
    homography = np.array([[1., .1, 10], [.1, 1, 10], [.001, .002, 1]])
    homogeneous = np.column_stack((segments.reshape(-1, 2), np.ones(segments.size // 2)))
    projected = homogeneous @ homography.T
    projected = (projected[:, :2] / projected[:, 2:]).reshape(-1, 4)
    clutter = np.array([[5., 15, 60, 45], [20, 95, 90, 70]])
    points, _ = estimate(np.concatenate((projected[:-1], clutter)), (120, 120), Settings())
    transform = normalisation((120, 120))
    for lines in (families[0], families[1][:2]):
        image_lines = lines @ np.linalg.inv(homography)
        angles = angular_residuals(image_lines @ transform, points @ np.linalg.inv(transform).T)
        assert np.any(np.max(angles, axis=1) < 1e-4)


def test_budget_and_fallback_match_original() -> None:
    families, _ = grid()
    points = np.array([[0., 1, 0], [1, 0, 0]])
    _, limited = select(families, points, (100, 100), detector.DEFAULT_SETTINGS, Settings(rectangles=3))
    assert len(set(limited['selected_pair_product_ids'])) == 3
    assert limited['budget_excluded'] == 6
    settings = replace(detector.DEFAULT_SETTINGS, max_rectangles=3)
    result, details = select(families, np.empty((0, 3)), (100, 100), settings, Settings())
    assert details['fallback']
    assert details['budget_excluded'] == 6
    np.testing.assert_array_equal(result, detector._image_rectangles(*families, (100, 100), settings))


def test_coverage_preserves_a_second_direction_under_crowding() -> None:
    masks = np.array([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 1]], dtype=bool)
    counts = masks.sum(axis=1)
    supports = counts.astype(float)
    ids = np.arange(len(masks))
    ranked, _ = retain_pencils(masks, counts, supports, ids, Settings(pencils=2))
    covered, _ = retain_pencils(masks, counts, supports, ids, Settings(pencils=2, pencil_selection='coverage'))
    assert ranked == [0, 1]
    assert covered == [0, 2]


def test_unmeasured_trace_rows_serialise_without_numeric_sentinels() -> None:
    dtype = [('generation_id', 'i8'), ('corners_px', 'f8', (4, 2)), ('one_fraction', 'f8'),
             ('two_fraction', 'f8'), ('floor_score', 'f8'), ('camera_error', 'f8'),
             ('family_support', 'f8', (2,)), ('line_counts', 'i8', (2,))]
    rows = np.zeros(1, dtype=dtype)
    rows['one_fraction'] = .5
    rows['floor_score'] = np.nan
    rows['camera_error'] = np.nan
    rows['family_support'] = np.nan
    rows['line_counts'] = -1
    reference = {'corners_px': np.zeros((4, 2)).tolist()}
    record = corner_summary({}, rows, reference)
    example = record['examples'][0]
    assert example['line_counts'] is None
    assert example['original_floor_score'] is None
    assert example['camera_error'] is None
    json.dumps(record, allow_nan=False)
    empty = corner_summary({}, rows[:0], reference)
    assert empty['counts']['geometry_before_players'] == 0
    assert empty['examples'] == []
