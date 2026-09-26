"""Synthetic checks for merge membership and anchor-dependent residuals (E0 and E1)."""

from dataclasses import replace

import numpy as np
import pytest
import vp_pruning
from membership import (
    MERGE_INPUT_CAP,
    angular_residuals_at,
    line_feet,
    longest_member,
    merge_lines_with_membership,
    project_onto_lines,
    residual_matrix,
    to_normalised,
    to_working,
)

from experiments.annotator.independent_court import detector

WORKING_SIZE = (960, 540)


def synthetic_fragments(seed: int, count: int) -> np.ndarray:
    """Fragments clustered on a few lines with jitter, plus scattered noise, so merging groups them."""
    generator = np.random.default_rng(seed)
    anchors = generator.uniform((0, 0), WORKING_SIZE, size=(6, 2))
    angles = generator.uniform(0, np.pi, size=6)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    rows = []
    for _ in range(count):
        if generator.uniform() < .75:
            line = generator.integers(6)
            start = anchors[line] + directions[line] * generator.uniform(-300, 300)
            start += generator.normal(scale=1.5, size=2)
            end = start + directions[line] * generator.uniform(15, 200)
        else:
            start = generator.uniform((0, 0), WORKING_SIZE)
            end = start + generator.uniform(-120, 120, size=2)
        rows.append(np.concatenate((start, end)))
    segments = np.asarray(rows)
    return segments[np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1) > 1]


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('count', [40, 360])
@pytest.mark.parametrize('cap', [300, 5])
def test_instrumented_merge_matches_detector(seed: int, count: int, cap: int) -> None:
    segments = synthetic_fragments(seed, count)
    settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=cap)
    expected = detector._merge_lines(segments, settings)
    lines, sidecar = merge_lines_with_membership(segments, settings)
    assert np.array_equal(expected, lines)
    assert len(sidecar) == len(lines)
    members = [member for entry in sidecar for member in entry['member_observation_ids']]
    assert len(members) == len(set(members))
    lengths = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1)
    merged_rows = set(np.argsort(-lengths, kind='stable')[:MERGE_INPUT_CAP].tolist())
    assert set(members) <= merged_rows
    if cap >= len(segments):
        assert set(members) == merged_rows
    for entry in sidecar:
        member_lengths = lengths[entry['member_observation_ids']]
        assert np.all(np.diff(member_lengths) <= 0), 'members must arrive in the merge loop order'


def test_anchor_at_foot_reproduces_angular_residuals() -> None:
    generator = np.random.default_rng(3)
    lines = generator.normal(size=(7, 3))
    points = generator.normal(size=(20, 3))
    points[5:8, 2] = 0.
    expected = vp_pruning.angular_residuals(lines, points)
    assert np.array_equal(angular_residuals_at(lines, points, line_feet(lines)), expected)
    assert np.array_equal(residual_matrix(lines, points, line_feet(lines), 6), expected)


def test_line_scale_sign_and_candidate_sign_preserve_angles() -> None:
    generator = np.random.default_rng(4)
    lines = generator.normal(size=(7, 3))
    midpoints = generator.normal(size=(7, 2))
    points = generator.normal(size=(20, 3))
    anchors = project_onto_lines(midpoints, lines)
    base = angular_residuals_at(lines, points, anchors)
    scales = generator.uniform(.1, 5., size=7) * generator.choice([-1., 1.], size=7)
    scaled_lines = lines * scales[:, None]
    scaled_anchors = project_onto_lines(midpoints, scaled_lines)
    np.testing.assert_allclose(scaled_anchors, anchors, atol=1e-9)
    np.testing.assert_allclose(angular_residuals_at(scaled_lines, points, scaled_anchors), base, atol=1e-9)
    signs = generator.choice([-1., 1.], size=(20, 1))
    np.testing.assert_allclose(angular_residuals_at(lines, points * signs, anchors), base, atol=1e-9)


def test_points_on_line_give_zero_except_the_coincident_case() -> None:
    line = np.array([[1., 2., -3.]])  # x + 2y = 3; tangent (2, -1)
    anchor = np.array([[1., 1.]])
    on_line = np.array([[5., -1., 1.]])
    at_anchor = np.array([[1., 1., 1.]])
    along = np.array([[2., -1., 0.]])
    across = np.array([[1., 2., 0.]])
    residuals = angular_residuals_at(line, np.concatenate((on_line, at_anchor, along, across)), anchor)
    np.testing.assert_allclose(residuals[:, 0], [0., 90., 0., 90.], atol=1e-12)


def test_infinity_directions_ignore_the_anchor() -> None:
    generator = np.random.default_rng(5)
    lines = generator.normal(size=(9, 3))
    points = generator.normal(size=(12, 3))
    points[:, 2] = 0.
    moved = angular_residuals_at(lines, points, generator.normal(size=(9, 2)) * 50)
    assert np.array_equal(moved, angular_residuals_at(lines, points, line_feet(lines)))


def test_longest_member_breaks_equal_lengths_by_observation_index() -> None:
    lengths = np.array([5., 9., 9., 3.])
    assert longest_member([3, 2, 1, 0], lengths) == 1
    assert longest_member([2, 1], lengths) == 1
    assert longest_member([0, 3], lengths) == 0
    assert longest_member([2], lengths) == 2


def test_projected_anchor_lies_on_the_line_in_both_frames() -> None:
    generator = np.random.default_rng(6)
    lines = generator.normal(size=(8, 3)) * np.array([1., 1., 200.])
    midpoints = generator.uniform((0, 0), WORKING_SIZE, size=(8, 2))
    anchors = project_onto_lines(midpoints, lines)
    residual = np.einsum('li,li->l', lines[:, :2], anchors) + lines[:, 2]
    np.testing.assert_allclose(residual, 0., atol=1e-9)
    normals = lines[:, :2] / np.linalg.norm(lines[:, :2], axis=1)[:, None]
    displacement = anchors - midpoints
    np.testing.assert_allclose(displacement[:, 0] * normals[:, 1] - displacement[:, 1] * normals[:, 0], 0., atol=1e-9)
    transform = vp_pruning.normalisation(WORKING_SIZE)
    normalised = to_normalised(anchors, transform)
    normalised_lines = lines @ transform
    np.testing.assert_allclose(np.einsum('li,li->l', normalised_lines[:, :2], normalised) + normalised_lines[:, 2], 0., atol=1e-9)
    np.testing.assert_allclose(to_working(normalised, transform), anchors, atol=1e-9)
