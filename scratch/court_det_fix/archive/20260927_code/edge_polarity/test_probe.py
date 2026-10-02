"""Check paint-side orientation against a synthetic bright stripe."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.annotator.independent_court import detector, fixed_stripe_refit
from scratch.court_det_fix.edge_polarity import run_probe as probe
from scratch.court_det_fix.w5_holistic import verifier


def stripe_context(horizontal: bool, reversed_endpoints: bool) -> tuple[SimpleNamespace, np.ndarray]:
    frame = np.full((100, 100, 3), 20, dtype=np.uint8)
    frame[:, 20:25] = 220
    samples = np.stack([np.column_stack((np.full(16, x), np.linspace(10, 90, 16))) for x in (20, 24, 22)])
    homography = np.array([[100., 0., 20.], [0., 100., 0.], [0., 0., 1.]])
    if horizontal:
        frame = frame.transpose(1, 0, 2).copy()
        samples = samples[..., ::-1].copy()
        homography = homography[[1, 0, 2]]
    if reversed_endpoints:
        samples = samples[:, ::-1].copy()
    segments = samples[:, [0, -1]]
    directions = segments[:, 1] - segments[:, 0]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    observations = SimpleNamespace(samples=samples, segments=segments, directions=directions, fragment_ids=np.arange(3))
    context = SimpleNamespace(frame=frame, size=(100, 100), mask_boxes=np.empty((0, 4)), observations=observations)
    return context, homography


@pytest.mark.parametrize("horizontal", [False, True])
@pytest.mark.parametrize("reversed_endpoints", [False, True])
@pytest.mark.parametrize("court_axis", [0, 1])
def test_bright_side_survives_orientation(horizontal: bool, reversed_endpoints: bool, court_axis: int) -> None:
    context, homography = stripe_context(horizontal, reversed_endpoints)
    if court_axis == 1:
        homography = homography[:, [1, 0, 2]]
    with probe.prepared_measurements(verifier) as counts:
        normals, contrast, valid_pairs = probe.brightness_profiles(context, verifier)
    expected = probe.expected_bright_side(
        homography, context.observations.segments.mean(axis=1), normals,
        np.full(3, 5 if court_axis else 0), np.array([1, 2, 0]), detector, .04,
    )
    np.testing.assert_array_equal(np.sign(contrast[:2, 0]), expected[:2])
    np.testing.assert_array_equal(valid_pairs, 14)
    assert contrast[2, 0] == 0
    assert counts["greyscale_conversions"] == 1


def test_relabels_inner_edge_without_moving_samples() -> None:
    context, homography = stripe_context(False, True)
    positions = np.array([1, 1, 0])
    constraints = fixed_stripe_refit.Constraints(
        points=context.observations.segments.mean(axis=1), intervals=np.zeros(3, dtype=int),
        positions=positions, weights=np.ones(3), fragment_ids=np.arange(3), sample_ids=np.zeros(3, dtype=int),
    )
    parent = {"homography_working": homography,
              "evidence": {"stripe_assignments": {"position": positions, "marking": np.zeros(3, dtype=int)}}}
    with probe.prepared_measurements(verifier):
        changed, _, metadata = probe.relabel(context, parent, constraints, verifier)
    np.testing.assert_array_equal(changed.positions, [1, 2, 0])
    np.testing.assert_array_equal(constraints.positions, [1, 1, 0])
    assert changed.points is constraints.points
    assert changed.weights is constraints.weights
    assert changed.intervals is constraints.intervals
    assert metadata["changed_fragment_count"] == 1


def test_masked_paint_leaves_assignment_unresolved() -> None:
    context, homography = stripe_context(False, False)
    context.mask_boxes = np.array([[0., 0., 100., 100.]])
    positions = np.ones(3, dtype=int)
    constraints = fixed_stripe_refit.Constraints(
        points=context.observations.segments.mean(axis=1), intervals=np.zeros(3, dtype=int),
        positions=positions, weights=np.ones(3), fragment_ids=np.arange(3), sample_ids=np.zeros(3, dtype=int),
    )
    parent = {"homography_working": homography,
              "evidence": {"stripe_assignments": {"position": positions, "marking": np.zeros(3, dtype=int)}}}
    with probe.prepared_measurements(verifier):
        changed, rows, _ = probe.relabel(context, parent, constraints, verifier)
    np.testing.assert_array_equal(changed.positions, positions)
    assert all(not row["strong_polarity"] for row in rows)
