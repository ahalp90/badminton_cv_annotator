"""Synthetic checks for the finite-marking diagnostic adapter."""

import cv2
import numpy as np
from run_diagnosis import finite_profiles

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from experiments.annotator.independent_court import stripe_observations as stripes


def test_frozen_identity_refit_and_profiles() -> None:
    size = (960, 540)
    truth = np.array([[300, 80], [660, 80], [840, 500], [120, 500]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, truth)
    segments, _ = detector.project(transform[None], detector.SEGMENTS_M)
    observations = assignment.prepare_observations(segments.reshape(-1, 4), size)
    weights = stripes.fragment_weights(observations)
    start = truth + np.array([[1., 1.], [-1., 1.], [-1., -1.], [1., -1.]])
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, start.astype(np.float32))
    evidence = stripes.measure(homography, observations, size)
    selected = stripes.score_model(evidence, weights, 1)['assignments']
    constraints = fitting.prepare(homography, observations, selected, weights)
    saved_ids = constraints.fragment_ids.copy()
    fitted = fitting.refine(start, constraints, size, False)
    assert fitted['successful'], fitted
    assert fitted['objective_after'] < fitted['objective_before'] * .01
    assert np.linalg.norm(np.asarray(fitted['corners_px']) - truth, axis=1).max() < .01
    np.testing.assert_array_equal(saved_ids, constraints.fragment_ids)
    assert 2 in constraints.intervals and 3 in constraints.intervals
    profiles = finite_profiles(transform, observations, size)
    assert len(profiles) == 12
    assert all(max(profile['distance_working_px']) < 1e-4 for profile in profiles)


def test_continuous_shortlist_distinguishes_displaced_far_end() -> None:
    from scan_population import continuous_support, geometry

    size = (960, 540)
    truth = np.array([[300, 80], [660, 80], [840, 500], [120, 500]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, truth)
    segments, _ = detector.project(transform[None], detector.SEGMENTS_M)
    maps = detector._distance_maps(detector._wide_line_families(segments.reshape(-1, 4)), size)
    displaced = truth.copy()
    displaced[:2, 1] += 8
    wrong = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, displaced)
    scores = continuous_support(np.stack([transform, wrong]), maps, size)
    assert scores[0] > scores[1] + .05
    valid, _ = geometry(np.stack([transform, wrong]), size)
    assert valid.all()


def test_vector_retention_matches_original_with_duplicates_and_ties() -> None:
    from dataclasses import replace

    from scan_population import retain

    random = np.random.default_rng(20260914)
    centres = random.normal(size=(20, 4, 2)) * 10
    candidates = [detector.Candidate(centres[index % 20] + random.normal(size=(4, 2)) * .1,
                                     float(index % 7), (0., 0.), (0, 0)) for index in range(100)]
    settings = replace(detector.DEFAULT_SETTINGS, keep_candidates=16, distinct_corner_distance=2)
    expected = detector._retain([], candidates, settings)
    actual = retain(candidates, settings)
    assert [id(candidate) for candidate in actual] == [id(candidate) for candidate in expected]
