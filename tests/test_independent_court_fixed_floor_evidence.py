"""Check observation access without changing court geometry or floor thresholds."""

import numpy as np

from experiments.annotator.independent_court import detector
from experiments.annotator.independent_court import fixed_floor_evidence as evidence

SIZE = (600, 300)
HOMOGRAPHY = np.array([[5.0, -35, 540], [35.0, 0, 20], [0, 0, 1]])
SETTINGS = detector.Settings(wide_families=True, min_supported_lines=3)


def test_exact_oblique_court_recovers_hidden_directional_support() -> None:
    fragments = detector.project(HOMOGRAPHY[None], detector.SEGMENTS_M)[0].reshape(-1, 4)
    observations = evidence.prepare(fragments, SIZE, SETTINGS)
    original = fragments.copy()
    result = evidence.measure(HOMOGRAPHY, observations, SIZE, SETTINGS)
    assert result["original_score_control_exact"]
    assert result["arms"]["finite_family"]["floor_score"] == -1
    directional = result["arms"]["finite_projected"]
    assert directional["family_means"] == [1, 1]
    assert directional["distinct_line_counts"] == [5, 6]
    assert directional["floor_score"] == 1
    np.testing.assert_array_equal(fragments, original)


def test_crossing_fragments_cannot_supply_parallel_support() -> None:
    projected = detector.project(HOMOGRAPHY[None], detector.SEGMENTS_M)[0].reshape(-1, 2, 2)
    samples, _ = detector._visible_samples(projected[None], SIZE, SETTINGS.samples_per_line)
    points = samples[0, 0]
    crossing = np.stack((points - [0, 3], points + [0, 3]), axis=1)
    fragments = np.concatenate((projected[1:], crossing)).reshape(-1, 4)
    result = evidence.measure(HOMOGRAPHY, evidence.prepare(fragments, SIZE, SETTINGS), SIZE, SETTINGS)
    assert result["arms"]["finite_all"]["interval_support"][0] == 1
    assert result["arms"]["finite_projected"]["interval_support"][0] == 0


def test_finite_fragment_does_not_support_its_unobserved_extension() -> None:
    projected = detector.project(HOMOGRAPHY[None], detector.SEGMENTS_M)[0].reshape(-1, 2, 2)
    projected[0, 1] = projected[0, 0] + .1 * (projected[0, 1] - projected[0, 0])
    fragments = projected.reshape(-1, 4)
    result = evidence.measure(HOMOGRAPHY, evidence.prepare(fragments, SIZE, SETTINGS), SIZE, SETTINGS)
    support = result["arms"]["finite_projected"]["interval_support"][0]
    assert 0 < support < .2
