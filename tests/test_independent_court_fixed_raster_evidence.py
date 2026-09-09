"""Check raster preservation and independent evidence-access arms."""

import numpy as np

from experiments.annotator.independent_court import detector
from experiments.annotator.independent_court import fixed_raster_evidence as evidence

SIZE = (600, 300)
HOMOGRAPHY = np.array([[5.0, -35, 540], [35.0, 0, 20], [0, 0, 1]])
SETTINGS = detector.Settings(wide_families=True, min_supported_lines=3)


def test_oblique_court_and_independent_access_arms() -> None:
    fragments = detector.project(HOMOGRAPHY[None], detector.SEGMENTS_M)[0].reshape(-1, 4)
    result = evidence.measure(HOMOGRAPHY, evidence.prepare(fragments, SIZE, SETTINGS), SIZE, SETTINGS)
    arms = result["arms"]
    assert result["original_score_control_exact"]
    assert arms["family_family"]["floor_score"] == -1
    assert arms["projected_all"]["floor_score"] == 1
    for access in ("family", "projected"):
        assert arms[f"{access}_family"]["covered_samples"] == arms[f"{access}_all"]["covered_samples"]
        assert arms[f"{access}_family"]["family_means"] == arms[f"{access}_all"]["family_means"]
    for pool in ("family", "all"):
        assert arms[f"family_{pool}"]["nearest_line_ids"] == arms[f"projected_{pool}"]["nearest_line_ids"]


def test_crossing_fragments_are_withheld_from_projected_raster() -> None:
    projected = detector.project(HOMOGRAPHY[None], detector.SEGMENTS_M)[0].reshape(-1, 2, 2)
    samples, _ = detector._visible_samples(projected[None], SIZE, SETTINGS.samples_per_line)
    points = samples[0, 0]
    crossing = np.stack((points - [0, 3], points + [0, 3]), axis=1)
    fragments = np.concatenate((projected[1:], crossing)).reshape(-1, 4)
    result = evidence.measure(HOMOGRAPHY, evidence.prepare(fragments, SIZE, SETTINGS), SIZE, SETTINGS)
    assert result["arms"]["all_all"]["interval_support"][0] == 1
    assert result["arms"]["projected_all"]["interval_support"][0] == 0


def test_projected_access_preserves_raster_rounding_at_support_boundary() -> None:
    homography = np.array([[30., 0, 20.9], [0, 15, 30.9], [0, 0, 1]])
    # The analytic distance is 4.5 pixels, but the original raster convention
    # measures from sample pixel x=20 to rounded observed pixel x=16: four pixels.
    fragments = np.array([[16.4, 20, 16.4, 250], [0, 280, 500, 280]])
    settings = detector.Settings(wide_families=True, min_supported_lines=1)
    result = evidence.measure(homography, evidence.prepare(fragments, SIZE, settings), SIZE, settings)
    assert result["original_score_control_exact"]
    for name in evidence.ARMS:
        assert result["arms"][name]["interval_support"][0] == 1
        assert result["arms"][name]["sample_raster_distances_px"][0] == [4.] * settings.samples_per_line
