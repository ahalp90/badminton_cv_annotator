"""Focused unit checks for the W5 evidence and ranking helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from verifier import (
    permutation_determinism,
    photometric_samples,
    rank_candidates,
)

from experiments.annotator.independent_court import junction_observations
from experiments.annotator.independent_court.assignment import prepare_observations
from experiments.annotator.independent_court.detector import SEGMENTS_M
from experiments.annotator.independent_court.paint_geometry import CENTRE_SEGMENTS_M


def candidate(origin_key: str, score: float, source_order: int, origin_index: int) -> dict:
    return {
        "origin_key": origin_key,
        "source_order": source_order,
        "origin_index": origin_index,
        "kind_order": 0,
        "hard_valid": True,
        "historical": {"historical_fullcourt": True, "historical_camera": True},
        "evidence": {
            "q_geom": score,
            "q_paint10": score,
            "exclusive_reverse": score,
        },
    }


def test_ranker_uses_stable_origin_order_for_ties() -> None:
    candidates = [
        candidate("G1:1", 0.8, 1, 1),
        candidate("G0:2", 0.8, 0, 2),
        candidate("G0:1", 0.8, 0, 1),
    ]
    ranking = rank_candidates(candidates)
    assert ranking["provisional_rank"] == ["G0:1", "G0:2", "G1:1"]
    assert ranking["criterion"] == "q_paint10"
    assert permutation_determinism(candidates)["match"]


def test_ranker_reports_sparse_evidence_without_a_gate() -> None:
    sparse = candidate("G0:0", 0.5, 0, 0)
    sparse["evidence"]["q_paint10"] = None
    sparse["evidence"]["q_geom"] = None
    result = rank_candidates([sparse])
    assert result["status"] == "evidence_sparse"
    assert result["selected_origin_key"] is None
    assert result["historical_camera_subset_rank"] == []


def test_photometry_keeps_raw_contrast_and_masks_unknown_samples() -> None:
    image = np.full((80, 120, 3), 20, dtype=np.uint8)
    image[40, 20:101] = 200
    samples = np.column_stack((np.linspace(20, 100, 17), np.full(17, 40.0)))
    contrast, p10 = photometric_samples(image, samples, np.array([1.0, 0.0]), np.empty((0, 4)))
    assert np.all(contrast > 100)
    assert np.all(p10)
    masked_contrast, masked_p10 = photometric_samples(
        image, samples, np.array([1.0, 0.0]), np.array([[0.0, 0.0, 119.0, 79.0]])
    )
    assert np.all(np.isnan(masked_contrast))
    assert not masked_p10.any()


def test_junction_helpers_accept_physical_centres() -> None:
    shifted = SEGMENTS_M.copy()
    shifted[[2, 3], :, 1] += 2.0
    assert junction_observations.expected_vertical_arms(0.0, shifted) == {
        "far": False,
        "near": False,
    }
    assert junction_observations.measure(
        np.eye(3),
        prepare_observations(np.empty((0, 4)), (960, 540)),
        np.empty((0, 4)),
        (960, 540),
        centres=CENTRE_SEGMENTS_M,
    )["usable_sites"] == 0
