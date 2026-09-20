"""Focused unit checks for the W5 evidence and ranking helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from run_w5 import ViewAmbiguity, canonicalise_populations
from verifier import (
    legacy_winners,
    permutation_determinism,
    photometric_samples,
    rank_candidates,
    raw_junctions,
)

from experiments.annotator.independent_court import detector, junction_observations
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
        "gates": {
            "camera_error": 0.05,
            "geometry_valid": True,
            "player_fractions": [1.0, 1.0],
            "family_support": [0.5, 0.5],
            "floor_score": 0.0,
            "line_counts": [1, 1],
        },
        "historical": {"historical_fullcourt": True, "historical_camera": True},
        "evidence": {
            "q_geom": score,
            "q_paint10": score,
            "q_geom_span_weighted": score,
            "q_paint10_span_weighted": score,
            "exclusive_reverse": score,
        },
    }


def population_entry(
    candidate_id: str,
    homography_offset: float = 0.0,
    profile_score: float = 0.5,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "corners_px": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        "homography_working": [
            [1.0, 0.0, homography_offset],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gates": {"camera_error": 0.05},
        "profile": {"score": profile_score},
        "stripe": {"exclusive": {"score": profile_score, "reverse": profile_score}},
    }


def test_source_qualified_identity_keeps_distinct_raw_id_collisions() -> None:
    records, resolution = canonicalise_populations(
        [population_entry("0:7")],
        [population_entry("0:7", homography_offset=1.0)],
    )

    assert [record["origin_key"] for record in records] == ["G0:0:7", "G1:0:7"]
    assert resolution["raw_id_collisions"] == ["0:7"]
    assert resolution["duplicate_group_count"] == 0
    by_origin = {record["origin_key"]: record for record in records}
    assert len(by_origin) == 2
    assert by_origin["G0:0:7"]["source_memberships"] == ["G0"]
    assert by_origin["G1:0:7"]["source_memberships"] == ["G1"]


def test_same_source_raw_id_collision_stops() -> None:
    with pytest.raises(ViewAmbiguity, match="candidate IDs are not unique"):
        canonicalise_populations([population_entry("0:7"), population_entry("0:7")], [])


def test_exact_geometry_duplicate_preserves_occurrences() -> None:
    records, resolution = canonicalise_populations(
        [population_entry("0:7")],
        [population_entry("3:99", profile_score=0.9)],
    )

    assert len(records) == 1
    assert records[0]["origin_key"] == "G0:0:7"
    assert records[0]["source_memberships"] == ["G0", "G1"]
    assert records[0]["occurrence_count"] == 2
    assert [item["candidate_id"] for item in records[0]["source_occurrences"]] == ["0:7", "3:99"]
    ranking = legacy_winners(records)
    assert ranking["eligible_count"] == 2
    assert ranking["paint"] == "G0:0:7"
    assert ranking["paint_occurrence_key"] == "G1:3:99"
    assert records[0]["_legacy_occurrences"][0]["parent_origin_key"] == "G0:0:7"
    assert resolution["duplicate_group_count"] == 1


def test_exact_geometry_duplicate_keeps_legacy_only_gate_differences() -> None:
    left = population_entry("0:7", profile_score=0.1)
    right = population_entry("3:99", profile_score=0.9)
    right["gates"].update({"family_support": [0.1, 0.2], "floor_score": -1.0, "line_counts": [2, 0]})

    records, _ = canonicalise_populations([left], [right])

    assert len(records) == 1
    assert records[0]["origin_key"] == "G0:0:7"
    assert records[0]["_legacy_occurrences"][1]["candidate_id"] == "3:99"
    assert records[0]["_legacy_occurrences"][1]["gates"] == right["gates"]
    ranking = legacy_winners(records)
    assert ranking["paint"] == "G0:0:7"
    assert ranking["paint_occurrence_key"] == "G1:3:99"
    assert ranking["paint_parent_origin_key"] == "G0:0:7"


def test_source_qualified_identity_set_is_order_independent() -> None:
    g0 = [population_entry("0:7"), population_entry("0:8", homography_offset=1.0)]
    g1 = [population_entry("0:7", homography_offset=2.0), population_entry("0:8", homography_offset=3.0)]
    records, _ = canonicalise_populations(g0, g1)
    permuted, _ = canonicalise_populations(list(reversed(g0)), list(reversed(g1)))

    assert sorted(record["origin_key"] for record in records) == sorted(
        record["origin_key"] for record in permuted
    )


def test_unexpected_duplicate_geometry_stops_instead_of_shrinking_pool() -> None:
    with pytest.raises(AssertionError, match="duplicate geometry"):
        canonicalise_populations([population_entry("0:7"), population_entry("0:8")], [])


def test_cross_source_duplicate_with_differing_w5_gates_stops() -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["gates"] = {"camera_error": 0.06}

    with pytest.raises(ViewAmbiguity, match="differing W5 gates"):
        canonicalise_populations([left], [right])


def test_cross_source_duplicate_with_differing_pair_metadata_stops() -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["pair_id"] = 4

    with pytest.raises(ViewAmbiguity, match="differing pair_id"):
        canonicalise_populations([left], [right])


def test_cross_source_duplicate_with_differing_corners_stops() -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["corners_px"][0][0] = 1.0

    with pytest.raises(ViewAmbiguity, match="differing corners"):
        canonicalise_populations([left], [right])


def test_merged_legacy_ties_keep_original_source_order() -> None:
    g0 = [population_entry("0:7"), population_entry("0:8", homography_offset=1.0)]
    g1 = [population_entry("3:99")]

    records, _ = canonicalise_populations(g0, g1)
    ranking = legacy_winners(records)

    assert ranking["paint"] == "G0:0:7"
    assert ranking["paint_occurrence_key"] == "G0:0:7"


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
    sparse["evidence"]["q_paint10_span_weighted"] = None
    sparse["evidence"]["q_geom_span_weighted"] = None
    result = rank_candidates([sparse])
    assert result["status"] == "evidence_sparse"
    assert result["selected_origin_key"] is None
    assert result["historical_camera_subset_rank"] == []


def test_ranker_applies_camera_limit_and_span_weighting() -> None:
    short_marking = candidate("G0:short", 0.8, 0, 0)
    long_marking = candidate("G0:long", 0.7, 0, 1)
    short_marking["evidence"].update({"q_paint10": 0.9, "q_paint10_span_weighted": 0.2})
    long_marking["evidence"].update({"q_paint10": 0.8, "q_paint10_span_weighted": 0.8})
    ranking = rank_candidates([short_marking, long_marking])
    assert ranking["r1_paint10_rank"] == ["G0:short", "G0:long"]
    assert ranking["r2_spanw_paint10_rank"] == ["G0:long", "G0:short"]
    assert ranking["provisional_rank"] == ["G0:long", "G0:short"]

    no_camera = candidate("G0:no-camera", 0.9, 0, 2)
    no_camera["gates"]["camera_error"] = 0.11
    result = rank_candidates([no_camera])
    assert result["status"] == "no_plausible_camera"
    assert result["selected_origin_key"] is None
    assert result["ungated_provisional_rank"] == ["G0:no-camera"]


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


def test_raw_junctions_use_projected_arm_direction() -> None:
    homography = cv2.getPerspectiveTransform(
        detector.CORNER_COURT_M.astype(np.float32),
        np.array([[100, 100], [850, 250], [780, 500], [180, 450]], dtype=np.float32),
    )
    physical = CENTRE_SEGMENTS_M
    junction_m = np.array([physical[2, 0, 0], physical[6, 0, 1]])
    court_arm = np.array([1.0, 0.0])
    along = np.linspace(
        junction_observations.ARM_START_M,
        junction_observations.ARM_END_M,
        junction_observations.ARM_SAMPLES,
    )
    court_samples = junction_m + along[:, None] * court_arm
    projected_samples, _ = detector.project(homography[None], court_samples)
    projected_samples = projected_samples[0]
    projected_direction = projected_samples[-1] - projected_samples[0]
    projected_direction /= np.linalg.norm(projected_direction)
    assert np.degrees(np.arccos(np.clip(court_arm @ projected_direction, -1.0, 1.0))) > 5.0

    observations = prepare_observations(
        projected_samples[[0, -1]].reshape(1, 4),
        (960, 540),
    )
    context = SimpleNamespace(
        observations=observations,
        mask_boxes=np.empty((0, 4)),
        size=(960, 540),
        frame=np.full((540, 960, 3), 20, dtype=np.uint8),
        same_image_mask_available=False,
    )
    result = raw_junctions(context, homography, {"exclusive": {"reverse": 0.0}})

    support = result["sites"][0]["arms"]["right"]["positions"][0]["fragment_support_mean"]
    assert support is not None
    assert support > 0.99
