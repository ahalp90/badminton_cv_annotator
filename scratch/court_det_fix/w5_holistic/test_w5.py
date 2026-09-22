"""Focused unit checks for the W5 evidence and ranking helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from line_template_source import (
    VISIBILITY_COLUMN_LABELS,
    _empty_metadata,
    attach_w5_gates,
    geometry_and_support,
    select_with_visibility_floor,
    union_distance_map,
    vector_camera_errors,
    visibility_eligible,
)
from render_gallery import load_native_frame, write_index
from run_w5 import (
    VISIBILITY_FLOOR_ARMS,
    ViewAmbiguity,
    canonicalise_populations,
    load_g0,
    parse_args,
    previous_stage5_anchors,
    run_pilot,
    validate_generation_record,
    validate_visibility_floor,
    validate_visibility_floors,
)
from verifier import (
    CASE_ORDER,
    CASE_PACKS,
    REGRESSION_CASE_ORDER,
    frame_path,
    has_same_image_boxes,
    legacy_winners,
    mask_boxes_working,
    permutation_determinism,
    photometric_samples,
    prepare_view,
    rank_candidates,
    raw_junctions,
    source_provenance,
)

from experiments.annotator.independent_court import detector, junction_observations
from experiments.annotator.independent_court.assignment import prepare_observations
from experiments.annotator.independent_court.case_provenance import (
    CaseProvenance,
    ImageKind,
    load_frozen_case_provenance,
)
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


def test_same_source_duplicate_geometry_preserves_both_occurrences() -> None:
    records, _ = canonicalise_populations([population_entry("0:7"), population_entry("0:8")], [])
    assert len(records) == 1
    assert records[0]["occurrence_count"] == 2
    assert records[0]["source_memberships"] == ["G0"]
    assert len(records[0]["_legacy_occurrences"]) == 2


@pytest.mark.parametrize("camera_error", [0.06, 0.10000000001, 3.099690687334811])
def test_cross_source_duplicate_with_differing_w5_gates_is_retained(camera_error) -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["gates"] = {**left["gates"], "camera_error": camera_error}

    records, resolution = canonicalise_populations([left], [right])
    assert [record["origin_key"] for record in records] == ["G0:0:7", "G1:3:99"]
    assert records[1]["entry"]["gates"]["camera_error"] == camera_error
    assert records[1]["source_occurrences"][0]["w5_gates"]["camera_error"] == camera_error
    assert resolution["conflicting_geometry_groups"] == [["G0:0:7", "G1:3:99"]]
    assert legacy_winners(records)["line"] == "G0:0:7"


def test_cross_source_duplicate_with_differing_pair_metadata_is_retained() -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["pair_id"] = 4

    records, _ = canonicalise_populations([left], [right])
    assert len(records) == 2


def test_cross_source_duplicate_with_differing_corners_is_retained() -> None:
    left = population_entry("0:7")
    right = population_entry("3:99")
    right["corners_px"][0][0] = 1.0

    records, _ = canonicalise_populations([left], [right])
    assert len(records) == 2


def test_merged_legacy_ties_keep_original_source_order() -> None:
    g0 = [population_entry("0:7"), population_entry("0:8", homography_offset=1.0)]
    g1 = [population_entry("3:99")]

    records, _ = canonicalise_populations(g0, g1)
    ranking = legacy_winners(records)

    assert ranking["paint"] == "G0:0:7"
    assert ranking["paint_occurrence_key"] == "G0:0:7"


def test_line_template_three_way_dedup_and_legacy_isolation() -> None:
    g0 = population_entry("g0")
    g1 = population_entry("g1")
    line_template = population_entry("rectangle_12:template_3")
    line_template["pair_id"] = 999
    line_template.update({"proposal_id": "rectangle_12:template_3", "rectangle_id": 12,
                          "rectangle_order": 4, "template_index": 3})
    line_template["line_template"] = {"admission_score": 0.9}
    line_template.pop("profile")
    line_template.pop("stripe")

    records, resolution = canonicalise_populations([g0], [g1], [line_template])

    assert len(records) == 1
    assert records[0]["source_memberships"] == ["G0", "G1", "line_template"]
    assert resolution["source_occurrence_counts"] == {"G0": 1, "G1": 1, "line_template": 1}
    assert len(records[0]["_legacy_occurrences"]) == 2
    assert records[0]["source_occurrences"][2]["candidate_id"] == "rectangle_12:template_3"
    assert records[0]["source_occurrences"][2]["rectangle_id"] == 12
    assert records[0]["source_occurrences"][2]["line_template"] == {"admission_score": 0.9}
    assert legacy_winners(records)["eligible_count"] == 2


def test_raw_id_collisions_include_line_template_source() -> None:
    line_template = population_entry("shared", homography_offset=1.0)
    records, resolution = canonicalise_populations([population_entry("shared")], [], [line_template])

    assert len(records) == 2
    assert resolution["raw_id_collisions"] == ["shared"]
    assert resolution["raw_id_collision_sources"] == {"shared": ["G0", "line_template"]}


def test_line_template_duplicate_requires_full_w5_gate_match() -> None:
    left = population_entry("legacy")
    line_template = population_entry("rectangle_1:template_1")
    left["gates"] = {"geometry_valid": True, "camera_error": 0.05, "player_fractions": [1.0, 0.5]}
    line_template["gates"] = {"geometry_valid": True, "camera_error": 0.05, "player_fractions": [0.8, 0.5]}

    records, _ = canonicalise_populations([left], [], [line_template])
    assert len(records) == 2


def test_line_template_full_w5_gates_are_attached_with_raw_maps() -> None:
    segments = np.asarray([[10, 10, 90, 10], [10, 90, 90, 90], [10, 10, 10, 90], [90, 10, 90, 90]], dtype=float)
    context = SimpleNamespace(
        source={"id": "synthetic"},
        segments=segments,
        families=(segments, segments),
        size=(100, 100),
    )
    calls = []

    def gate_evidence(corners, source, scale, size, families, maps, zone):
        calls.append((corners, source, scale, size, families, maps, zone))
        return {
            "geometry_valid": True,
            "player_fractions": [1.0, 0.5],
            "floor_score": 0.8,
            "family_support": [0.9, 0.8],
            "line_counts": [5, 5],
            "camera_error": 0.04,
        }

    entry = {"corners_px": [[10, 10], [90, 10], [90, 90], [10, 90]], "gates": {"camera_error": 0.2}}
    attach_w5_gates([entry], context, {"gate_evidence": gate_evidence, "zone": object()}, detector, np.ones(2))

    assert len(calls) == 1
    assert calls[0][5].shape == (2, 100, 100)
    assert entry["gates"]["player_fractions"] == [1.0, 0.5]
    assert entry["gates"]["line_counts"] == [5, 5]




def test_previous_stage5_anchor_keeps_only_final_c_selection(tmp_path: Path) -> None:
    path = tmp_path / "evidence/holistic_admission/runs/w5_stage5_20260920/review_candidates.json"
    path.parent.mkdir(parents=True)
    path.write_text(__import__("json").dumps({
        "case": {
            "selected": {"B": "b", "C": "c"},
            "A": {"line": "a", "paint": "a2"},
            "candidates": {"a": {}, "a2": {}, "b": {}, "c": {}},
        }
    }))

    anchors = previous_stage5_anchors(tmp_path, "case")

    assert anchors["selected"] == {"C": "c"}
    assert list(anchors["candidates"]) == ["c"]


def test_line_template_union_map_keeps_working_image_shape() -> None:
    segments = np.asarray([[10, 10, 90, 10], [10, 90, 90, 90], [10, 10, 10, 90], [90, 10, 90, 90]], dtype=float)
    distance_map = union_distance_map(segments, (100, 100))
    assert distance_map.shape == (100, 100)
    corners = np.asarray([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M.astype(np.float32), corners)
    projected, means, valid, visibility = geometry_and_support(
        homography[None], distance_map, (100, 100), detector,
    )
    assert valid.shape == (1,)
    assert len(projected) == len(means) == len(visibility)


def test_visibility_floor_names_and_columns_are_asymmetric_and_inclusive() -> None:
    visibility = np.asarray([[3, 4], [3, 2], [2, 4], [5, 3]], dtype=np.int16)
    np.testing.assert_array_equal(
        visibility_eligible(visibility, 3, 3),
        [True, False, False, True],
    )
    np.testing.assert_array_equal(
        visibility_eligible(visibility, 4, 3),
        [False, False, False, True],
    )
    assert VISIBILITY_COLUMN_LABELS == {
        "lengthwise": "first six projected court-template pieces (x-family): sidelines plus split centre",
        "cross_court": "second six projected court-template pieces (y-family): baselines and service lines",
    }


def test_visibility_floor_zero_preserves_old_admission_semantics() -> None:
    visibility = np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int16)
    np.testing.assert_array_equal(visibility_eligible(visibility, 0, 0), [True, True, True])


def test_visibility_filter_before_diversity_allows_later_refill() -> None:
    visibility = np.asarray([[0, 1], [1, 1], [1, 1]], dtype=np.int16)
    corners = np.asarray(
        [
            [[0.0, 0.0]] * 4,
            [[30.0, 0.0]] * 4,
            [[60.0, 0.0]] * 4,
        ]
    )
    scores = np.asarray([0.9, 0.8, 0.7])
    camera_eligible = np.asarray([True, True, True])
    rectangle_ids = np.asarray([0, 1, 2])
    templates = np.asarray([0, 0, 0])
    selection = select_with_visibility_floor(
        scores,
        camera_eligible,
        rectangle_ids,
        templates,
        corners,
        visibility,
        1,
        1,
        cap=2,
    )
    np.testing.assert_array_equal(selection.selected, [1, 2])
    np.testing.assert_array_equal(selection.floor_zero_selected, [0, 1])
    np.testing.assert_array_equal(selection.newly_admitted, [2])
    assert len(selection.newly_admitted) == 1
    zero_selection = select_with_visibility_floor(
        scores,
        camera_eligible,
        rectangle_ids,
        templates,
        corners,
        visibility,
        0,
        0,
        cap=2,
    )
    np.testing.assert_array_equal(zero_selection.selected, zero_selection.floor_zero_selected)
    assert len(zero_selection.newly_admitted) == 0
    assert zero_selection.scanned == zero_selection.floor_zero_scanned


def test_line_template_empty_metadata_records_visibility_floor_and_counts() -> None:
    metadata = _empty_metadata(
        {
            "min_visible_lengthwise": 4,
            "min_visible_cross_court": 3,
        },
        0.0,
        "empty",
    )
    assert metadata["settings"]["min_visible_lengthwise"] == 4
    assert metadata["settings"]["min_visible_cross_court"] == 3
    assert metadata["generation"]["visibility_admission"] == {
        "min_visible_lengthwise": 4,
        "min_visible_cross_court": 3,
        "visibility_columns": VISIBILITY_COLUMN_LABELS,
        "hypotheses_before": 0,
        "hypotheses_after": 0,
        "hypotheses_rejected": 0,
        "floor_zero_scanned_for_proposal_cap": 0,
        "floor_zero_selected_count": 0,
        "floor_zero_proposal_ids": [],
        "scanned_for_proposal_cap": 0,
        "removed_from_floor_zero_count": 0,
        "refilled_proposal_count": 0,
        "newly_admitted_indices": [],
        "newly_admitted_proposal_ids": [],
        "removed_from_floor_zero_indices": [],
        "removed_from_floor_zero_proposal_ids": [],
    }




def test_visibility_floor_validation_rejects_non_integer_and_negative_values() -> None:
    with pytest.raises(TypeError, match="min_visible_lengthwise"):
        validate_visibility_floor(True, "min_visible_lengthwise")
    with pytest.raises(TypeError, match="min_visible_cross_court"):
        validate_visibility_floors(3, 2.5)
    with pytest.raises(ValueError, match="min_visible_lengthwise"):
        validate_visibility_floor(-1, "min_visible_lengthwise")
    with pytest.raises(ValueError, match="min_visible_cross_court"):
        validate_visibility_floors(3, -1)


def test_planned_visibility_floor_arms_are_fixed() -> None:
    assert VISIBILITY_FLOOR_ARMS == ((3, 3), (4, 3), (5, 3))




def test_pilot_runs_without_preflight_and_reports_completion_order(tmp_path: Path, monkeypatch, capsys) -> None:
    verifier = {"ALL_CASE_IDS": ("case_a", "case_b")}
    monkeypatch.setattr("run_w5.load_runtime", lambda root: {"verifier": verifier, "paths": {}})
    dispatches = []
    packet = {}

    class ImmediateFuture:
        def __init__(self, case_id):
            self.case_id = case_id

        def result(self):
            return {
                "case_id": self.case_id,
                "B": {"selected_origin_key": "B"},
                "C": {"selected_origin_key": "C"},
            }

    class ImmediatePool:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, root, case_id, run_dir, **kwargs):
            dispatches.append((case_id, kwargs))
            return ImmediateFuture(case_id)

    monkeypatch.setattr("run_w5.ProcessPoolExecutor", ImmediatePool)
    monkeypatch.setattr("run_w5.as_completed", lambda futures: reversed(list(futures)))
    monkeypatch.setattr("run_w5.write_packet", lambda *args, **kwargs: packet.update(kwargs))
    results = run_pilot(
        tmp_path, tmp_path / "run", ["case_a", "case_b"], 2,
        min_visible_lengthwise=4, min_visible_cross_court=3,
    )
    assert dispatches == [
        (case_id, {"min_visible_lengthwise": 4, "min_visible_cross_court": 3})
        for case_id in ("case_a", "case_b")
    ]
    assert [result["case_id"] for result in results] == ["case_a", "case_b"]
    assert packet["requested_cases"] == ["case_a", "case_b"]
    assert packet["min_visible_lengthwise"] == 4
    assert packet["min_visible_cross_court"] == 3
    output = capsys.readouterr().out
    assert output.index("case_b complete") < output.index("case_a complete")
    assert "[1/2" in output and "[2/2" in output
    assert not (tmp_path / "run" / "preflight.json").exists()


def test_cli_rejects_old_scalar_floor_option(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_w5.py",
            "--run",
            "test",
            "--min-visible-markings",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as error:
        parse_args()
    assert error.value.code == 2


def test_cli_accepts_directional_floor_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_w5.py",
            "--run",
            "test",
            "--min-visible-lengthwise",
            "4",
            "--min-visible-cross-court",
            "3",
        ],
    )
    args = parse_args()
    assert args.min_visible_lengthwise == 4
    assert args.min_visible_cross_court == 3


def test_original_regression_case_order_remains_explicit() -> None:
    expected = (
        "gxBQ_window_00_frame_0",
        "gxBQ_window_00_frame_5",
        "am2_window_00_frame_150",
        "am2_window_01_frame_28019",
        "am3_window_00_frame_0",
        "shuttleset_03_scene_0017",
        "shuttleset_03_scene_0019",
        "shuttleset_03_scene_0016",
        "shuttleset_21_scene_0020",
    )
    assert tuple(case_id for case_id, _, _ in CASE_ORDER) == expected
    assert REGRESSION_CASE_ORDER == CASE_ORDER


def test_line_template_camera_vector_matches_flat_homography() -> None:
    homography = np.eye(3, dtype=float)[None]
    assert abs(vector_camera_errors(homography, (100, 100))[0]) < 1e-12


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


def test_gallery_index_orders_ranked_links_and_stopped_cells(tmp_path: Path) -> None:
    rendered_cases = [
        {
            "label": "Ordered",
            "rendered": {
                "a": {"links": ["a__crop.png"], "roles": ["A_paint"]},
                "b2": {"links": ["b2__crop.png"], "roles": ["B-alternative-2"]},
                "b1": {"links": ["b1__crop.png"], "roles": ["B"]},
                "c3": {"links": ["c3__crop.png"], "roles": ["C-alternative-3"]},
                "c1": {"links": ["c1__crop.png"], "roles": ["C"]},
            },
        },
        {"label": "Stopped", "stopped_reason": "ambiguous", "rendered": {}},
    ]

    write_index(tmp_path, rendered_cases)
    lines = (tmp_path / "gallery/index.md").read_text().splitlines()
    ordered = next(line for line in lines if line.startswith("| Ordered |"))
    assert ordered.index("b1__crop.png") < ordered.index("b2__crop.png")
    assert ordered.index("c1__crop.png") < ordered.index("c3__crop.png")
    assert "| Stopped | stopped: ambiguous | — | — | — | — | — |" in lines


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


def test_original_nine_use_typed_provenance_for_exactly_four_masks() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [True, False, True, True, True, False, False, False, False]
    contexts = [prepare_view(root, case_id) for case_id, _, _ in CASE_ORDER]
    assert [context.same_image_mask_available for context in contexts] == expected
    assert sum(expected) == 4
    assert sum(not available for available in expected) == 5
    marking = load_frozen_case_provenance(root / CASE_PACKS["amateur"])
    assert not has_same_image_boxes(marking["yellow_short_frame_14"])

    unavailable = [context for context in contexts if not context.same_image_mask_available]
    metadata = [source_provenance(context, "test") for context in unavailable]
    assert all(item["person_mask_unavailable_reason"] for item in metadata)
    assert [item["box_relation"] for item in metadata] == ["nearby_source_frame"] + ["composite"] * 4
    assert metadata[0]["anchor_frame"] == 5
    assert all(item["anchor_frame"] is None for item in metadata[1:])


def test_amateur_frame_path_rejects_mismatched_typed_frame() -> None:
    source = {"id": "am2_window_00_frame_150"}
    provenance = CaseProvenance(source["id"], ImageKind.SOURCE_FRAME, (151,), 151)

    with pytest.raises(ValueError, match="amateur frame path uses frame 150"):
        frame_path(Path("/tmp/root"), source, provenance)


def test_gallery_passes_typed_provenance_to_frame_resolution(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    assert cv2.imwrite(str(image_path), np.zeros((8, 12, 3), dtype=np.uint8))
    source = {"id": "case_a", "dimensions": {"width": 12, "height": 8}}
    provenance = object()
    received = []
    verifier = {
        "frame_path": lambda root, actual_source, actual_provenance: (
            received.append((actual_source, actual_provenance)) or image_path
        ),
    }

    frame = load_native_frame(tmp_path, source, provenance, verifier)

    assert frame.shape == (8, 12, 3)
    assert received == [(source, provenance)]


def test_g0_replay_passes_typed_provenance_to_frame_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    case_id = "case_a"
    replay_path = tmp_path / "automatic_axes_20260914/all_camera" / f"{case_id}.json.gz"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_bytes(b"saved record")
    provenance = object()
    context = SimpleNamespace(
        case_id=case_id,
        native_size=(12, 8),
        source={"id": case_id},
        observations=object(),
        size=(12, 8),
        segments=object(),
        families=object(),
        provenance=provenance,
    )
    saved = {
        "schema": "automatic-directions-axis-matching/1",
        "case_id": case_id,
        "pairs": [],
    }
    received = []
    verifier = {
        "read_json_gz": lambda path: saved,
        "frame_path": lambda root, source, actual_provenance: received.append(actual_provenance),
        "relative_path": lambda path, root: path.relative_to(root).as_posix(),
    }
    run_automatic = SimpleNamespace(frame_path=lambda source, root: None)
    entries = [{"candidate_id": f"candidate-{index}"} for index in range(256)]

    def evaluate_pool(*args):
        run_automatic.frame_path(context.source, tmp_path)
        return entries

    runtime = {
        "verifier": verifier,
        "select_pool": object(),
        "evaluate_pool": evaluate_pool,
        "zone": object(),
    }
    monkeypatch.setattr("run_w5.reconstruct_generation_entries", lambda *args: [])
    monkeypatch.setattr("run_w5.import_run_automatic", lambda: run_automatic)

    actual_entries, source = load_g0(tmp_path, context, runtime)

    assert actual_entries == entries
    assert source == f"replayed:automatic_axes_20260914/all_camera/{case_id}.json.gz"
    assert received == [provenance]


def test_saved_population_validation_rejects_swapped_or_unknown_membership() -> None:
    record = {
        "schema": "automatic-directions-axis-matching/1",
        "case_id": "case_a",
        "stage": "results",
        "pairs": [],
        "entries": [{"candidate_id": f"candidate-{index}"} for index in range(256)],
        "line_winner_id": "candidate-0",
        "paint_winner_id": "candidate-1",
    }
    validate_generation_record(record, "case_a", "saved G1", expected_stage="results")

    with pytest.raises(ValueError, match="case identity"):
        validate_generation_record(record, "case_b", "saved G1", expected_stage="results")

    record["paint_winner_id"] = "unknown"
    with pytest.raises(ValueError, match="outside its entries"):
        validate_generation_record(record, "case_a", "saved G1", expected_stage="results")


def test_unavailable_masks_do_not_read_boxes_but_available_masks_require_them() -> None:
    composite = CaseProvenance("composite", ImageKind.COMPOSITE, (1, 2, 3), 1)
    source = {"dimensions": {"width": 8, "height": 8}}
    assert mask_boxes_working(source, (8, 8), composite).shape == (0, 4)

    same_image = CaseProvenance("same", ImageKind.SOURCE_FRAME, (1,), 1)
    with pytest.raises(KeyError, match="bbox_px"):
        mask_boxes_working(source, (8, 8), same_image)
    assert has_same_image_boxes(composite) is False


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
