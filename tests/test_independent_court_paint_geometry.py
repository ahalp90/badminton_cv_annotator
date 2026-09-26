"""Check physical paint positions and their fixed-refit propagation."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import cv2
import numpy as np
import pytest

from experiments.annotator.independent_court import (
    check_paint_control,
    render_paint_refit,
    run_paint_refit,
)
from experiments.annotator.independent_court.run_junctions import provenance_binding
from scratch.court_det_fix.court_detector import paint_geometry as paint
from scratch.court_det_fix.court_detector import stripe_fitting as refit
from scratch.court_det_fix.court_detector import stripe_measurements as stripes
from scratch.court_det_fix.court_detector.geometry import (
    CORNER_COURT_M,
    SEGMENTS_M,
    project,
)
from scratch.court_det_fix.court_detector.image_sources import CaseProvenance, ImageKind
from scratch.court_det_fix.court_detector.line_observations import (
    Observations,
    prepare_observations,
)

IMAGE_SIZE = (2400, 1600)
KNOWN_CORNERS = np.array(
    [[250.0, 180.0], [2100.0, 220.0], [1960.0, 1450.0], [180.0, 1320.0]],
)
INTERVALS = np.arange(12, dtype=int)
POSITIONS = np.array([1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1], dtype=int)
EXPECTED_MARKINGS = np.array([0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=int)


def test_archive_control_detects_a_previously_successful_refit_disappearing() -> None:
    source = {"id": "parent/start", "parent_id": "parent", "model": "start",
              "corners_px": KNOWN_CORNERS.tolist(), "eligible": False}
    archived = {"records": [{"id": "frame", "entries": [source, {
        **source, "id": "parent/fixed_position", "model": "fixed_position",
    }]}]}
    fresh = [{"id": "frame", "entries": [{**source, "model": "legacy", "stage": "start"}]}]
    with pytest.raises(ValueError, match="Control candidate population changed"):
        run_paint_refit.verify_control(fresh, archived)


def test_paint_refit_direct_case_rejects_unsafe_boxes_before_prepare() -> None:
    legacy = SimpleNamespace(
        prepare_case=lambda _case: (_ for _ in ()).throw(AssertionError("prepared unsafe boxes")),
    )
    provenance = CaseProvenance("case", ImageKind.COMPOSITE, (1, 2), 1)
    with pytest.raises(ValueError, match="composite"):
        run_paint_refit.run_case({"id": "case"}, [], legacy, provenance)


def test_paint_refit_preflights_all_cases_before_import_or_output(monkeypatch, tmp_path: Path) -> None:
    recorded = tmp_path / "recorded"
    recorded.mkdir()
    (recorded / "marking_refit_replay.zip").write_bytes(b"replay bytes")
    pack_path = tmp_path / "marking_refit_inputs.json.gz"
    provenance = {
        "safe": CaseProvenance("safe", ImageKind.SOURCE_FRAME, (1,), 1),
        "unsafe": CaseProvenance("unsafe", ImageKind.COMPOSITE, (1, 2), 1),
    }
    monkeypatch.setattr(run_paint_refit, "load_frozen_case_provenance", lambda _path: provenance)
    monkeypatch.setattr(run_paint_refit, "require_replay_pack", lambda *_args: None)
    monkeypatch.setattr(
        run_paint_refit,
        "read_replay_bytes",
        lambda _bytes: ({"cases": [{"id": "safe"}, {"id": "unsafe"}]}, {"records": []}),
    )
    monkeypatch.setattr(
        run_paint_refit.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(AssertionError("legacy measurement imported before preflight")),
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_paint_refit.py", "--recorded", str(recorded), "--provenance-pack", str(pack_path),
         "--annotations", str(tmp_path / "annotations"), "--output", str(output)],
    )
    with pytest.raises(ValueError, match="composite"):
        run_paint_refit.main()
    assert not output.exists()


def test_paint_refit_output_binding_uses_artefact_spelling() -> None:
    binding = {"pack_filename": "pack", "pack_md5": "0" * 32}
    provenance = run_paint_refit.output_provenance(binding, b"replay", b"control", {"corners.csv": "1" * 32})
    assert run_paint_refit.OUTPUT_SCHEMA == "paired-paint-refit/2"
    assert provenance["replay_artefact_md5"] == run_paint_refit.bytes_md5(b"replay")
    assert provenance["control_artefact_md5"] == run_paint_refit.bytes_md5(b"control")


def _current_paint_result() -> dict:
    pack_path = Path("marking_refit_inputs.json.gz")
    binding = provenance_binding(pack_path)
    return {
        "schema": run_paint_refit.OUTPUT_SCHEMA,
        "provenance": run_paint_refit.output_provenance(
            binding,
            b"replay",
            b"control",
            {str(path): "1" * 32 for path in run_paint_refit.ANNOTATION_ARTEFACTS},
        ),
    }


@pytest.mark.parametrize("reader", [render_paint_refit, check_paint_control])
def test_paint_result_readers_reject_archived_schema_one(reader: ModuleType) -> None:
    with pytest.raises(ValueError, match="old or unsupported"):
        reader.validate_result_provenance({"schema": "paired-paint-refit/1"})


def test_paint_result_reader_accepts_bound_schema_two_and_checks_bytes() -> None:
    result = _current_paint_result()
    render_paint_refit.validate_result_provenance(result, replay_bytes=b"replay", control_bytes=b"control")
    with pytest.raises(ValueError, match="different replay bytes"):
        check_paint_control.validate_result_provenance(result, replay_bytes=b"other", control_bytes=b"control")


def test_paint_result_reader_rejects_non_marking_pinned_pack() -> None:
    result = _current_paint_result()
    result["provenance"]["pack_filename"] = "gx_extension_inputs.json.gz"
    with pytest.raises(ValueError, match="amateur replay pack"):
        run_paint_refit.validate_result_provenance(result)


@pytest.mark.parametrize("field", ["replay_artefact_md5", "control_artefact_md5", "annotation_artefacts_md5"])
@pytest.mark.parametrize(
    "malformed",
    [" " + "0" * 31, "0x" + "0" * 30, "+" + "0" * 31, "_" + "0" * 31, "A" + "0" * 31],
)
def test_paint_result_reader_rejects_noncanonical_md5(field: str, malformed: str) -> None:
    result = _current_paint_result()
    if field == "annotation_artefacts_md5":
        result["provenance"][field]["hand_corners.csv"] = malformed
    else:
        result["provenance"][field] = malformed
    with pytest.raises(ValueError, match="32-character lowercase MD5"):
        run_paint_refit.validate_result_provenance(result)


def _known_homography() -> np.ndarray:
    """Return a projective image transform with visible court boundaries."""
    return cv2.getPerspectiveTransform(CORNER_COURT_M, KNOWN_CORNERS.astype(np.float32))


def _projected_fragments(
    homography: np.ndarray, centres: np.ndarray, intervals: np.ndarray, positions: np.ndarray,
) -> np.ndarray:
    """Project finite fragments from the selected physical marking positions."""
    segments = paint.positioned_segments(centres, intervals, positions)
    vectors = segments[:, 1] - segments[:, 0]
    metric_fragments = np.stack((segments[:, 0] + 0.10 * vectors, segments[:, 0] + 0.90 * vectors), axis=1)
    projected, _ = project(homography[None], metric_fragments)
    return projected[0].reshape(-1, 4)


def _assert_evidence_equal(first: stripes.StripeEvidence, second: stripes.StripeEvidence) -> None:
    """Compare the array fields of two cached evidence records."""
    for first_row, second_row in zip(first.forward, second.forward):
        np.testing.assert_array_equal(first_row, second_row)
    np.testing.assert_array_equal(first.reverse, second.reverse)
    for first_row, second_row in zip(first.resolvable, second.resolvable):
        np.testing.assert_array_equal(first_row, second_row)
    np.testing.assert_array_equal(first.visible, second.visible)


def _assert_constraints_equal(first: refit.Constraints, second: refit.Constraints) -> None:
    """Compare frozen fitting inputs without relying on dataclass array equality."""
    for name in ("points", "intervals", "positions", "weights", "fragment_ids", "sample_ids"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_bwf_paint_edges_and_gaps_are_literal() -> None:
    """Check outer paint edges, inner gaps and short-service net-facing edges."""
    outer_edges = paint.positioned_segments(
        paint.CENTRE_SEGMENTS_M,
        np.array([0, 5, 6, 11]),
        np.array([1, 2, 1, 2]),
    )
    np.testing.assert_allclose(outer_edges[:, 0], [[0.0, 0.0], [6.10, 0.0], [0.0, 0.0], [0.0, 13.40]])

    inner_x_edges = paint.positioned_segments(
        paint.CENTRE_SEGMENTS_M,
        np.array([0, 1]),
        np.array([2, 1]),
    )
    inner_y_edges = paint.positioned_segments(
        paint.CENTRE_SEGMENTS_M,
        np.array([6, 7]),
        np.array([2, 1]),
    )
    assert inner_x_edges[1, 0, 0] - inner_x_edges[0, 0, 0] == 0.42
    assert inner_y_edges[1, 0, 1] - inner_y_edges[0, 0, 1] == 0.72

    short_service_edges = paint.positioned_segments(
        paint.CENTRE_SEGMENTS_M,
        np.array([8, 9]),
        np.array([2, 1]),
    )
    np.testing.assert_allclose(short_service_edges[:, 0, 1], [4.72, 8.68])


def test_positioned_fragments_measure_and_refine_with_physical_centres() -> None:
    """Recover a projective court when measurement and refit share paint centres."""
    homography = _known_homography()
    fragment_ids = np.arange(100, 112)
    observed = _projected_fragments(homography, paint.CENTRE_SEGMENTS_M, INTERVALS, POSITIONS)
    observations = prepare_observations(observed, IMAGE_SIZE, fragment_ids)
    measured = stripes.measure(homography, observations, IMAGE_SIZE, centres=paint.CENTRE_SEGMENTS_M)
    weights = stripes.fragment_weights(observations)
    assignment = stripes.compare(measured, weights)["stripe"]["assignments"]

    by_fragment = {
        int(fragment): (int(marking), int(position))
        for fragment, marking, position in zip(
            observations.fragment_ids, assignment["marking"], assignment["position"], strict=True,
        )
    }
    expected = dict(zip(fragment_ids, zip(EXPECTED_MARKINGS, POSITIONS, strict=True), strict=True))
    assert by_fragment == expected
    assert set(assignment["position"]) == {1, 2}

    constraints = refit.prepare(
        homography, observations, assignment, weights, centres=paint.CENTRE_SEGMENTS_M,
    )
    assert set(constraints.intervals) == set(INTERVALS)
    assert set(constraints.positions) == {1, 2}

    starting_corners = KNOWN_CORNERS + np.array(
        [[8.0, -6.0], [-9.0, 7.0], [11.0, 5.0], [-10.0, -8.0]],
    )
    fitted = refit.refine(
        starting_corners,
        constraints,
        IMAGE_SIZE,
        use_positions=True,
        centres=paint.CENTRE_SEGMENTS_M,
    )

    assert fitted["status"] == "converged"
    assert fitted["successful"] is True
    assert fitted["jacobian_rank"] == 8
    assert fitted["objective_after"] < fitted["objective_before"]
    np.testing.assert_allclose(fitted["corners_px"], KNOWN_CORNERS, atol=0.1)


def test_explicit_legacy_centres_match_defaults() -> None:
    """Keep omitted centre arguments exactly equivalent to the legacy template."""
    homography = _known_homography()
    observed = _projected_fragments(homography, SEGMENTS_M, INTERVALS, np.zeros(12, dtype=int))
    observations = prepare_observations(observed, IMAGE_SIZE)
    default_evidence = stripes.measure(homography, observations, IMAGE_SIZE)
    explicit_evidence = stripes.measure(homography, observations, IMAGE_SIZE, centres=SEGMENTS_M)
    _assert_evidence_equal(default_evidence, explicit_evidence)

    weights = stripes.fragment_weights(observations)
    assignment = stripes.compare(default_evidence, weights)["stripe"]["assignments"]
    default_constraints = refit.prepare(homography, observations, assignment, weights)
    explicit_constraints = refit.prepare(homography, observations, assignment, weights, centres=SEGMENTS_M)
    _assert_constraints_equal(default_constraints, explicit_constraints)

    starting_corners = KNOWN_CORNERS + np.array(
        [[3.0, -2.0], [-2.0, 3.0], [3.0, 2.0], [-3.0, -2.0]],
    )
    default_fit = refit.refine(starting_corners, default_constraints, IMAGE_SIZE, use_positions=True)
    explicit_fit = refit.refine(
        starting_corners, explicit_constraints, IMAGE_SIZE, use_positions=True, centres=SEGMENTS_M,
    )
    assert default_fit == explicit_fit


def test_boundary_tolerance_accepts_roundoff_but_not_real_out_of_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain endpoint support for roundoff while excluding a real outside point."""
    perspective = 0.001
    x_scale = 6.0 * (1.0 + perspective * 6.1) / 6.1
    homography = np.array([[x_scale, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective, 0.0, 1.0]])
    inverse = np.linalg.inv(homography)
    centres = paint.CENTRE_SEGMENTS_M
    centre_segment, _ = project(homography[None], centres[6])
    centre_segment = centre_segment[0]
    fractions = np.linspace(0.0, 1.0, 3)
    centre_samples = centre_segment[0] + fractions[:, None] * (centre_segment[1] - centre_segment[0])

    observed_segment = paint.positioned_segments(centres, np.array([6]), np.array([1]))[0]
    observed_segment, _ = project(homography[None], observed_segment)
    observed_segment = observed_segment[0]
    vector = observed_segment[1] - observed_segment[0]
    length = np.linalg.norm(vector)
    observations = Observations(
        segments=observed_segment[None],
        fragment_ids=np.array([0]),
        groups=(np.array([0]),),
        group_lengths=np.array([length]),
        directions=(vector / length)[None],
        lengths=np.array([length]),
        samples=(observed_segment[0] + fractions[:, None] * vector)[None],
    )

    ideal = stripes.interval_evidence(
        homography,
        inverse,
        6,
        centre_samples,
        observations,
        (7, 7),
        centres,
        boundary_tolerance_px=0.0,
    )
    original_project = stripes.project

    def run_with_roundoff(roundoff: float, tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls = 0

        def project_with_roundoff(homographies: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal calls
            calls += 1
            pixels, denominator = original_project(homographies, points)
            if calls == 3:
                # Only the round-trip shifted samples receive simulated projection roundoff.
                pixels = pixels.copy()
                sample_count = len(centre_samples)
                for position in range(3):
                    first = position * sample_count
                    pixels[0, first, 0] -= roundoff
                    pixels[0, first + sample_count - 1, 0] += roundoff
            return pixels, denominator

        monkeypatch.setattr(stripes, "project", project_with_roundoff)
        result = stripes.interval_evidence(
            homography,
            inverse,
            6,
            centre_samples,
            observations,
            (7, 7),
            centres,
            boundary_tolerance_px=tolerance,
        )
        assert calls == 3
        return result

    endpoint_indices = [0, len(centre_samples) - 1]
    tiny_without_tolerance = run_with_roundoff(1e-10, 0.0)
    tiny_with_tolerance = run_with_roundoff(1e-10, 1e-7)
    real_out_of_frame = run_with_roundoff(0.01, 1e-7)

    ideal_endpoint_support = ideal[0][:, endpoint_indices, 0]
    assert np.all(ideal_endpoint_support > 0.99)
    np.testing.assert_array_equal(tiny_without_tolerance[0][:, endpoint_indices, 0], 0.0)
    np.testing.assert_array_equal(tiny_without_tolerance[0][:, 1, 0], ideal[0][:, 1, 0])
    np.testing.assert_array_equal(tiny_with_tolerance[0], ideal[0])
    np.testing.assert_array_equal(real_out_of_frame[0][:, endpoint_indices, 0], 0.0)
