"""Replay fixed W5 fits after a saved court selection, with automatic stripe polarity."""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "wider_evaluation")]

from run_cases import load_runtime  # pyrefly: ignore[missing-import]

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.colour_consistency import (
    am1_recovery_trial,
    edge_auto_trial,
    observed_colour,
)
from scratch.court_det_fix.edge_polarity import run_probe as probe

MANIFEST = ROOT / "wider_evaluation/runs/20260922/manifest.json.gz"
CONTROL_PACK = ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz"
REPLAY_ATOL_NATIVE_PX = 1e-4


def checked_context(case_id: str, record: dict, verifier: object, manifest: dict):
    context = verifier.prepare_view(ROOT, case_id)
    provenance = record["provenance"]
    if record["case_id"] != case_id or provenance["case_id"] != case_id:
        raise ValueError(f"{case_id}: record case differs")
    if (provenance["frame_path"] != context.frame_relative_path
            or provenance["native_dimensions"] != list(context.native_size)
            or provenance["working_dimensions"] != list(context.size)):
        raise ValueError(f"{case_id}: source image or dimensions differ")
    matching = [row for row in manifest["cases"] if row["case_id"] == case_id]
    if len(matching) != 1 or matching[0]["image"] != context.frame_relative_path:
        raise ValueError(f"{case_id}: full manifest image differs")
    frame_md5 = hashlib.md5((ROOT / context.frame_relative_path).read_bytes()).hexdigest()
    if frame_md5 != matching[0]["image_md5"]:
        raise ValueError(f"{case_id}: image MD5 differs from full manifest")
    return context, frame_md5


def selected_parent(record: dict, origin_key: str) -> tuple[dict, dict, dict | None]:
    candidates = record["parents"] + record["valid_children"]
    by_key = {item["origin_key"]: item for item in candidates}
    if len(by_key) != len(candidates):
        raise ValueError("duplicate origin keys in source record")
    selected = by_key[origin_key]
    parent_key = selected.get("parent_origin_key") or origin_key
    parent = by_key[parent_key]
    if parent["kind"] == "child" or parent["origin_key"] != parent_key:
        raise ValueError(f"{origin_key}: invalid parent resolution")
    attempts = [row for row in record.get("fit_attempts", []) if row["origin_key"] == parent_key]
    if len(attempts) > 1:
        raise ValueError(f"{origin_key}: multiple saved fit attempts")
    attempt = attempts[0] if attempts else None
    if selected["kind"] == "child":
        if selected["parent_origin_key"] != parent_key or selected["origin_key"] != f"{parent_key}/child":
            raise ValueError(f"{origin_key}: child parent link differs")
        if attempt is not None:
            if attempt["child_origin_key"] != origin_key:
                raise ValueError(f"{origin_key}: saved attempt names another child")
            np.testing.assert_allclose(attempt["attempted_corners_native"], selected["corners_px"],
                                       rtol=0, atol=REPLAY_ATOL_NATIVE_PX)
    return selected, parent, attempt


def resolve_source_record(source_record: str) -> Path:
    source = Path(source_record)
    local = source if source.is_absolute() else REPO / source
    if local.is_file():
        return local.resolve()
    return am1_recovery_trial.resolve_saved_path(source_record)


def fit_geometry(fit: dict, context, verifier: object, runtime: dict, maps: np.ndarray) -> dict:
    result = {"fit": verifier.jsonable(fit), "status": fit.get("status"), "valid": False,
              "validity_reason": None, "measurement": None, "homography_working": None}
    corners = fit.get("corners_px")
    if corners is None:
        result["validity_reason"] = "no_fit_corners"
        return result
    working = np.asarray(corners, dtype=float)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    result["corners_working_px"] = working.tolist()
    result["corners_native_px"] = (working * scale).tolist()
    if not fit.get("successful", False):
        result["validity_reason"] = fit.get("status", "solver_failed")
        return result
    if fit.get("jacobian_rank") != 8 or fit.get("minimum_corner_denominator", 0) <= 1e-6:
        result["validity_reason"] = "fit_rank_or_depth_invalid"
        return result
    if not np.isfinite(working).all() or not verifier.convex_corners(working):
        result["validity_reason"] = "non_finite_or_non_convex_corners"
        return result
    homography = cv2.getPerspectiveTransform(verifier.detector.CORNER_COURT_M.astype(np.float32),
                                             working.astype(np.float32)).astype(float)
    result["homography_working"] = homography.tolist()
    measurement = probe.describe(working, context, verifier, runtime, maps)
    result["measurement"] = measurement
    valid, reason = verifier.hard_validity({"homography_working": homography, "gates": measurement["gates"]})
    result["valid"] = valid
    result["validity_reason"] = reason
    return result


def refit_selection(case_id: str, source_record: str, origin_key: str, label: str,
                    verifier: object, runtime: dict, manifest: dict) -> dict:
    started = perf_counter()
    path = resolve_source_record(source_record)
    record = verifier.read_json_gz(path)
    context, frame_md5 = checked_context(case_id, record, verifier, manifest)
    selected, parent, attempt = selected_parent(record, origin_key)
    homography = np.asarray(parent["homography_working"], dtype=float)
    constraints = fitting.prepare(homography, context.observations, parent["evidence"]["stripe_assignments"],
                                  context.weights, centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    starting = verifier.detector.project(homography[None], verifier.detector.CORNER_COURT_M)[0][0]
    original = fitting.refine(starting, constraints, context.size, use_positions=True,
                              centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    reference = attempt["attempted_corners_native"] if attempt is not None else selected["corners_px"]
    reference_name = "saved_fit_attempt" if attempt is not None else "saved_child_corners"
    if attempt is None and selected["kind"] != "child":
        raise ValueError(f"{origin_key}: parent has no saved attempt or child replay reference")
    if original.get("corners_px") is None:
        raise ValueError(f"{origin_key}: original fit returned no corners")
    replay_native = np.asarray(original["corners_px"], dtype=float) * scale
    difference = replay_native - np.asarray(reference, dtype=float)
    np.testing.assert_allclose(replay_native, reference, rtol=0, atol=REPLAY_ATOL_NATIVE_PX)
    replay_seconds = perf_counter() - started

    _, lab, grey, boxes = observed_colour.native_image(context, ROOT)
    _, old_rows, _ = probe.relabel(context, parent, constraints, verifier)
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    positions = constraints.positions.copy()
    rows = []
    for old in old_rows:
        fragment_id = old["raw_fragment_id"]
        index = lookup[fragment_id]
        segment = context.observations.segments[index]
        sampled = observed_colour.sample_fragment(segment.copy(), scale, lab, grey, boxes)
        inferred = edge_auto_trial.infer_polarity(sampled)
        direction = context.observations.directions[index]
        normal = np.array([[-direction[1], direction[0]]])
        expected = probe.expected_bright_side(
            homography, segment.mean(axis=0)[None], normal, np.array([old["marking"]]),
            np.array([1]), verifier.detector, verifier.paint_geometry.STRIPE_WIDTH_M,
        )[0]
        if expected == 0:
            raise ValueError(f"{case_id}: zero expected side for fragment {fragment_id}")
        new = edge_auto_trial.automatic_position(
            old["old_position"], old["signed_contrast_by_distance"][0],
            old["valid_pairs_by_distance"][0], expected, inferred["polarity"],
        )
        positions[constraints.fragment_ids == fragment_id] = new
        rows.append({"raw_fragment_id": fragment_id, "old_position": old["old_position"],
                     "automatic_position": new, "inferred_polarity": inferred["polarity"],
                     "usable_samples": inferred["usable_samples"]})
    revised = replace(constraints, positions=positions)
    for field in ("points", "intervals", "weights", "fragment_ids", "sample_ids"):
        if getattr(revised, field) is not getattr(constraints, field):
            raise AssertionError(f"{field} array was replaced")
        np.testing.assert_array_equal(getattr(revised, field), getattr(constraints, field))
    corrected = fitting.refine(starting, revised, context.size, use_positions=True,
                               centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments), context.size)
    geometry = fit_geometry(corrected, context, verifier, runtime, maps)
    corrected_seconds = perf_counter() - started - replay_seconds
    return {"case_id": case_id, "label": label, "source_record": str(path),
            "frame_md5": frame_md5, "selected_origin_key": origin_key,
            "selected_geometry": {"corners_native_px": selected["corners_px"],
                                  "homography_working": selected["homography_working"],
                                  "gates": selected["gates"]},
            "parent_origin_key": parent["origin_key"], "parent_homography_working": parent["homography_working"],
            "replay_reference": reference_name, "original_replay": {"status": "matched",
                "maximum_absolute_difference_native_px": float(np.abs(difference).max()),
                "corners_native_px": replay_native.tolist(), "fit_status": original.get("status")},
            "automatic": {"changed_fragment_count": sum(row["automatic_position"] != row["old_position"] for row in rows),
                          "unresolved_fragment_count": sum(row["inferred_polarity"] == 0 for row in rows),
                          "fragment_count": len(rows), "changed_point_count": int(np.sum(positions != constraints.positions)),
                          "fragments": rows, "unchanged_constraint_arrays": True},
            "corrected": geometry, "timings_seconds": {"original_replay": replay_seconds,
                                                        "automatic_and_corrected": corrected_seconds,
                                                        "total": perf_counter() - started}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    _, verifier, runtime = load_runtime(ROOT, CONTROL_PACK)
    requests = verifier.read_json_gz(arguments.requests)
    manifest = verifier.read_json_gz(MANIFEST)
    results = []
    for request in requests["selections"]:
        result = refit_selection(**request, verifier=verifier, runtime=runtime, manifest=manifest)
        results.append(result)
        print(f"{len(results)}/{len(requests['selections'])} {result['case_id']}: "
              f"{result['corrected']['status']}", flush=True)
    verifier.write_json_gz(arguments.output, {"schema": "selected-polarity-refit/1", "selections": results})


if __name__ == "__main__":
    main()
