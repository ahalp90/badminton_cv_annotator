"""Test image-side evidence while keeping the saved refit's points and weights."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "wider_evaluation"))
from measurement import prepared_measurements
from run_cases import load_runtime

BASE = ROOT / "wider_evaluation/runs/20260922"
RECORDS = ROOT / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/case_records"
CASES = tuple(f"shuttleset_03_scene_{scene:04d}" for scene in (16, 17, 19, 29, 34, 38)) + ("gxBQ_window_00_frame_5",)
CONTRAST_LEVELS = 10.0
PROFILE_DISTANCES = np.array([1.0, 2.0])


def read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def brightness_profiles(context: Any, verifier: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure raw-fragment polarity once, independently of court assignments."""
    directions = context.observations.directions
    normals = np.column_stack((-directions[:, 1], directions[:, 0]))
    samples = context.observations.samples[:, 1:-1]
    offsets = normals[:, None, None, :] * PROFILE_DISTANCES[None, None, :, None]
    positive, negative = samples[:, :, None, :] + offsets, samples[:, :, None, :] - offsets
    positive_valid = verifier.observable_points(positive, context.size, context.mask_boxes)
    negative_valid = verifier.observable_points(negative, context.size, context.mask_boxes)
    valid = positive_valid & negative_valid
    differences = (
        verifier.grayscale_sample(context.frame, positive) - verifier.grayscale_sample(context.frame, negative)
    )
    medians = np.zeros((len(samples), len(PROFILE_DISTANCES)))
    counts = valid.sum(axis=1)
    for fragment in range(len(samples)):
        for distance in range(len(PROFILE_DISTANCES)):
            if counts[fragment, distance]:
                medians[fragment, distance] = np.median(differences[fragment, valid[fragment, :, distance], distance])
    return normals, medians, counts


def expected_bright_side(
    homography: np.ndarray, midpoints: np.ndarray, normals: np.ndarray,
    markings: np.ndarray, positions: np.ndarray, detector: Any, stripe_width: float,
) -> np.ndarray:
    """Express the expected paint side in each raw fragment's normal convention."""
    world, _ = detector.project(np.linalg.inv(homography)[None], midpoints)
    world = world[0]
    shifted = world.copy()
    axis = np.where(markings < 5, 0, 1)
    shifted[np.arange(len(world)), axis] += stripe_width
    projected, _ = detector.project(homography[None], shifted)
    positive_axis = projected[0] - midpoints
    image_sign = np.sign(np.einsum("ij,ij->i", positive_axis, normals))
    return image_sign * np.where(positions == 1, 1, -1)


def relabel(
    context: Any, parent: dict, constraints: Any, verifier: Any,
) -> tuple[Any, list[dict], dict]:
    started = perf_counter()
    normals, contrasts, counts = brightness_profiles(context, verifier)
    assignments = parent["evidence"]["stripe_assignments"]
    positions = np.asarray(assignments["position"], dtype=int)
    markings = np.asarray(assignments["marking"], dtype=int)
    expected = expected_bright_side(
        np.asarray(parent["homography_working"]), context.observations.segments.mean(axis=1),
        normals, markings, positions, verifier.detector, verifier.paint_geometry.STRIPE_WIDTH_M,
    )
    minimum_samples = (context.observations.samples.shape[1] - 2 + 1) // 2
    edge = (positions == 1) | (positions == 2)
    sufficient = (counts[:, 0] >= minimum_samples) & (np.abs(contrasts[:, 0]) >= CONTRAST_LEVELS)
    contradicted = edge & sufficient & (contrasts[:, 0] * expected < 0)
    revised = np.where(contradicted, 3 - positions, positions)
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    new_positions = np.array([revised[lookup[int(raw_id)]] for raw_id in constraints.fragment_ids])
    changed = replace(constraints, positions=new_positions)
    retained = set(constraints.fragment_ids.tolist())
    rows = []
    for index, raw_id in enumerate(context.observations.fragment_ids):
        if int(raw_id) not in retained:
            continue
        rows.append({
            "raw_fragment_id": int(raw_id), "marking": int(markings[index]),
            "old_position": int(positions[index]), "new_position": int(revised[index]),
            "expected_bright_side": float(expected[index]) if edge[index] else None,
            "signed_contrast_by_distance": contrasts[index].tolist(),
            "valid_pairs_by_distance": counts[index].tolist(),
            "strong_polarity": bool(sufficient[index]), "changed": bool(contradicted[index]),
        })
    metadata = {"seconds": perf_counter() - started, "all_fragment_count": len(positions),
                "retained_fragment_count": len(retained),
                "changed_point_count": int(np.sum(new_positions != constraints.positions)),
                "changed_fragment_count": sum(row["changed"] for row in rows)}
    return changed, rows, metadata


def describe(corners_working: np.ndarray, context: Any, verifier: Any, runtime: dict, maps: np.ndarray) -> dict:
    detector = verifier.detector
    scale = np.asarray(context.native_size) / context.size
    corners_native = corners_working * scale
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, corners_working.astype(np.float32))
    evidence, _ = verifier.measure_candidate(context, {"homography_working": homography}, {})
    gates = runtime["gate_evidence"](
        corners_native, context.source, scale, context.size, context.families, maps, runtime["zone"],
    )
    return {"corners_native_px": corners_native.tolist(), "corners_working_px": corners_working.tolist(),
            "paint_score": evidence["q_paint10_span_weighted"], "geometry_score": evidence["q_geom_span_weighted"],
            "camera_eligible": verifier.camera_eligible({"gates": gates}),
            "historical": verifier.historical_predicates(gates), "gates": gates}


def run_case(case_id: str) -> dict:
    cv2.setNumThreads(1)
    _, verifier, runtime = load_runtime(ROOT)
    from experiments.annotator.independent_court import fixed_stripe_refit as fitting

    context = verifier.prepare_view(ROOT, case_id)
    comparison = next(case for case in read(BASE / "comparison.json.gz")["cases"] if case["case_id"] == case_id)
    selected_key = comparison["selections"]["full"]["gated"]
    record = read(RECORDS / f"{case_id}.json.gz")
    candidates = {entry["origin_key"]: entry for entry in record["parents"] + record["valid_children"]}
    selected = candidates[selected_key]
    parent_key = selected["parent_origin_key"] if selected["kind"] == "child" else selected_key
    parent = candidates[parent_key]
    attempt = next(row for row in record["fit_attempts"] if row["origin_key"] == parent_key)
    homography = np.asarray(parent["homography_working"])
    starting_corners = verifier.detector.project(homography[None], verifier.detector.CORNER_COURT_M)[0][0]
    constraints = fitting.prepare(
        homography, context.observations, parent["evidence"]["stripe_assignments"], context.weights,
        centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
    )
    scale = np.asarray(context.native_size) / context.size
    maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments), context.size)
    with prepared_measurements(verifier) as counts:
        changed, fragments, polarity = relabel(context, parent, constraints, verifier)
        fits = {}
        for label, current in (("baseline", constraints), ("polarity", changed)):
            fitted = fitting.refine(starting_corners, current, context.size, True,
                                    centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
            fitted["measurement"] = None
            if fitted["successful"]:
                fitted["measurement"] = describe(np.asarray(fitted["corners_px"]), context, verifier, runtime, maps)
            fits[label] = fitted
        assert fits["baseline"]["successful"], fits["baseline"]["status"]
        expected = np.asarray(attempt["attempted_corners_native"])
        actual = np.asarray(fits["baseline"]["corners_px"]) * scale
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
        selected_measurement = describe(np.asarray(selected["corners_px"]) / scale, context, verifier, runtime, maps)
    assert counts["greyscale_conversions"] == 1, counts
    movement = None
    if fits["polarity"]["successful"]:
        movement = (np.asarray(fits["polarity"]["corners_px"]) - fits["baseline"]["corners_px"]).tolist()
    result = {"case_id": case_id, "selected_key": selected_key, "parent_key": parent_key,
              "native_size": context.native_size, "working_size": context.size,
              "selected": selected_measurement, "polarity": polarity, "fragments": fragments, "fits": fits,
              "movement_working_px": movement,
              "baseline_max_absolute_native_difference": float(np.max(np.abs(actual - expected))),
              "measurement_counts": counts}
    print(case_id, "changed", polarity["changed_fragment_count"], "fit", fits["polarity"]["status"], flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json.gz"))
    args = parser.parse_args()
    if args.workers == 1:
        cases = [run_case(case_id) for case_id in args.cases]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
            cases = list(pool.map(run_case, args.cases))
    result = {"schema": "fixed-fragment-polarity-refit/1", "workers": args.workers,
              "contrast_threshold": CONTRAST_LEVELS, "profile_distances_working_px": PROFILE_DISTANCES.tolist(),
              "selection_uses_reference_labels": False, "fitted_point_membership_fixed": True, "cases": cases}
    args.output.write_bytes(gzip.compress(json.dumps(result, allow_nan=False, indent=2).encode(), mtime=0))


if __name__ == "__main__":
    main()
