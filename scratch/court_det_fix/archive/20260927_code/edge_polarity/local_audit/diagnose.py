"""Measure fixed centre-label counterfactuals and the constraints resisting SS03-34."""

from __future__ import annotations

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from experiments.annotator.independent_court.assignment import MARKINGS
from scratch.court_det_fix.edge_polarity import run_probe as probe

OUTPUT = Path(__file__).resolve().parent
CASES = ("shuttleset_03_scene_0034", "shuttleset_03_scene_0029", "shuttleset_03_scene_0019", "gxBQ_window_00_frame_5")
PROFILE_OFFSETS = np.linspace(-6, 6, 49)


def load_case(case_id: str, verifier: Any) -> tuple[Any, dict, np.ndarray, fitting.Constraints]:
    context = verifier.prepare_view(probe.ROOT, case_id)
    comparison = next(case for case in probe.read(probe.BASE / "comparison.json.gz")["cases"]
                      if case["case_id"] == case_id)
    record = probe.read(probe.RECORDS / f"{case_id}.json.gz")
    candidates = {entry["origin_key"]: entry for entry in record["parents"] + record["valid_children"]}
    selected = candidates[comparison["selections"]["full"]["gated"]]
    parent = candidates[selected["parent_origin_key"]] if selected["kind"] == "child" else selected
    homography = np.asarray(parent["homography_working"])
    corners = verifier.detector.project(homography[None], verifier.detector.CORNER_COURT_M)[0][0]
    constraints = fitting.prepare(homography, context.observations, parent["evidence"]["stripe_assignments"],
                                 context.weights, centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
    return context, parent, corners, constraints


def centre_choices(context: Any, parent: dict, rows: list[dict], verifier: Any) -> dict[int, int]:
    choices = {}
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    for row in rows:
        if row["new_position"] != 0 or not row["strong_polarity"]:
            continue
        index = lookup[row["raw_fragment_id"]]
        direction = context.observations.directions[index]
        normal = np.array([[-direction[1], direction[0]]])
        expected_negative = probe.expected_bright_side(
            np.asarray(parent["homography_working"]), context.observations.segments[index].mean(axis=0)[None],
            normal, np.array([row["marking"]]), np.array([1]), verifier.detector,
            verifier.paint_geometry.STRIPE_WIDTH_M,
        )[0]
        assert expected_negative != 0
        choices[row["raw_fragment_id"]] = 1 if expected_negative * row["signed_contrast_by_distance"][0] > 0 else 2
    return choices


def contributions(corners: np.ndarray, constraints: fitting.Constraints, context: Any, verifier: Any) -> np.ndarray:
    segments = fitting.shifted_intervals(constraints.intervals, constraints.positions,
                                        verifier.paint_geometry.CENTRE_SEGMENTS_M)
    court_scale = verifier.detector.CORNER_COURT_M.max(axis=0)
    scale = max(context.size)
    residuals = fitting.residual(
        fitting.initial_parameters(corners, context.size), constraints.points / scale, segments / court_scale,
        np.sqrt(constraints.weights / constraints.weights.sum()) * scale,
    ).reshape(-1, 2)
    return np.square(residuals).sum(axis=1)


def fragment_profile(context: Any, index: int, corners: np.ndarray, interval: int, verifier: Any) -> dict:
    direction = context.observations.directions[index]
    normal = np.array([-direction[1], direction[0]])
    samples = context.observations.samples[index, 1:-1]
    points = samples[:, None] + PROFILE_OFFSETS[None, :, None] * normal
    valid = verifier.observable_points(points, context.size, context.mask_boxes)
    values = verifier.grayscale_sample(context.frame, points)
    # The same cross-sections contribute at every offset, avoiding mask-induced gradients.
    complete = valid.all(axis=1)
    profile = np.median(values[complete], axis=0) if complete.any() else None
    homography = cv2.getPerspectiveTransform(verifier.detector.CORNER_COURT_M, corners.astype(np.float32))
    world = verifier.detector.project(np.linalg.inv(homography)[None], samples)[0][0]
    axis = 0 if interval < 6 else 1
    edge_offsets = []
    for position in (1, 2):
        segment = fitting.shifted_intervals(np.array([interval]), np.array([position]),
                                           verifier.paint_geometry.CENTRE_SEGMENTS_M)[0]
        edge_world = world.copy()
        edge_world[:, axis] = segment[0, axis]
        projected = verifier.detector.project(homography[None], edge_world)[0][0]
        edge_offsets.append((projected - samples) @ normal)
    widths = np.abs(edge_offsets[1] - edge_offsets[0])
    result = {"complete_cross_sections": int(complete.sum()), "offsets_px": PROFILE_OFFSETS.tolist(),
              "median_grey": profile.tolist() if profile is not None else None,
              "projected_edge_offsets_median_px": [float(np.median(offsets)) for offsets in edge_offsets],
              "projected_width_range_px": [float(widths.min()), float(widths.max())]}
    if profile is not None:
        gradient = np.diff(profile) / np.diff(PROFILE_OFFSETS)
        midpoints = (PROFILE_OFFSETS[1:] + PROFILE_OFFSETS[:-1]) / 2
        result["strongest_rise_offset_px"] = float(midpoints[np.argmax(gradient)])
        result["strongest_fall_offset_px"] = float(midpoints[np.argmin(gradient)])
    return result


def residual_report(context: Any, constraints: fitting.Constraints, rows: list[dict],
                    saved: dict, fits: dict, verifier: Any) -> dict:
    polarity_corners = np.asarray(fits["polarity"]["corners_px"])
    preferred = np.asarray(saved["fits"]["baseline"]["corners_px"])
    preferred[0] += [-3, -2]  # Reproduce the saved user diagnostic; never enter fitting.
    corrected_target = polarity_corners.copy()
    corrected_target[0] = preferred[0]
    arms = {"polarity": polarity_corners, "saved_preferred": preferred,
            "preferred_xy_only": corrected_target}
    # Axis-separated paths show which constraints resist the requested left/up movements.
    for axis, label in enumerate(("preferred_x_only", "preferred_y_only")):
        corners = polarity_corners.copy()
        corners[0, axis] = preferred[0, axis]
        arms[label] = corners
    objectives = {name: contributions(corners, constraints, context, verifier) for name, corners in arms.items()}
    np.testing.assert_allclose(objectives["polarity"].sum(), fits["polarity"]["objective_after"], atol=1e-5)
    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    fragments = []
    for row in rows:
        fragment_id = row["raw_fragment_id"]
        selected = constraints.fragment_ids == fragment_id
        totals = {name: float(values[selected].sum()) for name, values in objectives.items()}
        fragment = {**row, "marking_name": MARKINGS[row["marking"]], "point_count": int(selected.sum()),
                    "weight_fraction": float(constraints.weights[selected].sum() / constraints.weights.sum()),
                    "objectives_px2": totals,
                    "preferred_minus_polarity_px2": totals["preferred_xy_only"] - totals["polarity"]}
        fragment["profile"] = fragment_profile(context, lookup[fragment_id], polarity_corners,
                                                int(constraints.intervals[selected][0]), verifier)
        fragments.append(fragment)
    return {"corners_working_px": {name: corners.tolist() for name, corners in arms.items()},
            "objectives_px2": {name: float(values.sum()) for name, values in objectives.items()},
            "fragments": fragments}


def run_case(case_id: str) -> dict:
    cv2.setNumThreads(1)
    _, verifier, runtime = probe.load_runtime(probe.ROOT)
    context, parent, starting_corners, constraints = load_case(case_id, verifier)
    saved = next(case for case in probe.read(OUTPUT.parent / "results.json.gz")["cases"] if case["case_id"] == case_id)
    assert saved["parent_key"] == parent["origin_key"]
    maps = verifier.detector._distance_maps(verifier.detector._wide_line_families(context.segments), context.size)
    with probe.prepared_measurements(verifier) as counts:
        polarity, rows, _ = probe.relabel(context, parent, constraints, verifier)
        choices = centre_choices(context, parent, rows, verifier)
        changes = {"polarity": {}, "strong_centres": choices}
        if case_id == CASES[0]:
            changes.update({"fragment111": {111: choices[111]}, "fragment179": {179: choices[179]}})
        fits = {}
        for name, replacements in changes.items():
            positions = polarity.positions.copy()
            for fragment_id, position in replacements.items():
                positions[polarity.fragment_ids == fragment_id] = position
            revised = replace(polarity, positions=positions)
            for field in ("points", "intervals", "weights", "fragment_ids", "sample_ids"):
                assert getattr(revised, field) is getattr(polarity, field)
            fit = fitting.refine(starting_corners, revised, context.size, True,
                                 centres=verifier.paint_geometry.CENTRE_SEGMENTS_M)
            assert fit["successful"], (case_id, name, fit)
            fit["measurement"] = probe.describe(np.asarray(fit["corners_px"]), context, verifier, runtime, maps)
            fits[name] = fit
        difference = np.asarray(fits["polarity"]["corners_px"]) - saved["fits"]["polarity"]["corners_px"]
        np.testing.assert_allclose(difference, 0, atol=1e-7)
        report = residual_report(context, polarity, rows, saved, fits, verifier) if case_id == CASES[0] else None
        assert counts["greyscale_conversions"] == 1, counts
    result = {"case_id": case_id, "parent_key": parent["origin_key"], "centre_choices": choices,
              "polarity_baseline_max_difference_px": float(np.abs(difference).max()), "fits": fits,
              "frozen_arrays_preserved": True, "residual_report": report, "measurement_counts": counts}
    result["movements_working_px"] = {
        name: (np.asarray(fit["corners_px"]) - fits["polarity"]["corners_px"]).tolist() for name, fit in fits.items()
    }
    return result


def main() -> None:
    with ProcessPoolExecutor(max_workers=4) as executor:
        cases = list(executor.map(run_case, CASES))
    result = {"schema": "fixed-centre-label-diagnostic/1", "cases": cases}
    (OUTPUT / "diagnostics.json.gz").write_bytes(gzip.compress(json.dumps(result, indent=2, allow_nan=False).encode(), mtime=0))
    for case in cases:
        print(case["case_id"], case["centre_choices"], case["movements_working_px"]["strong_centres"][0])


if __name__ == "__main__":
    main()
