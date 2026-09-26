#!/usr/bin/env python3
"""Run the single highest-value SS03-34 local counterfactual.

Run from the pinned repository root in the same checkout that contains the local
case record used by scratch/court_det_fix/edge_polarity/run_probe.py::run_case.
No reference court labels are used. The script:

1. reconstructs the exact frozen constraints used by the saved probe;
2. applies the existing polarity swaps to positions 1/2;
3. resolves retained centre-assigned fragment 111 to the polarity-compatible
   edge using the existing brightness sign and expected_bright_side convention;
4. changes only fragment 111's position, preserving points, intervals, sample
   IDs, markings and weights;
5. compares the exact polarity fit with that one-fragment counterfactual.

The analytic proxy predicts an upper-left movement near (-1.154, +0.002)
working pixels. The exact result may be attenuated by the other constraints.
A non-leftward or negligible x movement rejects fragment 111 as a consequential
explanation of the remaining left inset, even if its physical identity is wrong.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.edge_polarity import run_probe as probe

CASE_ID = "shuttleset_03_scene_0034"
TARGET_FRAGMENT_ID = 111
PROXY_UPPER_LEFT_MOVEMENT = np.array([-1.154448, 0.001630])


def point_segment_distances(points: np.ndarray, segments: np.ndarray) -> np.ndarray:
    vectors = segments[:, 1] - segments[:, 0]
    fraction = np.einsum("pd,pd->p", points - segments[:, 0], vectors) / np.square(vectors).sum(axis=1)
    closest = segments[:, 0] + np.clip(fraction, 0.0, 1.0)[:, None] * vectors
    return np.linalg.norm(points - closest, axis=1)


def target_profile(
    constraints: fitting.Constraints,
    corners_px: np.ndarray,
    size: tuple[int, int],
    positions: tuple[int, ...],
    centres: np.ndarray,
    detector: Any,
) -> dict[str, dict[str, float | int]]:
    target = constraints.fragment_ids == TARGET_FRAGMENT_ID
    if not target.any():
        raise RuntimeError(f"fragment {TARGET_FRAGMENT_ID} has no retained constraint points")
    homography = cv2.getPerspectiveTransform(
        detector.CORNER_COURT_M.astype(np.float32), np.asarray(corners_px, dtype=np.float32)
    )
    result: dict[str, dict[str, float | int]] = {}
    total_weight = float(constraints.weights.sum())
    for position in positions:
        candidate_positions = np.full(int(target.sum()), position, dtype=int)
        segments_m = fitting.shifted_intervals(
            constraints.intervals[target], candidate_positions, centres=centres
        )
        projected, _ = detector.project(homography[None], segments_m)
        segments_px = projected.reshape(-1, 2, 2)
        distances = point_segment_distances(constraints.points[target], segments_px)
        normalized_weight = constraints.weights[target] / total_weight
        result[str(position)] = {
            "point_count": int(target.sum()),
            "median_distance_px": float(np.median(distances)),
            "mean_distance_px": float(np.mean(distances)),
            "maximum_distance_px": float(np.max(distances)),
            "objective_contribution_px2": float(np.dot(normalized_weight, np.square(distances))),
        }
    return result


def load_exact_case() -> tuple[Any, Any, dict, dict, np.ndarray, fitting.Constraints]:
    _, verifier, _ = probe.load_runtime(probe.ROOT)
    context = verifier.prepare_view(probe.ROOT, CASE_ID)

    comparison = next(
        case for case in probe.read(probe.BASE / "comparison.json.gz")["cases"]
        if case["case_id"] == CASE_ID
    )
    selected_key = comparison["selections"]["full"]["gated"]
    record_path = probe.RECORDS / f"{CASE_ID}.json.gz"
    if not record_path.exists():
        raise FileNotFoundError(
            f"Missing exact frozen case record: {record_path}\n"
            "The file is referenced by run_probe.py but is not committed at the pinned revision. "
            "Run this in the original evidence checkout; do not reconstruct a nearby subset."
        )
    record = probe.read(record_path)
    candidates = {entry["origin_key"]: entry for entry in record["parents"] + record["valid_children"]}
    selected = candidates[selected_key]
    parent_key = selected["parent_origin_key"] if selected["kind"] == "child" else selected_key
    parent = candidates[parent_key]

    homography = np.asarray(parent["homography_working"], dtype=float)
    starting_corners = verifier.detector.project(
        homography[None], verifier.detector.CORNER_COURT_M
    )[0][0]
    constraints = fitting.prepare(
        homography,
        context.observations,
        parent["evidence"]["stripe_assignments"],
        context.weights,
        centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
    )
    return context, verifier, parent, {"selected_key": selected_key, "parent_key": parent_key}, starting_corners, constraints


def choose_target_edge(context: Any, verifier: Any, parent: dict, polarity_rows: list[dict]) -> tuple[int, dict]:
    row = next(item for item in polarity_rows if item["raw_fragment_id"] == TARGET_FRAGMENT_ID)
    if row["old_position"] != 0 or row["new_position"] != 0:
        raise RuntimeError(f"expected fragment {TARGET_FRAGMENT_ID} to remain at centre: {row}")
    if not row["strong_polarity"]:
        raise RuntimeError(f"fragment {TARGET_FRAGMENT_ID} lacks strong polarity: {row}")

    lookup = {int(raw_id): index for index, raw_id in enumerate(context.observations.fragment_ids)}
    raw_index = lookup[TARGET_FRAGMENT_ID]
    direction = context.observations.directions[raw_index]
    normal = np.array([-direction[1], direction[0]], dtype=float)[None]
    midpoint = context.observations.segments[raw_index].mean(axis=0)[None]
    marking = np.array([row["marking"]], dtype=int)
    homography = np.asarray(parent["homography_working"], dtype=float)

    expected = {}
    for position in (1, 2):
        value = probe.expected_bright_side(
            homography,
            midpoint,
            normal,
            marking,
            np.array([position], dtype=int),
            verifier.detector,
            verifier.paint_geometry.STRIPE_WIDTH_M,
        )[0]
        expected[position] = int(np.sign(value))

    contrast = float(row["signed_contrast_by_distance"][0])
    contrast_sign = int(np.sign(contrast))
    matching = [position for position, sign in expected.items() if sign == contrast_sign]
    if len(matching) != 1:
        raise RuntimeError(
            f"polarity does not uniquely choose an edge: contrast={contrast}, expected={expected}"
        )
    position = matching[0]
    return position, {
        "contrast_px1": contrast,
        "contrast_sign": contrast_sign,
        "expected_contrast_sign": expected,
        "chosen_position": position,
        "chosen_position_name": ("centre", "negative_edge", "positive_edge")[position],
        "valid_pairs_px1": int(row["valid_pairs_by_distance"][0]),
    }


def run() -> dict:
    context, verifier, parent, keys, starting_corners, constraints = load_exact_case()

    with probe.prepared_measurements(verifier):
        polarity_constraints, polarity_rows, polarity_metadata = probe.relabel(
            context, parent, constraints, verifier
        )

    target_position, decision = choose_target_edge(context, verifier, parent, polarity_rows)
    target = polarity_constraints.fragment_ids == TARGET_FRAGMENT_ID
    if not target.any():
        raise RuntimeError(f"fragment {TARGET_FRAGMENT_ID} was not retained by fitting.prepare")
    if not np.all(polarity_constraints.positions[target] == 0):
        raise RuntimeError("target constraint rows are not all centre-assigned")

    revised_positions = polarity_constraints.positions.copy()
    revised_positions[target] = target_position
    counterfactual_constraints = replace(polarity_constraints, positions=revised_positions)

    fits = {}
    for name, current in (
        ("polarity", polarity_constraints),
        ("fragment111_edge", counterfactual_constraints),
    ):
        fit = fitting.refine(
            starting_corners,
            current,
            context.size,
            True,
            centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
        )
        if not fit["successful"]:
            raise RuntimeError(f"{name} fit failed: {fit}")
        fits[name] = fit

    polarity_corners = np.asarray(fits["polarity"]["corners_px"], dtype=float)
    edge_corners = np.asarray(fits["fragment111_edge"]["corners_px"], dtype=float)
    movement = edge_corners - polarity_corners

    profile = target_profile(
        polarity_constraints,
        polarity_corners,
        context.size,
        positions=(0, target_position),
        centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
        detector=verifier.detector,
    )

    return {
        "schema": "scene34-fragment111-centre-to-polarity-edge/1",
        "case_id": CASE_ID,
        "target_fragment_id": TARGET_FRAGMENT_ID,
        **keys,
        "working_size": list(context.size),
        "decision": decision,
        "polarity_metadata": polarity_metadata,
        "target_constraint_point_count": int(target.sum()),
        "frozen_arrays_preserved": {
            "points": counterfactual_constraints.points is polarity_constraints.points,
            "intervals": counterfactual_constraints.intervals is polarity_constraints.intervals,
            "weights": counterfactual_constraints.weights is polarity_constraints.weights,
            "fragment_ids": counterfactual_constraints.fragment_ids is polarity_constraints.fragment_ids,
            "sample_ids": counterfactual_constraints.sample_ids is polarity_constraints.sample_ids,
        },
        "no_refit_profile_at_polarity_fit": profile,
        "fits": fits,
        "corner_movement_working_px": movement.tolist(),
        "upper_left_movement_working_px": movement[0].tolist(),
        "proxy_upper_left_movement_working_px": PROXY_UPPER_LEFT_MOVEMENT.tolist(),
        "moves_upper_left_leftward": bool(movement[0, 0] < 0.0),
        "objective_change_px2": float(
            fits["fragment111_edge"]["objective_after"] - fits["polarity"]["objective_after"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("scene34_fragment111_counterfactual.json.gz"))
    args = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, allow_nan=False).encode()
    args.output.write_bytes(gzip.compress(payload, mtime=0))
    print(json.dumps({
        "chosen_position": result["decision"]["chosen_position_name"],
        "upper_left_movement_working_px": result["upper_left_movement_working_px"],
        "objective_change_px2": result["objective_change_px2"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
