"""Reproduce the scene34 fixed-assignment refit objective at selected corners."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[5]
COURT_ROOT = REPO_ROOT / "scratch" / "court_det_fix"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.annotator.independent_court import (
    fixed_stripe_refit as fitting,
)
from scratch.court_det_fix.w5_holistic import verifier

PAIR_PATH = HERE / "scene34_selected_pair.json.gz"
OUTPUT_PATH = HERE / "scene34_refit_objective_probe.json.gz"
CASE_ID = "shuttleset_03_scene_0034"


def objective(
    corners: np.ndarray, constraints: fitting.Constraints, size: tuple[int, int]
) -> float:
    image_scale = max(size)
    court_scale = verifier.detector.CORNER_COURT_M.max(axis=0)
    segments = (
        fitting.shifted_intervals(
            constraints.intervals,
            constraints.positions,
            verifier.paint_geometry.CENTRE_SEGMENTS_M,
        )
        / court_scale
    )
    points = constraints.points / image_scale
    weights = np.sqrt(constraints.weights / constraints.weights.sum()) * image_scale
    parameters = fitting.initial_parameters(corners, size)
    residual = fitting.residual(parameters, points, segments, weights)
    return float(residual @ residual)


def main() -> None:
    with gzip.open(PAIR_PATH, "rt") as stream:
        pair = json.load(stream)
    context = verifier.prepare_view(COURT_ROOT, CASE_ID)
    parent, child = pair["candidates"]
    parent_homography = np.asarray(parent["homography_working"], dtype=float)
    parent_corners = np.asarray(parent["corners_px"], dtype=float)
    child_corners = np.asarray(child["corners_px"], dtype=float)
    diagnostic_corners = child_corners.copy()
    diagnostic_corners[0] += (-3.0, -2.0)
    constraints = fitting.prepare(
        parent_homography,
        context.observations,
        parent["evidence"]["stripe_assignments"],
        context.weights,
        centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
    )

    def corners_record(corners: np.ndarray) -> dict:
        return {
            "corners_px": corners.tolist(),
            "objective": objective(corners, constraints, context.size),
        }

    diagnostic_fit = fitting.refine(
        diagnostic_corners,
        constraints,
        context.size,
        use_positions=True,
        centres=verifier.paint_geometry.CENTRE_SEGMENTS_M,
    )
    grid = {}
    for dx, dy in (
        (-6, -2),
        (-4, -2),
        (-3, -2),
        (-2, -2),
        (0, -2),
        (-3, -4),
        (-3, 0),
        (-3, 2),
        (0, 0),
    ):
        candidate = child_corners.copy()
        candidate[0] += (dx, dy)
        grid[f"{dx},{dy}"] = objective(candidate, constraints, context.size)
    result = {
        "case_id": CASE_ID,
        "objective": "fixed parent stripe assignments and prepared constraints",
        "constraint_points": len(constraints.points),
        "constraint_unique_fragments": len(np.unique(constraints.fragment_ids)),
        "fragment_weight_sum_before_refit_normalisation": float(
            constraints.weights.sum()
        ),
        "saved_parent": corners_record(parent_corners),
        "saved_child": corners_record(child_corners),
        "diagnostic_tl_minus_3_minus_2": {
            **corners_record(diagnostic_corners),
            "refine_from_diagnostic": {
                "status": diagnostic_fit["status"],
                "nfev": diagnostic_fit["nfev"],
                "objective_before": diagnostic_fit["objective_before"],
                "objective_after": diagnostic_fit["objective_after"],
                "jacobian_rank": diagnostic_fit["jacobian_rank"],
                "jacobian_condition": diagnostic_fit["jacobian_condition"],
                "corners_px": diagnostic_fit["corners_px"],
            },
        },
        "local_tl_objective_grid": {"relative_to_saved_child_px": grid},
    }
    OUTPUT_PATH.write_bytes(
        gzip.compress(json.dumps(result, allow_nan=False, indent=2).encode(), mtime=0)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
