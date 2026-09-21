#!/usr/bin/env python3
"""CPU-only checks for badminton_cv_annotator issue #148.

    python check_issue148.py
    python check_issue148.py --repo /path/to/badminton_cv_annotator

Default: run the bundled reviewed source extracts.
--repo: AST-extract the same pure functions from the local checkout, avoiding
torch/model imports. This mode executes source from the supplied checkout.
This does NOT rerun Hough extraction, CourtKeyNet, player tracking, or contacts.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
FIXTURES = PACK / "fixtures"

def extract_module(repo: Path):
    """Load named pure definitions, not the repository's import graph."""
    sources = [
        (repo / "src/courtkeynet/wrapper.py",
         {"DEFAULT_AREA_BOUNDS", "_is_convex", "_shoelace_area", "_quadrants_ok", "_geometry_flags"}),
        (repo / "src/courtkeynet/court_corners.py",
         {"CONSENSUS_FLAG_THRESHOLD_PX", "ConsensusRepair", "consensus_repair"}),
    ]
    nodes = []
    for path, names in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                nodes.append(node)
                found.add(node.name)
            elif isinstance(node, ast.Assign):
                assigned = {t.id for t in node.targets if isinstance(t, ast.Name)}
                if assigned & names:
                    nodes.append(node)
                    found.update(assigned & names)
        if names - found:
            raise ValueError(f"{path}: missing definitions {sorted(names - found)}; "
                             "adapt this diagnostic to the changed API.")
    module = types.ModuleType("_issue148_reviewed_functions")
    module.__dict__.update(np=np, dataclass=dataclass)
    sys.modules[module.__name__] = module
    tree = ast.Module(body=nodes, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<issue148 trusted repo extracts>", "exec"),
         module.__dict__)
    return module

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="Read pure functions from this trusted local checkout.")
    parser.add_argument("--output", type=Path, default=PACK / "checks" / "diagnostic_results.json")
    args = parser.parse_args()
    if args.repo:
        implementation = extract_module(args.repo.resolve())
        implementation_label = f"AST extracts from {args.repo.resolve()}"
    else:
        import reviewed_source_extracts as implementation
        implementation_label = "bundled reviewed source extracts at 94bc9cc"

    replay = json.loads((FIXTURES / "video53_nn_replay.json").read_text(encoding="utf-8"))
    good = np.asarray(replay["nn_median_corners_px"], dtype=np.float32)
    bad = np.asarray(replay["saved_scene"]["raw_corners_px"], dtype=np.float32)
    # Convexity and corner-order findings do not depend on the area gate or frame size.
    nn_flags = implementation._geometry_flags(good, (1920., 1080.), area_bounds=(0., float("inf")))
    fallback_flags = implementation._geometry_flags(bad, (1920., 1080.), area_bounds=(0., float("inf")))

    court = np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32)
    # Reconstructed from the FINAL stored quad. This is not the unsaved internal fit.
    homography = cv2.getPerspectiveTransform(court, bad)
    denominators = np.column_stack((court, np.ones(4))) @ homography[2]
    diag = replay["saved_scene"]["fallback_diagnostics"]

    alternate_view = good + np.array([200., 0.], dtype=np.float32)
    repair = implementation.consensus_repair(np.stack([good, good, good, alternate_view]))
    # Same relative geometry, different absolute pixel scale.
    near_threshold = np.stack([good, good, good, good + np.array([40., 0.], dtype=np.float32)])
    lower_res = implementation.consensus_repair(near_threshold)
    higher_res = implementation.consensus_repair(near_threshold * 1.5)
    result = {
        "implementation": implementation_label,
        "opencv_version": cv2.__version__,
        "actual_video53_scene334": {
            "nn_shape_flags": list(nn_flags),
            "saved_fallback_shape_flags": list(fallback_flags),
            "saved_tr_peak": replay["saved_scene"]["raw_peaks"][1],
            "corner_peak_floor": replay["corner_min_peak_conf"],
            "nn_tr_px": good[1].tolist(),
            "fallback_tr_px": bad[1].tolist(),
            "tr_displacement_px": float(np.linalg.norm(bad[1] - good[1])),
            "saved_line_residual_px": diag["reproj_line_px"],
            "saved_anchor_residual_px": diag["reproj_anchor_px"],
            "saved_normalised_residuals_pass_reviewed_thresholds":
                diag["gate_line_frac"] <= 0.010 and diag["gate_anchor_frac"] <= 0.020,
            "reconstructed_H_denominators_at_court_vertices": denominators.tolist(),
            "reconstructed_H_has_pole_through_court":
                bool(denominators.min() < 0 < denominators.max()),
        },
        "synthetic_camera_change_not_video17_replay": {
            "all_input_quads_shape_clean": all(
                not implementation._geometry_flags(q, (1920., 1080.), area_bounds=(0., float("inf")))
                for q in [good, alternate_view]
            ),
            "flagged": repair.flagged.tolist(),
            "valid_alternate_view_replaced": bool(
                np.allclose(repair.repaired_quads[-1], good)
            ),
        },
        "synthetic_scale_sensitivity": {
            "40px_shift_flagged": bool(lower_res.flagged[-1]),
            "same_geometry_at_1_5x_scale_flagged": bool(higher_res.flagged[-1]),
        },
        "limitations": [
            "No neural-network, Hough extraction, tracking, contact, or rally rerun.",
            "The reconstructed H is calculated from stored final corners, not the internal fit.",
            "The camera-change example is synthetic, not a replay of video 17.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
