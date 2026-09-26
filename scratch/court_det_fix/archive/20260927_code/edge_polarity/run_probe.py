"""Test image-side evidence while keeping the saved refit's points and weights."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(ROOT / "wider_evaluation")]
from measurement import prepared_measurements
from run_cases import load_runtime

# Other probe scripts reach brightness_profiles and expected_bright_side as probe.<name>.
from scratch.court_det_fix.court_detector.stripe_refit import (
    CONTRAST_LEVELS,
    PROFILE_DISTANCES,
    brightness_profiles,  # noqa: F401
    describe,
    expected_bright_side,  # noqa: F401
    relabel,
)

BASE = ROOT / "wider_evaluation/runs/20260922"
RECORDS = ROOT / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/case_records"
CASES = tuple(f"shuttleset_03_scene_{scene:04d}" for scene in (16, 17, 19, 29, 34, 38)) + ("gxBQ_window_00_frame_5",)


def read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


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
