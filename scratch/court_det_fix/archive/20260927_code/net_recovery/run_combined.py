"""Run one bounded seed generation, W5 replay, and net preference trial.

This historical replay mixes frozen and fresh paint measurements. Spot checks
found score drift on identical geometry. Its outputs establish candidate and
geometry retention, but do not isolate the seeds' effect on paint ranking.
Further comparisons need a freshly measured baseline in the same environment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT / "colour_consistency"), str(ROOT / "wider_evaluation")]

import am1_net_selection_trial as net
import am1_recovery_trial as recovery
import compare
import run_cases

COMPARISON = ROOT / "wider_evaluation/runs/20260922/comparison.json.gz"
NUMERIC = ROOT / "wider_evaluation/runs/20260922/numeric_fit.json.gz"


def selected_metrics(candidate: dict | None, context, rank: list[str], reference: dict,
                     landmarks: list[dict], verifier: dict) -> dict | None:
    if candidate is None:
        return None
    corners = np.asarray(candidate["corners_px"], dtype=float)
    far_midpoint = corners[:2].mean(axis=0)
    near_midpoint = corners[2:].mean(axis=0)
    diagnostics = verifier["projected_landmark_diagnostics"](
        np.asarray(candidate["homography_working"]), landmarks, context,
    )
    errors = np.asarray([item["error_px"] for item in diagnostics], dtype=float)
    corners_reference = reference.get("corners_px")
    return {
        "geometry": recovery.selected_geometry(candidate, context),
        "candidate_id": candidate["candidate_id"],
        "rank": rank.index(candidate["origin_key"]) + 1 if candidate["origin_key"] in rank else None,
        "gates": candidate["gates"],
        "score": {key: candidate.get("evidence", {}).get(key) for key in (
            "q_paint10", "q_paint10_span_weighted", "q_geom", "q_geom_span_weighted", "exclusive_score",
        )},
        "court_depth_native_px": float(np.linalg.norm(near_midpoint - far_midpoint)),
        "visible_landmark_error_native_px": {
            "count": len(errors), "maximum": float(errors.max()) if len(errors) else None,
            "median": float(np.median(errors)) if len(errors) else None,
        },
        "corner_reference_error_native_px": None if not corners_reference else verifier["reference_corner_error"](
            corners, np.asarray(corners_reference, dtype=float),
        ),
    }


def main(case_id: str, output_dir: Path, control_pack: Path | None = None) -> dict:
    cv2.setNumThreads(1)
    started = monotonic()
    rows = {row["case_id"]: row for row in recovery.read(COMPARISON)["cases"]}
    if case_id not in rows:
        raise ValueError(f"{case_id}: no row in {COMPARISON}")
    comparator = rows[case_id]
    replay = recovery.replay_case(case_id, comparator, control_pack)
    generation_seconds = monotonic() - started
    saved = replay["saved"]
    record = replay["trial_record"]
    context = replay["context"]
    verifier = replay["verifier"]
    candidates = record["parents"] + record["valid_children"]
    candidates_by_key = {candidate["origin_key"]: candidate for candidate in candidates}
    saved_candidates = {candidate["origin_key"]: candidate for candidate in saved["parents"] + saved["valid_children"]}
    saved_choices = compare.selections(saved)
    seeded_choices = compare.selections(record)
    timing = {"preparation": 0.0, "projection": 0.0, "scoring": 0.0}
    saved_net_case = net.evaluate_case("saved_pool", case_id, saved, context, timing)
    net_case = net.evaluate_case("seeded_pool", case_id, record, context, timing)
    choice_keys = {
        "saved_paint": saved_choices["full"]["gated"],
        "seeded_paint": seeded_choices["full"]["gated"],
        "seeded_net": None if net_case["trial"] is None else net_case["trial"]["origin_key"],
    }
    if net_case["original_comparator_full"] != seeded_choices["full"]:
        raise ValueError(f"{case_id}: net evaluator paint comparator differs")

    pool = dict(record)
    pool.update(schema="combined-seeded-w5-pool/1", case_id=case_id, provenance=saved["provenance"],
                min_visible_lengthwise=4, min_visible_cross_court=3,
                generation=replay["generated"].metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cases.write(output_dir / "scored_pool.json.gz", verifier["jsonable"](pool))

    # References are loaded only after all three selections have been fixed.
    numeric = next((row for row in recovery.read(NUMERIC)["cases"] if row["case_id"] == case_id), None)
    if numeric is None:
        raise ValueError(f"{case_id}: no numeric reference row in {NUMERIC}")
    if list(context.native_size) != numeric["native_size"] or list(context.size) != numeric["working_size"]:
        raise ValueError(f"{case_id}: numeric and prepared dimensions differ")
    reference = numeric["reference"]
    annotated_landmarks = []
    if case_id in verifier["PACK_OF"]:
        annotated_landmarks = recovery.run_w5.load_reference(ROOT, case_id, verifier).get("landmarks", [])
    width, height = context.native_size
    landmarks = [landmark for landmark in annotated_landmarks
                 if 0 <= landmark["image_px"][0] < width and 0 <= landmark["image_px"][1] < height]
    saved_rank = saved["rankings"]["C"]["provisional_rank"]
    seeded_rank = record["rankings"]["C"]["provisional_rank"]
    selected = {}
    for role, key in choice_keys.items():
        pool_map, rank = (saved_candidates, saved_rank) if role == "saved_paint" else (candidates_by_key, seeded_rank)
        selected[role] = selected_metrics(pool_map.get(key), context, rank, reference, landmarks, verifier)
        net_rows = saved_net_case["candidates"] if role == "saved_paint" else net_case["candidates"]
        net_row = next((row for row in net_rows if row["origin_key"] == key), None)
        if selected[role] is not None:
            selected[role]["net"] = None if net_row is None else {
                "coverage": dict(zip(net.PIECE_NAMES, net_row["coverage"])),
                "visible_samples": dict(zip(net.PIECE_NAMES, net_row["visible_samples"])),
                "strong_net": net_row["strong_net"],
            }
    saved_corners = None if selected["saved_paint"] is None else np.asarray(
        selected["saved_paint"]["geometry"]["corners_native_px"], dtype=float,
    )
    if saved_corners is not None:
        for role in ("seeded_paint", "seeded_net"):
            if selected[role] is not None:
                other = np.asarray(selected[role]["geometry"]["corners_native_px"], dtype=float)
                selected[role]["corner_gap_from_saved_native_px"] = {
                    "maximum": float(np.linalg.norm(other - saved_corners, axis=1).max()),
                    "median": float(np.median(np.linalg.norm(other - saved_corners, axis=1))),
                }
    strong = [row for row in net_case["candidates"] if row["strong_net"]]
    for row in strong:
        candidate = candidates_by_key[row["origin_key"]]
        row["score"] = {name: candidate.get("evidence", {}).get(name) for name in (
            "q_paint10", "q_paint10_span_weighted", "q_geom", "q_geom_span_weighted",
        )}
    shared_parent_keys = saved_candidates.keys() & candidates_by_key.keys()
    shared_parent_keys &= {parent["origin_key"] for parent in record["parents"]}
    exact_parent_geometry = [key for key in shared_parent_keys if (
        np.array_equal(saved_candidates[key]["corners_px"], candidates_by_key[key]["corners_px"])
        and np.array_equal(saved_candidates[key]["homography_working"], candidates_by_key[key]["homography_working"])
    )]
    original_key = choice_keys["saved_paint"]
    original_parent_key = None if original_key is None else (
        saved_candidates[original_key].get("parent_origin_key") or original_key
    )
    summary = {
        "schema": "combined-seed-net-trial/1", "case_id": case_id,
        "source_record": str(replay["saved_path"].relative_to(REPO)),
        "comparison_record": str(COMPARISON.relative_to(REPO)),
        "frame_path": context.frame_relative_path,
        "native_size_wh": list(context.native_size), "working_size_wh": list(context.size),
        "settings": {
            "seed_lengthwise_lines": recovery.SEED_LINE_COUNT, "seed_points": replay["seeds"].tolist(),
            "rectangle_cap": recovery.line_template_source.RECTANGLE_CAP,
            "proposal_cap": recovery.line_template_source.PROPOSAL_CAP,
            "min_visible_lengthwise": 4, "min_visible_cross_court": 3,
            "net_rule": net.RULE,
            "net_piece_names": net.PIECE_NAMES,
            "net_samples_per_piece": net.SAMPLES_PER_PIECE,
            "net_perpendicular_tolerance_working_px": net.PERPENDICULAR_TOLERANCE_WORKING_PX,
            "net_direction_tolerance_deg": net.DIRECTION_TOLERANCE_DEG,
            "net_extent_margin_working_px": net.EXTENT_MARGIN_WORKING_PX,
            "net_strong_support": net.STRONG_SUPPORT,
        },
        "generation": replay["generated"].metadata,
        "population_counts": {"G0": len(replay["g0"]), "G1": len(replay["g1"]),
                              "line_template": len(replay["templates"]),
                              "parents": len(record["parents"]), "children": len(record["valid_children"])},
        "candidate_changes": {"recomputed_parent_ids": replay["changed"],
                              "reused_parent_ids": replay["retained"],
                              "shared_parent_count": len(shared_parent_keys),
                              "exact_shared_parent_geometry_count": len(exact_parent_geometry),
                              "original_selected_parent_key": original_parent_key,
                              "original_selected_parent_geometry_exact": original_parent_key in exact_parent_geometry},
        "choices": choice_keys, "selected": selected,
        "saved_comparator_selections": saved_choices, "seeded_comparator_selections": seeded_choices,
        "net": {key: net_case[key] for key in (
            "ordering_criterion", "rank_status", "counts", "trial_reason", "changed",
        )},
        "strong_candidate_rows": strong,
        "net_candidate_rows": net_case["candidates"],
        "retrospective_reference_status": reference.get("reference_status"),
        "retrospective_landmark_count": len(landmarks),
        "timings_seconds": {"generation_and_replay": generation_seconds, **timing,
                            "total": monotonic() - started},
        "full_pool": "scored_pool.json.gz",
    }
    run_cases.write(output_dir / "summary.json.gz", verifier["jsonable"](summary))
    return {"case_id": case_id, "choices": choice_keys, "counts": summary["population_counts"],
            "seconds": summary["timings_seconds"]["total"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-pack", type=Path)
    arguments = parser.parse_args()
    print(main(arguments.case_id, arguments.output_dir, arguments.control_pack), flush=True)
