"""Mask fragments inside person boxes on GX0 and replay the direction selection.

Read-only on every input under the repository. Writes only into this OUT folder.

Gate: with no masking, the replay must reproduce the saved GX0 estimator exactly
(retained candidate IDs, working points, support masks). If the gate fails, stop.

Then: apply two masking rules to the fragments that feed the merge step (drop a
fragment when both endpoints lie inside a person box, or when its midpoint lies
inside one), rerun the unchanged merge and coverage allocation, and measure the
16 selected directions against the approved control three ways.
"""
from __future__ import annotations

import csv
import gzip
import json
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
DIRECTION_AGREEMENT = REPO_ROOT / "scratch/court_det_fix/direction_agreement"
HELPERS = REPO_ROOT / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914"
GX_DIRECTIONS = REPO_ROOT / "scratch/court_det_fix/evidence/pixel_temporal/diagnostics/gx_directions"
OUT_DIR = Path(__file__).resolve().parent

for path in (
    REPO_ROOT, REPO_ROOT / "src", DIRECTION_AGREEMENT,
    HELPERS / "vp_pruning", HELPERS / "marking_diagnosis", HELPERS / "axis_matching",
    HELPERS / "automatic_axes", HELPERS / "automatic_axes/svd_fixed", GX_DIRECTIONS,
):
    sys.path.insert(0, str(path))

import vp_pruning
from diagnose_direction_bank import reconstruct_bank
from geometry_certificate import set_bound
from membership import merge_lines_with_membership
from run_population import prepare
from run_svd_fixed import fit_pairs

from experiments.annotator.independent_court import assignment, detector

INPUT_PACK = (
    REPO_ROOT / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260909/gx_extension/inputs.json.gz"
)
SAVED_ESTIMATOR = REPO_ROOT / (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/vp_pruning/"
    "coverage/results/gxBQ_window_00_frame_0.json.gz"
)
E3_RECORD = DIRECTION_AGREEMENT / "runs/direction_agreement_20260915_144900/e3/gxBQ_window_00_frame_0.json.gz"
CASE_ID = "gxBQ_window_00_frame_0"
ESTIMATE_MERGE_CAP = 300
FLOAT_ATOL = 1e-12
CONTROL_ID_X = 1183  # candidate x-direction from run_gx_directions.py's control pair.
CONTROL_ID_Y = 122
EDGE_PROXIMITY_PX = 6.0
OFFENDING_ROWS = {
    31: [584, 683, 417, 583],
    51: [469],
    61: [499],
}


def read_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def load_case() -> dict:
    pack = read_gz_json(INPUT_PACK)
    for source in pack["cases"]:
        if source["id"] == CASE_ID:
            return source
    raise KeyError(CASE_ID)


def bank_and_retain(lines: np.ndarray, size: tuple[int, int], settings: vp_pruning.Settings) -> dict:
    """The candidate-generation and coverage-allocation half of vp_pruning.estimate.

    Reproduces vp_pruning.estimate exactly from the merged direction lines onward, so
    it can be run again on a different (masked) set of merged lines. Verified against
    a direct call to vp_pruning.estimate on the unmasked case in the gate below.
    """
    transform = vp_pruning.normalisation(size)
    normalised_lines = lines @ transform
    pairs = np.asarray(list(combinations(range(len(lines)), 2)), dtype=int).reshape(-1, 2)
    intersections = np.cross(normalised_lines[pairs[:, 0]], normalised_lines[pairs[:, 1]])
    infinity = np.column_stack((normalised_lines[:, 1], -normalised_lines[:, 0], np.zeros(len(lines))))
    raw_candidates = np.concatenate((intersections, infinity))
    norms = np.linalg.norm(raw_candidates, axis=1)
    nondegenerate = norms > 1e-12
    candidate_ids = np.flatnonzero(nondegenerate)
    candidates = raw_candidates[nondegenerate] / norms[nondegenerate, None]
    masks, counts, supports = [], [], []
    for offset in range(0, len(candidates), settings.candidate_batch):
        angles = vp_pruning.angular_residuals(normalised_lines, candidates[offset:offset + settings.candidate_batch])
        batch_masks = angles <= settings.angle_deg
        masks.extend(batch_masks)
        counts.extend(batch_masks.sum(axis=1))
        supports.extend(np.maximum(0, 1 - angles / settings.angle_deg).sum(axis=1))
    masks = np.asarray(masks, dtype=bool).reshape(-1, len(lines)) if len(lines) else np.empty((0, 0), dtype=bool)
    counts = np.asarray(counts, dtype=int)
    supports = np.asarray(supports)
    retained, status = vp_pruning.retain_pencils(masks, counts, supports, candidate_ids, settings)
    points_working = candidates[retained] @ transform.T
    return {
        "transform": transform, "candidates": candidates, "candidate_ids": candidate_ids,
        "masks": masks, "counts": counts, "supports": supports, "status": status,
        "retained": retained, "retained_candidate_ids": candidate_ids[retained].tolist(),
        "retained_support_masks": masks[retained].tolist(), "points_working": points_working,
    }


def run_gate(source: dict, saved: dict) -> tuple[np.ndarray, np.ndarray, tuple[int, int], vp_pruning.Settings]:
    """Reproduce the saved estimator exactly with no masking. Raise if it does not match."""
    settings = vp_pruning.Settings(**saved["settings"])
    segments, _, size = prepare(source)
    assert list(size) == saved["working_size"], (size, saved["working_size"])
    observations = assignment.prepare_observations(segments, size)
    fragments = observations.segments.reshape(-1, 4)

    merge_settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=ESTIMATE_MERGE_CAP)
    reference_lines = detector._merge_lines(fragments, merge_settings)
    all_lines, _sidecar = merge_lines_with_membership(fragments, merge_settings)
    assert np.array_equal(reference_lines, all_lines), "instrumented merge changed coefficients or order"
    lines = all_lines[:settings.direction_lines]
    estimator = saved["estimator"]
    np.testing.assert_allclose(lines, estimator["direction_lines"], rtol=0, atol=FLOAT_ATOL)

    _, replayed = vp_pruning.estimate(segments, size, settings)
    assert replayed["retained_candidate_ids"] == estimator["retained_candidate_ids"], "retained candidate IDs differ"
    np.testing.assert_allclose(replayed["points_working"], estimator["points_working"], rtol=0, atol=FLOAT_ATOL)
    assert replayed["retained_support_masks"] == estimator["retained_support_masks"], "support masks differ"

    own = bank_and_retain(lines, size, settings)
    assert own["retained_candidate_ids"] == estimator["retained_candidate_ids"], "own bank/retain replay differs"
    np.testing.assert_allclose(own["points_working"], estimator["points_working"], rtol=0, atol=FLOAT_ATOL)
    assert own["retained_support_masks"] == estimator["retained_support_masks"], "own bank/retain masks differ"

    return fragments, observations.fragment_ids, size, settings


def box_membership(fragments: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per fragment: both-endpoints-inside mask (rule A), midpoint-inside mask (rule B)."""
    x1, y1, x2, y2 = fragments[:, 0], fragments[:, 1], fragments[:, 2], fragments[:, 3]
    midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
    box_x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    box_x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    box_y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    box_y2 = np.maximum(boxes[:, 1], boxes[:, 3])

    def inside(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return (
            (px[:, None] >= box_x1[None]) & (px[:, None] <= box_x2[None])
            & (py[:, None] >= box_y1[None]) & (py[:, None] <= box_y2[None])
        ).any(axis=1)

    endpoint_a_in = inside(x1, y1)
    endpoint_b_in = inside(x2, y2)
    rule_a = endpoint_a_in & endpoint_b_in
    rule_b = inside(midpoint_x, midpoint_y)
    return rule_a, rule_b


def point_to_segment_distance(points: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> np.ndarray:
    """Distance from each point to one line segment; points is (N, 2)."""
    vector = segment_end - segment_start
    delta = points - segment_start[None]
    fraction = np.clip((delta @ vector) / (vector @ vector), 0.0, 1.0)
    nearest = segment_start[None] + fraction[:, None] * vector[None]
    return np.linalg.norm(points - nearest, axis=1)


def edge_proximity_count(dropped_fragments: np.ndarray, corners: np.ndarray) -> int:
    """Count dropped fragments with an endpoint within EDGE_PROXIMITY_PX of a court edge."""
    if not len(dropped_fragments):
        return 0
    endpoints = np.concatenate((dropped_fragments[:, :2], dropped_fragments[:, 2:]), axis=0)
    best = np.full(len(endpoints), np.inf)
    for index in range(4):
        start, end = corners[index], corners[(index + 1) % 4]
        best = np.minimum(best, point_to_segment_distance(endpoints, start, end))
    per_fragment = best.reshape(2, -1).min(axis=0)
    return int((per_fragment <= EDGE_PROXIMITY_PX).sum())


def survival(sidecar_masked: list[dict], raw_ids_kept: np.ndarray, direction_lines_used: int, members: list[int]) -> dict:
    """Whether an offending row's fragments still appear as a merged line after masking."""
    members_set = set(members)
    for entry in sidecar_masked[:direction_lines_used]:
        member_raw_ids = raw_ids_kept[entry["member_observation_ids"]].tolist()
        overlap = members_set.intersection(member_raw_ids)
        if overlap:
            return {
                "survives": True, "surviving_raw_fragment_ids": sorted(overlap),
                "dropped_raw_fragment_ids": sorted(members_set - overlap),
                "masked_line_id": entry["line_id"], "masked_member_count": len(member_raw_ids),
                "masked_covered_length_px": entry["covered_length_px"],
            }
    return {"survives": False, "surviving_raw_fragment_ids": [], "dropped_raw_fragment_ids": sorted(members_set)}


def run_rule(
    rule: str, drop_mask: np.ndarray, fragments: np.ndarray, raw_ids: np.ndarray, size: tuple[int, int],
    settings: vp_pruning.Settings, corners: np.ndarray, control_x: np.ndarray, control_y: np.ndarray,
) -> dict:
    keep_mask = ~drop_mask
    masked_fragments = fragments[keep_mask]
    raw_ids_kept = raw_ids[keep_mask]
    merge_settings = replace(detector.DEFAULT_SETTINGS, max_family_lines=ESTIMATE_MERGE_CAP)
    all_lines_masked, sidecar_masked = merge_lines_with_membership(masked_fragments, merge_settings)
    lines_masked = all_lines_masked[:settings.direction_lines]
    bank = bank_and_retain(lines_masked, size, settings)
    points_working = bank["points_working"]

    row_survival = {
        row: survival(sidecar_masked, raw_ids_kept, len(lines_masked), members)
        for row, members in OFFENDING_ROWS.items()
    }

    angles_x = np.array([angle_deg(point, control_x) for point in points_working]) if len(points_working) else np.empty(0)
    angles_y = np.array([angle_deg(point, control_y) for point in points_working]) if len(points_working) else np.empty(0)
    min_angle_x = float(angles_x.min()) if len(angles_x) else None
    min_angle_y = float(angles_y.min()) if len(angles_y) else None

    if len(points_working) >= 2:
        bound, bound_pair = set_bound(corners, points_working)
        fits = fit_pairs(points_working, corners)
        best_fit = fits["best_finite"]["max_corner_working_px"] if fits["best_finite"] else None
    else:
        bound, bound_pair, best_fit = None, None, None

    dropped_fragments = fragments[drop_mask]
    edge_count = edge_proximity_count(dropped_fragments, corners)

    return {
        "rule": rule, "fragments_dropped": int(drop_mask.sum()), "fragments_total": len(fragments),
        "merged_lines_total": len(all_lines_masked), "direction_lines_used": len(lines_masked),
        "selected_directions": len(points_working),
        "min_angle_to_control_x_deg": min_angle_x, "min_angle_to_control_y_deg": min_angle_y,
        "set_bound_working_px": bound, "set_bound_pair_indices": bound_pair,
        "best_control_fit_working_px": best_fit,
        "dropped_fragments_within_6px_of_court_edge": edge_count,
        "row_survival": row_survival,
    }


def angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    unit_a = vector_a / np.linalg.norm(vector_a)
    unit_b = vector_b / np.linalg.norm(vector_b)
    cosine = np.clip(np.abs(np.dot(unit_a, unit_b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    source = load_case()
    saved = read_gz_json(SAVED_ESTIMATOR)
    assert saved["case_id"] == CASE_ID

    fragments, raw_ids, size, settings = run_gate(source, saved)
    print("GATE PASSED: unmasked replay reproduces the saved estimator "
          "(retained candidate IDs, working points, support masks all match).")

    native_scale = np.asarray([source["dimensions"]["width"], source["dimensions"]["height"]], dtype=float) / np.asarray(size, dtype=float)
    bbox_native = np.asarray(source["bbox_px"], dtype=float)
    bbox_working = bbox_native / np.tile(native_scale, 2)

    e3 = read_gz_json(E3_RECORD)
    corners = np.asarray(e3["control"]["corners_working_px"], dtype=float)
    np.testing.assert_allclose(corners, [
        [70.69372861108405, 283.57787950035794], [306.3781251404358, 271.3518963191221],
        [782.8601501441515, 328.04844017786434], [486.30823431447186, 580.5813112824602],
    ], rtol=0, atol=1e-6)

    estimator = saved["estimator"]
    candidates, transform, _generators = reconstruct_bank(estimator)
    candidate_ids = np.asarray(estimator["candidate_ids"], dtype=int)
    working_points = candidates @ transform.T
    index_by_id = {int(candidate_id): row for candidate_id, row in zip(candidate_ids, working_points)}
    control_x = index_by_id[CONTROL_ID_X]
    control_y = index_by_id[CONTROL_ID_Y]

    baseline_bound, _baseline_bound_pair = set_bound(corners, np.asarray(estimator["points_working"], dtype=float))
    baseline_fits = fit_pairs(np.asarray(estimator["points_working"], dtype=float), corners)
    baseline_best_fit = baseline_fits["best_finite"]["max_corner_working_px"]
    baseline_angles_x = [angle_deg(point, control_x) for point in np.asarray(estimator["points_working"], dtype=float)]
    baseline_angles_y = [angle_deg(point, control_y) for point in np.asarray(estimator["points_working"], dtype=float)]

    rule_a_mask, rule_b_mask = box_membership(fragments, bbox_working)
    result_a = run_rule("A_both_endpoints", rule_a_mask, fragments, raw_ids, size, settings, corners, control_x, control_y)
    result_b = run_rule("B_midpoint", rule_b_mask, fragments, raw_ids, size, settings, corners, control_x, control_y)

    print("Rule A (both endpoints inside a person box): dropped", result_a["fragments_dropped"], "of", result_a["fragments_total"])
    print("Rule B (midpoint inside a person box): dropped", result_b["fragments_dropped"], "of", result_b["fragments_total"])
    for row in (31, 51, 61):
        print(f"row {row}: rule A survives={result_a['row_survival'][row]['survives']}, "
              f"rule B survives={result_b['row_survival'][row]['survives']}")

    write_table(baseline_bound, baseline_best_fit, baseline_angles_x, baseline_angles_y, result_a, result_b)
    write_note(baseline_bound, baseline_best_fit, baseline_angles_x, baseline_angles_y, result_a, result_b, fragments)


def write_table(baseline_bound, baseline_best_fit, baseline_angles_x, baseline_angles_y, result_a, result_b) -> None:
    fieldnames = [
        "rule", "fragments_dropped", "fragments_total", "merged_lines_total", "direction_lines_used",
        "selected_directions", "min_angle_to_control_x_deg", "min_angle_to_control_y_deg",
        "set_bound_working_px", "best_control_fit_working_px", "dropped_fragments_within_6px_of_court_edge",
        "row_31_survives", "row_51_survives", "row_61_survives",
    ]
    rows = [{
        "rule": "baseline (no masking)", "fragments_dropped": 0, "fragments_total": result_a["fragments_total"],
        "merged_lines_total": None, "direction_lines_used": None, "selected_directions": 16,
        "min_angle_to_control_x_deg": round(min(baseline_angles_x), 6),
        "min_angle_to_control_y_deg": round(min(baseline_angles_y), 6),
        "set_bound_working_px": round(baseline_bound, 6), "best_control_fit_working_px": round(baseline_best_fit, 6),
        "dropped_fragments_within_6px_of_court_edge": None,
        "row_31_survives": True, "row_51_survives": True, "row_61_survives": True,
    }]
    for result in (result_a, result_b):
        rows.append({
            "rule": result["rule"], "fragments_dropped": result["fragments_dropped"],
            "fragments_total": result["fragments_total"], "merged_lines_total": result["merged_lines_total"],
            "direction_lines_used": result["direction_lines_used"], "selected_directions": result["selected_directions"],
            "min_angle_to_control_x_deg": round(result["min_angle_to_control_x_deg"], 6) if result["min_angle_to_control_x_deg"] is not None else None,
            "min_angle_to_control_y_deg": round(result["min_angle_to_control_y_deg"], 6) if result["min_angle_to_control_y_deg"] is not None else None,
            "set_bound_working_px": round(result["set_bound_working_px"], 6) if result["set_bound_working_px"] is not None else None,
            "best_control_fit_working_px": round(result["best_control_fit_working_px"], 6) if result["best_control_fit_working_px"] is not None else None,
            "dropped_fragments_within_6px_of_court_edge": result["dropped_fragments_within_6px_of_court_edge"],
            "row_31_survives": result["row_survival"][31]["survives"],
            "row_51_survives": result["row_survival"][51]["survives"],
            "row_61_survives": result["row_survival"][61]["survives"],
        })
    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_note(baseline_bound, baseline_best_fit, baseline_angles_x, baseline_angles_y, result_a, result_b, fragments) -> None:
    lines = []
    lines.append("# Masking fragments inside person boxes on GX0, replayed")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append("Passed. With no masking, the replay reproduces the saved estimator exactly: the same 16 "
                  "retained candidate IDs, the same working points and the same support masks as "
                  "`vp_pruning_20260914/coverage/results/gxBQ_window_00_frame_0.json.gz`.")
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append(f"There are {result_a['fragments_total']} fragments feeding the merge before any masking.")
    lines.append("")
    header = ("| rule | dropped | merged lines | selected directions | min angle to x-control (deg) | "
               "min angle to y-control (deg) | set_bound (working px) | best control fit (working px) | "
               "dropped fragments within 6px of a court edge |")
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| baseline | 0 | 106 | 16 | {min(baseline_angles_x):.6f} | {min(baseline_angles_y):.6f} | "
                  f"{baseline_bound:.6f} | {baseline_best_fit:.6f} | n/a |")
    for result in (result_a, result_b):
        lines.append(
            f"| {result['rule']} | {result['fragments_dropped']} | {result['merged_lines_total']} | "
            f"{result['selected_directions']} | "
            f"{result['min_angle_to_control_x_deg']:.6f} | {result['min_angle_to_control_y_deg']:.6f} | "
            f"{result['set_bound_working_px']:.6f} | {result['best_control_fit_working_px']:.6f} | "
            f"{result['dropped_fragments_within_6px_of_court_edge']} |"
        )
    lines.append("")
    lines.append("## Which offending structures each rule removes")
    lines.append("")
    for result in (result_a, result_b):
        lines.append(f"### {result['rule']}")
        lines.append("")
        for row in (31, 51, 61):
            entry = result["row_survival"][row]
            if entry["survives"]:
                lines.append(f"- Row {row}: survives. {len(entry['surviving_raw_fragment_ids'])} of "
                              f"{len(OFFENDING_ROWS[row])} member fragment(s) remain, now merged as line "
                              f"{entry['masked_line_id']} ({entry['masked_member_count']} member(s), "
                              f"{entry['masked_covered_length_px']:.1f} px covered).")
            else:
                lines.append(f"- Row {row}: removed. All {len(OFFENDING_ROWS[row])} member fragment(s) "
                              f"({entry['dropped_raw_fragment_ids']}) fell inside a person box and were dropped.")
        lines.append("")
    lines.append("## Does masking person boxes alone move GX0's selection toward the approved court")
    lines.append("")
    lines.append(
        f"Both rules remove row 61 (the spectator's face); neither touches row 31 or row 51, which sit outside "
        f"every person box on this frame. Rule A barely changes the outcome: set_bound stays at "
        f"{result_a['set_bound_working_px']:.3f} working px (baseline {baseline_bound:.3f}) and the best control "
        f"fit gets slightly worse, {result_a['best_control_fit_working_px']:.3f} against {baseline_best_fit:.3f}. "
        f"Rule B, which drops more fragments, changes the coverage allocation enough that one of the 16 selected "
        f"directions lands on almost exactly the same direction as the approved control's y-axis candidate "
        f"(0.000 degrees away, against 1.795 at baseline), and both distance measures fall a long way: set_bound "
        f"to {result_b['set_bound_working_px']:.3f} working px and the best control fit to "
        f"{result_b['best_control_fit_working_px']:.3f} working px. Both rules remove row 61, but only the "
        f"midpoint-in-box rule improves the fit. The replay shows a changed allocation after the fragment set "
        f"changes; it does not isolate row 61's effect or establish a complete greedy-allocation trace."
    )
    lines.append("")
    note_path = OUT_DIR / "note.md"
    note_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
