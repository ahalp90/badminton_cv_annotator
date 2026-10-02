"""Check whether later GX frames' automatic direction selections keep the two
precise GX0 control directions (candidates 1183 for x, 122 for y).

Read-only on every input under the repository. Writes only into this OUT folder.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/automatic_axes"
)))
sys.path.insert(0, str(REPO_ROOT / (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/vp_pruning"
)))
sys.path.insert(0, str(REPO_ROOT / (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/marking_diagnosis"
)))

from diagnose_direction_bank import reconstruct_bank

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR))
from geometry_certificate import set_bound

TEMPORAL_RECORDS = REPO_ROOT / (
    "experiments/annotator/independent_court/recorded/player_guided/projective_patterns/"
    "evaluation/temporal_records.json.gz"
)
SAVED_GX0_ESTIMATOR = REPO_ROOT / (
    "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260914/vp_pruning/"
    "coverage/results/gxBQ_window_00_frame_0.json.gz"
)
E2_RECORD = REPO_ROOT / (
    "scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/"
    "e2/gxBQ_window_00_frame_0.json.gz"
)
E3_RECORD = REPO_ROOT / (
    "scratch/court_det_fix/direction_agreement/runs/direction_agreement_20260915_144900/"
    "e3/gxBQ_window_00_frame_0.json.gz"
)
GX0_CONTROL = REPO_ROOT / (
    "experiments/annotator/independent_court/recorded/player_guided/projective_patterns/"
    "gx0_control_measurements.json.gz"
)

CONTROL_ID_X = 1183
CONTROL_ID_Y = 122
VALID_FRAME_INDICES = {0, 5}


def read_gz_json(path: Path) -> dict:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def angle_deg(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Angle in degrees between two homogeneous 3-vectors, sign-invariant."""
    unit_a = vector_a / np.linalg.norm(vector_a)
    unit_b = vector_b / np.linalg.norm(vector_b)
    cosine = np.clip(np.abs(np.dot(unit_a, unit_b)), 0.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def run_gate() -> tuple[np.ndarray, dict[int, np.ndarray], float, float]:
    """Reconstruct the GX0 bank, cross-check it, and confirm the two bound values."""
    saved = read_gz_json(SAVED_GX0_ESTIMATOR)
    estimator = saved["estimator"]
    candidates, transform, _generators = reconstruct_bank(estimator)
    candidate_ids = np.asarray(estimator["candidate_ids"], dtype=int)
    working_points = candidates @ transform.T

    retained_ids = np.asarray(estimator["retained_candidate_ids"], dtype=int)
    selected_rows = np.searchsorted(candidate_ids, retained_ids)
    np.testing.assert_allclose(
        working_points[selected_rows], estimator["points_working"], rtol=1e-9, atol=1e-9
    )

    temporal = read_gz_json(TEMPORAL_RECORDS)
    frame0_row = next(
        row for row in temporal["rows"]
        if row["video_group"] == "gxBQ" and row["frame_index"] == 0
    )
    frame0_retained = np.asarray(
        frame0_row["direction_estimator"]["retained_candidate_ids"], dtype=int
    )
    if not np.array_equal(frame0_retained, retained_ids):
        raise AssertionError("Frame 0 temporal-record retained IDs do not match the saved GX0 bank.")

    e2 = read_gz_json(E2_RECORD)
    arm_b = e2["arms"]["B"]
    if not np.array_equal(np.asarray(arm_b["leader_candidate_ids"], dtype=int), retained_ids):
        raise AssertionError("E2 arm B leader IDs do not match the saved GX0 bank.")
    b_points_working = np.asarray(arm_b["points_working"], dtype=float)
    np.testing.assert_allclose(
        b_points_working, working_points[selected_rows], rtol=1e-6, atol=1e-6
    )

    gx0_control = read_gz_json(GX0_CONTROL)
    approved_native = np.asarray(gx0_control["approved_corners_native_px"], dtype=float)
    approved_working = approved_native / 2.0

    e3 = read_gz_json(E3_RECORD)
    e3_working = np.asarray(e3["control"]["corners_working_px"], dtype=float)
    np.testing.assert_allclose(approved_working, e3_working, rtol=1e-9, atol=1e-6)

    bound_full, _ = set_bound(approved_working, b_points_working)
    print(f"Gate 1 (16 GX0 directions vs approved control): {bound_full:.6f}")

    full_bank_index = {int(candidate_id): row for candidate_id, row in zip(candidate_ids, working_points)}
    pair_points = np.stack([full_bank_index[CONTROL_ID_X], full_bank_index[CONTROL_ID_Y]])
    bound_pair, _ = set_bound(approved_working, pair_points)
    print(f"Gate 2 (candidates 1183, 122 vs approved control): {bound_pair:.6f}")

    return approved_working, full_bank_index, bound_full, bound_pair


def gate_passes(bound_full: float, bound_pair: float) -> bool:
    return abs(bound_full - 6.022569) < 5e-7 and abs(bound_pair - 0.423931) < 5e-7


def main() -> None:
    approved_working, gx0_id_to_row, bound_full, bound_pair = run_gate()
    if not gate_passes(bound_full, bound_pair):
        print("GATE FAILED. Stopping before computing per-frame results.")
        sys.exit(1)

    control_x = gx0_id_to_row[CONTROL_ID_X]
    control_y = gx0_id_to_row[CONTROL_ID_Y]

    temporal = read_gz_json(TEMPORAL_RECORDS)
    gx_rows = [row for row in temporal["rows"] if row["video_group"] == "gxBQ"]
    gx_rows.sort(key=lambda row: row["frame_index"])

    table_rows = []
    for row in gx_rows:
        frame_index = row["frame_index"]
        nominal_seconds = row["nominal_seconds_from_frame_zero"]
        direction_estimator = row["direction_estimator"]
        retained_ids = np.asarray(direction_estimator["retained_candidate_ids"], dtype=int)
        points_working = np.asarray(direction_estimator["points_working"], dtype=float)

        angles_x = np.array([angle_deg(point, control_x) for point in points_working])
        angles_y = np.array([angle_deg(point, control_y) for point in points_working])
        best_x_index = int(np.argmin(angles_x))
        best_y_index = int(np.argmin(angles_y))

        bound, bound_pair_indices = set_bound(approved_working, points_working)
        validity = "valid" if frame_index in VALID_FRAME_INDICES else "indicative only, camera may have moved"

        table_rows.append({
            "frame_index": frame_index,
            "nominal_seconds": round(float(nominal_seconds), 3),
            "min_angle_to_1183_deg": round(float(angles_x[best_x_index]), 6),
            "closest_retained_id_to_1183": int(retained_ids[best_x_index]),
            "min_angle_to_122_deg": round(float(angles_y[best_y_index]), 6),
            "closest_retained_id_to_122": int(retained_ids[best_y_index]),
            "set_bound_working_px": round(float(bound), 6),
            "set_bound_pair_indices": bound_pair_indices,
            "validity": validity,
        })

    write_outputs(bound_full, bound_pair, table_rows)


def write_outputs(bound_full: float, bound_pair: float, table_rows: list[dict]) -> None:
    import csv

    csv_path = OUT_DIR / "table.csv"
    fieldnames = [
        "frame_index", "nominal_seconds",
        "min_angle_to_1183_deg", "closest_retained_id_to_1183",
        "min_angle_to_122_deg", "closest_retained_id_to_122",
        "set_bound_working_px", "validity",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({key: row[key] for key in fieldnames})

    within_one_degree = [
        row for row in table_rows
        if row["min_angle_to_1183_deg"] <= 1.0 and row["min_angle_to_122_deg"] <= 1.0
    ]
    under_bound = [row for row in table_rows if row["set_bound_working_px"] < 3.5]

    lines = []
    lines.append("# Do later GX frames keep the precise directions")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"Gate 1 (16 GX0 directions vs approved control): {bound_full:.6f}")
    lines.append(f"Gate 2 (candidates 1183, 122 vs approved control): {bound_pair:.6f}")
    lines.append("Both match the required values to six decimals. Coordinates are correct.")
    lines.append("")
    lines.append("## Per-frame table")
    lines.append("")
    header = "| frame | seconds | min angle to 1183 (deg) | closest ID | min angle to 122 (deg) | closest ID | set_bound (working px) | note |"
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in table_rows:
        lines.append(
            f"| {row['frame_index']} | {row['nominal_seconds']} | "
            f"{row['min_angle_to_1183_deg']:.6f} | {row['closest_retained_id_to_1183']} | "
            f"{row['min_angle_to_122_deg']:.6f} | {row['closest_retained_id_to_122']} | "
            f"{row['set_bound_working_px']:.6f} | {row['validity']} |"
        )
    lines.append("")
    lines.append("## Answers")
    lines.append("")
    if within_one_degree:
        frames = ", ".join(str(row["frame_index"]) for row in within_one_degree)
        lines.append(
            f"Frame(s) {frames} keep both control directions within about one degree."
        )
    else:
        lines.append("No frame keeps both control directions within about one degree.")
    if under_bound:
        frames = ", ".join(str(row["frame_index"]) for row in under_bound)
        lines.append(
            f"Frame(s) {frames} have a set_bound under 3.5 working pixels."
        )
    else:
        lines.append("No frame's set_bound falls under 3.5 working pixels.")
    lines.append("")

    note_path = OUT_DIR / "note.md"
    note_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
