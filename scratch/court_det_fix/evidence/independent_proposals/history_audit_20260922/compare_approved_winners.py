"""Trace eight previously approved gallery winners through six current pools."""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("scratch/court_det_fix")
OLD = (
    ROOT
    / "evidence/independent_proposals/development/player_guided/20260914/automatic_axes/collected/all_camera"
)
CURRENT = (
    ROOT
    / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/case_records"
)
JOBS = {
    "am2_window_00_frame_150": ["30:33"],
    "am3_window_00_frame_0": ["43:22603", "43:22627"],
    "shuttleset_03_scene_0016": ["1:60", "1:90"],
    "shuttleset_03_scene_0017": ["1:80"],
    "shuttleset_03_scene_0019": ["1:60"],
    "shuttleset_21_scene_0020": ["0:2"],
}


def read(path: Path) -> dict:
    return json.loads(gzip.decompress(path.read_bytes()))


def compare(job: tuple[str, list[str], dict]) -> dict:
    case_id, candidate_ids, numeric = job
    old, current = (
        read(OLD / f"{case_id}.json.gz"),
        read(CURRENT / f"{case_id}.json.gz"),
    )
    candidates = current["parents"] + current["valid_children"]
    current_by_id = {candidate["origin_key"]: candidate for candidate in candidates}
    scale = np.asarray(numeric["working_size"]) / numeric["native_size"]
    rows = []
    for candidate_id in candidate_ids:
        source = next(
            entry for entry in old["entries"] if entry["candidate_id"] == candidate_id
        )
        target = np.asarray(source["corners_px"])
        comparisons = []
        for candidate in candidates:
            points = np.asarray(candidate["corners_px"])
            distance = min(
                np.linalg.norm((points - target) * scale, axis=1).max(),
                np.linalg.norm(
                    (np.roll(points, 2, axis=0) - target) * scale, axis=1
                ).max(),
            )
            comparisons.append(
                {
                    "origin_key": candidate["origin_key"],
                    "max_corner_disagreement_working_px": float(distance),
                }
            )
        lookup = {entry["origin_key"]: entry for entry in comparisons}
        parent_key = "G0:" + candidate_id
        parent = current_by_id[parent_key]
        lookup[parent_key].update(
            camera_eligible=parent["camera_eligible"],
            player_gate=parent["historical"]["historical_fullcourt"],
            full_rank_position=current["rankings"]["C"]["provisional_rank"].index(
                parent_key
            )
            + 1,
            paint_score=parent["evidence"]["q_paint10_span_weighted"],
        )
        rows.append(
            {
                "old_candidate_id": candidate_id,
                "old_corners_native_px": target.tolist(),
                "same_id_current_parent": lookup.get("G0:" + candidate_id),
                "nearest_current": min(
                    comparisons,
                    key=lambda entry: entry["max_corner_disagreement_working_px"],
                ),
                "selected": {
                    arm: lookup[selection["gated"]]
                    for arm, selection in numeric["selections"].items()
                },
            }
        )
    return {"case_id": case_id, "witnesses": rows}


if __name__ == "__main__":
    numeric = read(ROOT / "wider_evaluation/runs/20260922/numeric_fit.json.gz")
    by_id = {case["case_id"]: case for case in numeric["cases"]}
    jobs = [(case_id, ids, by_id[case_id]) for case_id, ids in JOBS.items()]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(compare, jobs))
    destination = (
        ROOT
        / "evidence/independent_proposals/history_audit_20260922/approved_winner_retention.json.gz"
    )
    result = {
        "workers": 6,
        "scope": "Eight prior visual approvals across six views; chosen after historical review, not one automatic policy.",
        "rulings": "experiments/annotator/independent_court/recorded/player_guided/projective_patterns/automatic_axes_visual_judgements.md",
        "old_record_root": str(OLD),
        "current_record_root": str(CURRENT),
        "cases": rows,
    }
    destination.write_bytes(
        gzip.compress(json.dumps(result, allow_nan=False).encode(), mtime=0)
    )
    for case in rows:
        for witness in case["witnesses"]:
            print(
                case["case_id"],
                witness["old_candidate_id"],
                witness["same_id_current_parent"],
                witness["selected"],
                flush=True,
            )
