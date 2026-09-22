"""Compare saved line/template shortlists with the prior W5 anchor geometries."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

import audit_admission as audit


ROOT = Path(__file__).resolve().parents[5]
AUDIT_ROOT = Path(__file__).resolve().parent
W5_ROOT = ROOT / "scratch/court_det_fix/evidence/holistic_admission/runs/w5_stage5_20260920"
CASES = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "am2_window_00_frame_150",
    "am2_window_01_frame_28019",
    "am3_window_00_frame_0",
    "shuttleset_03_scene_0017",
    "shuttleset_03_scene_0019",
    "shuttleset_03_scene_0016",
    "shuttleset_21_scene_0020",
)


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def output_dir(case_id: str) -> Path:
    if case_id == "gxBQ_window_00_frame_5":
        return AUDIT_ROOT / "results_gx5_camera"
    return AUDIT_ROOT / "results_w5_camera"


def anchor_for(case_id: str, review: dict) -> tuple[np.ndarray, dict]:
    if case_id == "gxBQ_window_00_frame_5":
        target = json.loads((AUDIT_ROOT / "gx5_posthoc_target.json").read_text())
        return np.asarray(target["corners_px"], dtype=np.float64), {
            "source": target["source"],
            "origin_key": f"rectangle_{target['pair_product_id']}:template_{target['template_index']}",
            "reference_distance_px": None,
            "camera_error": None,
        }
    nearest = review[case_id]["reference_near"]
    candidate = review[case_id]["candidates"][nearest["origin_key"]]
    return np.asarray(candidate["corners_px"], dtype=np.float64), {
        "source": "W5 stage-5 approved supplied-direction control nearest to the saved reference",
        "origin_key": nearest["origin_key"],
        "reference_distance_px": float(nearest["distance_px"]),
        "camera_error": candidate["gates"].get("camera_error"),
    }


def assess_case(case_id: str, review: dict) -> dict:
    directory = output_dir(case_id)
    automatic_name = "automatic.json.gz" if case_id == "gxBQ_window_00_frame_5" else f"{case_id}.json.gz"
    automatic = read_json_gz(directory / automatic_name)
    with np.load(directory / f"{case_id}.npz") as data:
        arrays = {name: data[name] for name in data.files}
    anchor_native, anchor_metadata = anchor_for(case_id, review)
    native_scale = np.asarray(automatic["coordinate_semantics"]["native_scale"], dtype=np.float64)
    anchor_working = anchor_native / native_scale
    camera_eligible = arrays["camera_errors"] <= audit.CAMERA_LIMIT
    eligible_indices = np.flatnonzero(camera_eligible)
    result = {
        "case_id": case_id,
        "anchor": {**anchor_metadata, "corners_native": anchor_native.tolist(), "corners_working": anchor_working.tolist()},
        "variants": {},
    }
    for variant_name, mode in (
        ("family_map_mean_directions", "family_means"),
        ("family_map_minimum_directions", "family_means"),
        ("union_map_mean_directions", "union_means"),
        ("union_map_minimum_directions", "union_means"),
    ):
        means = arrays[mode]
        scores = means.mean(axis=1) if "mean" in variant_name else means.min(axis=1)
        order = eligible_indices[np.lexsort((
            arrays["templates"][eligible_indices],
            arrays["rectangle_ids"][eligible_indices],
            -scores[eligible_indices],
        ))]
        selected, scanned = audit.greedy_diverse(order, arrays["corners"], max(audit.CAPS))
        pool_errors = np.linalg.norm(arrays["corners"][order] - anchor_working, axis=2).max(axis=1)
        caps = {}
        for cap in audit.CAPS:
            prefix = selected[:cap]
            errors = np.linalg.norm(arrays["corners"][prefix] - anchor_working, axis=2).max(axis=1)
            best_index = int(prefix[np.argmin(errors)]) if len(errors) else None
            caps[str(cap)] = {
                "count": int(len(prefix)),
                "closest_max_corner_error_working": float(errors.min()) if len(errors) else None,
                "closest_proposal_id": (
                    f"rectangle_{int(arrays['rectangle_ids'][best_index])}:template_{int(arrays['templates'][best_index])}"
                    if best_index is not None else None
                ),
            }
        pool_index = int(order[np.argmin(pool_errors)]) if len(pool_errors) else None
        result["variants"][variant_name] = {
            "geometry_valid_count": int(len(scores)),
            "camera_eligible_count": int(len(order)),
            "scanned_for_256": int(scanned),
            "closest_pool_max_corner_error_working": float(pool_errors.min()) if len(pool_errors) else None,
            "closest_pool_proposal_id": (
                f"rectangle_{int(arrays['rectangle_ids'][pool_index])}:template_{int(arrays['templates'][pool_index])}"
                if pool_index is not None else None
            ),
            "caps": caps,
        }
    return result


def main() -> None:
    review = json.loads((W5_ROOT / "review_candidates.json").read_text())
    result = {
        "schema": "line-template-admission-anchor-posthoc/1",
        "automatic_outputs": "saved before W5 review anchors were loaded",
        "cases": {case_id: assess_case(case_id, review) for case_id in CASES},
    }
    path = AUDIT_ROOT / "anchor_posthoc.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(path)


if __name__ == "__main__":
    main()
