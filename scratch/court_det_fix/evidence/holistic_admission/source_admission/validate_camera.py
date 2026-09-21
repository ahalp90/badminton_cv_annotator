"""Compare the audit camera vectorisation with the frozen scalar helper."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import audit_admission as audit


ROOT = Path(__file__).resolve().parents[5]
OUTPUT = Path(__file__).resolve().parent / "results_gx5"
CASE_ID = "gxBQ_window_00_frame_5"


def main() -> None:
    detector = audit.load_helpers(ROOT)[0]
    camera_helper = audit.load_camera_helper(ROOT)
    automatic = audit.read_json_gz(OUTPUT / f"{CASE_ID}.json.gz")
    with np.load(OUTPUT / f"{CASE_ID}.npz") as data:
        corners = data["corners"]
        rectangle_ids = data["rectangle_ids"]
        templates = data["templates"]

    working_size = (960, 540)
    native_size = (1920, 1080)
    native_scale = np.asarray(native_size, dtype=np.float64) / working_size
    pack_root = ROOT if (ROOT / "frozen_views").exists() else ROOT / "scratch/court_det_fix"
    source, _ = audit.load_case(pack_root, CASE_ID)
    _, families, size = audit.prepare_source(source, detector)
    _, _, _, rectangle_population, _ = audit.load_helpers(ROOT)
    quads, _, all_ids = rectangle_population(families, size)
    selected_ids = automatic["ordering"]["selected_rectangle_ids_first_cap"]
    quad_by_id = {int(rectangle_id): quad for rectangle_id, quad in zip(all_ids, quads, strict=True)}
    rectangle_matrices = {
        int(rectangle_id): cv2.getPerspectiveTransform(
            detector.UNIT_CORNERS, quad_by_id[int(rectangle_id)].astype(np.float32)
        )
        for rectangle_id in selected_ids
        if cv2.contourArea(quad_by_id[int(rectangle_id)].astype(np.float32)) >= 100
    }
    vector_errors = np.empty(len(corners), dtype=np.float64)
    for start in range(0, len(corners), 2048):
        stop = min(start + 2048, len(corners))
        homographies = np.asarray(
            [rectangle_matrices[int(rectangle_id)] @ detector.TEMPLATE_TRANSFORMS[int(template)]
             for rectangle_id, template in zip(rectangle_ids[start:stop], templates[start:stop], strict=True)],
            dtype=np.float64,
        )
        vector_errors[start:stop] = audit.vector_camera_errors(homographies, working_size)
    nearest = np.argsort(np.abs(vector_errors - audit.CAMERA_LIMIT), kind="stable")[:512]
    evenly_spaced = np.linspace(0, len(corners) - 1, 177, dtype=np.int64)
    target_matches = np.flatnonzero((rectangle_ids == 85795) & (templates == 72))
    target = target_matches[:1]
    indices = np.unique(np.concatenate((nearest, evenly_spaced, target)))
    scalar_errors = np.empty(len(indices), dtype=np.float64)
    sample_homographies = np.asarray(
        [rectangle_matrices[int(rectangle_ids[index])] @ detector.TEMPLATE_TRANSFORMS[int(templates[index])]
         for index in indices],
        dtype=np.float64,
    )
    sample_corners, _ = detector.project(sample_homographies, detector.CORNER_COURT_M)
    for position, native_corners_working in enumerate(sample_corners):
        native_corners = (native_corners_working * native_scale).astype(np.float32)
        scalar_errors[position] = camera_helper.camera(native_corners, native_size)[0]
    vector_sample = vector_errors[indices]
    differences = np.abs(vector_sample - scalar_errors)
    result = {
        "case_id": CASE_ID,
        "source": "cached ungated GX5 audit arrays; no references used",
        "scalar_reconstruction": "original proposal homography projected to float32 native corners, matching W5 gate",
        "all_real_candidates": int(len(corners)),
        "sample_count": int(len(indices)),
        "near_cutoff_sample_count": int(np.sum(np.abs(vector_sample - audit.CAMERA_LIMIT) <= 1e-3)),
        "working_size": list(working_size),
        "native_size": list(native_size),
        "camera_limit": audit.CAMERA_LIMIT,
        "max_abs_difference": float(np.max(differences)),
        "p99_abs_difference": float(np.quantile(differences, 0.99)),
        "threshold_crossings": int(np.sum((vector_sample <= audit.CAMERA_LIMIT) != (scalar_errors <= audit.CAMERA_LIMIT))),
        "scalar_eligible_count": int(np.sum(scalar_errors <= audit.CAMERA_LIMIT)),
        "vector_eligible_count": int(np.sum(vector_sample <= audit.CAMERA_LIMIT)),
        "target_in_sample": bool(len(target)),
        "target_vector_error": float(vector_errors[target[0]]) if len(target) else None,
        "target_scalar_error": float(scalar_errors[np.flatnonzero(indices == target[0])[0]]) if len(target) else None,
    }
    (Path(__file__).resolve().parent / "camera_validation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
