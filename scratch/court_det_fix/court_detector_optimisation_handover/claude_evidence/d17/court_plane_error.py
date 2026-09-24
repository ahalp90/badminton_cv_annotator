"""Pixel error and court-floor error (cm) for budget 16, budget 12 and the accepted court (throwaway).

Floor error: fit one homography from the reference clicks to their court positions, map each
predicted reference point back onto the court floor through it, and measure the distance to
where that point should be. Landmark views use every clicked landmark (least squares);
corner-only views use the four clicked corners (exact fit, so no click-noise averaging).

Usage (repo root, scratch/court_det_fix/d17_timing on PYTHONPATH):
    court_plane_error.py RUN_DIR ACCEPTED_RESULTS
RUN_DIR holds budget16/d17/*.json.gz and budget12/d17/*.json.gz.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

import compare_d17 as c

# Detector corner order in court metres (across width, along length); checked against
# the inverse of a corrected homography.
CORNERS_M = np.array([[0.0, 0.0], [6.1, 0.0], [6.1, 13.4], [0.0, 13.4]])


def reference_points(reference: dict, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Court positions (m) and clicked pixels (native) of the reference points."""
    landmarks = reference.get("landmarks")
    if landmarks:
        return (np.asarray([landmark["court_m"] for landmark in landmarks], dtype=float),
                np.asarray([landmark["image_px"] for landmark in landmarks], dtype=float))
    clicked = np.asarray(reference["corners_px"], dtype=float)
    predicted = np.asarray(geometry["corners_native_px"], dtype=float)
    # Same 180-degree choice as paired_reference_analysis.reference_errors; the court is
    # symmetric under that rotation, so floor distances do not depend on it.
    direct = np.linalg.norm(predicted - clicked, axis=1).max()
    rotated = np.linalg.norm(predicted[[2, 3, 0, 1]] - clicked, axis=1).max()
    court_m = CORNERS_M[[2, 3, 0, 1]] if rotated < direct else CORNERS_M
    return court_m, clicked


def errors(summary_sizes: dict, geometry: dict, reference: dict) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(summary_sizes["native_size_wh"], float) / np.asarray(summary_sizes["working_size_wh"], float)
    court_m, clicked_px = reference_points(reference, geometry)
    reference_h, _ = cv2.findHomography(court_m, clicked_px, 0)
    predicted_px = c.statistics.project(geometry["homography_working"], court_m.tolist(),
                                        summary_sizes["native_size_wh"], summary_sizes["working_size_wh"])
    pixel_error = np.linalg.norm((predicted_px - clicked_px) / scale, axis=1)
    predicted_m = cv2.perspectiveTransform(np.asarray(predicted_px, float)[None], np.linalg.inv(reference_h))[0]
    return pixel_error, 100 * np.linalg.norm(predicted_m - court_m, axis=1)


def main() -> None:
    run_dir, accepted_path = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = c.statistics.read(c.statistics.MANIFEST)
    rows = {row["case_id"]: row for row in manifest["cases"]}
    references = c.statistics.load_references(manifest)
    accepted = {row["case_id"]: row for row in c.read(accepted_path)["selections"]}
    print("case | reference | px median/max: b16, b12, accepted | floor cm median/max: b16, b12, accepted")
    for path in sorted((run_dir / "budget16/d17").glob("*.json.gz")):
        case_id = path.name.removesuffix(".json.gz")
        reference = c.measurable_reference(case_id, rows, references)
        if reference is None:
            continue
        geometries, sizes = {}, None
        for budget in ("16", "12"):
            summary = c.read(run_dir / f"budget{budget}/d17/{case_id}.json.gz")
            sizes = {"native_size_wh": summary["native_size_wh"], "working_size_wh": summary["working_size_wh"]}
            geometries[f"b{budget}"] = c.corrected_geometry(summary["selection"]["polarity_refit"])
        geometries["accepted"] = c.corrected_geometry(accepted[case_id]) if case_id in accepted else None
        pixel_cells, floor_cells = [], []
        for geometry in geometries.values():
            if geometry is None:
                pixel_cells.append("-")
                floor_cells.append("-")
                continue
            pixel_error, floor_error_cm = errors(sizes, geometry, reference)
            pixel_cells.append(f"{np.median(pixel_error):.1f}/{pixel_error.max():.1f}")
            floor_cells.append(f"{np.median(floor_error_cm):.0f}/{floor_error_cm.max():.0f}")
        kind = f"{len(reference['landmarks'])} landmarks" if reference.get("landmarks") else "4 corners"
        print(f"{case_id} | {kind} | {', '.join(pixel_cells)} | {', '.join(floor_cells)}")


if __name__ == "__main__":
    main()
