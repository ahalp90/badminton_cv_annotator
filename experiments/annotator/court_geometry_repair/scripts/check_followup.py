"""Check retained grouping outputs, fresh-court equality and pilot aggregation.

Run from any directory with Python. An optional positional path selects another
archive, allowing the same checks to inspect a copied or modified bundle.
This reads saved results; it does not run detection or regenerate player features.
"""

import argparse
import gzip
import json
from itertools import combinations
from pathlib import Path
from statistics import fmean, median

import cv2
import numpy as np


def group_disagreement(group: dict, original_records: list[dict]) -> tuple[float, float]:
    """Compare original calibrations at the same 105 image points and four corners."""
    unit_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 15), np.linspace(0, 1, 7)), axis=-1)
    shared = np.asarray(group["corners_native_px"], dtype=np.float32)
    to_image = cv2.getPerspectiveTransform(unit_corners, shared)
    image_points = cv2.perspectiveTransform(grid.reshape(1, -1, 2), to_image)
    by_index = {record["scene_index"]: record for record in original_records}
    projections = []
    corners = []
    for scene_index in group["members"]:
        original = np.asarray(by_index[scene_index]["active_corners_native_px"], dtype=np.float32)
        to_court = cv2.getPerspectiveTransform(original, unit_corners)
        projections.append(cv2.perspectiveTransform(image_points, to_court)[0] * [6.1, 13.4])
        corners.append(original.astype(float))
    max_metres = 0.0
    max_pixels = 0.0
    for first, second in combinations(range(len(corners)), 2):
        max_metres = max(max_metres, float(np.linalg.norm(projections[first] - projections[second], axis=1).max()))
        max_pixels = max(max_pixels, float(np.linalg.norm(corners[first] - corners[second], axis=1).max()))
    return max_metres, max_pixels


def without_case_ids(value: object) -> object:
    """Remove the before/after labels used by the original comparison runner."""
    if isinstance(value, dict):
        return {key: without_case_ids(child) for key, child in value.items() if key != "case_id"}
    if isinstance(value, list):
        return [without_case_ids(child) for child in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1] / "evidence/followup.json.gz")
    args = parser.parse_args()
    with gzip.open(args.archive, "rt") as stream:
        archive = json.load(stream)

    for video_id, expected_accepted, expected_shared, expected_metres, expected_pixels in [
        (17, 28, 27, 0.077, 3.135), (53, 97, 97, 0.661, 24.799),
    ]:
        groups = archive[f"grouping_results/grouped{video_id}_courts.json.gz"]["groups"]
        records = archive[
            f"grouping_results/grouped{video_id}/stages/court/{video_id}/court_evidence.json.gz"
        ]["scene_records"]
        by_index = {record["scene_index"]: record for record in records}
        assert sum(record["scene_valid"] for record in records) == expected_accepted
        assert sum(len(group["members"]) for group in groups) == expected_shared
        for group in groups:
            for scene_index in group["members"]:
                assert by_index[scene_index]["active_corners_native_px"] == group["corners_native_px"]
        original_path = Path(__file__).resolve().parents[1] / f"evidence/video_{video_id}/after/court_evidence.json.gz"
        with gzip.open(original_path, "rt") as stream:
            original = json.load(stream)
        assert len(groups) == 1
        metres, pixels = group_disagreement(groups[0], original["scene_records"])
        assert round(metres, 3) == expected_metres
        assert round(pixels, 3) == expected_pixels
        print(f"Video {video_id}: {expected_accepted} accepted, {expected_shared} share identical corners")
        print(f"  Before sharing: {metres:.3f} m coordinate disagreement, {pixels:.3f} px corner variation")

    for video_id, expected_scenes, expected_accepted in [(8, 636, 90), (9, 317, 66), (10, 375, 64)]:
        prefix = f"followup_results/video_{video_id:02d}"
        before = archive[f"{prefix}/before/court_evidence.json.gz"]
        after = archive[f"{prefix}/after/court_evidence.json.gz"]
        assert without_case_ids(before) == without_case_ids(after), f"video {video_id}: court payload changed"
        records = after["scene_records"]
        assert len(records) == expected_scenes
        assert sum(record["scene_valid"] for record in records) == expected_accepted
        masks = archive["mask_runs"][f"video_{video_id:02d}"]
        assert masks["before"] == masks["after"], f"video {video_id}: mask changed"
        for runs in masks["after"].values():
            previous_end = -1
            for start, end in runs:
                assert previous_end < start < end <= masks["n_frames"]
                previous_end = end
        print(f"Video {video_id}: {expected_scenes} scenes, {expected_accepted} accepted, courts and masks equal")

    cases = {case["case_id"]: case for case in archive["line_support.json.gz"]["cases"]}
    for case_id, candidate, expected_percent in [
        ("17_0", "active", [48, 58]), ("53_223", "raw", [64, 47]),
        ("53_223", "active", [91, 96]), ("53_310", "active", [5, 2]),
    ]:
        result = cases[case_id]["candidates"][candidate]["10"]
        line_medians = [median(samples) for samples in zip(*result["vectors"])]
        assert line_medians == result["median_by_line"]
        family_means = [fmean(line_medians[:6]), fmean(line_medians[6:])]
        assert [round(100 * value) for value in family_means] == expected_percent
        print(f"Pilot {case_id} {candidate}: {family_means[0]:.4f}, {family_means[1]:.4f}")


if __name__ == "__main__":
    main()
