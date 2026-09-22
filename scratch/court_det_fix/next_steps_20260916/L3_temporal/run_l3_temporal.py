"""Run the bounded cached-evidence temporal scoring pilot for the GX video.

The panel is reference-blind: its courts come only from the two saved automatic
all-camera records.  The scorer is the existing stripe/profile evaluator.  This
script does not generate new directions or courts.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_ROOT = REPO_ROOT / "scratch/court_det_fix/frozen_helpers_20260914"
SRC_ROOT = REPO_ROOT / "src"
AUTOMATIC_ROOT = HELPER_ROOT / "automatic_axes"
AXIS_ROOT = HELPER_ROOT / "axis_matching"
MARKING_ROOT = HELPER_ROOT / "marking_diagnosis"
VP_ROOT = HELPER_ROOT / "vp_pruning"
LEGACY_ROOT = HELPER_ROOT / "legacy"
CAMERA_HELPERS = REPO_ROOT / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260908"
for import_root in (
    REPO_ROOT,
    SRC_ROOT,
    CAMERA_HELPERS,
    LEGACY_ROOT,
    VP_ROOT,
    MARKING_ROOT,
    AXIS_ROOT,
    AUTOMATIC_ROOT,
):
    sys.path.insert(0, str(import_root))

from projective_seed import corner_errors
from run_automatic import evaluate_pool
from run_population import prepare

from experiments.annotator.independent_court import assignment

OUTPUT_ROOT = REPO_ROOT / "scratch/court_det_fix/next_steps_20260916/L3_temporal"
PACK_PATH = REPO_ROOT / "scratch/court_det_fix/frozen_views/packs/gx_extension_inputs.json.gz"
LOCAL_PACK_PATH = (
    REPO_ROOT / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260909/gx_extension/inputs.json.gz"
)
IMAGE_ROOT = REPO_ROOT / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/20260909"
ORIGIN_ROOT = (
    REPO_ROOT
    / "scratch/court_det_fix/evidence/independent_proposals/development/player_guided/"
    "20260914/automatic_axes/collected/all_camera"
)

ORIGIN_CASES = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
)
SCORING_CASES = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "gxBQ_window_00_frame_689",
    "gxBQ_window_01_frame_5111",
    "gxBQ_window_02_frame_5766",
    "gxBQ_window_03_frame_77876",
    "gxBQ_window_04_frame_86088",
)
TOP_K = 8
CAMERA_ERROR_LIMIT = 0.1
WORKING_SIZE = (960, 540)
REFERENCE_SCALE = np.asarray(WORKING_SIZE, dtype=float) / np.asarray((1920, 1080), dtype=float)

SIFT_FEATURES = 4_000
MATCH_RATIO = 0.75
RANSAC_THRESHOLD_PX = 3.0
HOLDOUT_EVERY = 5
GRID_COLUMNS = 4
GRID_ROWS = 3
OVERLAY_SIZE = WORKING_SIZE


def read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def current_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def source_image_path(source: dict[str, Any], image_root: Path) -> Path:
    if not source["id"].startswith("gxBQ"):
        raise ValueError(f"This pilot accepts GX cases only: {source['id']}")
    return image_root / "gx_extension/people" / source["image"]


def check_image(source: dict[str, Any], image_root: Path) -> Path:
    path = source_image_path(source, image_root)
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(path)
    expected = (source["dimensions"]["height"], source["dimensions"]["width"])
    if frame.shape[:2] != expected:
        raise ValueError(f"{source['id']}: image shape {frame.shape[:2]} != {expected}")
    return path


def rank_eligible(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reproduce the existing line and paint rank keys with fixed identity ties."""
    limit = float(record.get("camera_error_limit", CAMERA_ERROR_LIMIT))
    eligible = []
    for entry in record["entries"]:
        gates = entry.get("gates", {})
        profile = entry.get("profile", {})
        camera_error = finite_number(gates.get("camera_error"))
        if camera_error is not None and camera_error <= limit and profile.get("score") is not None:
            eligible.append(entry)
    line = sorted(
        eligible,
        key=lambda entry: (-float(entry["stripe"]["exclusive"]["score"]), str(entry["candidate_id"])),
    )
    paint = sorted(
        eligible,
        key=lambda entry: (
            -float(entry["profile"]["score"]),
            -float(entry["stripe"]["exclusive"]["score"]),
            str(entry["candidate_id"]),
        ),
    )
    return line, paint


def panel_metadata(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "court_id": entry["court_id"],
        "origin_key": entry["origin_key"],
        "origin_candidate_id": entry["origin_candidate_id"],
        "selection_roles": entry["selection_roles"],
        "line_rank_at_origin": entry["line_rank_at_origin"],
        "paint_rank_at_origin": entry["paint_rank_at_origin"],
        "origin_line_score": finite_number(entry["stripe"]["exclusive"]["score"]),
        "origin_paint_profile_score": finite_number(entry["profile"].get("score")),
        "corners_px": entry["corners_px"],
    }


def build_panel(
    origin_records: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    panel: dict[str, dict[str, Any]] = {}
    origin_manifest = []
    stages = {record.get("selection_stage") for record in origin_records.values()}
    if len(stages) != 1:
        raise ValueError(f"Origin records use different selection stages: {sorted(stages)}")
    stage = stages.pop()
    for origin_key in ORIGIN_CASES:
        record = origin_records[origin_key]
        line, paint = rank_eligible(record)
        selected_by_role = {
            "line_top8": [(entry, rank) for rank, entry in enumerate(line[:TOP_K], 1)],
            "paint_top8": [(entry, rank) for rank, entry in enumerate(paint[:TOP_K], 1)],
        }
        for role, ranked in selected_by_role.items():
            for entry, rank in ranked:
                origin_candidate_id = str(entry["candidate_id"])
                court_id = f"{origin_key}::{origin_candidate_id}"
                if court_id not in panel:
                    panel[court_id] = {
                        **entry,
                        "court_id": court_id,
                        "candidate_id": court_id,
                        "origin_key": origin_key,
                        "origin_candidate_id": origin_candidate_id,
                        "selection_roles": [],
                        "line_rank_at_origin": None,
                        "paint_rank_at_origin": None,
                    }
                panel_entry = panel[court_id]
                panel_entry["selection_roles"].append(role)
                if role == "line_top8":
                    panel_entry["line_rank_at_origin"] = rank
                else:
                    panel_entry["paint_rank_at_origin"] = rank
        origin_manifest.append(
            {
                "origin_key": origin_key,
                "record_path": relative_path(ORIGIN_ROOT / f"{origin_key}.json.gz"),
                "record_md5": md5(ORIGIN_ROOT / f"{origin_key}.json.gz"),
                "selection_stage": stage,
                "record_entry_count": len(record["entries"]),
                "eligible_entry_count": len(line),
                "existing_line_winner_id": record.get("line_winner_id"),
                "existing_paint_winner_id": record.get("paint_winner_id"),
                "line_top8": [
                    {
                        "origin_candidate_id": str(entry["candidate_id"]),
                        "rank": rank,
                        "score": finite_number(entry["stripe"]["exclusive"]["score"]),
                    }
                    for entry, rank in selected_by_role["line_top8"]
                ],
                "paint_top8": [
                    {
                        "origin_candidate_id": str(entry["candidate_id"]),
                        "rank": rank,
                        "score": finite_number(entry["profile"].get("score")),
                    }
                    for entry, rank in selected_by_role["paint_top8"]
                ],
            }
        )
    ordered_panel = [panel[court_id] for court_id in sorted(panel)]
    if len(ordered_panel) > 32:
        raise ValueError(f"Panel exceeds the task bound: {len(ordered_panel)} courts")
    return ordered_panel, origin_manifest


def prepare_context(source: dict[str, Any], image_root: Path) -> dict[str, Any]:
    image_path = check_image(source, image_root)
    image_md5 = md5(image_path)
    segments, families, size = prepare(source)
    observations = assignment.prepare_observations(segments, size)
    if tuple(size) != WORKING_SIZE:
        raise ValueError(f"{source['id']}: unexpected working size {size}")
    feet = source["all_feet_px"]
    missing_feet = sum(foot is None for frame in feet for foot in frame)
    return {
        "source": source,
        "segments": segments,
        "families": families,
        "size": size,
        "observations": observations,
        "image_path": image_path,
        "image_md5": image_md5,
        "summary": {
            "case_id": source["id"],
            "frame_index": int(source["id"].rsplit("_", 1)[1]),
            "source_video": source["provenance"]["source_video"],
            "source_fps": source["provenance"]["source_fps"],
            "native_dimensions": source["dimensions"],
            "working_size": list(size),
            "image_path": relative_path(image_path),
            "image_md5": image_md5,
            "native_fragment_count": len(source["segments_px"]),
            "working_visible_fragment_count": len(observations.segments),
            "working_group_count": len(observations.groups),
            "person_box_count": len(source["bbox_px"]),
            "feet_sample_count": len(feet),
            "feet_slot_count": len(feet[0]) if feet else 0,
            "missing_feet_count": missing_feet,
        },
    }


def diagonal_fields(record: dict[str, Any]) -> dict[str, float | None]:
    gates = record.get("gates", {})
    stripe = record.get("stripe", {}).get("exclusive", {})
    profile = record.get("profile", {})
    return {
        "line_score": finite_number(stripe.get("score")),
        "line_forward": finite_number(stripe.get("forward")),
        "line_reverse": finite_number(stripe.get("reverse")),
        "paint_profile_score": finite_number(profile.get("score")),
        "floor_score": finite_number(gates.get("floor_score")),
        "camera_error": finite_number(gates.get("camera_error")),
    }


def verify_diagonals(
    panel: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    origin_records: dict[str, dict[str, Any]],
    zone: Any,
    image_root: Path,
) -> list[dict[str, Any]]:
    checks = []
    for origin_key in ORIGIN_CASES:
        entry = next(candidate for candidate in panel if candidate["origin_key"] == origin_key)
        context = contexts[origin_key]
        measured = evaluate_pool(
            context["source"],
            [entry],
            context["observations"],
            context["size"],
            context["segments"],
            context["families"],
            zone,
            image_root,
        )[0]
        saved = next(
            candidate
            for candidate in origin_records[origin_key]["entries"]
            if str(candidate["candidate_id"]) == entry["origin_candidate_id"]
        )
        saved_fields, measured_fields = diagonal_fields(saved), diagonal_fields(measured)
        differences = {
            key: None
            if saved_fields[key] is None or measured_fields[key] is None
            else abs(saved_fields[key] - measured_fields[key])
            for key in saved_fields
        }
        finite_differences = [value for value in differences.values() if value is not None]
        maximum = max(finite_differences, default=0.0)
        if maximum > 1e-10:
            raise RuntimeError(f"{origin_key}: saved diagonal differs by {maximum}")
        checks.append(
            {
                "origin_key": origin_key,
                "court_id": entry["court_id"],
                "saved": saved_fields,
                "recomputed": measured_fields,
                "absolute_differences": differences,
                "max_absolute_difference": maximum,
                "verified": True,
            }
        )
    return checks


def score_payload(record: dict[str, Any], score_source: str) -> dict[str, Any]:
    gates = record.get("gates", {})
    stripe = record.get("stripe", {})
    exclusive = stripe.get("exclusive", {})
    profile = record.get("profile", {})
    geometry_valid = gates.get("geometry_valid")
    camera_error = finite_number(gates.get("camera_error"))
    line_score = finite_number(exclusive.get("score"))
    paint_profile_score = finite_number(profile.get("score"))
    if geometry_valid is False:
        status = "invalid_geometry"
    elif camera_error is None or camera_error > CAMERA_ERROR_LIMIT:
        status = "camera_ineligible"
    elif line_score is None:
        status = "missing_line_score"
    elif paint_profile_score is None:
        status = "missing_paint_profile"
    else:
        status = "ok"
    return {
        "status": status,
        "score_source": score_source,
        "geometry_valid": geometry_valid,
        "camera_error": camera_error,
        "floor_score": finite_number(gates.get("floor_score")),
        "line_score": line_score,
        "line_forward": finite_number(exclusive.get("forward")),
        "line_reverse": finite_number(exclusive.get("reverse")),
        "paint_profile_score": paint_profile_score,
        "paint_profile_vector": profile.get("marking_ridge"),
        "paint_interval_visible": profile.get("interval_visible"),
        "paint_interval_vector": profile.get("interval_ridge"),
    }


def score_panel_on_frame(
    target_key: str,
    context: dict[str, Any],
    panel: list[dict[str, Any]],
    zone: Any,
    image_root: Path,
) -> dict[str, dict[str, Any]]:
    target_entries = [entry for entry in panel if entry["origin_key"] != target_key]
    measured_by_id = {
        entry["court_id"]: entry
        for entry in evaluate_pool(
            context["source"],
            target_entries,
            context["observations"],
            context["size"],
            context["segments"],
            context["families"],
            zone,
            image_root,
        )
    }
    scores = {}
    for entry in panel:
        if entry["origin_key"] == target_key:
            scores[entry["court_id"]] = score_payload(entry, "saved_origin_record")
        else:
            scores[entry["court_id"]] = score_payload(
                measured_by_id[entry["court_id"]], "rescored_target_frame"
            )
    return scores


def identity_grid(homography: np.ndarray, size: tuple[int, int]) -> dict[str, Any]:
    width, height = size
    xs = np.linspace(0.05 * (width - 1), 0.95 * (width - 1), GRID_COLUMNS)
    ys = np.linspace(0.05 * (height - 1), 0.95 * (height - 1), GRID_ROWS)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points[None], homography)[0]
    displacements = np.linalg.norm(mapped - points, axis=1)
    rows = [
        {
            "target_xy": [float(value) for value in point],
            "mapped_to_anchor_xy": [float(value) for value in mapped_point],
            "identity_displacement_px": float(displacement),
        }
        for point, mapped_point, displacement in zip(points, mapped, displacements, strict=True)
    ]
    return {
        "columns": GRID_COLUMNS,
        "rows": GRID_ROWS,
        "points": rows,
        "summary_px": {
            "median": float(np.median(displacements)),
            "mean": float(np.mean(displacements)),
            "maximum": float(np.max(displacements)),
        },
    }


def grid_cell_counts(points: np.ndarray, size: tuple[int, int]) -> list[int]:
    width, height = size
    columns = np.minimum((points[:, 0] / width * GRID_COLUMNS).astype(int), GRID_COLUMNS - 1)
    rows = np.minimum((points[:, 1] / height * GRID_ROWS).astype(int), GRID_ROWS - 1)
    counts = np.zeros(GRID_COLUMNS * GRID_ROWS, dtype=int)
    for row, column in zip(rows, columns, strict=True):
        counts[row * GRID_COLUMNS + column] += 1
    return counts.tolist()


def residual_summary(values: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"count": 0, "median_px": None, "p90_px": None, "maximum_px": None}
    return {
        "count": len(values),
        "median_px": float(np.median(values)),
        "p90_px": float(np.percentile(values, 90)),
        "maximum_px": float(np.max(values)),
    }


def register_pair(anchor_path: Path, target_path: Path, overlay_path: Path) -> dict[str, Any]:
    """Fit one static-feature homography and report its departure from identity."""
    anchor = cv2.imread(str(anchor_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if anchor is None or target is None:
        raise FileNotFoundError(f"Could not read registration pair {anchor_path}, {target_path}")
    if anchor.shape != target.shape:
        raise ValueError(f"Registration dimensions differ: {anchor.shape} and {target.shape}")
    gray_anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    anchor_keypoints, anchor_descriptors = sift.detectAndCompute(gray_anchor, None)
    target_keypoints, target_descriptors = sift.detectAndCompute(gray_target, None)
    if anchor_descriptors is None or target_descriptors is None:
        return {"status": "insufficient_features"}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidates = matcher.knnMatch(target_descriptors, anchor_descriptors, k=2)
    good = []
    for pair in candidates:
        if len(pair) == 2 and pair[0].distance < MATCH_RATIO * pair[1].distance:
            good.append(pair[0])
    good.sort(key=lambda match: (match.queryIdx, match.trainIdx, match.distance))
    fit_matches = [match for index, match in enumerate(good) if index % HOLDOUT_EVERY]
    heldout_matches = [match for index, match in enumerate(good) if not index % HOLDOUT_EVERY]
    if len(fit_matches) < 4:
        return {
            "status": "insufficient_matches",
            "anchor_keypoints": len(anchor_keypoints),
            "target_keypoints": len(target_keypoints),
            "ratio_matches": len(good),
            "fit_matches": len(fit_matches),
            "heldout_matches": len(heldout_matches),
        }
    target_points = np.asarray([target_keypoints[match.queryIdx].pt for match in fit_matches], dtype=np.float32)
    anchor_points = np.asarray([anchor_keypoints[match.trainIdx].pt for match in fit_matches], dtype=np.float32)
    cv2.setRNGSeed(0)
    homography, inlier_mask = cv2.findHomography(
        target_points, anchor_points, cv2.RANSAC, RANSAC_THRESHOLD_PX
    )
    if homography is None or inlier_mask is None:
        return {
            "status": "homography_failed",
            "anchor_keypoints": len(anchor_keypoints),
            "target_keypoints": len(target_keypoints),
            "ratio_matches": len(good),
            "fit_matches": len(fit_matches),
            "heldout_matches": len(heldout_matches),
        }
    homography = homography / homography[2, 2]
    inlier_mask = inlier_mask.ravel().astype(bool)
    inlier_target = target_points[inlier_mask]
    inlier_count = int(inlier_mask.sum())
    heldout_target = np.asarray([target_keypoints[match.queryIdx].pt for match in heldout_matches], dtype=np.float32)
    heldout_anchor = np.asarray([anchor_keypoints[match.trainIdx].pt for match in heldout_matches], dtype=np.float32)
    if len(heldout_target):
        heldout_mapped = cv2.perspectiveTransform(heldout_target[None], homography)[0]
        heldout_residuals = np.linalg.norm(heldout_mapped - heldout_anchor, axis=1)
    else:
        heldout_residuals = np.empty(0)
    warped = cv2.warpPerspective(target, homography, (anchor.shape[1], anchor.shape[0]))
    valid = cv2.warpPerspective(
        np.ones(anchor.shape[:2], dtype=np.uint8), homography, (anchor.shape[1], anchor.shape[0])
    )
    blended = cv2.addWeighted(anchor, 0.5, warped, 0.5, 0.0)
    overlay = np.where(valid[..., None] > 0, blended, anchor)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(overlay_path), cv2.resize(overlay, OVERLAY_SIZE, interpolation=cv2.INTER_AREA)):
        raise OSError(f"Could not write {overlay_path}")
    return {
        "status": "ok",
        "method": "SIFT descriptors, BF ratio test, RANSAC homography",
        "anchor_keypoints": len(anchor_keypoints),
        "target_keypoints": len(target_keypoints),
        "ratio_matches": len(good),
        "fit_matches": len(fit_matches),
        "heldout_matches": len(heldout_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / len(fit_matches)),
        "inlier_target_bbox_px": {
            "min": [float(value) for value in inlier_target.min(axis=0)],
            "max": [float(value) for value in inlier_target.max(axis=0)],
        },
        "inlier_target_grid_counts": grid_cell_counts(inlier_target, (anchor.shape[1], anchor.shape[0])),
        "inlier_target_grid_coverage_fraction": float(
            np.count_nonzero(grid_cell_counts(inlier_target, (anchor.shape[1], anchor.shape[0])))
            / (GRID_COLUMNS * GRID_ROWS)
        ),
        "heldout_residuals": residual_summary(heldout_residuals),
        "homography_target_to_anchor": homography.tolist(),
        "identity_grid": identity_grid(homography, (anchor.shape[1], anchor.shape[0])),
        "overlay_path": relative_path(overlay_path),
        "overlay_dimensions": {"width": OVERLAY_SIZE[0], "height": OVERLAY_SIZE[1]},
    }


def run_registration(
    contexts: dict[str, dict[str, Any]], output_root: Path, anchor_key: str
) -> dict[str, Any]:
    results = {}
    anchor_path = contexts[anchor_key]["image_path"]
    for target_key in SCORING_CASES:
        if target_key == anchor_key:
            results[target_key] = {
                "status": "anchor",
                "homography_target_to_anchor": np.eye(3).tolist(),
                "identity_grid": identity_grid(np.eye(3), (1920, 1080)),
            }
            continue
        overlay_path = output_root / f"alignment_{anchor_key.rsplit('_', 1)[1]}_to_{target_key.rsplit('_', 1)[1]}.png"
        results[target_key] = register_pair(anchor_path, contexts[target_key]["image_path"], overlay_path)
    return {
        "schema": "cached-temporal-static-registration/1",
        "anchor_case_id": anchor_key,
        "coordinate_space": "native image pixels; identity is target-to-anchor pixel identity",
        "selection_independent": True,
        "reference_blind": True,
        "moving_content_policy": "RANSAC-consistent feature matches only; references and courts are unused",
        "settings": {
            "sift_features": SIFT_FEATURES,
            "ratio_test": MATCH_RATIO,
            "ransac_reprojection_threshold_px": RANSAC_THRESHOLD_PX,
            "holdout_every": HOLDOUT_EVERY,
            "identity_grid": [GRID_COLUMNS, GRID_ROWS],
        },
        "pairs": results,
    }


def attach_reference_metrics(
    panel: list[dict[str, Any]],
    score_matrix: dict[str, dict[str, Any]],
    sources: dict[str, dict[str, Any]],
    evaluation_ids: set[str],
) -> None:
    """Join exact existing reference corners for locked witness courts only."""
    for entry in panel:
        if entry["court_id"] not in evaluation_ids:
            continue
        corners = np.asarray(entry["corners_px"], dtype=float) * REFERENCE_SCALE
        for case_id, score in score_matrix[entry["court_id"]].items():
            reference = np.asarray(sources[case_id]["reference"]["corners_px"], dtype=float) * REFERENCE_SCALE
            score["reference_max_corner_working_px"] = float(corner_errors(corners, reference))
            score["reference_status"] = sources[case_id]["reference"].get("reference_status")


def common_subset(
    panel: list[dict[str, Any]], score_matrix: dict[str, dict[str, Any]], scoring_cases: tuple[str, ...]
) -> tuple[list[str], list[dict[str, Any]]]:
    common = []
    exclusions = []
    for entry in panel:
        court_id = entry["court_id"]
        scores = score_matrix[court_id]
        invalid = [
            {"case_id": case_id, "status": scores[case_id]["status"]}
            for case_id in scoring_cases
            if scores[case_id]["status"] != "ok"
        ]
        if invalid:
            exclusions.append({"court_id": court_id, "invalid_frames": invalid})
        else:
            common.append(court_id)
    return common, exclusions


def choose_winners(
    common_ids: list[str], score_matrix: dict[str, dict[str, Any]], scoring_cases: tuple[str, ...]
) -> dict[str, Any]:
    independent = {}
    for case_id in scoring_cases:
        winner = min(common_ids, key=lambda court_id: (-score_matrix[court_id][case_id]["line_score"], court_id))
        independent[case_id] = {
            "court_id": winner,
            "line_score": score_matrix[winner][case_id]["line_score"],
            "line_vector": [score_matrix[winner][other]["line_score"] for other in scoring_cases],
            "paint_profile_vector": [
                score_matrix[winner][other]["paint_profile_score"] for other in scoring_cases
            ],
        }
    medians = {
        court_id: float(np.median([score_matrix[court_id][case_id]["line_score"] for case_id in scoring_cases]))
        for court_id in common_ids
    }
    shared = min(common_ids, key=lambda court_id: (-medians[court_id], court_id))
    return {
        "metric": "stripe.exclusive.score",
        "tie_order": "ascending court_id",
        "independent": independent,
        "shared": {
            "aggregation": "median across all seven selected GX frames",
            "court_id": shared,
            "median_line_score": medians[shared],
            "line_vector": [score_matrix[shared][case_id]["line_score"] for case_id in scoring_cases],
            "paint_profile_vector": [
                score_matrix[shared][case_id]["paint_profile_score"] for case_id in scoring_cases
            ],
        },
    }


def dominance(
    common_ids: list[str], score_matrix: dict[str, dict[str, Any]], scoring_cases: tuple[str, ...]
) -> list[dict[str, Any]]:
    vectors = {
        court_id: np.asarray([score_matrix[court_id][case_id]["line_score"] for case_id in scoring_cases])
        for court_id in common_ids
    }
    rows = []
    for court_id in common_ids:
        beaten_by = [
            other_id
            for other_id in common_ids
            if other_id != court_id
            and np.all(vectors[other_id] >= vectors[court_id])
            and np.any(vectors[other_id] > vectors[court_id])
        ]
        if beaten_by:
            rows.append({"court_id": court_id, "dominated_by": sorted(beaten_by)})
    return rows


def write_score_csv(
    path: Path,
    panel: list[dict[str, Any]],
    score_matrix: dict[str, dict[str, Any]],
    source_summaries: dict[str, dict[str, Any]],
) -> None:
    fieldnames = [
        "court_id",
        "origin_key",
        "origin_candidate_id",
        "selection_roles",
        "line_rank_at_origin",
        "paint_rank_at_origin",
        "scoring_case_id",
        "frame_index",
        "nominal_seconds",
        "score_source",
        "status",
        "geometry_valid",
        "camera_error",
        "floor_score",
        "line_score",
        "line_forward",
        "line_reverse",
        "paint_profile_score",
        "paint_profile_vector",
        "reference_max_corner_working_px",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for entry in panel:
            for case_id, score in score_matrix[entry["court_id"]].items():
                summary = source_summaries[case_id]
                fps = summary["source_fps"]
                writer.writerow(
                    {
                        "court_id": entry["court_id"],
                        "origin_key": entry["origin_key"],
                        "origin_candidate_id": entry["origin_candidate_id"],
                        "selection_roles": ";".join(entry["selection_roles"]),
                        "line_rank_at_origin": entry["line_rank_at_origin"],
                        "paint_rank_at_origin": entry["paint_rank_at_origin"],
                        "scoring_case_id": case_id,
                        "frame_index": summary["frame_index"],
                        "nominal_seconds": None if fps is None else summary["frame_index"] / float(fps),
                        "score_source": score["score_source"],
                        "status": score["status"],
                        "geometry_valid": score["geometry_valid"],
                        "camera_error": score["camera_error"],
                        "floor_score": score["floor_score"],
                        "line_score": score["line_score"],
                        "line_forward": score["line_forward"],
                        "line_reverse": score["line_reverse"],
                        "paint_profile_score": score["paint_profile_score"],
                        "paint_profile_vector": json.dumps(score["paint_profile_vector"], separators=(",", ":")),
                        "reference_max_corner_working_px": score.get("reference_max_corner_working_px"),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--pack", type=Path, default=PACK_PATH)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--origin-root", type=Path, default=ORIGIN_ROOT)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)

    packed = read_gzip_json(args.pack)
    input_pack_md5 = md5(args.pack)
    local_pack_md5 = md5(LOCAL_PACK_PATH)
    if args.pack.resolve() == PACK_PATH.resolve() and local_pack_md5 != input_pack_md5:
        raise ValueError("Frozen and local GX input packs differ")
    sources = {source["id"]: source for source in packed["cases"]}
    required = set(SCORING_CASES)
    if required - sources.keys():
        raise ValueError(f"Missing scoring cases: {sorted(required - sources.keys())}")
    if set(packed.get("references", {})) != required:
        raise ValueError("Selected scoring cases do not have exactly matching references")
    if any(not case_id.startswith("gxBQ") for case_id in SCORING_CASES):
        raise ValueError("The pilot must use one GX video")

    origin_records = {}
    for case_id in ORIGIN_CASES:
        path = args.origin_root / f"{case_id}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        origin_records[case_id] = read_gzip_json(path)
    panel, origin_manifest = build_panel(origin_records)
    contexts = {case_id: prepare_context(sources[case_id], args.image_root) for case_id in SCORING_CASES}
    source_summaries = {case_id: context["summary"] for case_id, context in contexts.items()}

    registration = run_registration(contexts, args.output, SCORING_CASES[0])
    write_json(args.output / "view_alignment.json", registration)

    zone = importlib.import_module("zone_net")
    diagonal_checks = verify_diagonals(panel, contexts, origin_records, zone, args.image_root)
    score_matrix = {entry["court_id"]: {} for entry in panel}
    for case_id in SCORING_CASES:
        frame_scores = score_panel_on_frame(case_id, contexts[case_id], panel, zone, args.image_root)
        for court_id, score in frame_scores.items():
            score_matrix[court_id][case_id] = score

    common_ids, exclusions = common_subset(panel, score_matrix, SCORING_CASES)
    if not common_ids:
        raise RuntimeError("No common eligible panel court has valid scores on every selected frame")
    selection = choose_winners(common_ids, score_matrix, SCORING_CASES)
    contrary_ids = set()
    for origin_key, record in origin_records.items():
        for winner_id in (record.get("line_winner_id"), record.get("paint_winner_id")):
            if winner_id is not None:
                contrary_ids.add(f"{origin_key}::{winner_id}")
    selected_ids = {selection["shared"]["court_id"]}
    selected_ids.update(row["court_id"] for row in selection["independent"].values())
    witness_ids = selected_ids | contrary_ids
    # Outputs are now locked. Existing references are diagnostic only from this point onward.
    attach_reference_metrics(
        panel,
        score_matrix,
        {case_id: {"reference": packed["references"][case_id]} for case_id in SCORING_CASES},
        witness_ids,
    )
    dominance_rows = dominance(common_ids, score_matrix, SCORING_CASES)

    panel_rows = [panel_metadata(entry) for entry in panel]
    for row in panel_rows:
        row["contrary_example"] = row["court_id"] in contrary_ids
    matrix_payload = {
        "schema": "cached-temporal-score-matrix/1",
        "basis_commit": current_commit(),
        "video_group": "GX",
        "source_video": contexts[SCORING_CASES[0]]["source"]["provenance"]["source_video"],
        "scope_note": "User-authorised seven-frame representation extension; one video, two origins, at most 32 courts.",
        "selection_metric": "stripe.exclusive.score",
        "selection_is_reference_blind": True,
        "coordinate_convention": "maximum corner distance uses 960x540 working pixels and existing 180-degree relabelling",
        "scoring_cases": source_summaries,
        "panel": panel_rows,
        "scores": [
            {
                "court_id": entry["court_id"],
                "origin_key": entry["origin_key"],
                "origin_candidate_id": entry["origin_candidate_id"],
                "line_score_vector": [score_matrix[entry["court_id"]][case_id]["line_score"] for case_id in SCORING_CASES],
                "paint_profile_score_vector": [
                    score_matrix[entry["court_id"]][case_id]["paint_profile_score"] for case_id in SCORING_CASES
                ],
                "by_frame": score_matrix[entry["court_id"]],
            }
            for entry in panel
        ],
        "diagonal_verification": diagonal_checks,
        "common_eligible_subset": {
            "count": len(common_ids),
            "court_ids": common_ids,
            "excluded_count": len(exclusions),
            "exclusions": exclusions,
            "status_counts_across_matrix": dict(
                Counter(score["status"] for row in score_matrix.values() for score in row.values())
            ),
        },
        "selection": selection,
        "reference_evaluation_ids": sorted(witness_ids),
        "line_componentwise_dominance": {
            "common_subset_only": True,
            "dominated_count": len(dominance_rows),
            "rows": dominance_rows,
        },
        "contrary_examples": sorted(contrary_ids),
        "references_joined_after_selection_lock": True,
    }
    source_manifest = {
        "schema": "cached-temporal-source-manifest/1",
        "basis_commit": current_commit(),
        "task_scope": "one GX video; seven cached scoring frames; two origin records; no new detector generation",
        "input_pack": {
            "path": relative_path(args.pack),
            "md5": input_pack_md5,
            "local_comparison_path": relative_path(LOCAL_PACK_PATH),
            "local_comparison_md5": local_pack_md5,
            "matches_local_comparison": input_pack_md5 == local_pack_md5,
        },
        "image_root": relative_path(args.image_root),
        "origin_records": origin_manifest,
        "scoring_frames": list(source_summaries.values()),
        "panel_count": len(panel),
        "panel": panel_rows,
        "scorer": {
            "evaluator": relative_path(AUTOMATIC_ROOT / "run_automatic.py"),
            "stripe_scorer": relative_path(REPO_ROOT / "experiments/annotator/independent_court/stripe_observations.py"),
            "paint_profile_scorer": relative_path(AXIS_ROOT / "inspect_appearance.py"),
            "camera_error_limit": CAMERA_ERROR_LIMIT,
            "stage": origin_manifest[0]["selection_stage"],
        },
        "registration_output": "view_alignment.json",
        "score_output": "score_matrix.json",
    }
    write_json(args.output / "source_manifest.json", source_manifest)
    write_json(args.output / "score_matrix.json", matrix_payload)
    write_score_csv(args.output / "score_matrix.csv", panel, score_matrix, source_summaries)
    print(
        "completed",
        "panel=", len(panel),
        "common=", len(common_ids),
        "frames=", len(SCORING_CASES),
        "registration=", [registration["pairs"][case_id]["status"] for case_id in SCORING_CASES],
        "shared=", selection["shared"]["court_id"],
        "independent=", [row["court_id"] for row in selection["independent"].values()],
    )


if __name__ == "__main__":
    main()
