"""Replay fixed S0 scores on complete G0/G1 candidate unions.

The script keeps candidate geometry in one anchor coordinate system and writes a
scalar score checkpoint for each target frame. It does not generate directions
or consult references before the selection lock.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def find_repo_root() -> Path:
    for parent in (HERE, *HERE.parents):
        if (parent / "src").is_dir() and (parent / "scratch/court_det_fix").is_dir():
            return parent
    raise RuntimeError("Could not locate the repository root")


REPO_ROOT = find_repo_root()
COURT_DET_FIX = REPO_ROOT / "scratch/court_det_fix"
HELPERS = COURT_DET_FIX / "frozen_helpers_20260914"
line_identity_override = os.environ.get("COURT_DET_FIX_LINE_IDENTITY")
LINE_IDENTITY = Path(line_identity_override) if line_identity_override else COURT_DET_FIX / "line_identity"
LEGACY = HELPERS / "legacy"
view_alignment_override = os.environ.get("COURT_DET_FIX_VIEW_ALIGNMENT")
VIEW_ALIGNMENT = (
    Path(view_alignment_override)
    if view_alignment_override
    else COURT_DET_FIX / "next_steps_20260916/L3_temporal/view_alignment.json"
)
POPULATION_ROOT = COURT_DET_FIX / "evidence/g0_g1/evaluation_20260922/populations"
OUTPUT_ROOT = HERE / "results"

GX_CASES = (
    "gxBQ_window_00_frame_0",
    "gxBQ_window_00_frame_5",
    "gxBQ_window_00_frame_689",
    "gxBQ_window_01_frame_5111",
    "gxBQ_window_02_frame_5766",
    "gxBQ_window_03_frame_77876",
    "gxBQ_window_04_frame_86088",
)
AM3_CASES = ("am3_window_00_frame_0", "am3_window_01_frame_10514")
WORKING_SIZE = (960, 540)
CAMERA_ERROR_LIMIT = 0.1
ARMS = ("G0", "G1")

for import_root in (
    REPO_ROOT,
    REPO_ROOT / "src",
    LINE_IDENTITY,
    HELPERS / "vp_pruning",
    HELPERS / "marking_diagnosis",
    HELPERS / "axis_matching",
    HELPERS / "automatic_axes",
    HELPERS / "automatic_axes/svd_fixed",
    LEGACY,
):
    sys.path.insert(0, str(import_root))

import shared

shared.add_helper_paths()
sys.path.insert(0, str(LEGACY))

import run_automatic

run_automatic.frame_path = lambda source, _root: shared.frame_path(source)

import zone_net
from inspect_appearance import profiles
from run_automatic import evaluate_pool
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector


def batched_profiles(frame: np.ndarray, homographies: np.ndarray) -> list[dict]:
    # OpenCV remap has a 32767-row limit. Each court contributes up to 12
    # visible intervals; 256 courts stays well below that implementation limit.
    result = []
    for start in range(0, len(homographies), 256):
        result.extend(profiles(frame, homographies[start:start + 256]))
    return result


run_automatic.profiles = batched_profiles


def load_l3_helpers() -> Any:
    l3_override = os.environ.get("COURT_DET_FIX_L3_TEMPORAL")
    path = (
        Path(l3_override)
        if l3_override
        else COURT_DET_FIX / "next_steps_20260916/L3_temporal/run_l3_temporal.py"
    )
    spec = importlib.util.spec_from_file_location("cached_l3_temporal_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L3 = load_l3_helpers()


@dataclass(slots=True)
class ViewContext:
    case_id: str
    source: dict[str, Any]
    segments: np.ndarray
    families: tuple[Any, Any]
    size: tuple[int, int]
    observations: assignment.Observations
    image_path: Path


@dataclass(slots=True)
class CandidateGeometry:
    court_id: str
    origin_case: str
    origin_arm: str
    origin_candidate_id: str
    source_entry: dict[str, Any]
    anchor_homography_working: np.ndarray


def read_json_gz(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_json_gz(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(gzip.compress(payload, mtime=0))
    temporary.replace(path)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def normalise_homography(homography: np.ndarray) -> np.ndarray:
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("homography must be a finite, non-singular 3x3 matrix")
    return matrix / matrix[2, 2]


def validate_homography(homography: np.ndarray) -> np.ndarray:
    """Validate a candidate matrix without changing its saved numerical scale."""
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        raise ValueError("homography must be a finite, non-singular 3x3 matrix")
    return matrix


def source_for(case_id: str) -> dict[str, Any]:
    return shared.load_source(case_id)


def prepare_context(case_id: str) -> ViewContext:
    source = source_for(case_id)
    segments, families, size = prepare(source)
    if tuple(size) != WORKING_SIZE:
        raise ValueError(f"{case_id}: working size {size} != {WORKING_SIZE}")
    image_path = shared.frame_path(source)
    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(image_path)
    expected_shape = (source["dimensions"]["height"], source["dimensions"]["width"])
    if frame.shape[:2] != expected_shape:
        raise ValueError(f"{case_id}: frame shape {frame.shape[:2]} != {expected_shape}")
    return ViewContext(
        case_id=case_id,
        source=source,
        segments=segments,
        families=families,
        size=tuple(size),
        observations=assignment.prepare_observations(segments, size),
        image_path=image_path,
    )


def useful_registration(case_id: str, record: dict[str, Any]) -> None:
    status = record.get("status")
    if status == "anchor":
        return
    if status != "ok":
        raise RuntimeError(f"{case_id}: registration status is {status!r}")
    if int(record.get("inlier_count", 0)) < 20:
        raise RuntimeError(f"{case_id}: registration has too few inliers")
    if float(record.get("inlier_ratio", 0.0)) < 0.5:
        raise RuntimeError(f"{case_id}: registration inlier ratio is too low")
    coverage = float(record.get("inlier_target_grid_coverage_fraction", 0.0))
    if coverage < 0.5:
        raise RuntimeError(f"{case_id}: registration coverage is too low")
    median_residual = record.get("heldout_residuals", {}).get("median_px")
    if median_residual is not None and float(median_residual) > 5.0:
        raise RuntimeError(f"{case_id}: registration held-out residual is too high")


def compact_registration(record: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "status": record.get("status"),
        "method": record.get("method"),
        "inlier_count": record.get("inlier_count"),
        "inlier_ratio": record.get("inlier_ratio"),
        "inlier_target_grid_coverage_fraction": record.get("inlier_target_grid_coverage_fraction"),
        "heldout_residuals": record.get("heldout_residuals"),
        "homography_target_to_anchor": record.get("homography_target_to_anchor"),
        "overlay_path": record.get("overlay_path"),
    }
    identity_grid = record.get("identity_grid", {})
    if identity_grid.get("summary_px") is not None:
        compact["identity_grid_summary_px"] = identity_grid["summary_px"]
    return compact


def working_alignment(
    homography_native: np.ndarray, target_source: dict[str, Any], anchor_source: dict[str, Any]
) -> np.ndarray:
    target_size = np.asarray(
        [target_source["dimensions"]["width"], target_source["dimensions"]["height"]], dtype=float
    )
    anchor_size = np.asarray(
        [anchor_source["dimensions"]["width"], anchor_source["dimensions"]["height"]], dtype=float
    )
    target_scale = np.diag([WORKING_SIZE[0] / target_size[0], WORKING_SIZE[1] / target_size[1], 1.0])
    anchor_scale = np.diag([WORKING_SIZE[0] / anchor_size[0], WORKING_SIZE[1] / anchor_size[1], 1.0])
    return normalise_homography(anchor_scale @ homography_native @ np.linalg.inv(target_scale))


def load_alignments(
    case_ids: tuple[str, ...], cohort: str, output: Path
) -> tuple[str, dict[str, np.ndarray], dict[str, Any]]:
    if cohort == "gx":
        anchor_case = GX_CASES[0]
        anchor_source = source_for(anchor_case)
        alignment = read_json_gz(VIEW_ALIGNMENT) if VIEW_ALIGNMENT.suffix == ".gz" else json.loads(VIEW_ALIGNMENT.read_text())
        records = alignment.get("pairs", {})
        result = {}
        for case_id in case_ids:
            record = records.get(case_id)
            if record is None:
                raise FileNotFoundError(f"{VIEW_ALIGNMENT}: missing {case_id}")
            useful_registration(case_id, record)
            result[case_id] = working_alignment(
                np.asarray(record["homography_target_to_anchor"], dtype=float),
                source_for(case_id),
                anchor_source,
            )
        return anchor_case, result, {
            "source": str(VIEW_ALIGNMENT),
            "records": {case_id: compact_registration(records[case_id]) for case_id in case_ids},
        }

    if cohort != "am3":
        raise ValueError(cohort)
    anchor_case = AM3_CASES[0]
    anchor_source = source_for(anchor_case)
    records: dict[str, Any] = {}
    result = {}
    records[anchor_case] = {"status": "anchor", "homography_target_to_anchor": np.eye(3).tolist()}
    for case_id in case_ids:
        if case_id == anchor_case:
            result[case_id] = np.eye(3)
            continue
        overlay = output / "alignment" / f"{anchor_case}_to_{case_id}.png"
        record = L3.register_pair(
            shared.frame_path(anchor_source), shared.frame_path(source_for(case_id)), overlay
        )
        useful_registration(case_id, record)
        result[case_id] = working_alignment(
            np.asarray(record["homography_target_to_anchor"], dtype=float),
            source_for(case_id),
            anchor_source,
        )
        records[case_id] = compact_registration(record)
    write_json(output / "alignment.json", {"schema": "temporal-registration/1", "anchor_case": anchor_case, "pairs": records})
    return anchor_case, result, {"source": "L3.register_pair", "records": records}


def load_candidates(
    case_ids: tuple[str, ...], population_root: Path, alignments: dict[str, np.ndarray]
) -> tuple[list[CandidateGeometry], dict[str, dict[str, Any]]]:
    candidates: list[CandidateGeometry] = []
    sources: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for case_id in case_ids:
        path = population_root / f"{case_id}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        record = read_json_gz(path)
        if record.get("case_id") != case_id:
            raise ValueError(f"{path}: case_id mismatch")
        sources[case_id] = {"path": str(path), "md5": md5(path), "g0_source": record.get("g0_source")}
        for arm in ARMS:
            entries = record.get(arm)
            if not isinstance(entries, list) or len(entries) != 256:
                raise ValueError(f"{path}: {arm} must contain exactly 256 entries")
            for entry in entries:
                candidate_id = str(entry.get("candidate_id"))
                court_id = f"{case_id}::{arm}::{candidate_id}"
                if court_id in seen:
                    raise ValueError(f"duplicate candidate {court_id}")
                seen.add(court_id)
                # The matcher stores a projectively equivalent matrix whose scale
                # affects its floating-point score path. Preserve it for the
                # native diagonal check; only alignment matrices are normalised.
                homography = validate_homography(np.asarray(entry.get("homography_working"), dtype=float))
                candidates.append(
                    CandidateGeometry(
                        court_id=court_id,
                        origin_case=case_id,
                        origin_arm=arm,
                        origin_candidate_id=candidate_id,
                        source_entry=entry,
                        anchor_homography_working=alignments[case_id] @ homography,
                    )
                )
    candidates.sort(key=lambda candidate: candidate.court_id)
    return candidates, sources


def population_identity(population_sources: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the path-independent identity used to validate score checkpoints."""
    return {
        case_id: {"md5": source["md5"], "g0_source": source.get("g0_source")}
        for case_id, source in population_sources.items()
    }


def corners_for(homography: np.ndarray, source: dict[str, Any]) -> np.ndarray | None:
    corners_working, denominator = detector.project(homography[None], detector.CORNER_COURT_M)
    if not np.isfinite(corners_working).all() or not np.all(denominator > 1e-9):
        return None
    native_scale = np.asarray(
        [source["dimensions"]["width"], source["dimensions"]["height"]], dtype=float
    ) / np.asarray(WORKING_SIZE, dtype=float)
    return corners_working[0] * native_scale


def transformed_entry(
    candidate: CandidateGeometry, context: ViewContext, alignment: np.ndarray
) -> dict[str, Any] | None:
    if candidate.origin_case == context.case_id:
        # Keep the exact source matrix for the native diagonal. The scoring
        # helpers are projectively invariant in theory but not bitwise invariant
        # after repeated inversion and multiplication.
        target_homography = validate_homography(
            np.asarray(candidate.source_entry["homography_working"], dtype=float)
        )
    else:
        target_homography = np.linalg.inv(alignment) @ candidate.anchor_homography_working
        validate_homography(target_homography)
    corners = corners_for(target_homography, context.source)
    if corners is None:
        return None
    entry = {
        key: value
        for key, value in candidate.source_entry.items()
        if key not in {"candidate_id", "homography_working", "corners_px", "stripe", "gates", "profile"}
    }
    entry.update(
        {
            "candidate_id": candidate.court_id,
            "homography_working": target_homography.tolist(),
            "corners_px": corners.tolist(),
            "shortlist_score": float(candidate.source_entry.get("shortlist_score", 0.0)),
            "origin_key": candidate.court_id,
        }
    )
    return entry


def score_metadata(payload: dict[str, Any], candidate: CandidateGeometry) -> dict[str, Any]:
    scalar_keys = (
        "status",
        "score_source",
        "geometry_valid",
        "camera_error",
        "floor_score",
        "line_score",
        "line_forward",
        "line_reverse",
        "paint_profile_score",
    )
    result = {key: payload.get(key) for key in scalar_keys}
    result.update(
        {
            "origin_case": candidate.origin_case,
            "origin_arm": candidate.origin_arm,
            "origin_candidate_id": candidate.origin_candidate_id,
        }
    )
    return result


def diagonal_difference(saved: dict[str, Any], measured: dict[str, Any]) -> tuple[bool, float, str | None]:
    keys = ("status", "geometry_valid", "camera_error", "floor_score", "line_score", "paint_profile_score")
    maximum = 0.0
    for key in keys:
        left, right = saved.get(key), measured.get(key)
        if isinstance(left, (int, float)) or isinstance(right, (int, float)):
            if left is None or right is None:
                return False, maximum, key
            difference = abs(float(left) - float(right))
            maximum = max(maximum, difference)
            if difference > 1e-8:
                return False, maximum, key
        elif left != right:
            return False, maximum, key
    return True, maximum, None


def score_target(
    target: ViewContext,
    candidates: list[CandidateGeometry],
    alignments: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], float]:
    started = monotonic()
    entries = []
    invalid_projection = set()
    for candidate in candidates:
        entry = transformed_entry(candidate, target, alignments[target.case_id])
        # A transferred proposal can cross the projective horizon even when
        # its source-frame geometry was valid. Exclude it on this target only.
        if entry is None:
            invalid_projection.add(candidate.court_id)
        else:
            entries.append(entry)
    evaluated = evaluate_pool(
        target.source,
        entries,
        target.observations,
        target.size,
        target.segments,
        target.families,
        zone_net,
        COURT_DET_FIX,
    )
    measured_by_id = {str(entry["candidate_id"]): entry for entry in evaluated}
    scores: dict[str, dict[str, Any]] = {}
    reused = 0
    mismatches = 0
    maximum_difference = 0.0
    mismatch_examples = []
    for candidate in candidates:
        if candidate.court_id in invalid_projection:
            scores[candidate.court_id] = score_metadata(
                {"status": "invalid_projection", "geometry_valid": False,
                 "score_source": "target_frame_projection"}, candidate
            )
            continue
        measured_payload = L3.score_payload(measured_by_id[candidate.court_id], "reevaluated_target_frame")
        payload = measured_payload
        if candidate.origin_case == target.case_id:
            saved_entry = dict(candidate.source_entry)
            saved_entry["stripe"] = candidate.source_entry.get(
                "stripe_s0", candidate.source_entry.get("stripe", {})
            )
            saved_payload = L3.score_payload(saved_entry, "saved_origin_record")
            matches, difference, field = diagonal_difference(saved_payload, measured_payload)
            maximum_difference = max(maximum_difference, difference)
            if matches:
                payload = saved_payload
                reused += 1
            else:
                mismatches += 1
                if len(mismatch_examples) < 5:
                    mismatch_examples.append({"court_id": candidate.court_id, "field": field, "difference": difference})
        scores[candidate.court_id] = score_metadata(payload, candidate)
    diagonal = {
        "target_case": target.case_id,
        "native_candidate_count": sum(candidate.origin_case == target.case_id for candidate in candidates),
        "native_saved_reused": reused,
        "native_reevaluated": mismatches,
        "maximum_absolute_difference": maximum_difference,
        "mismatch_examples": mismatch_examples,
        "invalid_projection_count": len(invalid_projection),
        "invalid_projection_ids": sorted(invalid_projection),
    }
    return scores, diagonal, monotonic() - started


def candidate_manifest(
    candidates: list[CandidateGeometry], anchor_case: str, population_sources: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema": "temporal-candidate-geometry/1",
        "coordinate_system": "anchor working pixels at 960x540",
        "anchor_case": anchor_case,
        "candidate_count": len(candidates),
        "population_sources": population_sources,
        "population_identity": population_identity(population_sources),
        "candidates": [
            {
                "court_id": candidate.court_id,
                "origin_case": candidate.origin_case,
                "origin_arm": candidate.origin_arm,
                "origin_candidate_id": candidate.origin_candidate_id,
                "homography_anchor_working": candidate.anchor_homography_working.tolist(),
            }
            for candidate in candidates
        ],
    }


def candidate_ids(candidates: list[CandidateGeometry]) -> list[str]:
    return [candidate.court_id for candidate in candidates]


def cached_target_checkpoint(
    path: Path,
    target_case: str,
    candidates: list[CandidateGeometry],
    population_identity_record: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        record = read_json_gz(path)
    except (OSError, EOFError, json.JSONDecodeError):
        return None
    expected_ids = candidate_ids(candidates)
    scores = record.get("scores")
    if (
        record.get("schema") != "temporal-score-matrix-target/1"
        or record.get("target_case") != target_case
        or record.get("observation_arm") != "S0"
        or record.get("candidate_count") != len(expected_ids)
        or record.get("candidate_ids") != expected_ids
        or record.get("population_identity") != population_identity_record
        or not isinstance(scores, dict)
        or list(scores) != expected_ids
    ):
        return None
    return record


def eligible_ids(
    candidates: list[CandidateGeometry],
    matrix: dict[str, dict[str, dict[str, Any]]],
    scoring_cases: tuple[str, ...],
    arms: set[str] | None = None,
    origin_case: str | None = None,
) -> list[str]:
    result = []
    for candidate in candidates:
        if arms is not None and candidate.origin_arm not in arms:
            continue
        if origin_case is not None and candidate.origin_case != origin_case:
            continue
        if all(matrix[case_id][candidate.court_id]["status"] == "ok" for case_id in scoring_cases):
            result.append(candidate.court_id)
    return result


def winner_record(court_id: str, scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    score = scores[court_id]
    return {
        "court_id": court_id,
        "origin_case": score["origin_case"],
        "origin_arm": score["origin_arm"],
        "origin_candidate_id": score["origin_candidate_id"],
        "line_score": score["line_score"],
        "paint_profile_score": score["paint_profile_score"],
    }


def frame_winners(ids: list[str], target_case: str, matrix: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    if not ids:
        return {"eligible_count": 0, "line": None, "paint": None}
    scores = matrix[target_case]
    line = min(ids, key=lambda court_id: (-scores[court_id]["line_score"], court_id))
    paint = min(
        ids,
        key=lambda court_id: (
            -scores[court_id]["paint_profile_score"],
            -scores[court_id]["line_score"],
            court_id,
        ),
    )
    return {"eligible_count": len(ids), "line": winner_record(line, scores), "paint": winner_record(paint, scores)}


def temporal_winners(
    ids: list[str], scoring_cases: tuple[str, ...], matrix: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    if not ids:
        return {"eligible_count": 0, "shared_line": None, "shared_paint": None, "l3": None}
    l3_matrix = {court_id: {case_id: matrix[case_id][court_id] for case_id in scoring_cases} for court_id in ids}
    l3_selection = L3.choose_winners(ids, l3_matrix, scoring_cases)
    l3_selection["shared"]["aggregation"] = f"median across {len(scoring_cases)} selected frames"
    medians = {
        court_id: (
            float(np.median([matrix[case_id][court_id]["paint_profile_score"] for case_id in scoring_cases])),
            float(np.median([matrix[case_id][court_id]["line_score"] for case_id in scoring_cases])),
        )
        for court_id in ids
    }
    shared_paint = min(ids, key=lambda court_id: (-medians[court_id][0], -medians[court_id][1], court_id))
    shared_line = l3_selection["shared"]["court_id"]
    return {
        "eligible_count": len(ids),
        "shared_line": {
            "court_id": shared_line,
            "median_line_score": medians[shared_line][1],
            "line_vector": [matrix[case_id][shared_line]["line_score"] for case_id in scoring_cases],
        },
        "shared_paint": {
            "court_id": shared_paint,
            "median_paint_profile_score": medians[shared_paint][0],
            "median_line_score": medians[shared_paint][1],
            "paint_vector": [matrix[case_id][shared_paint]["paint_profile_score"] for case_id in scoring_cases],
        },
        "l3": l3_selection,
    }


def build_selection(candidates: list[CandidateGeometry], matrix: dict[str, dict[str, dict[str, Any]]], case_ids: tuple[str, ...]) -> dict[str, Any]:
    common_union = eligible_ids(candidates, matrix, case_ids)
    access = {}
    for label, arms in (("G0", {"G0"}), ("G1", {"G1"}), ("union", {"G0", "G1"})):
        common_ids = eligible_ids(candidates, matrix, case_ids, arms=arms)
        access[label] = {
            "common_union_count": len(common_ids),
            "per_frame": {
                case_id: frame_winners(common_ids, case_id, matrix) for case_id in case_ids
            },
            "target_eligible_per_frame": {
                case_id: frame_winners(
                    eligible_ids(candidates, matrix, (case_id,), arms=arms), case_id, matrix
                )
                for case_id in case_ids
            },
            "temporal": temporal_winners(common_ids, case_ids, matrix),
        }
    native_per_frame = {
        case_id: frame_winners(
            eligible_ids(candidates, matrix, (case_id,), arms={"G0", "G1"}, origin_case=case_id),
            case_id,
            matrix,
        )
        for case_id in case_ids
    }
    common_per_frame = {
        case_id: frame_winners(common_union, case_id, matrix) for case_id in case_ids
    }
    target_eligible_per_frame = {
        case_id: frame_winners(eligible_ids(candidates, matrix, (case_id,)), case_id, matrix)
        for case_id in case_ids
    }
    return {
        "schema": "temporal-union-selection/1",
        "scoring_cases": list(case_ids),
        "common_union_count": len(common_union),
        "native_per_frame": native_per_frame,
        "common_union_per_frame": common_per_frame,
        "target_eligible_per_frame": target_eligible_per_frame,
        "common_union_temporal": temporal_winners(common_union, case_ids, matrix),
        "access_arms": access,
    }


def render_winners(
    output: Path,
    case_ids: tuple[str, ...],
    contexts: dict[str, ViewContext],
    candidates: dict[str, CandidateGeometry],
    alignments: dict[str, np.ndarray],
    selection: dict[str, Any],
) -> list[str]:
    sys.path.insert(0, str(COURT_DET_FIX / "w5_holistic"))
    from render_gallery import render_prediction

    rendered = []
    temporal_ids = {
        selection["common_union_temporal"][key]["court_id"]
        for key in ("shared_line", "shared_paint")
        if selection["common_union_temporal"].get(key) is not None
    }
    for case_id in case_ids:
        chosen: dict[str, list[str]] = {}
        for section in ("native_per_frame", "common_union_per_frame"):
            for kind in ("line", "paint"):
                record = selection[section][case_id].get(kind)
                if record is not None:
                    chosen.setdefault(record["court_id"], []).append(f"{section}-{kind}")
        for court_id in temporal_ids:
            chosen.setdefault(court_id, []).append("temporal-shared")
        frame = cv2.imread(str(contexts[case_id].image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(contexts[case_id].image_path)
        render_context = {
            "id": case_id,
            "dimensions": contexts[case_id].source["dimensions"],
            "working_dimensions": list(WORKING_SIZE),
        }
        for court_id, roles in chosen.items():
            entry = transformed_entry(candidates[court_id], contexts[case_id], alignments[case_id])
            if entry is None:
                raise AssertionError(f"selected candidate has invalid target geometry: {court_id}")
            stem = output / "overlays" / f"{case_id}__{court_id.replace(':', '_')}"
            stem.parent.mkdir(parents=True, exist_ok=True)
            links = render_prediction(frame, entry, render_context, roles, stem, None)
            rendered.append(f"{case_id}:{court_id}:{','.join(roles)}:{','.join(links)}")
    return rendered


def historical_w4() -> dict[str, Any] | None:
    path = COURT_DET_FIX / "evidence/pixel_temporal/w4/results/l3_rank_sum.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "winner": record.get("winner"),
        "shared_median_line_winner": record.get("shared_median_line_winner"),
        "common_eligible_count": record.get("common_eligible_count"),
        "note": "Historical rank-sum panel; not merged with the complete G0/G1 union.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("gx", "am3"), default="gx")
    parser.add_argument(
        "--cases",
        nargs="+",
        help="Population-origin cases; defaults to the full selected cohort",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        help=(
            "Target cases to score. With a strict subset, write partial checkpoints "
            "only; rerun without --targets to aggregate the full cohort."
        ),
    )
    parser.add_argument("--population-root", type=Path, default=POPULATION_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "gx_full")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--include-w4", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(1)
    cohort_cases = GX_CASES if args.cohort == "gx" else AM3_CASES
    population_cases = tuple(args.cases or cohort_cases)
    target_cases = tuple(args.targets or population_cases)
    unknown = (set(population_cases) | set(target_cases)) - set(cohort_cases)
    if unknown:
        raise ValueError(f"Cases do not belong to {args.cohort}: {sorted(unknown)}")
    if not population_cases or not target_cases:
        raise ValueError("At least one population case and target case are required")
    partial_targets = args.targets is not None and set(target_cases) != set(cohort_cases)
    if partial_targets and args.render:
        raise ValueError("--render requires the full target cohort after checkpoint aggregation")
    args.output.mkdir(parents=True, exist_ok=True)
    alignment_cases = tuple(dict.fromkeys((*population_cases, *target_cases)))
    contexts = {case_id: prepare_context(case_id) for case_id in target_cases}
    anchor_case, alignments, alignment_manifest = load_alignments(alignment_cases, args.cohort, args.output)
    candidates, population_sources = load_candidates(population_cases, args.population_root, alignments)
    population_identity_record = population_identity(population_sources)
    write_json_gz(args.output / "candidates.json.gz", candidate_manifest(candidates, anchor_case, population_sources))
    matrix: dict[str, dict[str, dict[str, Any]]] = {}
    target_timings = {}
    diagonal_checks = {}
    target_reused = {}
    expected_ids = candidate_ids(candidates)
    for case_id in target_cases:
        checkpoint_path = args.output / "scores" / f"{case_id}.json.gz"
        cached = cached_target_checkpoint(checkpoint_path, case_id, candidates, population_identity_record)
        if cached is not None:
            matrix[case_id] = cached["scores"]
            diagonal_checks[case_id] = cached["diagonal_check"]
            target_timings[case_id] = cached["elapsed_s"]
            target_reused[case_id] = True
            print(case_id, "reused checkpoint", "candidates", len(expected_ids), flush=True)
            continue
        scores, diagonal, elapsed = score_target(contexts[case_id], candidates, alignments)
        matrix[case_id] = scores
        diagonal_checks[case_id] = diagonal
        target_timings[case_id] = elapsed
        target_reused[case_id] = False
        write_json_gz(
            checkpoint_path,
            {
                "schema": "temporal-score-matrix-target/1",
                "target_case": case_id,
                "observation_arm": "S0",
                "candidate_count": len(candidates),
                "candidate_ids": expected_ids,
                "population_sources": population_sources,
                "population_identity": population_identity_record,
                "elapsed_s": elapsed,
                "diagonal_check": diagonal,
                "scores": scores,
            },
        )
        print(case_id, "scored", len(candidates), "candidates", "seconds", round(elapsed, 1), flush=True)
    selection = None
    rendered = []
    if not partial_targets:
        selection = build_selection(candidates, matrix, target_cases)
        selection["selection_locked"] = True
        write_json(args.output / "selection.json", selection)
        if args.render:
            rendered = render_winners(
                args.output,
                target_cases,
                contexts,
                {candidate.court_id: candidate for candidate in candidates},
                alignments,
                selection,
            )
    manifest = {
        "schema": "temporal-union-replay/1",
        "cohort": args.cohort,
        "population_cases": list(population_cases),
        "scoring_cases": list(target_cases),
        "population_sources": population_sources,
        "population_root": str(args.population_root),
        "candidate_count": len(candidates),
        "candidates_per_case_and_arm": 256,
        "observation_convention": "S0 original source segments and fixed native 960x540 prepare context",
        "alignment": alignment_manifest,
        "anchor_case": anchor_case,
        "diagonal_checks": diagonal_checks,
        "target_timings_s": target_timings,
        "target_reused": target_reused,
        "selection_locked": selection is not None,
        "selection_path": str(args.output / "selection.json") if selection is not None else None,
        "rendered": rendered,
        "w4_rank_sum": historical_w4() if args.include_w4 else None,
        "references_loaded_before_selection": False,
    }
    write_json(args.output / "manifest.json", manifest)
    if selection is None:
        print("partial checkpoints written", "candidates=", len(candidates), "frames=", len(target_cases), flush=True)
    else:
        print("selection locked", "candidates=", len(candidates), "frames=", len(target_cases), flush=True)


if __name__ == "__main__":
    main()
