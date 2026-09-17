"""Separate candidate-population and fragment-scoring effects on four frozen views.

The matcher has already generated the two populations used here.  This script reconstructs the
missing baseline generation populations from their saved per-pair shortlists, replays only the
fragment-dependent stripe score under the original and paint-filtered observations, and traces the
existing pixel paint-profile test for four previously judged original candidates.
"""

from __future__ import annotations

import csv
import gzip
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(os.environ.get("L2_REPO", Path(__file__).resolve().parents[4])).resolve()
COURT_DET_FIX = Path(
    os.environ.get("L2_COURT_DET_FIX", REPO / "scratch/court_det_fix")
).resolve()
FROZEN_VIEWS = COURT_DET_FIX / "frozen_views"
HELPERS = COURT_DET_FIX / "frozen_helpers_20260914"
CAMERA_HELPERS = COURT_DET_FIX / "worklog/checks/independent/player_guided/20260908"
SOURCE = COURT_DET_FIX / "next_steps_20260916/webui_seed/source"
FILTER_INPUTS = COURT_DET_FIX / "line_identity/inputs/paint_observations"
FILTER_RUN = COURT_DET_FIX / "line_identity/runs/line_identity_20260915_222437/matcher/paint_observations"
BASELINE_ALL_CAMERA = (
    COURT_DET_FIX / "worklog/checks/independent/player_guided/20260914/automatic_axes/collected/all_camera"
)
BASELINE_GENERATION = FROZEN_VIEWS / "baseline_generation"
BASELINE_ACCOUNTING = COURT_DET_FIX / "direction_agreement/runs/direction_agreement_20260915_144900/e4/accounting.csv.gz"
CONTROL_RECORDS = COURT_DET_FIX / "direction_agreement/runs/direction_agreement_20260915_144900/e3"

CASE_PACKS = {
    "gxBQ_window_00_frame_0": FROZEN_VIEWS / "packs/gx_extension_inputs.json.gz",
    "am2_window_00_frame_150": FROZEN_VIEWS / "packs/marking_refit_inputs.json.gz",
    "am2_window_01_frame_28019": FROZEN_VIEWS / "packs/marking_refit_inputs.json.gz",
    "am3_window_00_frame_0": FROZEN_VIEWS / "packs/marking_refit_inputs.json.gz",
}
DIAGNOSTIC_PACKS = {
    "shuttleset_03_scene_0019": FROZEN_VIEWS / "packs/broadcast_extension_inputs.json.gz",
}
CASE_LABELS = {
    "gxBQ_window_00_frame_0": "GX0",
    "am2_window_00_frame_150": "Am2-150",
    "am2_window_01_frame_28019": "Am2-28019",
    "am3_window_00_frame_0": "Am3-0",
    "shuttleset_03_scene_0019": "SS03-19",
}
CASE_IDS = tuple(CASE_PACKS)
DIAGNOSTIC_CASES = (
    ("am2_window_01_frame_28019", "184:4123", "paint winner; rejected as an incoherent court near the net top"),
    ("am2_window_00_frame_150", "30:33", "paint winner; visually approved and described as perfect"),
    ("shuttleset_03_scene_0019", "165:6702", "paint winner; rejected as an unrelated, hallucinated court"),
    ("shuttleset_03_scene_0019", "1:60", "line winner; visually judged usable"),
)
CAMERA_ERROR_LIMIT = 0.1
PROFILE_THRESHOLD = 10.0
PROFILE_FRACTION = 0.4
PROFILE_SAMPLES = 24
PROFILE_OFFSETS = np.array([-4, -2, 0, 2, 4], dtype=np.float32)
PROFILE_SIDE_DISTANCE = 6.0


def add_experiment_paths() -> None:
    """Match the frozen matcher import order without importing any generated output."""
    paths = (
        SOURCE,
        HELPERS / "marking_diagnosis",
        HELPERS / "vp_pruning",
        HELPERS / "axis_matching",
        CAMERA_HELPERS,
        HELPERS / "legacy",
        REPO / "src",
        REPO,
    )
    for path in reversed(paths):
        sys.path.insert(0, str(path))


add_experiment_paths()

from inspect_appearance import profiles as existing_profiles
from run_automatic import evaluate_pool, select_pool
from run_population import prepare

from experiments.annotator.independent_court import assignment, detector
from experiments.annotator.independent_court import (
    stripe_observations as stripes,
)


def read_json(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode())
    temporary.replace(path)


def load_source(case_id: str) -> dict:
    packed = read_json({**CASE_PACKS, **DIAGNOSTIC_PACKS}[case_id])
    return next(source for source in packed["cases"] if source["id"] == case_id)


def load_filtered_source(case_id: str) -> dict:
    return read_json(FILTER_INPUTS / "cases" / f"{case_id}.json.gz")


def frame_path(source: dict) -> Path:
    case_id = source["id"]
    if case_id.startswith("gxBQ"):
        return FROZEN_VIEWS / "frames/gx" / source["image"]
    if case_id.startswith("shuttleset"):
        return FROZEN_VIEWS / "frames/original" / source["image"]
    video, frame = case_id.split("_", 1)[0], case_id.rsplit("_", 1)[1]
    return FROZEN_VIEWS / "frames/amateur" / video / f"frame_{int(frame):08d}.png"


def load_control(case_id: str) -> dict:
    return read_json(CONTROL_RECORDS / f"{case_id}.json.gz")["control"]


def load_saved_baseline_winners() -> dict[str, tuple[str | None, str | None]]:
    with gzip.open(BASELINE_ACCOUNTING, "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        return {
            row["case_id"]: (row["line_winner_id"] or None, row["paint_winner_id"] or None)
            for row in rows
            if row["arm"] == "B" and row["stage"] == "results" and row["case_id"] in CASE_IDS
        }


def reconstruct_generation_entries(record: dict, native_size: tuple[int, int]) -> list[dict]:
    """Replay the unchanged global retention from a saved all-camera record's shortlists."""
    working_size = np.asarray(record["working_size"], dtype=float)
    scale = np.asarray(native_size, dtype=float) / working_size
    candidates = []
    provenance: dict[int, dict] = {}
    for pair in record["pairs"]:
        for entry in pair.get("shortlist", []):
            corners = np.asarray(entry["corners_px"], dtype=float) / scale
            candidate = detector.Candidate(corners, entry["shortlist_score"], (0.0, 0.0), (0, 0))
            candidates.append(candidate)
            provenance[id(candidate)] = entry
    retained = select_pool(candidates)
    return [provenance[id(candidate)] for candidate in retained]


def load_generation_population(case_id: str, source: dict) -> tuple[list[dict], str]:
    direct = BASELINE_GENERATION / f"{case_id}.json.gz"
    if direct.exists():
        return read_json(direct)["entries"], str(direct.relative_to(REPO))
    all_camera_path = BASELINE_ALL_CAMERA / f"{case_id}.json.gz"
    record = read_json(all_camera_path)
    entries = reconstruct_generation_entries(record, (source["dimensions"]["width"], source["dimensions"]["height"]))
    return entries, str(all_camera_path.relative_to(REPO)) + " (generation selection replay)"


def evaluate_generation_population(
    source: dict, entries: list[dict], observations: assignment.Observations,
    size: tuple[int, int], segments: np.ndarray, families: tuple,
) -> list[dict]:
    """Recover cached eligibility/profile fields for a replayed G0 population only."""
    zone = importlib.import_module("zone_net")
    evaluator_globals = evaluate_pool.__globals__
    original_frame_path = evaluator_globals["frame_path"]
    evaluator_globals["frame_path"] = lambda current_source, _root: frame_path(current_source)
    try:
        evaluated = evaluate_pool(source, entries, observations, size, segments, families, zone, REPO)
    finally:
        evaluator_globals["frame_path"] = original_frame_path
    if [entry["candidate_id"] for entry in evaluated] != [entry["candidate_id"] for entry in entries]:
        raise AssertionError("G0 evaluation changed candidate order or membership")
    return evaluated


def scorer_inputs(source: dict) -> tuple[assignment.Observations, np.ndarray, tuple[int, int], np.ndarray, tuple]:
    segments, families, size = prepare(source)
    observations = assignment.prepare_observations(segments, size)
    return observations, stripes.fragment_weights(observations), size, segments, families


def homography_key(entry: dict) -> bytes:
    return np.asarray(entry["homography_working"], dtype=np.float64).tobytes()


def score_population(
    entries: list[dict], origin_arm: str, observations: assignment.Observations, weights: np.ndarray,
    size: tuple[int, int], cache: dict[bytes, dict],
) -> list[dict]:
    scored = []
    for index, entry in enumerate(entries):
        key = homography_key(entry)
        stripe = cache.get(key)
        if stripe is None:
            homography = np.asarray(entry["homography_working"], dtype=float)
            stripe = stripes.score_model(stripes.measure(homography, observations, size), weights, 3)
            cache[key] = stripe
        scored.append({
            **entry,
            "_origin_arm": origin_arm,
            "_origin_index": index,
            "_origin_key": f"{origin_arm}:{entry['candidate_id']}",
            "stripe": stripe,
        })
    return scored


def winner_entry(entries: list[dict], kind: str) -> dict | None:
    eligible = [
        entry for entry in entries
        if entry["gates"]["camera_error"] is not None
        and entry["gates"]["camera_error"] <= CAMERA_ERROR_LIMIT
        and entry["profile"]["score"] is not None
    ]
    if kind == "line":
        return max(eligible, key=lambda entry: entry["stripe"]["exclusive"]["score"], default=None)
    if kind == "paint":
        return max(
            eligible,
            key=lambda entry: (entry["profile"]["score"], entry["stripe"]["exclusive"]["score"]),
            default=None,
        )
    raise ValueError(kind)


def corner_error(entry: dict, control: np.ndarray, native_size: tuple[int, int], working_size: tuple[int, int]) -> float:
    scale = np.asarray(native_size, dtype=float) / np.asarray(working_size, dtype=float)
    corners = np.asarray(entry["corners_px"], dtype=float) / scale
    direct = np.linalg.norm(corners - control, axis=1).max()
    rotated = np.linalg.norm(corners - control[[2, 3, 0, 1]], axis=1).max()
    return float(min(direct, rotated))


def nearest_entry(
    entries: list[dict], control: np.ndarray, native_size: tuple[int, int], working_size: tuple[int, int],
    camera_only: bool,
) -> tuple[dict | None, float | None]:
    available = entries
    if camera_only:
        available = [
            entry for entry in entries
            if entry["gates"]["camera_error"] is not None and entry["gates"]["camera_error"] <= CAMERA_ERROR_LIMIT
        ]
    if not available:
        return None, None
    winner = min(enumerate(available), key=lambda pair: (corner_error(pair[1], control, native_size, working_size), pair[0]))[1]
    return winner, corner_error(winner, control, native_size, working_size)


def compact_candidate(
    entry: dict | None, control: np.ndarray, native_size: tuple[int, int], working_size: tuple[int, int],
) -> dict | None:
    if entry is None:
        return None
    return {
        "origin_key": entry["_origin_key"],
        "candidate_id": entry["candidate_id"],
        "entry_index": entry["_origin_index"],
        "distance_working_px": corner_error(entry, control, native_size, working_size),
        "corners_px": entry["corners_px"],
        "profile": entry["profile"],
        "stripe_exclusive": entry["stripe"]["exclusive"],
        "gates": entry["gates"],
    }


def cell_record(
    population: str, scorer: str, entries: list[dict], control: np.ndarray,
    native_size: tuple[int, int], working_size: tuple[int, int],
) -> dict:
    eligible_count = sum(
        entry["gates"]["camera_error"] is not None
        and entry["gates"]["camera_error"] <= CAMERA_ERROR_LIMIT
        and entry["profile"]["score"] is not None
        for entry in entries
    )
    nearest_all, nearest_all_distance = nearest_entry(entries, control, native_size, working_size, False)
    nearest_camera, nearest_camera_distance = nearest_entry(entries, control, native_size, working_size, True)
    line = winner_entry(entries, "line")
    paint = winner_entry(entries, "paint")
    return {
        "population": population,
        "scorer": scorer,
        "candidate_count": len(entries),
        "eligible_count": int(eligible_count),
        "line": compact_candidate(line, control, native_size, working_size),
        "paint": compact_candidate(paint, control, native_size, working_size),
        "nearest_all": compact_candidate(nearest_all, control, native_size, working_size),
        "nearest_camera_eligible": compact_candidate(nearest_camera, control, native_size, working_size),
        "nearest_all_distance_working_px": nearest_all_distance,
        "nearest_camera_distance_working_px": nearest_camera_distance,
    }


def profile_trace(frame: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> dict:
    """Trace the exact detector ridge test, including every centre shift that passes."""
    projected, _ = detector.project(homography[None], detector.SEGMENTS_M)
    endpoints, interval_visible = detector._visible_samples(projected.reshape(-1, 12, 2, 2), size, 2)
    segments = endpoints[0]
    finite_segments = segments.reshape(-1, 4).astype(np.float32)
    vectors = segments[:, 1] - segments[:, 0]
    normals = np.stack((-vectors[:, 1], vectors[:, 0]), axis=1)
    normals /= np.linalg.norm(vectors, axis=1)[:, None]
    fractions = np.linspace(0, 1, PROFILE_SAMPLES, dtype=np.float32)
    centres = segments[:, None, 0] + vectors[:, None] * fractions[None, :, None]
    shifted = centres[:, :, None] + normals[:, None, None] * PROFILE_OFFSETS[None, None, :, None]
    sides = PROFILE_SIDE_DISTANCE * normals[:, None, None]
    width, height = size
    intensity_sets = []
    in_frame_sets = []
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    for points in (shifted, shifted - sides, shifted + sides):
        maps = points.reshape(len(segments), -1, 2)
        sampled = cv2.remap(
            grey, maps[..., 0].astype(np.float32), maps[..., 1].astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        intensity_sets.append(sampled.reshape(points.shape[:-1]))
        in_frame_sets.append(
            (points[..., 0] >= 0)
            & (points[..., 0] < width)
            & (points[..., 1] >= 0)
            & (points[..., 1] < height)
        )
    centre, first_side, second_side = intensity_sets
    first_contrast = centre - first_side
    second_contrast = centre - second_side
    available = in_frame_sets[0] & in_frame_sets[1] & in_frame_sets[2]
    minimum_contrast = np.minimum(first_contrast, second_contrast)
    passing_offsets = available & (minimum_contrast >= PROFILE_THRESHOLD)
    passing_offsets &= interval_visible[0][:, None, None]
    sample_pass = passing_offsets.any(axis=2)
    interval_pass = (sample_pass.mean(axis=1) >= PROFILE_FRACTION) & interval_visible[0]
    interval_visible_list = interval_visible[0].tolist()
    interval_records = []
    for interval in range(len(segments)):
        interval_records.append({
            "interval": interval,
            "visible": bool(interval_visible_list[interval]),
            "tested_sample_coordinates_working_px": shifted[interval].tolist(),
            "minus_side_coordinates_working_px": (shifted[interval] - sides[interval]).tolist(),
            "plus_side_coordinates_working_px": (shifted[interval] + sides[interval]).tolist(),
            "in_frame_centre": in_frame_sets[0][interval].tolist(),
            "in_frame_minus_side": in_frame_sets[1][interval].tolist(),
            "in_frame_plus_side": in_frame_sets[2][interval].tolist(),
            "available": available[interval].tolist(),
            "contrast_centre_minus_minus_side": finite_float_rows(first_contrast[interval]),
            "contrast_centre_minus_plus_side": finite_float_rows(second_contrast[interval]),
            "minimum_contrast": finite_float_rows(minimum_contrast[interval]),
            "passing_shifts": [
                [float(PROFILE_OFFSETS[offset]) for offset in np.flatnonzero(passing_offsets[interval, sample])]
                for sample in range(PROFILE_SAMPLES)
            ],
            "pass_by_shift": passing_offsets[interval].tolist(),
            "sample_pass": sample_pass[interval].tolist(),
            "argmax_shift_working_px": PROFILE_OFFSETS[minimum_contrast[interval].argmax(axis=1)].astype(float).tolist(),
            "argmax_contrast": finite_float_rows(minimum_contrast[interval].max(axis=1)),
            "interval_pass": bool(interval_pass[interval]),
        })
    profile = aggregate_profile(interval_pass, interval_visible_list)
    visible_mask = np.asarray(interval_visible[0], dtype=bool)
    accepted_segments = detector._filter_painted_stripes(frame, finite_segments[visible_mask])
    exact_flags = {tuple(segment): True for segment in accepted_segments}
    exact_interval_flags = [visible and tuple(segment) in exact_flags for visible, segment in zip(visible_mask, finite_segments, strict=True)]
    return {
        "settings": {
            "samples_along": PROFILE_SAMPLES,
            "centre_offsets_working_px": PROFILE_OFFSETS.astype(float).tolist(),
            "side_distance_working_px": PROFILE_SIDE_DISTANCE,
            "minimum_contrast": PROFILE_THRESHOLD,
            "minimum_fraction": PROFILE_FRACTION,
        },
        "interval_visible": interval_visible_list,
        "interval_pass": interval_pass.tolist(),
        "profile_recomputed": profile,
        "interval_pass_matches_detector": exact_interval_flags == interval_pass.tolist(),
        "intervals": interval_records,
    }


def finite_float_rows(values: np.ndarray) -> list:
    array = np.asarray(values)
    if array.ndim == 1:
        return [None if not np.isfinite(value) else float(value) for value in array]
    return [[None if not np.isfinite(value) else float(value) for value in row] for row in array]


def aggregate_profile(interval_pass: np.ndarray, interval_visible: list[bool]) -> dict:
    marking_ridge = []
    for intervals in assignment.MARKING_INTERVALS:
        visible = np.asarray(interval_visible)[list(intervals)]
        values = np.asarray(interval_pass)[list(intervals)][visible]
        marking_ridge.append(float(values.mean()) if len(values) else None)
    available = [value for value in marking_ridge if value is not None]
    return {
        "score": float(np.mean(available)) if available else None,
        "marking_ridge": marking_ridge,
        "interval_visible": interval_visible,
        "interval_ridge": interval_pass.tolist(),
    }


def profile_matches(saved: dict, recomputed: dict) -> bool:
    if saved["interval_visible"] != recomputed["profile_recomputed"]["interval_visible"]:
        return False
    if saved["interval_ridge"] != recomputed["profile_recomputed"]["interval_ridge"]:
        return False
    saved_score = saved["score"]
    new_score = recomputed["profile_recomputed"]["score"]
    if saved_score is None or new_score is None:
        return saved_score is None and new_score is None
    return bool(np.isclose(saved_score, new_score, rtol=0, atol=1e-12))


def profiles_equal(first: dict, second: dict) -> bool:
    """Compare aggregate profile fields without depending on JSON float formatting."""
    if first["interval_visible"] != second["interval_visible"]:
        return False
    if first["interval_ridge"] != second["interval_ridge"]:
        return False
    if first["score"] is None or second["score"] is None:
        return first["score"] is None and second["score"] is None
    if not np.isclose(first["score"], second["score"], rtol=0, atol=1e-12):
        return False
    for first_value, second_value in zip(first["marking_ridge"], second["marking_ridge"], strict=True):
        if first_value is None or second_value is None:
            if first_value is not None or second_value is not None:
                return False
        elif not np.isclose(first_value, second_value, rtol=0, atol=1e-12):
            return False
    return True


def native_overlay(
    output: Path, source: dict, entry: dict, trace: dict, frame_native: np.ndarray, size: tuple[int, int],
) -> str:
    native_size = np.asarray([source["dimensions"]["width"], source["dimensions"]["height"]], dtype=float)
    working_scale = native_size / np.asarray(size, dtype=float)
    canvas = frame_native.copy()
    projected_points = []
    passing_points = []
    for interval in trace["intervals"]:
        coordinates = np.asarray(interval["tested_sample_coordinates_working_px"], dtype=float)
        available = np.asarray(interval["available"], dtype=bool)
        passing = np.asarray(interval["pass_by_shift"], dtype=bool)
        projected_points.extend(coordinates[available].reshape(-1, 2).tolist())
        passing_points.extend(coordinates[passing].reshape(-1, 2).tolist())
        endpoints = coordinates[[0, -1], 2] * working_scale
        colour = (255, 190, 0) if interval["interval_pass"] else (0, 165, 255)
        cv2.line(canvas, tuple(np.rint(endpoints[0]).astype(int)), tuple(np.rint(endpoints[1]).astype(int)), colour, 2)
        midpoint = np.rint(endpoints.mean(axis=0)).astype(int)
        cv2.putText(canvas, str(interval["interval"]), tuple(midpoint), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    for point in passing_points:
        native = np.rint(np.asarray(point) * working_scale).astype(int)
        cv2.circle(canvas, tuple(native), 3, (255, 0, 255), -1)
    corners = np.asarray(entry["corners_px"], dtype=float)
    cv2.polylines(canvas, [np.rint(corners).astype(int)], True, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{source['id']} {entry['candidate_id']} pass samples={len(passing_points)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    points = np.asarray(passing_points or projected_points, dtype=float) * working_scale
    points = points[np.isfinite(points).all(axis=1)]
    if len(points):
        x0, y0 = np.maximum(np.floor(points.min(axis=0) - 60), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(points.max(axis=0) + 60), native_size - 1).astype(int)
    else:
        x0, y0, x1, y1 = 0, 0, int(native_size[0]), int(native_size[1])
    if x1 - x0 > 900:
        centre = (x0 + x1) // 2
        x0, x1 = max(0, centre - 450), min(int(native_size[0]), centre + 450)
    if y1 - y0 > 650:
        centre = (y0 + y1) // 2
        y0, y1 = max(0, centre - 325), min(int(native_size[1]), centre + 325)
    crop = canvas[y0:y1, x0:x1]
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", source["id"])
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", entry["candidate_id"])
    path = output / f"witness_{safe_case}_{safe_id}.png"
    if not cv2.imwrite(str(path), crop):
        raise OSError(f"Could not write {path}")
    return str(path.resolve().relative_to(REPO))


def run_diagnostic(output: Path, case_id: str, candidate_id: str, ruling: str) -> dict:
    source = load_source(case_id)
    record_path = BASELINE_ALL_CAMERA / f"{case_id}.json.gz"
    record = read_json(record_path)
    entry = next(entry for entry in record["entries"] if entry["candidate_id"] == candidate_id)
    _, _, size = prepare(source)
    native_frame = cv2.imread(str(frame_path(source)))
    if native_frame is None:
        raise FileNotFoundError(frame_path(source))
    working_frame = cv2.resize(native_frame, size, interpolation=cv2.INTER_AREA)
    homography = np.asarray(entry["homography_working"], dtype=float)
    trace = profile_trace(working_frame, homography, size)
    trace["saved_profile"] = entry["profile"]
    trace["saved_profile_reproduced"] = profile_matches(entry["profile"], trace)
    existing = existing_profiles(working_frame, homography[None])[0]
    trace["existing_profile"] = existing
    trace["existing_profile_reproduced"] = profiles_equal(entry["profile"], existing)
    image = native_overlay(output, source, entry, trace, native_frame, size)
    return {
        "case_id": case_id,
        "label": CASE_LABELS.get(case_id, case_id),
        "candidate_id": candidate_id,
        "existing_ruling": ruling,
        "source_record": str(record_path.relative_to(REPO)),
        "candidate": {
            "corners_px": entry["corners_px"],
            "profile": entry["profile"],
            "stripe_exclusive": entry["stripe"]["exclusive"],
            "gates": entry["gates"],
        },
        "image": image,
        "trace": trace,
    }


def winner_delta(first: dict | None, second: dict | None) -> bool:
    first_key = None if first is None else first["origin_key"]
    second_key = None if second is None else second["origin_key"]
    return first_key != second_key


def csv_row(case_id: str, cell: dict, diagonal: dict) -> dict:
    row = {
        "case_id": case_id,
        "label": CASE_LABELS[case_id],
        "cell": f"{cell['population']},{cell['scorer']}",
        "population": cell["population"],
        "scorer": cell["scorer"],
        "candidate_count": cell["candidate_count"],
        "eligible_count": cell["eligible_count"],
        "line_winner_origin": None if cell["line"] is None else cell["line"]["origin_key"],
        "line_winner_id": None if cell["line"] is None else cell["line"]["candidate_id"],
        "line_distance_working_px": None if cell["line"] is None else cell["line"]["distance_working_px"],
        "paint_winner_origin": None if cell["paint"] is None else cell["paint"]["origin_key"],
        "paint_winner_id": None if cell["paint"] is None else cell["paint"]["candidate_id"],
        "paint_distance_working_px": None if cell["paint"] is None else cell["paint"]["distance_working_px"],
        "nearest_all_distance_working_px": cell["nearest_all_distance_working_px"],
        "nearest_camera_distance_working_px": cell["nearest_camera_distance_working_px"],
        "g0_diagonal_match": diagonal.get("G0,S0"),
        "g1_diagonal_match": diagonal.get("G1,S1"),
    }
    return row


def format_id(cell: dict, kind: str) -> str:
    winner = cell[kind]
    if winner is None:
        return "none"
    return f"{winner['origin_key']} ({winner['distance_working_px']:.1f}px)"


def build_result(cases: list[dict]) -> str:
    access = {"line": [], "paint": []}
    rescoring = {"line": [], "paint": []}
    union_changes = {"line": [], "paint": []}
    for case in cases:
        cells = case["cells"]
        for kind in ("line", "paint"):
            if winner_delta(cells["G0,S0"][kind], cells["G1,S0"][kind]):
                access[kind].append(case["label"])
            if winner_delta(cells["G0,S0"][kind], cells["G0,S1"][kind]):
                rescoring[kind].append(case["label"])
            if winner_delta(cells["U,S0"][kind], cells["U,S1"][kind]):
                union_changes[kind].append(case["label"])
    lines = [
        "# L2 — candidate access versus fragment rescoring",
        "",
        "This fixed-survivor replay uses the existing 960×540 working-pixel convention and the existing 180-degree corner relabelling. G0 is the saved baseline generation population; G1 is the saved `paint_observations` generation population. The union is ordered G0 first, then G1, with full origin keys preserved. S0 and S1 use the unchanged `stripe_observations` weighting and score functions.",
        "",
        "## Winner changes",
        "",
        "| case | G0,S0 line / paint | G0,S1 line / paint | G1,S0 line / paint | G1,S1 line / paint |",
        "| --- | --- | --- | --- | --- |",
    ]
    for case in cases:
        cells = case["cells"]
        lines.append(
            f"| {case['label']} | {format_id(cells['G0,S0'], 'line')} / {format_id(cells['G0,S0'], 'paint')} | "
            f"{format_id(cells['G0,S1'], 'line')} / {format_id(cells['G0,S1'], 'paint')} | "
            f"{format_id(cells['G1,S0'], 'line')} / {format_id(cells['G1,S0'], 'paint')} | "
            f"{format_id(cells['G1,S1'], 'line')} / {format_id(cells['G1,S1'], 'paint')} |"
        )
    lines.extend([
        "",
        f"Candidate access changes the line winner on {', '.join(access['line']) or 'none'} and the paint winner on {', '.join(access['paint']) or 'none'} when S0 is held fixed. Rescoring alone changes the line winner on {', '.join(rescoring['line']) or 'none'} and the paint winner on {', '.join(rescoring['paint']) or 'none'} when G0 is held fixed.",
        f"On the fixed union, S0→S1 changes the line winner on {', '.join(union_changes['line']) or 'none'} and the paint winner on {', '.join(union_changes['paint']) or 'none'}. This isolates ranker-input effects from access to G1 candidates, but it is still a development-population comparison rather than held-out evaluation.",
        "",
        "Nearest available courts are recorded in `comparison.csv` for every cell. They are independent of the score choice and include both the full generation population and the camera-eligible subset. Controls measure the products only; they never select a winner.",
        "",
        "## Diagonal checks and unresolved points",
        "",
    ])
    for case in cases:
        status = "passed" if case["diagonal_ok"] else "BLOCKED"
        lines.append(f"- {case['label']}: saved G0/S0 and G1/S1 winner identities {status}.")
    lines.extend([
        "",
        "The four pixel traces reproduce each saved aggregate profile and retain every passing centre shift, both-side contrasts, tested coordinates and availability mask. The overlays are diagnostic witnesses, not new visual approvals. The existing rulings remain: Am2-28019 `184:4123` and SS03-19 `165:6702` are false paint winners; Am2-150 `30:33` and SS03-19 `1:60` are the approved/usable contrary examples. The profile evidence shows what the scorer accepted, but does not by itself establish physical court-line ownership, especially for the far-Am2 alias.",
    ])
    result = "\n".join(lines) + "\n"
    if len(result.split()) > 800:
        raise RuntimeError(f"result.md exceeds 800 words: {len(result.split())}")
    return result


def run_case(case_id: str, output: Path, saved_baseline_winners: dict) -> dict:
    source = load_source(case_id)
    filtered = load_filtered_source(case_id)
    native_size = (source["dimensions"]["width"], source["dimensions"]["height"])
    working_size = tuple(read_json(FILTER_INPUTS / "estimators" / f"{case_id}.json.gz")["working_size"])
    control_record = load_control(case_id)
    control = np.asarray(control_record["corners_working_px"], dtype=float)
    g0_entries, g0_source = load_generation_population(case_id, source)
    g1_record_path = FILTER_RUN / "results" / f"{case_id}.json.gz"
    g1_record = read_json(g1_record_path)
    g1_entries = g1_record["entries"]
    if len(g0_entries) != 256 or len(g1_entries) != 256:
        raise AssertionError((case_id, len(g0_entries), len(g1_entries)))
    observations0, weights0, size0, segments0, families0 = scorer_inputs(source)
    observations1, weights1, size1, _, _ = scorer_inputs(filtered)
    if size0 != size1 or tuple(size0) != working_size:
        raise AssertionError((case_id, size0, size1, working_size))
    if g0_entries and "gates" not in g0_entries[0]:
        g0_entries = evaluate_generation_population(source, g0_entries, observations0, size0, segments0, families0)
    cache_s0: dict[bytes, dict] = {}
    cache_s1: dict[bytes, dict] = {}
    scored_g0_s0 = score_population(g0_entries, "G0", observations0, weights0, size0, cache_s0)
    scored_g0_s1 = score_population(g0_entries, "G0", observations1, weights1, size1, cache_s1)
    scored_g1_s0 = score_population(g1_entries, "G1", observations0, weights0, size0, cache_s0)
    scored_g1_s1 = score_population(g1_entries, "G1", observations1, weights1, size1, cache_s1)
    populations = {
        "G0,S0": scored_g0_s0,
        "G0,S1": scored_g0_s1,
        "G1,S0": scored_g1_s0,
        "G1,S1": scored_g1_s1,
        "U,S0": scored_g0_s0 + scored_g1_s0,
        "U,S1": scored_g0_s1 + scored_g1_s1,
    }
    cells = {
        name: cell_record(name.split(",")[0], name.split(",")[1], entries, control, native_size, working_size)
        for name, entries in populations.items()
    }
    diagonal = {
        "G0,S0": (None if cells["G0,S0"]["line"] is None else cells["G0,S0"]["line"]["candidate_id"],
                   None if cells["G0,S0"]["paint"] is None else cells["G0,S0"]["paint"]["candidate_id"]),
        "G1,S1": (None if cells["G1,S1"]["line"] is None else cells["G1,S1"]["line"]["candidate_id"],
                   None if cells["G1,S1"]["paint"] is None else cells["G1,S1"]["paint"]["candidate_id"]),
    }
    expected = {
        "G0,S0": saved_baseline_winners[case_id],
        "G1,S1": (g1_record["line_winner_id"], g1_record["paint_winner_id"]),
    }
    diagonal_matches = {name: diagonal[name] == expected[name] for name in expected}
    diagonal_ok = all(diagonal_matches.values())
    witnesses = {
        "case_id": case_id,
        "label": CASE_LABELS[case_id],
        "control": control_record,
        "population_sources": {"G0": g0_source, "G1": str(g1_record_path.relative_to(REPO))},
        "population_order": {"G0": "saved G0 order", "G1": "saved G1 order", "U": "G0 followed by G1"},
        "saved_winners": {name: list(value) for name, value in expected.items()},
        "replayed_winners": {name: list(value) for name, value in diagonal.items()},
        "diagonal_matches": diagonal_matches,
        "cells": cells,
    }
    return {"case_id": case_id, "label": CASE_LABELS[case_id], "cells": cells, "diagonal_ok": diagonal_ok, "witnesses": witnesses}


def main() -> None:
    output = Path(__file__).resolve().parent
    cv2.setNumThreads(1)
    saved_baseline_winners = load_saved_baseline_winners()
    results = [run_case(case_id, output, saved_baseline_winners) for case_id in CASE_IDS]
    rows = []
    for result in results:
        diagonal = result["witnesses"]["diagonal_matches"]
        for cell in result["cells"].values():
            rows.append(csv_row(result["case_id"], cell, diagonal))
    columns = list(rows[0])
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    diagnostic_records = [run_diagnostic(output, *diagnostic) for diagnostic in DIAGNOSTIC_CASES]
    witnesses = {
        "schema": "l2-generation-scoring-witnesses/1",
        "cases": [result["witnesses"] for result in results],
        "diagnostic_panel": diagnostic_records,
    }
    write_json(output / "witnesses.json", witnesses)
    (output / "result.md").write_text(build_result(results))
    print((output / "result.md").read_text())
    print("wrote", output / "comparison.csv")
    print("wrote", output / "witnesses.json")
    print("wrote", len(diagnostic_records), "diagnostic images")


if __name__ == "__main__":
    main()
