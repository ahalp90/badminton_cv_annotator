"""Experimental net-support preference over the saved W5 full-court choice.

Each candidate's court corners imply a net: two tape halves and two posts. This
trial measures how much of each projected piece lies on cached DeepLSD segments
from the same image. Among full-court candidates, in the original W5 camera-valid
order, it prefers the first one with strong net support. With no strongly
supported candidate, it keeps the original W5 winner.

Weak or missing net support never rejects a candidate; it only withholds the
preference. References enter only after every selection is fixed, as a labelled
retrospective diagnostic.

Limits: no occlusion masking (players and the net mesh can hide or mimic
fragments); the net comes from ``net_geometry.project_net``, which assumes a
pinhole camera with square pixels and a centred principal point; post and tape
heights are regulation values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "w5_holistic"))
sys.path.insert(0, str(ROOT / "wider_evaluation"))

import compare  # pyrefly: ignore[missing-import]
import run_w5  # pyrefly: ignore[missing-import]

run_w5.add_helper_paths(ROOT)
import verifier  # pyrefly: ignore[missing-import]

from experiments.annotator.independent_court import net_geometry

AM1 = "am1_window_00_frame_54"
RECORDS = ROOT / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/case_records"
COMPARISON = ROOT / "wider_evaluation/runs/20260922/comparison.json.gz"
NUMERIC = ROOT / "wider_evaluation/runs/20260922/numeric_fit.json.gz"
SAVED_CASES = (
    ("am1_saved_baseline", AM1),
    ("gx5_retention", "gxBQ_window_00_frame_5"),
    ("letterboxed45_retention", "letterboxed_short_frame_45"),
)
POOL_LABEL = "am1_seeded_pool"

# Same measurement as the net-feasibility probe (net_fragment_probe.py).
PIECE_NAMES = ("tape_left", "tape_right", "post_left", "post_right")
SAMPLES_PER_PIECE = 24
PERPENDICULAR_TOLERANCE_WORKING_PX = 4.0
DIRECTION_TOLERANCE_DEG = 8.0
EXTENT_MARGIN_WORKING_PX = 2.0
STRONG_SUPPORT = 0.75
RULE = (
    "Among full-court-gated candidates in stored W5 C provisional_rank order, choose the first whose "
    "tape_left and tape_right coverage are both >= 0.75 and at least one post coverage is >= 0.75. "
    "Missing coverage never counts as support. With no such candidate, keep the original W5 full/gated winner."
)


def piece_coverage(piece: np.ndarray, segments: np.ndarray, working_size: tuple[int, int]) -> tuple[float | None, int]:
    """Measure how much of one projected piece lies on an aligned image fragment.

    Reproduces ``piece_support`` from the net-feasibility probe, returning only
    the coverage and the in-frame sample count.

    :param piece: Working-pixel piece endpoints, shape (2, 2) as (start/end, x/y).
    :param segments: Working-pixel DeepLSD segments, one per row as x1, y1, x2, y2.
    :param working_size: Working image size as (width, height).
    :return: Covered fraction of in-frame samples (None when no sample is in
        frame), and the number of in-frame samples.
    """
    start, end = piece
    direction = (end - start) / np.linalg.norm(end - start)
    samples = start + np.linspace(0, 1, SAMPLES_PER_PIECE)[:, None] * (end - start)  # (samples, x/y)
    sample_x, sample_y = samples.T
    width, height = working_size
    in_frame = (sample_x >= 0) & (sample_x < width) & (sample_y >= 0) & (sample_y < height)
    segment_start, segment_end = segments[:, :2], segments[:, 2:]
    segment_vector = segment_end - segment_start
    segment_length = np.linalg.norm(segment_vector, axis=1)
    segment_direction = segment_vector / segment_length[:, None]
    aligned = np.abs(segment_direction @ direction) >= np.cos(np.radians(DIRECTION_TOLERANCE_DEG))
    covered = np.zeros(SAMPLES_PER_PIECE, dtype=bool)
    for sample_index, sample in enumerate(samples):
        offset = sample - segment_start
        along = (offset * segment_direction).sum(axis=1)
        within_extent = (along >= -EXTENT_MARGIN_WORKING_PX) & (along <= segment_length + EXTENT_MARGIN_WORKING_PX)
        perpendicular = np.abs(offset[:, 0] * segment_direction[:, 1] - offset[:, 1] * segment_direction[:, 0])
        near = aligned & within_extent & (perpendicular <= PERPENDICULAR_TOLERANCE_WORKING_PX)
        covered[sample_index] = near.any()
    visible = int(in_frame.sum())
    if visible == 0:
        return None, 0
    return float((covered & in_frame).sum() / visible), visible


def project_pieces(corners_native: list, context) -> dict:
    """Project the net from native corners, then convert its pieces to working pixels once."""
    try:
        projection = net_geometry.project_net(corners_native, context.native_size)
    except ValueError as error:
        return {"state": "projection_failed", "reason": str(error)}
    native_per_working = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    return {
        "state": "measured",
        "camera_error": projection.camera_error,
        "focal_widths": projection.focal_widths,
        "pieces_native_px": projection.segments_px,  # (piece, start/end, x/y)
        "pieces_working_px": projection.segments_px / native_per_working,
    }


def is_strong(coverage: list[float | None]) -> bool:
    supported = [value is not None and value >= STRONG_SUPPORT for value in coverage]
    tape_left, tape_right, post_left, post_right = supported
    return tape_left and tape_right and (post_left or post_right)


def selection_geometry(candidate: dict, context, net: dict) -> dict:
    corners_native = np.asarray(candidate["corners_px"], dtype=float)
    native_per_working = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    return {
        "origin_key": candidate["origin_key"],
        "candidate_id": candidate["candidate_id"],
        "source": candidate["source"],
        "native_size_wh": list(context.native_size),
        "working_size_wh": list(context.size),
        "corners_native_px": corners_native,
        "corners_working_px": corners_native / native_per_working,
        "homography_working": candidate["homography_working"],
        "gates": candidate["gates"],
        "historical": candidate["historical"],
        "net": net,
    }


def check_record(label: str, case_id: str, record: dict, path: Path, context, comparator: dict | None) -> None:
    """Fail loudly unless the record is one self-consistent W5 ranking for this view."""
    if record["case_id"] != case_id:
        raise ValueError(f"{label}: record case {record['case_id']} is not {case_id}")
    if record["provenance"]["frame_path"] != context.frame_relative_path:
        raise ValueError(f"{label}: frame {record['provenance']['frame_path']} is not {context.frame_relative_path}")
    candidates = record["parents"] + record["valid_children"]
    if len({candidate["origin_key"] for candidate in candidates}) != len(candidates):
        raise ValueError(f"{label}: duplicate origin keys")
    if verifier.rank_candidates(candidates) != record["rankings"]["C"]:
        raise ValueError(f"{label}: stored C ranking does not match a fresh W5 ranking of the same candidates")
    if comparator is not None:
        if Path(comparator["record"]).resolve() != path.resolve():
            raise ValueError(f"{label}: comparator row points at {comparator['record']}")
        if compare.selections(record) != comparator["selections"]:
            raise ValueError(f"{label}: comparator selections differ from the saved record")


def evaluate_case(label: str, case_id: str, record: dict, context, timings: dict) -> dict:
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    ranking = record["rankings"]["C"]["provisional_rank"]
    original = compare.selections(record)["full"]
    rows = []
    nets = {}
    full_court_rank = 0
    for original_rank, key in enumerate(ranking, start=1):
        candidate = candidates[key]
        tick = perf_counter()
        net = project_pieces(candidate["corners_px"], context)
        timings["projection"] += perf_counter() - tick
        tick = perf_counter()
        coverage = [None] * len(PIECE_NAMES)
        visible = [0] * len(PIECE_NAMES)
        if net["state"] == "measured":
            for piece_index, piece in enumerate(net["pieces_working_px"]):
                coverage[piece_index], visible[piece_index] = piece_coverage(piece, context.segments, context.size)
            net["coverage"] = dict(zip(PIECE_NAMES, coverage))
            net["visible_samples"] = dict(zip(PIECE_NAMES, visible))
        timings["scoring"] += perf_counter() - tick
        full_court = candidate["historical"]["historical_fullcourt"]
        if full_court:
            full_court_rank += 1
        nets[key] = net
        rows.append({
            "origin_key": key,
            "candidate_id": candidate["candidate_id"],
            "source": candidate["source"],
            "original_rank": original_rank,
            "full_court_rank": full_court_rank if full_court else None,
            "camera_eligible": candidate["camera_eligible"],
            "gate_camera_error": candidate["gates"]["camera_error"],
            "historical_fullcourt": full_court,
            "net_state": net["state"],
            "net_camera_error": net.get("camera_error"),
            "coverage": coverage,
            "visible_samples": visible,
            "strong_net": is_strong(coverage),
        })

    full_court_rows = [row for row in rows if row["historical_fullcourt"]]
    baseline_key = full_court_rows[0]["origin_key"] if full_court_rows else None
    # The trial must start from exactly the comparator's full/gated winner.
    if baseline_key != original["gated"]:
        raise ValueError(f"{label}: gated order winner {baseline_key} is not comparator {original['gated']}")
    strong_rows = [row for row in full_court_rows if row["strong_net"]]
    trial_key = strong_rows[0]["origin_key"] if strong_rows else baseline_key
    ranked_keys = set(ranking)
    return {
        "label": label,
        "case_id": case_id,
        "frame_path": context.frame_relative_path,
        "native_size_wh": list(context.native_size),
        "working_size_wh": list(context.size),
        "ordering_criterion": record["rankings"]["C"]["r2_criterion"],
        "rank_status": record["rankings"]["C"]["status"],
        "original_comparator_full": original,
        "counts": {
            "candidates": len(candidates),
            "ranked_camera_eligible": len(rows),
            "full_court_gated": len(full_court_rows),
            "projection_failed": sum(row["net_state"] == "projection_failed" for row in rows),
            "strong_net_all_ranked": sum(row["strong_net"] for row in rows),
            "strong_net_full_court": len(strong_rows),
        },
        "trial_reason": "first_strong_net_full_court" if strong_rows else "no_strong_net_kept_original",
        "changed": trial_key != baseline_key,
        "baseline": None if baseline_key is None else selection_geometry(
            candidates[baseline_key], context, nets[baseline_key]),
        "trial": None if trial_key is None else selection_geometry(candidates[trial_key], context, nets[trial_key]),
        "candidates": rows,
        "unranked_origin_keys": [key for key in candidates if key not in ranked_keys],
    }


def retrospective_errors(case: dict, reference_corners: list | None) -> dict:
    """Corner error against the stored reference, computed after selection for diagnosis only."""
    if not reference_corners:
        return {"status": "no_reference_corners"}
    reference = np.asarray(reference_corners, dtype=float)
    errors = {"status": "measured"}
    for role in ("baseline", "trial"):
        selected = case[role]
        errors[role] = None if selected is None else verifier.reference_corner_error(
            np.asarray(selected["corners_native_px"]), reference)
    return errors


def main(output: Path, am1_pool: Path | None) -> None:
    started = perf_counter()
    timings = {"preparation": 0.0, "projection": 0.0, "scoring": 0.0}
    tick = perf_counter()
    comparison = {case["case_id"]: case for case in verifier.read_json_gz(COMPARISON)["cases"]}
    inputs = []
    skipped = []
    if am1_pool is None:
        skipped.append({"label": POOL_LABEL, "reason": "--am1-pool not given"})
    else:
        if not am1_pool.is_file():
            raise FileNotFoundError(am1_pool)
        inputs.append((POOL_LABEL, AM1, am1_pool, None))
    for label, case_id in SAVED_CASES:
        path = RECORDS / f"{case_id}.json.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs.append((label, case_id, path, comparison[case_id]))
    timings["preparation"] += perf_counter() - tick

    contexts = {}
    cases = []
    for label, case_id, path, comparator in inputs:
        tick = perf_counter()
        if case_id not in contexts:
            contexts[case_id] = verifier.prepare_view(ROOT, case_id)
        context = contexts[case_id]
        record = verifier.read_json_gz(path)
        check_record(label, case_id, record, path, context, comparator)
        timings["preparation"] += perf_counter() - tick
        case = evaluate_case(label, case_id, record, context, timings)
        case["source_record"] = str(path.resolve().relative_to(REPO))
        cases.append(case)
        print(f"{label}: baseline {case['baseline'] and case['baseline']['origin_key']} "
              f"trial {case['trial'] and case['trial']['origin_key']} ({case['trial_reason']})", flush=True)

    # Every selection above is fixed before any reference is read.
    numeric = {case["case_id"]: case for case in verifier.read_json_gz(NUMERIC)["cases"]}
    for case in cases:
        reference = numeric.get(case["case_id"], {}).get("reference", {}).get("corners_px")
        case["retrospective_reference_error_diagnostic_only"] = retrospective_errors(case, reference)

    result = {
        "schema": "am1-net-selection-trial/1",
        "settings": {
            "rule": RULE,
            "piece_names": PIECE_NAMES,
            "samples_per_piece": SAMPLES_PER_PIECE,
            "perpendicular_tolerance_working_px": PERPENDICULAR_TOLERANCE_WORKING_PX,
            "direction_tolerance_deg": DIRECTION_TOLERANCE_DEG,
            "extent_margin_working_px": EXTENT_MARGIN_WORKING_PX,
            "strong_support": STRONG_SUPPORT,
            "fragments": "all cached DeepLSD segments from verifier.prepare_view, no occlusion mask",
            "coverage_denominator": "in-frame samples only; null coverage when no sample is in frame",
        },
        "sources": {
            "am1_pool": None if am1_pool is None else str(am1_pool.resolve()),
            "saved_records": str(RECORDS.relative_to(REPO)),
            "comparison": str(COMPARISON.relative_to(REPO)),
            "reference_for_retrospective_only": str(NUMERIC.relative_to(REPO)),
        },
        "skipped": skipped,
        "changed_selection_count": sum(case["changed"] for case in cases),
        "cases": cases,
        "reference_note": "Retrospective only; mixed annotation conventions rule out claims from small drifts.",
        "timings_seconds": {**timings, "total": perf_counter() - started},
    }
    verifier.write_json_gz(output, result)
    print(f"changed selections: {result['changed_selection_count']} of {len(cases)}; wrote {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--am1-pool", type=Path, help="gzipped W5 record for the seeded Am1 pool")
    arguments = parser.parse_args()
    main(arguments.output, arguments.am1_pool)
