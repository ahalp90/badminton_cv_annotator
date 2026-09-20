"""Run the W5 replay preflight and the bounded holistic court pilot."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def add_helper_paths(root: Path) -> None:
    """Load the L2 seed snapshot and frozen helper snapshots in their intended order."""
    paths = (
        root / "next_steps_20260916/webui_seed/source",
        root / "frozen_helpers_20260914/marking_diagnosis",
        root / "frozen_helpers_20260914/vp_pruning",
        root / "frozen_helpers_20260914/axis_matching",
        root / "frozen_helpers_20260914/legacy",
        root / "src",
        root,
    )
    for path in reversed(paths):
        if path.exists():
            sys.path.insert(0, str(path))


def module_paths(root: Path) -> dict[str, str]:
    names = ("run_automatic", "run_given", "run_population", "run_diagnosis", "zone_net")
    paths = {}
    for name in names:
        module = importlib.import_module(name)
        paths[name] = str(Path(module.__file__).resolve().relative_to(root.resolve()))
    return paths


def import_runtime(root: Path) -> dict[str, Any]:
    add_helper_paths(root)
    from run_automatic import evaluate_pool, select_pool
    from run_diagnosis import gate_evidence

    return {
        "evaluate_pool": evaluate_pool,
        "select_pool": select_pool,
        "gate_evidence": gate_evidence,
        "zone": importlib.import_module("zone_net"),
        "paths": module_paths(root),
    }


def load_verifier(root: Path):
    sys.path.insert(0, str(root / "w5_holistic"))
    from verifier import (
        CAMERA_LIMIT,
        CASE_IDS,
        CASE_LABELS,
        CASE_ORDER,
        CASE_PACKS,
        PACK_OF,
        PHOTO_CENTRE_OFFSETS_PX,
        PHOTO_SIDE_DISTANCE_PX,
        WORKING_SIZE,
        ViewContext,
        camera_eligible,
        candidate_review,
        convex_corners,
        frame_path,
        hard_validity,
        historical_predicates,
        jsonable,
        legacy_winners,
        load_source,
        measure_candidate,
        permutation_determinism,
        prepare_view,
        project_corners,
        projected_landmark_diagnostics,
        rank_candidates,
        read_json_gz,
        reference_corner_error,
        relative_path,
        source_provenance,
        write_json_gz,
    )
    return {
        "CASE_IDS": CASE_IDS,
        "CASE_LABELS": CASE_LABELS,
        "CASE_ORDER": CASE_ORDER,
        "CASE_PACKS": CASE_PACKS,
        "PACK_OF": PACK_OF,
        "CAMERA_LIMIT": CAMERA_LIMIT,
        "PHOTO_CENTRE_OFFSETS_PX": PHOTO_CENTRE_OFFSETS_PX,
        "PHOTO_SIDE_DISTANCE_PX": PHOTO_SIDE_DISTANCE_PX,
        "WORKING_SIZE": WORKING_SIZE,
        "ViewContext": ViewContext,
        "candidate_review": candidate_review,
        "camera_eligible": camera_eligible,
        "convex_corners": convex_corners,
        "frame_path": frame_path,
        "hard_validity": hard_validity,
        "historical_predicates": historical_predicates,
        "legacy_winners": legacy_winners,
        "load_source": load_source,
        "measure_candidate": measure_candidate,
        "permutation_determinism": permutation_determinism,
        "projected_landmark_diagnostics": projected_landmark_diagnostics,
        "project_corners": project_corners,
        "prepare_view": prepare_view,
        "reference_corner_error": reference_corner_error,
        "rank_candidates": rank_candidates,
        "read_json_gz": read_json_gz,
        "relative_path": relative_path,
        "source_provenance": source_provenance,
        "write_json_gz": write_json_gz,
        "jsonable": jsonable,
    }


KNOWN_CONTROLS = {
    "am2_window_00_frame_150": {
        "30:33": "positive_approved",
    },
    "shuttleset_03_scene_0019": {
        "1:60": "positive_usable",
        "165:6702": "negative_rejected_hallucinated",
    },
    "am2_window_01_frame_28019": {
        "184:4123": "negative_rejected_false_paint",
    },
}


class ViewAmbiguity(AssertionError):
    """A candidate identity conflict makes one view unsafe to score."""
L2_COMPARISON_CASES = (
    "gxBQ_window_00_frame_0",
    "am2_window_00_frame_150",
    "am2_window_01_frame_28019",
    "am3_window_00_frame_0",
)
EXPECTED_MASKS = {
    "gxBQ_window_00_frame_0": True,
    "gxBQ_window_00_frame_5": False,
    "am2_window_00_frame_150": True,
    "am2_window_01_frame_28019": True,
    "am3_window_00_frame_0": True,
    "shuttleset_03_scene_0017": False,
    "shuttleset_03_scene_0019": False,
    "shuttleset_03_scene_0016": False,
    "shuttleset_21_scene_0020": False,
}


def load_runtime(root: Path) -> dict[str, Any]:
    return {"verifier": load_verifier(root), **import_runtime(root)}


def reconstruct_generation_entries(record: dict, native_size: tuple[int, int], select_pool, detector) -> list[dict]:
    """Replay L2's unchanged global selection from its saved per-pair shortlists."""
    working_size = np.asarray(record["working_size"], dtype=float)
    native_scale = np.asarray(native_size, dtype=float) / working_size
    candidates = []
    provenance: dict[int, dict] = {}
    for pair in record["pairs"]:
        for entry in pair.get("shortlist", []):
            candidate = detector.Candidate(
                np.asarray(entry["corners_px"], dtype=float) / native_scale,
                float(entry["shortlist_score"]),
                (0.0, 0.0),
                (0, 0),
            )
            candidates.append(candidate)
            provenance[id(candidate)] = entry
    retained = select_pool(candidates)
    return [provenance[id(candidate)] for candidate in retained]


def load_g0(root: Path, context, runtime: dict[str, Any]) -> tuple[list[dict], str]:
    verifier = runtime["verifier"]
    direct = root / "frozen_views/baseline_generation" / f"{context.case_id}.json.gz"
    if direct.exists():
        entries = verifier["read_json_gz"](direct)["entries"]
        if len(entries) != 256:
            raise ValueError(f"{context.case_id}: direct G0 has {len(entries)} entries")
        return entries, "direct:" + verifier["relative_path"](direct, root)
    record_path = root / "automatic_axes_20260914/all_camera" / f"{context.case_id}.json.gz"
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    record = verifier["read_json_gz"](record_path)
    shortlist = reconstruct_generation_entries(
        record,
        context.native_size,
        runtime["select_pool"],
        import_detector(),
    )
    run_automatic = import_run_automatic()
    original_frame_path = run_automatic.frame_path
    run_automatic.frame_path = lambda source, _root: verifier["frame_path"](root, source)
    try:
        entries = runtime["evaluate_pool"](
            context.source,
            shortlist,
            context.observations,
            context.size,
            context.segments,
            context.families,
            runtime["zone"],
            root,
        )
    finally:
        run_automatic.frame_path = original_frame_path
    if len(entries) != 256:
        raise ValueError(f"{context.case_id}: replayed G0 has {len(entries)} entries")
    return entries, "replayed:" + verifier["relative_path"](record_path, root)


def import_detector():
    from experiments.annotator.independent_court import detector

    return detector


def import_run_automatic():
    return importlib.import_module("run_automatic")


def load_g1(root: Path, context, verifier: dict[str, Any]) -> tuple[list[dict], str]:
    path = root / "line_identity/runs/line_identity_20260915_222437/matcher/paint_observations/results" / (
        f"{context.case_id}.json.gz"
    )
    record = verifier["read_json_gz"](path)
    entries = record["entries"]
    if len(entries) != 256:
        raise ValueError(f"{context.case_id}: saved G1 has {len(entries)} entries")
    return entries, verifier["relative_path"](path, root)


def load_populations(root: Path, context, runtime: dict[str, Any]) -> tuple[list[dict], list[dict], dict]:
    g0, g0_source = load_g0(root, context, runtime)
    g1, g1_source = load_g1(root, context, runtime["verifier"])
    return g0, g1, {"G0": g0_source, "G1": g1_source}


def expected_legacy_winners(root: Path, case_id: str, verifier: dict[str, Any]) -> dict:
    accounting = root / "direction_agreement/runs/direction_agreement_20260915_144900/e4/accounting.csv.gz"
    import gzip

    with gzip.open(accounting, "rt", newline="") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            if row["stage"] == "results" and row["arm"] == "B" and row["case_id"] == case_id:
                return {"line": row["line_winner_id"] or None, "paint": row["paint_winner_id"] or None}
    raise KeyError(case_id)


def forbidden_automatic_key(key: str) -> bool:
    lower = key.lower()
    return any(token in lower for token in ("reference", "manual", "ruling", "landmark", "control"))


def find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if forbidden_automatic_key(str(key)):
                found.append(child_path)
            found.extend(find_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(find_forbidden_keys(child, f"{path}[{index}]"))
    return found


def preflight_determinism(verifier: dict[str, Any]) -> dict:
    candidates = []
    for source_order, source in enumerate(("G0", "G1")):
        for origin_index, score in enumerate((0.7, 0.7, 0.6)):
            candidates.append({
                "origin_key": f"{source}:{origin_index}",
                "source_order": source_order,
                "origin_index": origin_index,
                "kind_order": 0,
                "hard_valid": True,
                "historical": {"historical_fullcourt": True, "historical_camera": True},
                "evidence": {"q_geom": score, "q_paint10": score, "exclusive_reverse": score},
            })
    return verifier["permutation_determinism"](candidates)


def run_preflight(root: Path, run_dir: Path) -> dict:
    runtime = load_runtime(root)
    verifier = runtime["verifier"]
    results = {
        "schema": "w5-preflight/1",
        "status": "passed",
        "case_ids": list(verifier["CASE_IDS"]),
        "module_paths": runtime["paths"],
        "working_dimensions": {},
        "g0_source": {},
        "mask_availability": {},
        "automatic_reference_fields": {},
        "l2_replay": [],
        "determinism": preflight_determinism(verifier),
        "failures": [],
    }
    if results["case_ids"] != [case_id for case_id, _, _ in verifier["CASE_ORDER"]]:
        results["failures"].append("case ID order does not match the W5 contract")
    populations = {}
    for case_id in verifier["CASE_IDS"]:
        context = verifier["prepare_view"](root, case_id)
        results["working_dimensions"][case_id] = list(context.size)
        if context.size != verifier["WORKING_SIZE"]:
            results["failures"].append(f"{case_id}: working dimensions {context.size}")
        actual_mask = context.same_image_mask_available
        results["mask_availability"][case_id] = {
            "actual": actual_mask,
            "expected": EXPECTED_MASKS[case_id],
            "match": actual_mask == EXPECTED_MASKS[case_id],
        }
        if actual_mask != EXPECTED_MASKS[case_id]:
            results["failures"].append(f"{case_id}: same-image mask provenance differs")
        g0_path = root / "frozen_views/baseline_generation" / f"{case_id}.json.gz"
        if g0_path.exists():
            g0_source = "direct:" + verifier["relative_path"](g0_path, root)
        else:
            replay_path = root / "automatic_axes_20260914/all_camera" / f"{case_id}.json.gz"
            g0_source = "replayed:" + verifier["relative_path"](replay_path, root)
        results["g0_source"][case_id] = g0_source
        g0, g1, sources = load_populations(root, context, runtime)
        populations[case_id] = (g0, g1, sources)
        automatic_entries = g0 + g1
        forbidden = []
        for index, entry in enumerate(automatic_entries):
            forbidden.extend(find_forbidden_keys(entry, f"{case_id}.automatic[{index}]"))
        results["automatic_reference_fields"][case_id] = {"match": not forbidden, "fields": forbidden}
        if forbidden:
            results["failures"].append(f"{case_id}: automatic candidate path contains reference fields")
    for case_id in L2_COMPARISON_CASES:
        context = verifier["prepare_view"](root, case_id)
        g0, g1, sources = populations[case_id]
        actual_g0 = verifier["legacy_winners"](g0)
        actual_g1 = verifier["legacy_winners"](g1)
        expected_g0 = expected_legacy_winners(root, case_id, verifier)
        g1_record = verifier["read_json_gz"](
            root / "line_identity/runs/line_identity_20260915_222437/matcher/paint_observations/results" /
            f"{case_id}.json.gz"
        )
        expected_g1 = {"line": g1_record["line_winner_id"], "paint": g1_record["paint_winner_id"]}
        for label, expected, actual in (("G0,S0", expected_g0, actual_g0), ("G1,S1", expected_g1, actual_g1)):
            match = {key: expected[key] == actual[key] for key in ("line", "paint")}
            results["l2_replay"].append({
                "case_id": case_id,
                "identity": label,
                "expected": expected,
                "reproduced": {"line": actual["line"], "paint": actual["paint"]},
                "match": match,
                "population_count": len(g0 if label.startswith("G0") else g1),
                "population_source": sources["G0" if label.startswith("G0") else "G1"],
                "cause_of_difference": None if all(match.values()) else "population or legacy scorer path differs",
            })
            if not all(match.values()):
                results["failures"].append(f"{case_id} {label}: legacy winner identity mismatch")
    if not results["determinism"]["match"]:
        results["failures"].append("ranker permutation determinism check failed")
    results["status"] = "passed" if not results["failures"] else "failed"
    verifier["write_json_gz"](run_dir / "preflight.json.gz", results)
    (run_dir / "preflight.json").write_text(
        __import__("json").dumps(verifier["jsonable"](results), indent=2, sort_keys=True) + "\n"
    )
    if results["status"] != "passed":
        raise RuntimeError("W5 preflight failed; see preflight.json")
    return results


def compact_legacy(entry: dict) -> dict:
    stripe = entry.get("stripe", {})
    exclusive = stripe.get("exclusive", {})
    profile = entry.get("profile", {})
    return {
        "profile_score": profile.get("score"),
        "stripe_exclusive_score": exclusive.get("score"),
        "stripe_exclusive_reverse": exclusive.get("reverse"),
    }


def candidate_geometry_key(entry: dict) -> tuple[tuple[int, ...], bytes]:
    homography = np.asarray(entry["homography_working"], dtype=float)
    return homography.shape, homography.tobytes()


def values_equal_with_nan(left: Any, right: Any) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            values_equal_with_nan(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def w5_gate_fields(entry: dict) -> tuple[Any, Any, Any]:
    gates = entry.get("gates", {})
    return (
        gates.get("geometry_valid"),
        gates.get("camera_error"),
        gates.get("player_fractions"),
    )


def source_occurrence(entry: dict, source: str, source_order: int, origin_index: int) -> dict:
    candidate_id = str(entry["candidate_id"])
    return {
        "origin_key": f"{source}:{candidate_id}",
        "source": source,
        "source_order": source_order,
        "origin_index": origin_index,
        "candidate_id": candidate_id,
        "pair_id": entry.get("pair_id"),
        "axis_ids": entry.get("axis_ids"),
        "rotated_180": entry.get("rotated_180"),
    }


def canonicalise_populations(g0: list[dict], g1: list[dict]) -> tuple[list[dict], dict]:
    """Build collision-safe parent identities while retaining source occurrences."""
    source_entries = (("G0", g0), ("G1", g1))
    source_ids = {
        source: [str(entry["candidate_id"]) for entry in entries]
        for source, entries in source_entries
    }
    for source, ids in source_ids.items():
        if len(ids) != len(set(ids)):
            raise ViewAmbiguity(f"{source} candidate IDs are not unique within their source")

    by_geometry: dict[tuple[tuple[int, ...], bytes], dict] = {}
    records = []
    for source_order, (source, entries) in enumerate(source_entries):
        for origin_index, entry in enumerate(entries):
            geometry_key = candidate_geometry_key(entry)
            occurrence = source_occurrence(entry, source, source_order, origin_index)
            legacy_occurrence = {
                **occurrence,
                "gates": entry.get("gates", {}),
                "legacy": compact_legacy(entry),
            }
            record = by_geometry.get(geometry_key)
            if record is None:
                record = {
                    "entry": entry,
                    "origin_key": occurrence["origin_key"],
                    "candidate_id": occurrence["candidate_id"],
                    "source": source,
                    "source_order": source_order,
                    "origin_index": origin_index,
                    "source_memberships": [source],
                    "source_occurrences": [occurrence],
                    "occurrence_count": 1,
                    "_legacy_occurrences": [legacy_occurrence],
                }
                by_geometry[geometry_key] = record
                records.append(record)
                continue
            if source in record["source_memberships"]:
                raise ViewAmbiguity(
                    f"{source}: duplicate geometry for {occurrence['candidate_id']} and {record['candidate_id']}"
                )
            reference_entry = record["entry"]
            if not np.array_equal(
                np.asarray(entry["corners_px"], dtype=float),
                np.asarray(reference_entry["corners_px"], dtype=float),
            ):
                raise ViewAmbiguity(
                    f"{source}: duplicate homography has differing corners for "
                    f"{occurrence['candidate_id']} and {record['candidate_id']}"
                )
            for field in ("pair_id", "rotated_180"):
                if entry.get(field) != reference_entry.get(field):
                    raise ViewAmbiguity(
                        f"{source}: duplicate homography has differing {field} for "
                        f"{occurrence['candidate_id']} and {record['candidate_id']}"
                    )
            if not all(
                values_equal_with_nan(left, right)
                for left, right in zip(w5_gate_fields(entry), w5_gate_fields(reference_entry), strict=True)
            ):
                raise ViewAmbiguity(
                    f"{source}: duplicate geometry has differing W5 gates for "
                    f"{occurrence['candidate_id']} and {record['candidate_id']}"
                )
            record["source_memberships"].append(source)
            record["source_occurrences"].append(occurrence)
            record["occurrence_count"] += 1
            record["_legacy_occurrences"].append(legacy_occurrence)

    for record in records:
        for occurrence in record["_legacy_occurrences"]:
            occurrence["parent_origin_key"] = record["origin_key"]

    raw_id_collisions = sorted(set(source_ids["G0"]) & set(source_ids["G1"]))
    duplicate_groups = [
        {
            "origin_key": record["origin_key"],
            "occurrence_count": record["occurrence_count"],
            "source_memberships": record["source_memberships"],
            "source_occurrences": record["source_occurrences"],
            "legacy_occurrences": record["_legacy_occurrences"],
        }
        for record in records
        if record["occurrence_count"] > 1
    ]
    return records, {
        "policy": "source-qualified canonical origin_key; exact cross-source geometry duplicates retain all source occurrences when W5-relevant gates agree",
        "source_occurrence_counts": {"G0": len(g0), "G1": len(g1)},
        "source_occurrence_count": len(g0) + len(g1),
        "canonical_parent_count": len(records),
        "raw_id_collisions": raw_id_collisions,
        "raw_id_collision_count": len(raw_id_collisions),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
    }


def make_parent_record(
    context,
    entry: dict,
    source: str,
    source_order: int,
    origin_index: int,
    runtime: dict[str, Any],
    cache: dict,
    identity: dict | None = None,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    verifier = runtime["verifier"]
    if identity is None:
        occurrence = source_occurrence(entry, source, source_order, origin_index)
        identity = {
            "origin_key": occurrence["origin_key"],
            "candidate_id": occurrence["candidate_id"],
            "source": source,
            "source_order": source_order,
            "origin_index": origin_index,
            "source_memberships": [source],
            "source_occurrences": [occurrence],
            "occurrence_count": 1,
            "_legacy_occurrences": [{
                **occurrence,
                "gates": entry.get("gates", {}),
                "legacy": compact_legacy(entry),
            }],
        }
    candidate = {
        "origin_key": identity["origin_key"],
        "candidate_id": identity["candidate_id"],
        "kind": "parent",
        "source": identity["source"],
        "source_order": identity["source_order"],
        "origin_index": identity["origin_index"],
        "kind_order": 0,
        "parent_origin_key": None,
        "source_memberships": identity["source_memberships"],
        "source_occurrences": identity["source_occurrences"],
        "occurrence_count": identity["occurrence_count"],
        "corners_px": entry["corners_px"],
        "homography_working": entry["homography_working"],
        "gates": entry.get("gates", {}),
        "historical": verifier["historical_predicates"](entry.get("gates", {})),
        "legacy": compact_legacy(entry),
        "_legacy_occurrences": identity["_legacy_occurrences"],
    }
    valid, reason = verifier["hard_validity"](entry)
    candidate["hard_valid"] = valid
    candidate["hard_validity_reason"] = reason
    candidate["camera_eligible"] = verifier["camera_eligible"](candidate)
    if not valid:
        return candidate, None
    evidence, arrays = verifier["measure_candidate"](context, entry, cache)
    candidate["evidence"] = evidence
    candidate["_stripe_score"] = {
        "assignments": evidence["stripe_assignments"],
    }
    candidate["_entry"] = entry
    candidate["_arrays"] = arrays
    return candidate, arrays


def fit_row_base(parent: dict, status: str, reason: str | None = None) -> dict:
    return {
        "origin_key": parent["origin_key"],
        "candidate_id": parent["candidate_id"],
        "status": status,
        "rejection_reason": reason,
        "nfev": None,
        "solver_status": None,
        "successful": False,
        "jacobian_rank": None,
        "jacobian_condition": None,
        "minimum_corner_denominator": None,
        "objective_before": None,
        "objective_after": None,
        "objective_improvement": None,
        "fit_fragment_count": 0,
        "fit_fragment_ids": [],
        "assignment_strengths": [],
        "near_selected_strengths": [],
        "attempted_corners_native": None,
        "child_origin_key": None,
    }


def attempt_refit(context, parent: dict, runtime: dict[str, Any], cache: dict) -> tuple[dict, dict | None, dict[str, np.ndarray] | None]:
    verifier = runtime["verifier"]
    row = fit_row_base(parent, "not_attempted")
    if not parent["hard_valid"] or "evidence" not in parent:
        row["status"] = "parent_invalid"
        row["rejection_reason"] = parent.get("hard_validity_reason", "no_evidence")
        return row, None, None
    assignments = parent["_stripe_score"]["assignments"]
    homography = np.asarray(parent["homography_working"], dtype=float)
    constraints = import_fitting().prepare(
        homography,
        context.observations,
        assignments,
        context.weights,
        centres=import_paint_geometry().CENTRE_SEGMENTS_M,
    )
    row["fit_fragment_count"] = len(np.unique(constraints.fragment_ids))
    row["fit_fragment_ids"] = constraints.fragment_ids.tolist()
    row["assignment_strengths"] = assignments["strength"]
    row["near_selected_strengths"] = assignments["alternative_strength"]
    corners_working, _ = verifier["project_corners"](homography) if "project_corners" in verifier else (
        import_detector().project(homography[None], import_detector().CORNER_COURT_M)[0][0], None
    )
    fit = import_fitting().refine(
        corners_working,
        constraints,
        context.size,
        use_positions=True,
        centres=import_paint_geometry().CENTRE_SEGMENTS_M,
    )
    for key in ("nfev", "solver_status", "successful", "jacobian_rank", "jacobian_condition",
                "minimum_corner_denominator", "objective_before", "objective_after"):
        row[key] = fit.get(key)
    if fit.get("objective_before") is not None and fit.get("objective_after") is not None:
        row["objective_improvement"] = fit["objective_before"] - fit["objective_after"]
    attempted = fit.get("corners_px")
    if attempted is not None:
        attempted_working = np.asarray(attempted, dtype=float)
        native_scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
        attempted_native = attempted_working * native_scale
        row["attempted_corners_native"] = attempted_native.tolist()
    else:
        attempted_working = None
    reason = None
    if not fit.get("successful", False):
        reason = fit.get("status", "solver_failed")
    elif fit.get("jacobian_rank") != 8:
        reason = "jacobian_rank_not_8"
    elif fit.get("minimum_corner_denominator", 0.0) <= 1e-6:
        reason = "minimum_corner_denominator_not_positive"
    elif attempted_working is None or not np.isfinite(attempted_working).all():
        reason = "non_finite_attempted_corners"
    elif not verifier["convex_corners"](attempted_working):
        reason = "non_convex_attempted_corners"
    if reason is not None:
        row["status"] = fit.get("status", "rejected")
        row["rejection_reason"] = reason
        return row, None, None
    detector = import_detector()
    child_homography = cv2.getPerspectiveTransform(
        detector.CORNER_COURT_M.astype(np.float32), attempted_working.astype(np.float32)
    ).astype(float)
    native_corners = row["attempted_corners_native"]
    maps = detector._distance_maps(detector._wide_line_families(context.segments), context.size)
    child_gates = runtime["gate_evidence"](
        np.asarray(native_corners),
        context.source,
        np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float),
        context.size,
        context.families,
        maps,
        runtime["zone"],
    )
    child_entry = {
        "candidate_id": f"{parent['candidate_id']}/child",
        "corners_px": native_corners,
        "homography_working": child_homography.tolist(),
        "gates": child_gates,
        "profile": {},
        "stripe": {},
    }
    valid, geometry_reason = verifier["hard_validity"](child_entry)
    if not valid:
        row["status"] = "invalid_child"
        row["rejection_reason"] = geometry_reason or "child_geometry_invalid"
        return row, None, None
    child_evidence, child_arrays = verifier["measure_candidate"](context, child_entry, cache)
    child = {
        "origin_key": f"{parent['origin_key']}/child",
        "candidate_id": child_entry["candidate_id"],
        "kind": "child",
        "source": parent["source"],
        "source_order": parent["source_order"],
        "origin_index": parent["origin_index"],
        "kind_order": 1,
        "parent_origin_key": parent["origin_key"],
        "source_memberships": parent.get("source_memberships", []),
        "source_occurrences": parent.get("source_occurrences", []),
        "occurrence_count": parent.get("occurrence_count", 1),
        "corners_px": native_corners,
        "homography_working": child_homography.tolist(),
        "gates": child_gates,
        "historical": verifier["historical_predicates"](child_gates),
        "camera_eligible": verifier["camera_eligible"]({"gates": child_gates}),
        "hard_valid": True,
        "hard_validity_reason": None,
        "evidence": child_evidence,
        "refit": {key: fit.get(key) for key in fit if key != "corners_px"},
        "_stripe_score": {"assignments": child_evidence["stripe_assignments"]},
        "_entry": child_entry,
        "_arrays": child_arrays,
    }
    row["status"] = "valid_child"
    row["successful"] = True
    row["child_origin_key"] = child["origin_key"]
    parent["refit"] = {"status": "valid_child", "child_origin_key": child["origin_key"]}
    return row, child, child_arrays


def import_fitting():
    from experiments.annotator.independent_court import fixed_stripe_refit

    return fixed_stripe_refit


def import_paint_geometry():
    from experiments.annotator.independent_court import paint_geometry

    return paint_geometry


def public_candidate(candidate: dict) -> dict:
    public = {key: value for key, value in candidate.items() if not key.startswith("_")}
    if candidate.get("occurrence_count", 1) > 1:
        public["legacy_occurrences"] = candidate.get("_legacy_occurrences", [])
    return public


def load_control_entry(root: Path, case_id: str, candidate_id: str, verifier: dict[str, Any]) -> dict:
    path = root / "automatic_axes_20260914/all_camera" / f"{case_id}.json.gz"
    record = verifier["read_json_gz"](path)
    for entry in record.get("entries", []):
        if entry["candidate_id"] == candidate_id:
            return entry
    raise KeyError((case_id, candidate_id))


def has_source_occurrence(candidate: dict, source: str, candidate_id: str) -> bool:
    return any(
        occurrence["source"] == source and occurrence["candidate_id"] == candidate_id
        for occurrence in candidate.get("source_occurrences", [])
    )


def process_case(root: Path, case_id: str, run_dir: Path) -> dict:
    cv2.setNumThreads(1)
    runtime = load_runtime(root)
    verifier = runtime["verifier"]
    context = verifier["prepare_view"](root, case_id)
    g0, g1, sources = load_populations(root, context, runtime)
    parent_identities, identity_resolution = canonicalise_populations(g0, g1)
    cache: dict[bytes, tuple[dict, dict[str, np.ndarray]]] = {}
    parents = []
    all_arrays: dict[str, np.ndarray] = {}
    for identity in parent_identities:
        parent, arrays = make_parent_record(
            context,
            identity["entry"],
            identity["source"],
            identity["source_order"],
            identity["origin_index"],
            runtime,
            cache,
            identity=identity,
        )
        parents.append(parent)
        if arrays is not None:
            for key, value in arrays.items():
                all_arrays[f"{parent['origin_key']}::{key}"] = value
    fit_rows = []
    children = []
    for parent in parents:
        row, child, arrays = attempt_refit(context, parent, runtime, cache)
        fit_rows.append(row)
        if child is not None:
            children.append(child)
            for key, value in arrays.items():
                all_arrays[f"{child['origin_key']}::{key}"] = value
    b_candidates = [parent for parent in parents if parent.get("hard_valid") and "evidence" in parent]
    c_candidates = b_candidates + children
    b_rankings = verifier["rank_candidates"](b_candidates)
    c_rankings = verifier["rank_candidates"](c_candidates)
    determinism = verifier["permutation_determinism"](c_candidates)
    if not determinism["match"]:
        raise RuntimeError(f"{case_id}: W5 ranker is not permutation-deterministic")
    control_candidates = []
    for control_id, expected in KNOWN_CONTROLS.get(case_id, {}).items():
        entry = load_control_entry(root, case_id, control_id, verifier)
        control, _ = make_parent_record(context, entry, "diagnostic", 2, 0, runtime, cache)
        control["origin_key"] = f"diagnostic:{case_id}:{control_id}"
        control["candidate_id"] = control_id
        control["source_occurrences"][0]["origin_key"] = control["origin_key"]
        control["expected_ruling"] = expected
        control["in_automatic_pool"] = any(
            has_source_occurrence(parent, "G0", control_id) for parent in parents
        )
        control_candidates.append(control)
    controls = []
    for control in control_candidates:
        ranking_with_control = verifier["rank_candidates"](c_candidates + [control])

        control_ranks = {}
        for name in ("ungated_provisional_rank", "r1_paint10_rank", "provisional_rank"):
            ranking = ranking_with_control[name]
            control_ranks[name] = (
                ranking.index(control["origin_key"]) + 1
                if control["origin_key"] in ranking else None
            )

        review = verifier["candidate_review"](control)
        review["ranks"] = {
            "pilot": control_ranks["ungated_provisional_rank"],
            "r1": control_ranks["r1_paint10_rank"],
            "r2": control_ranks["provisional_rank"],
        }
        controls.append(review)
    provenance = verifier["source_provenance"](context, sources["G0"])
    public_parents = [public_candidate(parent) for parent in parents]
    public_children = [public_candidate(child) for child in children]
    full_record = {
        "schema": "w5-case-evidence/1",
        "case_id": case_id,
        "provenance": provenance,
        "population_sources": sources,
        "population_counts": {"G0": len(g0), "G1": len(g1), "union": len(parents)},
        "identity_resolution": identity_resolution,
        "parents": public_parents,
        "valid_children": public_children,
        "diagnostic_controls": controls,
        "fit_attempts": fit_rows,
        "rankings": {"B": b_rankings, "C": c_rankings},
        "determinism": determinism,
    }
    case_dir = run_dir / "case_records"
    case_dir.mkdir(parents=True, exist_ok=True)
    verifier["write_json_gz"](case_dir / f"{case_id}.json.gz", full_record)
    array_path = run_dir / "arrays" / f"{case_id}.npz"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **all_arrays)
    return {
        "schema": "w5-case-result/1",
        "case_id": case_id,
        "label": verifier["CASE_LABELS"][case_id],
        "provenance": provenance,
        "population_sources": sources,
        "population_counts": {"G0": len(g0), "G1": len(g1), "union": len(parents)},
        "identity_resolution": identity_resolution,
        "A": verifier["legacy_winners"](parents),
        "B": b_rankings,
        "C": c_rankings,
        "fit_attempt_count": len(fit_rows),
        "valid_child_count": len(children),
        "diagnostic_controls": controls,
        "determinism": determinism,
        "array_file": verifier["relative_path"](array_path, run_dir),
        "case_record": verifier["relative_path"](case_dir / f"{case_id}.json.gz", run_dir),
        "fit_rows": fit_rows,
        "review_candidates": {
            candidate["origin_key"]: verifier["candidate_review"](candidate)
            for candidate in b_candidates + children
        },
    }


def load_reference(root: Path, case_id: str, verifier: dict[str, Any]) -> dict:
    source_pack = verifier["read_json_gz"](root / verifier["CASE_PACKS"][verifier["PACK_OF"][case_id]])
    reference = source_pack.get("references", {}).get(case_id, {})
    names = reference.get("landmark_names", [])
    if names and reference.get("landmarks"):
        reference = dict(reference)
        reference["landmarks"] = [
            dict(landmark, name=names[index]) if index < len(names) else landmark
            for index, landmark in enumerate(reference["landmarks"])
        ]
    return reference


def load_supplied_control(root: Path, case_id: str, verifier: dict[str, Any]) -> dict | None:
    path = root / "direction_agreement/runs/direction_agreement_20260915_144900/e3" / f"{case_id}.json.gz"
    if not path.exists():
        return None
    return verifier["read_json_gz"](path).get("control")


def reference_diagnostics(root: Path, case_results: list[dict], verifier: dict[str, Any]) -> dict:
    diagnostics = {}
    for result in case_results:
        context = verifier["prepare_view"](root, result["case_id"])
        reference = load_reference(root, result["case_id"], verifier)
        supplied = load_supplied_control(root, result["case_id"], verifier)
        selected = {name: result[name].get("selected_origin_key") for name in ("B", "C")}
        selected["A_line"] = result["A"].get("line")
        selected["A_paint"] = result["A"].get("paint")
        record = {"case_id": result["case_id"], "selected_origin_keys": selected, "candidates": {}}
        candidate_map = result["review_candidates"]
        selected_origins = {value for value in selected.values() if value}
        for origin_key in selected_origins:
            candidate = candidate_map.get(origin_key)
            if candidate is None:
                continue
            metrics = {}
            corners = np.asarray(candidate["corners_px"], dtype=float) if "corners_px" in candidate else None
            if corners is not None and reference.get("corners_px"):
                metrics["frozen_case_reference"] = verifier["reference_corner_error"](
                    corners, np.asarray(reference["corners_px"], dtype=float)
                )
            if corners is not None and supplied and supplied.get("corners_working_px"):
                scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
                control_native = np.asarray(supplied["corners_working_px"], dtype=float) * scale
                metrics["approved_supplied_direction_control"] = verifier["reference_corner_error"](
                    corners, control_native
                )
            if candidate.get("homography_working") is not None:
                metrics["visible_landmarks"] = verifier["projected_landmark_diagnostics"](
                    np.asarray(candidate["homography_working"], dtype=float), reference.get("landmarks", []), context
                )
            record["candidates"][origin_key] = metrics
        diagnostics[result["case_id"]] = record
    return diagnostics


def add_reference_near_candidates(root: Path, case_results: list[dict], case_packets: dict, verifier: dict[str, Any]) -> None:
    for result in case_results:
        context = verifier["prepare_view"](root, result["case_id"])
        reference = load_reference(root, result["case_id"], verifier)
        supplied = load_supplied_control(root, result["case_id"], verifier)
        target = None
        target_kind = None
        if supplied and supplied.get("corners_working_px"):
            scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
            target = np.asarray(supplied["corners_working_px"], dtype=float) * scale
            target_kind = "approved_supplied_direction_control"
        elif reference.get("corners_px"):
            target = np.asarray(reference["corners_px"], dtype=float)
            target_kind = "frozen_case_reference"
        if target is None:
            continue
        candidates = []
        for candidate in case_packets[result["case_id"]]["review_candidates"].values():
            if not candidate.get("hard_valid") or not candidate.get("corners_px"):
                continue
            error = verifier["reference_corner_error"](np.asarray(candidate["corners_px"]), target)
            candidates.append((error["maximum"], candidate["origin_key"]))
        if candidates:
            distance, origin_key = min(candidates)
            result["reference_near"] = {"origin_key": origin_key, "distance_px": distance, "kind": target_kind}


def write_fit_attempts(path: Path, case_results: list[dict]) -> None:
    fields = [
        "case_id", "origin_key", "candidate_id", "status", "rejection_reason", "nfev", "solver_status",
        "successful", "jacobian_rank", "jacobian_condition", "minimum_corner_denominator", "objective_before",
        "objective_after", "objective_improvement", "fit_fragment_count", "fit_fragment_ids",
        "assignment_strengths", "near_selected_strengths", "attempted_corners_native", "child_origin_key",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for result in case_results:
            for row in result["fit_rows"]:
                output = {field: row.get(field) for field in fields}
                for field in ("fit_fragment_ids", "assignment_strengths", "near_selected_strengths",
                              "attempted_corners_native"):
                    output[field] = json_text(output[field])
                output["case_id"] = result["case_id"]
                writer.writerow(output)


def json_text(value: Any) -> str:
    import json

    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def probe_paint_evidence(candidate: dict, arrays: Any, probe: float) -> dict:
    q_values = []
    markings = candidate.get("markings", [])
    for marking_index, marking in enumerate(markings):
        support = np.asarray(
            arrays[f"{candidate['origin_key']}::marking_{marking_index}_c_support"],
            dtype=float,
        )
        ridge = np.asarray(
            arrays[f"{candidate['origin_key']}::marking_{marking_index}_ridge_contrast"],
            dtype=float,
        )
        known = np.isfinite(ridge)
        q_values.append(float(np.mean(support[known] * (ridge[known] >= probe))) if known.any() else None)

    def plain_mean(indices: range) -> float | None:
        values = [q_values[index] for index in indices if q_values[index] is not None]
        return float(np.mean(values)) if values else None

    def span_mean(indices: range) -> float | None:
        weighted = []
        total_span = 0.0
        for index in indices:
            value = q_values[index]
            span = float(markings[index].get("projected_visible_span_px") or 0.0)
            if value is None or span <= 0.0:
                continue
            weighted.append(value * span)
            total_span += span
        return sum(weighted) / total_span if total_span else None

    lengthwise = plain_mean(range(5))
    transverse = plain_mean(range(5, 11))
    lengthwise_span = span_mean(range(5))
    transverse_span = span_mean(range(5, 11))
    return {
        "q_paint10": min(lengthwise, transverse) if lengthwise is not None and transverse is not None else None,
        "q_paint10_span_weighted": (
            min(lengthwise_span, transverse_span)
            if lengthwise_span is not None and transverse_span is not None else None
        ),
        "directional": {
            "lengthwise_q_paint10": lengthwise,
            "transverse_q_paint10": transverse,
            "lengthwise_q_paint10_span_weighted": lengthwise_span,
            "transverse_q_paint10_span_weighted": transverse_span,
        },
    }


def sensitivity_candidate(candidate: dict, arrays: Any, probe: float) -> dict:
    evidence = probe_paint_evidence(candidate, arrays, probe)
    evidence.update({
        "q_geom": candidate.get("q_geom"),
        "q_geom_span_weighted": candidate.get("q_geom_span_weighted"),
        "exclusive_reverse": candidate.get("exclusive_reverse"),
    })
    return {
        "origin_key": candidate["origin_key"],
        "source_order": candidate.get("source_order"),
        "origin_index": candidate.get("origin_index"),
        "kind_order": candidate.get("kind_order"),
        "hard_valid": candidate.get("hard_valid", False),
        "gates": candidate.get("gates", {}),
        "historical": candidate.get("historical", {}),
        "evidence": evidence,
    }


def sensitivity_target(root: Path, case_id: str, context, verifier: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    supplied = load_supplied_control(root, case_id, verifier)
    if supplied and supplied.get("corners_working_px"):
        scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
        return np.asarray(supplied["corners_working_px"], dtype=float) * scale, "approved_supplied_direction_control"
    reference = load_reference(root, case_id, verifier)
    if reference.get("corners_px"):
        return np.asarray(reference["corners_px"], dtype=float), "frozen_case_reference"
    return None, None


def write_sensitivity(root: Path, run_dir: Path, case_results: list[dict], verifier: dict[str, Any]) -> dict:
    probes = (5.0, 10.0, 15.0, 20.0)
    sensitivity = {"schema": "w5-p10-sensitivity/1", "probes": list(probes), "cases": {}}
    for result in case_results:
        context = verifier["prepare_view"](root, result["case_id"])
        target, target_kind = sensitivity_target(root, result["case_id"], context, verifier)
        candidates = list(result["review_candidates"].values())
        array_path = run_dir / result["array_file"]
        with np.load(array_path, allow_pickle=False) as arrays:
            probe_records = {}
            for probe in probes:
                ranked_candidates = [
                    sensitivity_candidate(candidate, arrays, probe)
                    for candidate in candidates
                ]
                ranking = verifier["rank_candidates"](ranked_candidates)
                origin_key = ranking["selected_origin_key"]
                ungated = False
                if origin_key is None and ranking["ungated_provisional_rank"]:
                    origin_key = ranking["ungated_provisional_rank"][0]
                    ungated = True
                control_error = None
                if origin_key is not None and target is not None:
                    selected = next(candidate for candidate in candidates if candidate["origin_key"] == origin_key)
                    control_error = verifier["reference_corner_error"](
                        np.asarray(selected["corners_px"], dtype=float), target,
                    )
                probe_records[str(int(probe))] = {
                    "rank1_origin_key": origin_key,
                    "rank1_was_ungated": ungated,
                    "status": ranking["status"],
                    "r1_paint10_rank": ranking["r1_paint10_rank"],
                    "r2_spanw_paint10_rank": ranking["r2_spanw_paint10_rank"],
                    "provisional_rank": ranking["provisional_rank"],
                    "control_error": control_error,
                }
        sensitivity["cases"][result["case_id"]] = {
            "target_kind": target_kind,
            "probes": probe_records,
        }
    (run_dir / "p10_sensitivity.json").write_text(
        __import__("json").dumps(verifier["jsonable"](sensitivity), indent=2) + "\n"
    )
    return sensitivity


def helper_hashes(root: Path, runtime_paths: dict[str, str]) -> dict:
    imported = {}
    for module_name, relative in sorted(runtime_paths.items()):
        path = root / relative
        imported[module_name] = {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    readme_relative = "frozen_helpers_20260914/README.md"
    readme_path = root / readme_relative
    if not readme_path.exists():
        return {
            "imported": imported,
            "readme": {"path": readme_relative, "available": False, "hash_checks": []},
        }
    checks = []
    pattern = re.compile(r"^\| `([^`]+)` \| [0-9,]+ \| `([0-9a-f]+)` \|$")
    for line in readme_path.read_text().splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        relative = f"frozen_helpers_20260914/{match.group(1)}"
        path = root / relative
        actual_md5 = hashlib.md5(path.read_bytes()).hexdigest()
        checks.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "readme_md5": match.group(2),
            "readme_md5_matches": actual_md5 == match.group(2),
        })
    return {
        "imported": imported,
        "readme": {"path": readme_relative, "available": True, "hash_checks": checks},
    }


def write_packet(
    root: Path,
    run_dir: Path,
    case_results: list[dict],
    verifier: dict[str, Any],
    runtime_paths: dict,
    stopped_views: list[dict] | None = None,
) -> None:
    case_results = sorted(case_results, key=lambda result: verifier["CASE_IDS"].index(result["case_id"]))
    stopped_views = stopped_views or []
    packets = {result["case_id"]: result for result in case_results}
    sensitivity = write_sensitivity(root, run_dir, case_results, verifier)
    add_reference_near_candidates(root, case_results, packets, verifier)
    review = {}
    for result in case_results:
        candidates = dict(result["review_candidates"])
        keys = set()
        for arm in ("B", "C"):
            for rank_name in (
                "provisional_rank", "ungated_provisional_rank", "r1_paint10_rank", "r2_spanw_paint10_rank",
            ):
                keys.update(result[arm][rank_name][:3])
            if result[arm].get("selected_origin_key"):
                keys.add(result[arm]["selected_origin_key"])
        keys.update(value for value in (result["A"].get("line"), result["A"].get("paint")) if value)
        if result.get("reference_near"):
            keys.add(result["reference_near"]["origin_key"])
        for control in result["diagnostic_controls"]:
            keys.add(control["origin_key"])
        result["review_candidates"] = {key: candidates[key] for key in sorted(keys) if key in candidates}
        review[result["case_id"]] = {
            "selected": {arm: result[arm].get("selected_origin_key") for arm in ("B", "C")},
            "A": result["A"],
            "candidates": result["review_candidates"],
            "diagnostic_controls": result["diagnostic_controls"],
            "reference_near": result.get("reference_near"),
        }
    rankings = {
        "schema": "w5-rankings/1",
        "cases": {
            result["case_id"]: {
                "A": result["A"],
                "B": result["B"],
                "C": result["C"],
                "identity_resolution": result["identity_resolution"],
                "determinism": result["determinism"],
            }
            for result in case_results
        },
    }
    manifest = {
        "schema": "w5-manifest/1",
        "run_id": run_dir.name,
        "cases": [result["case_id"] for result in case_results],
        "stopped_views": stopped_views,
        "global_parameters": {
            "working_size": list(verifier["WORKING_SIZE"]),
            "camera_limit": verifier["CAMERA_LIMIT"],
            "camera_error_limit_historical": verifier["CAMERA_LIMIT"],
            "physical_centres": "paint_geometry.CENTRE_SEGMENTS_M",
            "photometric_offsets_working_px": verifier["PHOTO_CENTRE_OFFSETS_PX"].tolist(),
            "photometric_side_distance_working_px": verifier["PHOTO_SIDE_DISTANCE_PX"],
            "photometric_probe_threshold_historical": 10.0,
            "refit_max_evaluations": import_fitting().MAX_EVALUATIONS,
            "workers": int(os.environ.get("W5_WORKERS", "1")),
        },
        "source_stage": {
            "G0": "frozen_views/baseline_generation when present, otherwise automatic_axes_20260914/all_camera with L2 selection replay",
            "G1": "line_identity/runs/line_identity_20260915_222437/matcher/paint_observations/results",
            "automatic_path_reference_fields": False,
        },
        "imported_helper_paths": runtime_paths,
        "imported_helper_hashes": helper_hashes(root, runtime_paths),
        "view_provenance": {result["case_id"]: result["provenance"] for result in case_results},
        "candidate_identity": {
            "policy": "source-qualified canonical origin_key; exact cross-source geometry duplicates retain all source occurrences when W5-relevant gates agree",
            "views": {result["case_id"]: result["identity_resolution"] for result in case_results},
        },
        "steering_rule_revision": {
            "status": "R1 camera eligibility + R2 span-weighted directional means",
            "changed_global_rule": "R1 and R2 from steering_record.md",
            "source": "scratch/court_det_fix/w5_holistic/steering_record.md",
            "expensive_geometry_rerun": True,
        },
    }
    per_view_fields = [
        "case_id", "label", "view_status", "g0_source", "working_width", "working_height", "image_kind",
        "same_image_mask_available", "G0_count", "G1_count", "canonical_parent_count",
        "raw_id_collision_count", "duplicate_group_count", "A_line", "A_paint", "A_eligible_count",
        "B_status", "B_selected", "B_r1_selected", "B_r2_selected", "B_pilot_selected",
        "C_status", "C_selected", "C_r1_selected", "C_r2_selected", "C_pilot_selected",
        "valid_children", "fit_attempts",
        "determinism_match", "controls",
    ]
    with (run_dir / "per_view.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=per_view_fields, lineterminator="\n")
        writer.writeheader()
        for result in case_results:
            provenance = result["provenance"]
            writer.writerow({
                "case_id": result["case_id"], "label": result["label"], "g0_source": provenance["g0_source"],
                "view_status": "completed",
                "working_width": provenance["working_dimensions"][0], "working_height": provenance["working_dimensions"][1],
                "image_kind": provenance["image_kind"], "same_image_mask_available": provenance["same_image_mask_available"],
                "G0_count": result["population_counts"]["G0"], "G1_count": result["population_counts"]["G1"],
                "canonical_parent_count": result["identity_resolution"]["canonical_parent_count"],
                "raw_id_collision_count": result["identity_resolution"]["raw_id_collision_count"],
                "duplicate_group_count": result["identity_resolution"]["duplicate_group_count"],
                "A_line": result["A"]["line"], "A_paint": result["A"]["paint"],
                "A_eligible_count": result["A"]["eligible_count"], "B_status": result["B"]["status"],
                "B_selected": result["B"]["selected_origin_key"],
                "B_r1_selected": result["B"]["r1_selected_origin_key"],
                "B_r2_selected": result["B"]["r2_selected_origin_key"],
                "B_pilot_selected": (result["B"]["ungated_provisional_rank"] or [None])[0],
                "C_status": result["C"]["status"], "C_selected": result["C"]["selected_origin_key"],
                "C_r1_selected": result["C"]["r1_selected_origin_key"],
                "C_r2_selected": result["C"]["r2_selected_origin_key"],
                "C_pilot_selected": (result["C"]["ungated_provisional_rank"] or [None])[0],
                "valid_children": result["valid_child_count"],
                "fit_attempts": result["fit_attempt_count"], "determinism_match": result["determinism"]["match"],
                "controls": ";".join(control["origin_key"] for control in result["diagnostic_controls"]),
            })
        for stopped in stopped_views:
            writer.writerow({
                "case_id": stopped["case_id"],
                "label": verifier["CASE_LABELS"].get(stopped["case_id"], stopped["case_id"]),
                "view_status": "stopped",
                "B_status": stopped["reason"],
            })
    (run_dir / "manifest.json").write_text(__import__("json").dumps(verifier["jsonable"](manifest), indent=2) + "\n")
    (run_dir / "rankings.json").write_text(__import__("json").dumps(verifier["jsonable"](rankings), indent=2) + "\n")
    refs = reference_diagnostics(root, case_results, verifier)
    verifier["write_json_gz"](run_dir / "reference_diagnostics.json.gz", refs)
    (run_dir / "review_candidates.json").write_text(__import__("json").dumps(verifier["jsonable"](review), indent=2) + "\n")
    (run_dir / "visual_rulings.json").write_text(
        '{"schema":"w5-visual-rulings/1","status":"pending_review","rulings":[]}\n'
    )
    write_fit_attempts(run_dir / "fit_attempts.csv.gz", case_results)
    write_result(run_dir / "result.md", case_results, sensitivity, refs, stopped_views)


def write_result(
    path: Path,
    case_results: list[dict],
    sensitivity: dict,
    references: dict,
    stopped_views: list[dict] | None = None,
) -> None:
    stopped_views = stopped_views or []
    controls = [
        (result["label"], control)
        for result in case_results
        for control in result["diagnostic_controls"]
    ]
    positive_controls = [
        control for _, control in controls
        if str(control.get("expected_ruling", "")).startswith("positive")
    ]
    negative_controls = [
        control for _, control in controls
        if str(control.get("expected_ruling", "")).startswith("negative")
    ]
    positive_geometry = [control.get("q_geom") for control in positive_controls]
    negative_geometry = [control.get("q_geom") for control in negative_controls]
    positive_paint = [control.get("q_paint10") for control in positive_controls]
    negative_paint = [control.get("q_paint10") for control in negative_controls]
    controls_directionally_consistent = all(
        values and all(value is not None for value in values)
        for values in (positive_geometry, negative_geometry, positive_paint, negative_paint)
    ) and (
        min(positive_geometry) > max(negative_geometry)
        and min(positive_paint) > max(negative_paint)
    )

    def metric_text(value: float | None) -> str:
        return "unknown" if value is None else f"{value:.6g}"

    lines = [
        "# W5 holistic court-detector pilot",
        "",
        "This is a development/fit packet on the frozen corpus. The B and C orders are provisional readouts, not an acceptance rule.",
        "",
        "## Completed views",
        "",
        "| view | A paint-first | C pilot | C R1 | C R2 | C status | children | determinism |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for result in case_results:
        lines.append(
            f"| {result['label']} | {result['A']['paint'] or 'none'} | "
            f"{(result['C']['ungated_provisional_rank'] or [None])[0] or 'none'} | "
            f"{result['C']['r1_selected_origin_key'] or 'none'} | "
            f"{result['C']['selected_origin_key'] or 'none'} | "
            f"{result['C']['status']} | "
            f"{result['valid_child_count']} | {'pass' if result['determinism']['match'] else 'FAIL'} |"
        )
    if stopped_views:
        lines.extend([
            "",
            "## Stopped views",
            "",
            "The following view stopped before scoring. The reason is recorded below.",
            "",
            "| view | reason |",
            "| --- | --- |",
        ])
        for stopped in stopped_views:
            lines.append(f"| {stopped['case_id']} | {stopped['reason']} |")
    lines.extend([
        "",
        "## Collision resolution",
        "",
        "Parent candidates use source-qualified `origin_key` values for every join, ranking, diagnostic and gallery lookup. Raw candidate IDs remain source-local provenance. Exact cross-source geometry duplicates retain all source occurrences under one canonical parent when their W5-relevant gates agree; legacy-only source metadata stays attached to each occurrence. A same-source geometry duplicate, metadata mismatch or W5-gate mismatch stops that view as ambiguous.",
        "",
        "| view | source occurrences | canonical parents | raw-ID collisions | duplicate geometry groups |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for result in case_results:
        resolution = result["identity_resolution"]
        lines.append(
            f"| {result['label']} | {resolution['source_occurrence_count']} | "
            f"{resolution['canonical_parent_count']} | {resolution['raw_id_collision_count']} | "
            f"{resolution['duplicate_group_count']} |"
        )
    lines.extend([
        "",
        "## R1 and R2 rank-1 selections",
        "",
        "The C pool contains parents and valid children. R1 is the camera-eligible plain-mean paint order. R2 is the camera-eligible span-weighted order, with the span-weighted geometry fallback when needed.",
        "",
        "| view | R1 rank-1 | R2 rank-1 | status |",
        "| --- | --- | --- | --- |",
    ])
    for result in case_results:
        lines.append(
            f"| {result['label']} | {result['C']['r1_selected_origin_key'] or 'none'} | "
            f"{result['C']['selected_origin_key'] or 'none'} | {result['C']['status']} |"
        )
    lines.extend([
        "",
        "## Known diagnostic controls",
        "",
        (
            "The positive and negative controls are directionally separated in the saved "
            "two-direction Q readouts, which is consistent with their prior rulings. "
            "This is a diagnostic result only: no threshold or automatic pass/fail was "
            "applied."
            if controls_directionally_consistent else
            "The control readouts do not show a consistent positive-versus-negative "
            "separation in both two-direction Q measures. This is a diagnostic result "
            "only: no threshold or automatic pass/fail was applied."
        ),
        "",
        "| view | origin | raw candidate | expected role | automatic pool | hard-valid | camera eligible | pilot rank | R1 rank | R2 rank | Q_geom | Q_paint10 | status |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for label, control in controls:
        lines.append(
            f"| {label} | {control['origin_key']} | {control['candidate_id']} | {control.get('expected_ruling', 'unspecified')} | "
            f"{'yes' if control.get('in_automatic_pool') else 'no'} | "
            f"{'yes' if control.get('hard_valid') else 'no'} | "
            f"{'yes' if control.get('camera_eligible') else 'no'} | "
            f"{control.get('ranks', {}).get('pilot') or '—'} | {control.get('ranks', {}).get('r1') or '—'} | "
            f"{control.get('ranks', {}).get('r2') or '—'} | {metric_text(control.get('q_geom'))} | "
            f"{metric_text(control.get('q_paint10'))} | diagnostic-only; no gate |"
        )
    lines.extend([
        "",
        "## Contrast-probe sensitivity",
        "",
        "Each row reranks the C pool under the same R1 + R2 rules from the saved raw ridge arrays. Probe 10 remains the named pilot setting.",
        "",
        "| view | probe | rank-1 origin | status | control target | control error |",
        "| --- | ---: | --- | --- | --- | ---: |",
    ])
    for result in case_results:
        case_sensitivity = sensitivity["cases"][result["case_id"]]
        target_kind = case_sensitivity.get("target_kind") or "none"
        for probe in sensitivity["probes"]:
            record = case_sensitivity["probes"][str(int(probe))]
            error = record.get("control_error")
            maximum = error.get("maximum") if error else None
            lines.append(
                f"| {result['label']} | {int(probe)} | {record.get('rank1_origin_key') or 'none'} | "
                f"{record['status']}{' (ungated)' if record.get('rank1_was_ungated') else ''} | "
                f"{target_kind} | {metric_text(maximum)} |"
            )
    lines.extend([
        "",
        "## Reference diagnostics after ranking lock",
        "",
        "Reference metrics were joined after the automatic rankings and sensitivity orders were written.",
        "",
        "| view | origin | supplied-control error | frozen-reference error | visible-landmark max error |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for result in case_results:
        reference = references.get(result["case_id"], {})
        for origin_key, metrics in reference.get("candidates", {}).items():
            supplied = metrics.get("approved_supplied_direction_control", {}).get("maximum")
            frozen = metrics.get("frozen_case_reference", {}).get("maximum")
            landmarks = metrics.get("visible_landmarks", [])
            landmark_max = max((item["error_px"] for item in landmarks), default=None)
            lines.append(
                f"| {result['label']} | {origin_key} | {metric_text(supplied)} | {metric_text(frozen)} | "
                f"{metric_text(landmark_max)} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.",
        "",
    ])
    path.write_text("\n".join(lines))


def run_pilot(root: Path, run_dir: Path, cases: list[str], workers: int) -> list[dict]:
    runtime = load_runtime(root)
    verifier = runtime["verifier"]
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.exists():
        raise FileNotFoundError(f"preflight result is required before Stage 2: {preflight_path}")
    preflight = __import__("json").loads(preflight_path.read_text())
    if preflight.get("status") != "passed":
        raise RuntimeError("Stage 2 is blocked by a failed W5 preflight")
    workers = max(1, min(int(workers), 10, len(cases)))
    os.environ["W5_WORKERS"] = str(workers)
    print(f"W5 pilot cases={cases} workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_case, root, case_id, run_dir) for case_id in cases]
        results = []
        stopped_views = []
        for case_id, future in zip(cases, futures, strict=True):
            try:
                result = future.result()
            except ViewAmbiguity as error:
                reason = str(error)
                stopped_views.append({"case_id": case_id, "reason": reason})
                print(case_id, "stopped", reason, flush=True)
                continue
            results.append(result)
            print(result["case_id"], "complete", result["B"]["selected_origin_key"], result["C"]["selected_origin_key"], flush=True)
    write_packet(root, run_dir, results, verifier, runtime["paths"], stopped_views)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", required=True)
    parser.add_argument("--stage", choices=("preflight", "pilot"), required=True)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = root / "w5_holistic/runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        run_preflight(root, run_dir)
        print("W5 preflight passed", flush=True)
        return
    verifier = load_verifier(root)
    cases = args.cases or [
        "gxBQ_window_00_frame_0",
        "am2_window_00_frame_150",
        "am2_window_01_frame_28019",
        "am3_window_00_frame_0",
        "shuttleset_03_scene_0019",
    ]
    unknown = sorted(set(cases) - set(verifier["CASE_IDS"]))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    run_pilot(root, run_dir, cases, args.workers)
    print("W5 pilot packet written", run_dir, flush=True)


if __name__ == "__main__":
    main()
