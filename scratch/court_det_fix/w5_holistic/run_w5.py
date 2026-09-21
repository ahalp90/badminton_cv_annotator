"""Run the bounded W5 holistic court experiment."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import monotonic
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
    names = (
        "run_automatic",
        "run_given",
        "run_population",
        "run_diagnosis",
        "zone_net",
        "camera_diagnostic",
        "line_template_source",
    )
    paths = {}
    for name in names:
        module = importlib.import_module(name)
        paths[name] = str(Path(module.__file__).resolve().relative_to(root.resolve()))
    return paths


def import_runtime(root: Path) -> dict[str, Any]:
    add_helper_paths(root)
    from line_template_source import generate

    # run_diagnosis prepends its sibling snapshot while importing. Pin this
    # byte-identical seed helper first so module provenance stays deterministic.
    importlib.import_module("run_population")
    from run_automatic import evaluate_pool, select_pool
    from run_diagnosis import gate_evidence

    return {
        "evaluate_pool": evaluate_pool,
        "select_pool": select_pool,
        "gate_evidence": gate_evidence,
        "line_template": generate,
        "zone": importlib.import_module("zone_net"),
        "paths": module_paths(root),
    }


def load_verifier(root: Path):
    sys.path.insert(0, str(root / "w5_holistic"))
    from verifier import (
        ALL_CASE_IDS,
        CAMERA_LIMIT,
        CASE_IDS,
        CASE_LABELS,
        CASE_ORDER,
        CASE_PACKS,
        PACK_OF,
        PHOTO_CENTRE_OFFSETS_PX,
        PHOTO_SIDE_DISTANCE_PX,
        REGRESSION_CASE_IDS,
        UNUSED_CASE_IDS,
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
        load_case_provenance,
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
        "ALL_CASE_IDS": ALL_CASE_IDS,
        "CASE_LABELS": CASE_LABELS,
        "CASE_ORDER": CASE_ORDER,
        "CASE_PACKS": CASE_PACKS,
        "PACK_OF": PACK_OF,
        "REGRESSION_CASE_IDS": REGRESSION_CASE_IDS,
        "UNUSED_CASE_IDS": UNUSED_CASE_IDS,
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
        "load_case_provenance": load_case_provenance,
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


VISIBILITY_COLUMNS = {
    "lengthwise": "first six projected court-template pieces (x-family): sidelines plus split centre",
    "cross_court": "second six projected court-template pieces (y-family): baselines and service lines",
}
VISIBILITY_FLOOR_ARMS = ((3, 3), (4, 3), (5, 3))



def load_runtime(root: Path) -> dict[str, Any]:
    return {"verifier": load_verifier(root), **import_runtime(root)}


def validate_visibility_floor(value: int, name: str) -> int:
    """Validate one inclusive projected-piece floor at a run boundary."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(value)


def validate_visibility_floors(
    min_visible_lengthwise: int,
    min_visible_cross_court: int,
) -> tuple[int, int]:
    """Validate both visibility columns without applying a scalar alias."""
    return (
        validate_visibility_floor(min_visible_lengthwise, "min_visible_lengthwise"),
        validate_visibility_floor(min_visible_cross_court, "min_visible_cross_court"),
    )


def nonnegative_int(value: str) -> int:
    """Parse a non-negative integer argparse value."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def resolve_case_ids(verifier: dict[str, Any], cases: list[str] | tuple[str, ...] | None) -> list[str]:
    """Return a validated, ordered case list for one scoped W5 run."""
    requested = list(cases) if cases is not None else list(verifier["REGRESSION_CASE_IDS"])
    if not requested:
        raise ValueError("at least one case ID is required")
    unknown = sorted(set(requested) - set(verifier["ALL_CASE_IDS"]))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("case IDs must be unique")
    return requested


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


def validate_population_entries(entries: Any, case_id: str, source: str) -> list[dict]:
    """Validate the candidate identities used by one frozen population."""
    if not isinstance(entries, list):
        raise TypeError(f"{case_id}: {source} entries must be a list")
    if len(entries) != 256:
        raise ValueError(f"{case_id}: {source} has {len(entries)} entries")
    if not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{case_id}: {source} entries must be objects")
    candidate_ids = [entry.get("candidate_id") for entry in entries]
    if not all(isinstance(candidate_id, str) and candidate_id for candidate_id in candidate_ids):
        raise ValueError(f"{case_id}: {source} has an invalid candidate ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{case_id}: {source} candidate IDs are not unique")
    return entries


def validate_generation_record(
    record: Any,
    case_id: str,
    source: str,
    *,
    expected_stage: str | None = None,
    validate_entries: bool = True,
) -> dict:
    """Validate the saved identity and membership used by a G0 or G1 population."""
    if not isinstance(record, dict):
        raise TypeError(f"{case_id}: {source} record must be an object")
    if record.get("schema") != "automatic-directions-axis-matching/1":
        raise ValueError(f"{case_id}: {source} has an unsupported schema")
    if record.get("case_id") != case_id:
        raise ValueError(f"{case_id}: {source} case identity does not match")
    if expected_stage is not None and record.get("stage") != expected_stage:
        raise ValueError(f"{case_id}: {source} stage is not {expected_stage}")
    if not isinstance(record.get("pairs"), list):
        raise TypeError(f"{case_id}: {source} generation pairs must be a list")
    if validate_entries:
        entries = validate_population_entries(record.get("entries"), case_id, source)
        candidate_ids = {entry["candidate_id"] for entry in entries}
        for winner_name in ("line_winner_id", "paint_winner_id"):
            winner_id = record.get(winner_name)
            if winner_id is not None and winner_id not in candidate_ids:
                raise ValueError(f"{case_id}: {source} {winner_name} is outside its entries")
    return record


def load_g0(root: Path, context, runtime: dict[str, Any]) -> tuple[list[dict], str]:
    verifier = runtime["verifier"]
    direct = root / "frozen_views/baseline_generation" / f"{context.case_id}.json.gz"
    if direct.exists():
        record = validate_generation_record(
            verifier["read_json_gz"](direct), context.case_id, "direct G0"
        )
        entries = record["entries"]
        return entries, "direct:" + verifier["relative_path"](direct, root)
    record_path = root / "automatic_axes_20260914/all_camera" / f"{context.case_id}.json.gz"
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    record = validate_generation_record(
        verifier["read_json_gz"](record_path),
        context.case_id,
        "automatic G0 replay input",
        validate_entries=False,
    )
    shortlist = reconstruct_generation_entries(
        record,
        context.native_size,
        runtime["select_pool"],
        import_detector(),
    )
    run_automatic = import_run_automatic()
    original_frame_path = run_automatic.frame_path
    run_automatic.frame_path = lambda source, _root: verifier["frame_path"](
        root, source, context.provenance
    )
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
    validate_population_entries(entries, context.case_id, "replayed G0")
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
    record = validate_generation_record(
        verifier["read_json_gz"](path), context.case_id, "saved G1", expected_stage="results"
    )
    entries = record["entries"]
    return entries, verifier["relative_path"](path, root)


def load_populations(
    root: Path,
    context,
    runtime: dict[str, Any],
    *,
    include_line_template: bool = True,
    min_visible_lengthwise: int = 0,
    min_visible_cross_court: int = 0,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    g0, g0_source = load_g0(root, context, runtime)
    g1, g1_source = load_g1(root, context, runtime["verifier"])
    if include_line_template:
        generated = runtime["line_template"](
            context,
            runtime,
            import_detector(),
            min_visible_lengthwise=min_visible_lengthwise,
            min_visible_cross_court=min_visible_cross_court,
        )
        line_template = list(generated.entries)
        line_template_source = generated.metadata
    else:
        line_template = []
        line_template_source = {
            "name": "line_template",
            "status": "not_loaded",
            "reason": "legacy-only population requested",
        }
    return g0, g1, line_template, {
        "G0": g0_source,
        "G1": g1_source,
        "line_template": line_template_source,
    }




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


def compatible_duplicate(entry: dict, record: dict, source: str) -> bool:
    reference = record["entry"]
    if not np.array_equal(entry["corners_px"], reference["corners_px"]):
        return False
    legacy_metadata_differs = (
        source in {"G0", "G1"} and record["source"] in {"G0", "G1"}
        and any(entry.get(field) != reference.get(field) for field in ("pair_id", "rotated_180"))
    )
    if legacy_metadata_differs:
        return False
    return all(
        values_equal_with_nan(left, right)
        for left, right in zip(w5_gate_fields(entry), w5_gate_fields(reference), strict=True)
    )


def source_occurrence(entry: dict, source: str, source_order: int, origin_index: int) -> dict:
    candidate_id = str(entry["candidate_id"])
    occurrence = {
        "origin_key": f"{source}:{candidate_id}",
        "source": source,
        "source_order": source_order,
        "origin_index": origin_index,
        "candidate_id": candidate_id,
        "pair_id": entry.get("pair_id"),
        "axis_ids": entry.get("axis_ids"),
        "rotated_180": entry.get("rotated_180"),
        "w5_gates": dict(zip(("geometry_valid", "camera_error", "player_fractions"), w5_gate_fields(entry), strict=True)),
    }
    for field in ("proposal_id", "rectangle_id", "rectangle_order", "template_index"):
        if field in entry:
            occurrence[field] = entry[field]
    if source == "line_template" and "line_template" in entry:
        occurrence["line_template"] = entry["line_template"]
    return occurrence


def canonicalise_populations(
    g0: list[dict], g1: list[dict], line_template: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Build collision-safe parent identities while retaining source occurrences."""
    source_entries = (("G0", g0), ("G1", g1), ("line_template", line_template or []))
    legacy_sources = {"G0", "G1"}
    source_ids = {
        source: [str(entry["candidate_id"]) for entry in entries]
        for source, entries in source_entries
    }
    for source, ids in source_ids.items():
        if len(ids) != len(set(ids)):
            raise ViewAmbiguity(f"{source} candidate IDs are not unique within their source")

    by_geometry: dict[tuple[tuple[int, ...], bytes], list[dict]] = {}
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
            geometry_records = by_geometry.setdefault(geometry_key, [])
            record = next(
                (existing for existing in geometry_records if compatible_duplicate(entry, existing, source)),
                None,
            )
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
                    "_legacy_occurrences": [legacy_occurrence] if source in legacy_sources else [],
                }
                geometry_records.append(record)
                records.append(record)
                continue
            if source not in record["source_memberships"]:
                record["source_memberships"].append(source)
            record["source_occurrences"].append(occurrence)
            record["occurrence_count"] += 1
            if source in legacy_sources:
                record["_legacy_occurrences"].append(legacy_occurrence)

    for record in records:
        for occurrence in record["_legacy_occurrences"]:
            occurrence["parent_origin_key"] = record["origin_key"]

    raw_id_sources: dict[str, set[str]] = {}
    for source, ids in source_ids.items():
        for candidate_id in ids:
            raw_id_sources.setdefault(candidate_id, set()).add(source)
    raw_id_collisions = sorted(
        candidate_id for candidate_id, sources in raw_id_sources.items() if len(sources) >= 2
    )
    raw_id_collision_sources = {
        candidate_id: sorted(raw_id_sources[candidate_id]) for candidate_id in raw_id_collisions
    }
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
        "policy": "merge compatible geometry duplicates; retain conflicting gates or metadata as separate source-qualified candidates",
        "conflicting_geometry_groups": [
            [record["origin_key"] for record in geometry_records]
            for geometry_records in by_geometry.values() if len(geometry_records) > 1
        ],
        "source_occurrence_counts": {
            "G0": len(g0), "G1": len(g1), "line_template": len(line_template or []),
        },
        "source_occurrence_count": len(g0) + len(g1) + len(line_template or []),
        "legacy_source_occurrence_count": len(g0) + len(g1),
        "canonical_parent_count": len(records),
        "raw_id_collisions": raw_id_collisions,
        "raw_id_collision_count": len(raw_id_collisions),
        "raw_id_collision_sources": raw_id_collision_sources,
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
            "_legacy_occurrences": ([{
                **occurrence,
                "gates": entry.get("gates", {}),
                "legacy": compact_legacy(entry),
            }] if source in {"G0", "G1"} else []),
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
    line_template_occurrences = [
        occurrence for occurrence in identity["source_occurrences"] if occurrence["source"] == "line_template"
    ]
    if line_template_occurrences:
        candidate["line_template_provenance"] = line_template_occurrences
    for field in ("proposal_id", "rectangle_id", "rectangle_order", "template_index", "line_template"):
        if field in entry:
            candidate[field] = entry[field]
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
        "_legacy_occurrences": parent.get("_legacy_occurrences", []),
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
    for field in ("proposal_id", "rectangle_id", "rectangle_order", "template_index", "line_template"):
        if field in parent:
            child[field] = parent[field]
    if "line_template_provenance" in parent:
        child["line_template_provenance"] = parent["line_template_provenance"]
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
    record = validate_generation_record(
        verifier["read_json_gz"](path),
        case_id,
        "diagnostic control source",
        validate_entries=False,
    )
    entries = record.get("entries")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(f"{case_id}: diagnostic control entries must be objects")
    candidate_ids = [entry.get("candidate_id") for entry in entries]
    if not all(isinstance(saved_id, str) and saved_id for saved_id in candidate_ids):
        raise ValueError(f"{case_id}: diagnostic control source has an invalid candidate ID")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{case_id}: diagnostic control candidate IDs are not unique")
    for entry in entries:
        if entry["candidate_id"] == candidate_id:
            return entry
    raise KeyError((case_id, candidate_id))


def has_source_occurrence(candidate: dict, source: str, candidate_id: str) -> bool:
    return any(
        occurrence["source"] == source and occurrence["candidate_id"] == candidate_id
        for occurrence in candidate.get("source_occurrences", [])
    )


def process_case(
    root: Path,
    case_id: str,
    run_dir: Path,
    *,
    min_visible_lengthwise: int = 0,
    min_visible_cross_court: int = 0,
) -> dict:
    cv2.setNumThreads(1)
    started = monotonic()

    def progress(message: str) -> None:
        print(f"[{case_id} +{monotonic() - started:.0f}s] {message}", flush=True)

    progress("loading view and candidate populations")
    min_visible_lengthwise, min_visible_cross_court = validate_visibility_floors(
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    runtime = load_runtime(root)
    verifier = runtime["verifier"]
    context = verifier["prepare_view"](root, case_id)
    g0, g1, line_template, sources = load_populations(
        root,
        context,
        runtime,
        min_visible_lengthwise=min_visible_lengthwise,
        min_visible_cross_court=min_visible_cross_court,
    )
    automatic_entries = g0 + g1 + line_template
    contamination_fields = []
    for index, entry in enumerate(automatic_entries):
        contamination_fields.extend(find_forbidden_keys(entry, f"{case_id}.automatic[{index}]"))
    if contamination_fields:
        raise ViewAmbiguity(
            f"{case_id}: automatic candidate path contains reference fields: {contamination_fields}"
        )
    parent_identities, identity_resolution = canonicalise_populations(g0, g1, line_template)
    progress(
        f"measuring {len(parent_identities)} parents "
        f"(G0={len(g0)}, G1={len(g1)}, line_template={len(line_template)}; "
        f"conflicting geometry groups={len(identity_resolution['conflicting_geometry_groups'])})"
    )
    cache: dict[bytes, tuple[dict, dict[str, np.ndarray]]] = {}
    parents = []
    all_arrays: dict[str, np.ndarray] = {}
    for parent_index, identity in enumerate(parent_identities, start=1):
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
        if parent_index % 250 == 0:
            progress(f"parents {parent_index}/{len(parent_identities)}")
    fit_rows = []
    children = []
    progress(f"refitting {len(parents)} parents")
    for parent_index, parent in enumerate(parents, start=1):
        row, child, arrays = attempt_refit(context, parent, runtime, cache)
        fit_rows.append(row)
        if child is not None:
            children.append(child)
            for key, value in arrays.items():
                all_arrays[f"{child['origin_key']}::{key}"] = value
        if parent_index % 250 == 0:
            progress(f"refits {parent_index}/{len(parents)}; valid children={len(children)}")
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
        control, _ = make_parent_record(context, entry, "diagnostic", 3, 0, runtime, cache)
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
    provenance["line_template_source"] = sources["line_template"]
    public_parents = [public_candidate(parent) for parent in parents]
    public_children = [public_candidate(child) for child in children]
    progress("ranking sensitivity thresholds")
    review_candidates = {
        candidate["origin_key"]: verifier["candidate_review"](candidate)
        for candidate in c_candidates
    }
    sensitivity = rank_sensitivity(list(review_candidates.values()), all_arrays, verifier)
    progress("saving evidence and arrays")
    full_record = {
        "schema": "w5-case-evidence/2",
        "case_id": case_id,
        "min_visible_lengthwise": min_visible_lengthwise,
        "min_visible_cross_court": min_visible_cross_court,
        "visibility_columns": dict(VISIBILITY_COLUMNS),
        "provenance": provenance,
        "population_sources": sources,
        "population_counts": {
            "G0": len(g0), "G1": len(g1), "line_template": len(line_template), "union": len(parents),
        },
        "automatic_contamination_check": {"match": not contamination_fields, "fields": contamination_fields},
        "identity_resolution": identity_resolution,
        "parents": public_parents,
        "valid_children": public_children,
        "diagnostic_controls": controls,
        "fit_attempts": fit_rows,
        "rankings": {"B": b_rankings, "C": c_rankings},
        "determinism": determinism,
        "sensitivity": sensitivity,
    }
    case_dir = run_dir / "case_records"
    case_dir.mkdir(parents=True, exist_ok=True)
    verifier["write_json_gz"](case_dir / f"{case_id}.json.gz", full_record)
    array_path = run_dir / "arrays" / f"{case_id}.npz"
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **all_arrays)
    progress("saved")
    return {
        "schema": "w5-case-result/2",
        "case_id": case_id,
        "min_visible_lengthwise": min_visible_lengthwise,
        "min_visible_cross_court": min_visible_cross_court,
        "visibility_columns": dict(VISIBILITY_COLUMNS),
        "label": verifier["CASE_LABELS"][case_id],
        "provenance": provenance,
        "population_sources": sources,
        "population_counts": {
            "G0": len(g0), "G1": len(g1), "line_template": len(line_template), "union": len(parents),
        },
        "automatic_contamination_check": {"match": not contamination_fields, "fields": contamination_fields},
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
        "review_candidates": review_candidates,
        "sensitivity": sensitivity,
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
        if result.get("reference_near"):
            selected_origins.add(result["reference_near"]["origin_key"])
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


SENSITIVITY_PROBES = (5.0, 10.0, 15.0, 20.0)


def probe_paint_evidence(candidate: dict, arrays: Any) -> list[dict]:
    """Measure all thresholds together, reading each marking's arrays once."""
    q_values = []
    markings = candidate.get("markings", [])
    thresholds = np.asarray(SENSITIVITY_PROBES)[:, None]
    for marking_index in range(len(markings)):
        support = np.asarray(
            arrays[f"{candidate['origin_key']}::marking_{marking_index}_c_support"],
            dtype=float,
        )
        ridge = np.asarray(
            arrays[f"{candidate['origin_key']}::marking_{marking_index}_ridge_contrast"],
            dtype=float,
        )
        known = np.isfinite(ridge)
        q_values.append(
            np.mean(support[known][None, :] * (ridge[known][None, :] >= thresholds), axis=1).tolist()
            if known.any() else [None] * len(SENSITIVITY_PROBES)
        )
    return [paint_probe_summary(list(values), markings) for values in zip(*q_values, strict=True)]


def paint_probe_summary(q_values: list[float | None], markings: list[dict]) -> dict:

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


def sensitivity_candidate(candidate: dict, evidence: dict) -> dict:
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


def rank_sensitivity(candidates: list[dict], arrays: Any, verifier: dict[str, Any]) -> dict:
    evidence_by_candidate = [probe_paint_evidence(candidate, arrays) for candidate in candidates]
    probe_records = {}
    for probe_index, probe in enumerate(SENSITIVITY_PROBES):
        ranking = verifier["rank_candidates"]([
            sensitivity_candidate(candidate, evidence[probe_index])
            for candidate, evidence in zip(candidates, evidence_by_candidate, strict=True)
        ])
        origin_key = ranking["selected_origin_key"]
        ungated = origin_key is None and bool(ranking["ungated_provisional_rank"])
        if ungated:
            origin_key = ranking["ungated_provisional_rank"][0]
        probe_records[str(int(probe))] = {
            "rank1_origin_key": origin_key,
            "rank1_was_ungated": ungated,
            "status": ranking["status"],
            "r1_paint10_rank": ranking["r1_paint10_rank"],
            "r2_spanw_paint10_rank": ranking["r2_spanw_paint10_rank"],
            "provisional_rank": ranking["provisional_rank"],
        }
    return probe_records


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
    sensitivity = {"schema": "w5-p10-sensitivity/1", "probes": list(SENSITIVITY_PROBES), "cases": {}}
    for result in case_results:
        context = verifier["prepare_view"](root, result["case_id"])
        target, target_kind = sensitivity_target(root, result["case_id"], context, verifier)
        probe_records = {}
        for probe, record in result["sensitivity"].items():
            origin_key = record["rank1_origin_key"]
            control_error = None
            if origin_key is not None and target is not None:
                selected = result["review_candidates"][origin_key]
                control_error = verifier["reference_corner_error"](
                    np.asarray(selected["corners_px"], dtype=float), target,
                )
            probe_records[probe] = {**record, "control_error": control_error}
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
    code_paths = runtime_paths | {"run_w5": "w5_holistic/run_w5.py", "verifier": "w5_holistic/verifier.py"}
    for module_name, relative in sorted(code_paths.items()):
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


def previous_stage5_anchors(root: Path, case_id: str) -> dict:
    """Load only the selected stage-5 identities for review continuity."""
    run_name = "w5_stage5_20260920"
    path = root / "w5_holistic/runs" / run_name / "review_candidates.json"
    if not path.exists():
        return {"available": False, "run": run_name, "selected": {}, "candidates": {}}
    records = __import__("json").loads(path.read_text())
    record = records.get(case_id)
    if record is None:
        return {"available": False, "run": run_name, "selected": {}, "candidates": {}}
    selected = {}
    origin_key = record.get("selected", {}).get("C")
    if origin_key:
        selected["C"] = origin_key
    candidates = {
        origin_key: record["candidates"][origin_key]
        for origin_key in selected.values()
        if origin_key in record.get("candidates", {})
    }
    return {
        "available": True,
        "run": run_name,
        "selected": selected,
        "candidates": candidates,
    }


def write_packet(
    root: Path,
    run_dir: Path,
    case_results: list[dict],
    verifier: dict[str, Any],
    runtime_paths: dict,
    stopped_views: list[dict] | None = None,
    *,
    requested_cases: list[str],
    min_visible_lengthwise: int,
    min_visible_cross_court: int,
) -> None:
    case_order = verifier.get("ALL_CASE_IDS", verifier["CASE_IDS"])
    case_results = sorted(case_results, key=lambda result: case_order.index(result["case_id"]))
    stopped_views = stopped_views or []
    packets = {result["case_id"]: result for result in case_results}
    sensitivity = write_sensitivity(root, run_dir, case_results, verifier)
    add_reference_near_candidates(root, case_results, packets, verifier)
    full_run_contamination_checks = {
        result["case_id"]: result.get("automatic_contamination_check", {"match": False, "fields": []})
        for result in case_results
    }
    automatic_path_free_of_reference_fields = (
        set(full_run_contamination_checks) == set(requested_cases)
        and all(check["match"] and not check["fields"] for check in full_run_contamination_checks.values())
    )
    review = {}
    for result in case_results:
        candidates = dict(result["review_candidates"])
        anchors = previous_stage5_anchors(root, result["case_id"])
        for origin_key, candidate in anchors["candidates"].items():
            candidates.setdefault(origin_key, candidate)
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
        keys.update(anchors["candidates"])
        result["review_candidates"] = {key: candidates[key] for key in sorted(keys) if key in candidates}
        result["previous_stage5_anchor"] = {
            "available": anchors["available"],
            "run": anchors["run"],
            "selected": anchors["selected"],
            "retained_origins": sorted(anchors["candidates"]),
        }
        review[result["case_id"]] = {
            "selected": {arm: result[arm].get("selected_origin_key") for arm in ("B", "C")},
            "A": result["A"],
            "candidates": result["review_candidates"],
            "diagnostic_controls": result["diagnostic_controls"],
            "reference_near": result.get("reference_near"),
            "previous_stage5_anchor": result["previous_stage5_anchor"],
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
        "schema": "w5-manifest/2",
        "run_id": run_dir.name,
        "cases": [result["case_id"] for result in case_results],
        "requested_cases": requested_cases,
        "min_visible_lengthwise": min_visible_lengthwise,
        "min_visible_cross_court": min_visible_cross_court,
        "visibility_columns": dict(VISIBILITY_COLUMNS),
        "stopped_views": stopped_views,
        "global_parameters": {
            "working_size": list(verifier["WORKING_SIZE"]),
            "camera_limit": verifier["CAMERA_LIMIT"],
            "min_visible_lengthwise": min_visible_lengthwise,
            "min_visible_cross_court": min_visible_cross_court,
            "visibility_columns": dict(VISIBILITY_COLUMNS),
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
            "line_template": "cached W5 fragments + frozen coverage VP ordering + union-map support",
            "automatic_path_reference_fields": not automatic_path_free_of_reference_fields,
            "automatic_path_free_of_reference_fields": automatic_path_free_of_reference_fields,
            "full_run_contamination_checks": full_run_contamination_checks,
        },
        "imported_helper_paths": runtime_paths,
        "imported_helper_hashes": helper_hashes(root, runtime_paths),
        "view_provenance": {result["case_id"]: result["provenance"] for result in case_results},
        "line_template_sources": {
            result["case_id"]: result["population_sources"].get("line_template")
            for result in case_results
        },
        "line_template_admission": {
            "min_visible_lengthwise": min_visible_lengthwise,
            "min_visible_cross_court": min_visible_cross_court,
            "visibility_columns": dict(VISIBILITY_COLUMNS),
            "cases": {
                result["case_id"]: result["population_sources"].get("line_template", {}).get(
                    "generation", {}
                ).get("visibility_admission", {})
                for result in case_results
            },
        },
        "candidate_identity": {
            "policy": "merge compatible geometry duplicates; retain conflicting gates or metadata as separate source-qualified candidates",
            "views": {result["case_id"]: result["identity_resolution"] for result in case_results},
        },
        "steering_rule_revision": {
            "status": "camera plausibility filter + visible-span weighting",
            "changed_global_rule": "camera plausibility filter and visible-span weighting from steering_record.md",
            "source": "scratch/court_det_fix/w5_holistic/steering_record.md",
            "expensive_geometry_rerun": True,
        },
    }
    per_view_fields = [
        "case_id", "label", "view_status", "g0_source", "working_width", "working_height", "image_kind",
        "same_image_mask_available", "G0_count", "G1_count", "line_template_count", "canonical_parent_count",
        "min_visible_lengthwise", "min_visible_cross_court",
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
                "line_template_count": result["population_counts"]["line_template"],
                "min_visible_lengthwise": min_visible_lengthwise,
                "min_visible_cross_court": min_visible_cross_court,
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
    write_result(
        run_dir / "result.md",
        case_results,
        sensitivity,
        refs,
        stopped_views,
        min_visible_lengthwise=min_visible_lengthwise,
        min_visible_cross_court=min_visible_cross_court,
    )


def write_result(
    path: Path,
    case_results: list[dict],
    sensitivity: dict,
    references: dict,
    stopped_views: list[dict] | None = None,
    *,
    min_visible_lengthwise: int = 0,
    min_visible_cross_court: int = 0,
) -> None:
    min_visible_lengthwise, min_visible_cross_court = validate_visibility_floors(
        min_visible_lengthwise,
        min_visible_cross_court,
    )
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
        (
            "This is a development/fit packet on the frozen corpus. It compares the legacy paint-first baseline, "
            "whole-court scoring of original candidates, and the same scoring over original plus locally adjusted "
            "candidates. The automatic pool includes the cached-fragment `line_template` source. The JSON and CSV "
            "retain `A`, `B` and `C` as historical field names."
        ),
        "",
        "## Line-template admission",
        "",
        (
            f"The inclusive projected court-template piece floors are "
            f"`min_visible_lengthwise={min_visible_lengthwise}` and "
            f"`min_visible_cross_court={min_visible_cross_court}`. "
            f"Lengthwise covers the first six template pieces ({VISIBILITY_COLUMNS['lengthwise']}); "
            f"cross-court covers the second six ({VISIBILITY_COLUMNS['cross_court']}). "
            "The filter runs before score ordering, diversity and the 256-entry cap, "
            "so later supported hypotheses can refill the pool."
        ),
        "Cross-court visibility partly reflects camera framing: head-height, cropped or even overhead footage can omit horizontal baselines or service lines. Those pieces still contribute to the support score whenever they are present.",
        "",
        "## Completed views",
        "",
        (
            "| view | legacy paint-first | original + adjusted: initial | original + adjusted: camera-filtered | "
            "original + adjusted: span-weighted | final status | adjusted candidates | determinism |"
        ),
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
        "Parent candidates use source-qualified `origin_key` values for joins and rankings. Compatible geometry duplicates share one parent and retain every source occurrence. Conflicting gates or metadata retain separate candidates with their original gates; `conflicting_geometry_groups` records those shared homographies. Neither source is treated as authoritative. Duplicate IDs within a source remain an error.",
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
    if any(result["case_id"] == "am2_window_00_frame_150" for result in case_results):
        lines.extend([
            "",
            (
                "Exact-geometry deduplication can change pool counts and diagnostic ranks without changing the "
                "evidence for retained geometries. Per-view counts are recorded in `per_view.csv`; the `30:33` "
                "control remains available for the Am2-150 comparison."
            ),
        ])
    lines.extend([
        "",
        "## Legacy baseline source provenance",
        "",
        (
            "The legacy baseline reports the canonical parent used for joins, rankings, diagnostics and gallery "
            "lookups. For an exact-geometry merge, the occurrence column identifies the source record that supplied "
            "the winning legacy score. Both parent and occurrence identities remain available in the packet records."
        ),
        "",
        "| view | line parent | line occurrence | paint parent | paint occurrence |",
        "| --- | --- | --- | --- | --- |",
    ])
    for result in case_results:
        arm_a = result["A"]
        lines.append(
            f"| {result['label']} | {arm_a.get('line') or 'none'} | "
            f"{arm_a.get('line_occurrence_key') or 'none'} | "
            f"{arm_a.get('paint') or 'none'} | "
            f"{arm_a.get('paint_occurrence_key') or 'none'} |"
        )
    lines.extend([
        "",
        "## Top candidates after each scoring change",
        "",
        (
            "This pool contains original and valid locally adjusted candidates. The first order rejects implausible "
            "camera geometry and uses the plain mean of each marking's paint evidence. The final order weights each "
            "marking by its visible span, with the same weighting applied to the geometry fallback when needed."
        ),
        "",
        "| view | camera-filtered top candidate | span-weighted top candidate | status |",
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
        (
            "| view | origin | raw candidate | expected role | automatic pool | hard-valid | camera eligible | "
            "initial rank | camera-filtered rank | span-weighted rank | Q_geom | Q_paint10 | status |"
        ),
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
        (
            "Each row reranks the original-plus-adjusted pool with the camera filter and visible-span weighting, "
            "using the saved raw ridge arrays. The table reports the top span-weighted candidate. The JSON retains "
            "the separate historical `r1` and `r2` orders for compatibility. Probe 10 remains the named pilot setting."
        ),
        "",
        "| view | probe | span-weighted top candidate | status | control target | control error |",
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
        "| view | role | origin | supplied-control error | frozen-reference error | visible-landmark max error |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for result in case_results:
        reference = references.get(result["case_id"], {})
        for origin_key, metrics in reference.get("candidates", {}).items():
            role = "reference-near" if result.get("reference_near", {}).get("origin_key") == origin_key else "ranked"
            supplied = metrics.get("approved_supplied_direction_control", {}).get("maximum")
            frozen = metrics.get("frozen_case_reference", {}).get("maximum")
            landmarks = metrics.get("visible_landmarks", [])
            landmark_max = max((item["error_px"] for item in landmarks), default=None)
            lines.append(
                f"| {result['label']} | {role} | {origin_key} | {metric_text(supplied)} | {metric_text(frozen)} | "
                f"{metric_text(landmark_max)} |"
            )
    lines.extend([
        "",
        "## Notes",
        "",
        "The packet keeps historical player/camera subsets, raw junction continuation evidence and refit attempts visible. It makes no claim beyond this development corpus.",
        "Visual review is still pending: `visual_rulings.json` has status `pending_review` and no rulings have been applied.",
        "",
    ])
    path.write_text("\n".join(lines))


def run_pilot(
    root: Path,
    run_dir: Path,
    cases: list[str],
    workers: int,
    *,
    min_visible_lengthwise: int = 0,
    min_visible_cross_court: int = 0,
) -> list[dict]:
    for output_name in ("case_records", "arrays", "manifest.json"):
        if (run_dir / output_name).exists():
            raise FileExistsError(f"W5 output already exists: {run_dir / output_name}; use a new run name")
    min_visible_lengthwise, min_visible_cross_court = validate_visibility_floors(
        min_visible_lengthwise,
        min_visible_cross_court,
    )
    runtime = load_runtime(root)
    verifier = runtime["verifier"]
    cases = resolve_case_ids(verifier, cases)
    workers = max(1, min(int(workers), 10, len(cases)))
    os.environ["W5_WORKERS"] = str(workers)
    print(f"W5 pilot cases={cases} workers={workers}", flush=True)
    started = monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                process_case,
                root,
                case_id,
                run_dir,
                min_visible_lengthwise=min_visible_lengthwise,
                min_visible_cross_court=min_visible_cross_court,
            ): case_id
            for case_id in cases
        }
        results = []
        stopped_views = []
        for completed, future in enumerate(as_completed(futures), start=1):
            case_id = futures[future]
            try:
                result = future.result()
            except ViewAmbiguity as error:
                reason = str(error)
                stopped_views.append({"case_id": case_id, "reason": reason})
                print(f"[{completed}/{len(cases)}] {case_id} stopped: {reason}", flush=True)
                continue
            results.append(result)
            print(
                f"[{completed}/{len(cases)} +{monotonic() - started:.0f}s] {case_id} complete: "
                f"B={result['B']['selected_origin_key']} C={result['C']['selected_origin_key']}",
                flush=True,
            )
    results.sort(key=lambda result: cases.index(result["case_id"]))
    stopped_views.sort(key=lambda result: cases.index(result["case_id"]))
    print("Writing combined packet", flush=True)
    write_packet(
        root,
        run_dir,
        results,
        verifier,
        runtime["paths"],
        stopped_views,
        requested_cases=cases,
        min_visible_lengthwise=min_visible_lengthwise,
        min_visible_cross_court=min_visible_cross_court,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", required=True)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument("--min-visible-lengthwise", type=nonnegative_int, default=0)
    parser.add_argument("--min-visible-cross-court", type=nonnegative_int, default=0)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    run_dir = root / "w5_holistic/runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    verifier = load_verifier(root)
    cases = resolve_case_ids(verifier, args.cases)
    run_pilot(
        root,
        run_dir,
        cases,
        args.workers,
        min_visible_lengthwise=args.min_visible_lengthwise,
        min_visible_cross_court=args.min_visible_cross_court,
    )
    print("W5 pilot packet written", run_dir, flush=True)


if __name__ == "__main__":
    main()
