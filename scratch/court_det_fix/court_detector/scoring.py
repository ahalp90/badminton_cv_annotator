"""Merge duplicate courts, measure them, refit their stripes and rank the results."""


from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import cv2
import numpy as np


class ViewAmbiguity(AssertionError):
    """A candidate identity conflict makes one view unsafe to score."""


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


def import_detector():
    from . import geometry as detector

    return detector


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


def view_line_maps(context) -> np.ndarray:
    """The view's two wide-family distance maps. They depend only on the view, so every refit shares one copy."""
    detector = import_detector()
    maps = detector._distance_maps(detector._wide_line_families(context.segments), context.size)
    # Shared across every refit in the view, so a stray in-place write should fail loudly.
    maps.flags.writeable = False
    return maps


def attempt_refit(
    context, parent: dict, runtime: dict[str, Any], cache: dict, line_maps: np.ndarray,
) -> tuple[dict, dict | None, dict[str, np.ndarray] | None]:
    """Refit one parent court; line_maps is view_line_maps(context)."""
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
    child_gates = runtime["gate_evidence"](
        np.asarray(native_corners),
        context.source,
        np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float),
        context.size,
        context.families,
        line_maps,
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
    from . import stripe_fitting as fixed_stripe_refit

    return fixed_stripe_refit


def import_paint_geometry():
    from . import paint_geometry

    return paint_geometry


def public_candidate(candidate: dict) -> dict:
    public = {key: value for key, value in candidate.items() if not key.startswith("_")}
    if candidate.get("occurrence_count", 1) > 1:
        public["legacy_occurrences"] = candidate.get("_legacy_occurrences", [])
    return public


class ScoredPopulations(NamedTuple):
    """One view's W5 parents, refitted children and C ranking."""

    parents: list[dict]
    children: list[dict]
    fit_rows: list[dict]
    arrays: dict[str, np.ndarray]  # per-candidate arrays, keyed "<origin_key>::<name>"
    identity_resolution: dict
    b_candidates: list[dict]  # hard-valid parents with evidence
    c_candidates: list[dict]  # b_candidates, then the valid children
    c_rankings: dict
    determinism: dict | None  # None with self-checks off
    line_maps: np.ndarray  # the view's wide-family distance maps, shared by every refit


def score_populations(
    context,
    g0: list[dict],
    g1: list[dict],
    line_template: list[dict],
    runtime: dict[str, Any],
    cache: dict[bytes, tuple[dict, dict[str, np.ndarray]]],
    progress: Callable[[str], None],
    *,
    self_checks: bool = True,
) -> ScoredPopulations:
    """Merge the three populations, measure every parent, refit each once and rank them.

    :param cache: make_parent_record's measurement cache. process_case reuses it for its
        diagnostic controls.
    :param progress: Receives one progress line at a time.
    :param self_checks: Reject automatic entries that carry reference fields, and require
        the C ranking not to depend on candidate order. Both raise on failure.
    """
    case_id = context.case_id
    verifier = runtime["verifier"]
    if self_checks:
        contamination_fields = []
        for index, entry in enumerate(g0 + g1 + line_template):
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
    line_maps = view_line_maps(context)
    for parent_index, parent in enumerate(parents, start=1):
        row, child, arrays = attempt_refit(context, parent, runtime, cache, line_maps)
        fit_rows.append(row)
        if child is not None:
            children.append(child)
            for key, value in arrays.items():
                all_arrays[f"{child['origin_key']}::{key}"] = value
        if parent_index % 250 == 0:
            progress(f"refits {parent_index}/{len(parents)}; valid children={len(children)}")
    b_candidates = [parent for parent in parents if parent.get("hard_valid") and "evidence" in parent]
    c_candidates = b_candidates + children
    c_rankings = verifier["rank_candidates"](c_candidates)
    determinism = None
    if self_checks:
        determinism = verifier["permutation_determinism"](c_candidates)
        if not determinism["match"]:
            raise RuntimeError(f"{case_id}: W5 ranker is not permutation-deterministic")
    return ScoredPopulations(
        parents, children, fit_rows, all_arrays, identity_resolution, b_candidates, c_candidates, c_rankings,
        determinism, line_maps,
    )
