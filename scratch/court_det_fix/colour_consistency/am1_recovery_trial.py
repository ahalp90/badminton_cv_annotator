"""Bounded top-three vanishing-point seed replay through the saved W5 ranker."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from time import monotonic

for thread_limit in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_limit] = "1"

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(ROOT / "w5_holistic"))
sys.path.insert(0, str(ROOT / "wider_evaluation"))

import compare
import line_template_source
import run_cases
import run_w5

run_w5.add_helper_paths(ROOT)
import vp_pruning

CASE = "am1_window_00_frame_54"
SAVED = ROOT / (
    "evidence/holistic_admission/directional_20260921_r5/"
    "w5_directional_20260921_r5_43/case_records/am1_window_00_frame_54.json.gz"
)
COMPARISON = ROOT / "wider_evaluation/runs/20260922/comparison.json.gz"
NUMERIC = ROOT / "wider_evaluation/runs/20260922/numeric_fit.json.gz"
SEED_LINE_COUNT = 3


def read(path: Path) -> dict:
    with gzip.open(path, "rt") as source:
        return json.load(source)


def seed_points(lengthwise_lines: np.ndarray) -> np.ndarray:
    """Use each pair of the longest merged lengthwise lines, including infinity."""
    points = []
    for first, second in combinations(lengthwise_lines[:SEED_LINE_COUNT], 2):
        meeting = np.cross(first, second)
        norm = np.linalg.norm(meeting)
        if norm > 0:
            points.append(meeting / norm)
    return np.asarray(points, dtype=float).reshape(-1, 3)


def generate_seeded(context, runtime, detector):
    original = vp_pruning.estimate
    seeds = seed_points(context.families[0])

    def estimate(segments, size, settings):
        points, details = original(segments, size, settings)
        return np.concatenate((points, seeds)), details

    vp_pruning.estimate = estimate
    try:
        generated = line_template_source.generate(
            context, runtime, detector, min_visible_lengthwise=4, min_visible_cross_court=3,
        )
    finally:
        vp_pruning.estimate = original
    assert vp_pruning.estimate is original
    return generated, seeds


def entry_matches_parent(entry: dict, parent: dict) -> bool:
    # Extra seeds change rectangle enumeration order without changing scoring
    # inputs. The replay updates source occurrences when it reuses a record.
    return (
        entry["candidate_id"] == parent["candidate_id"]
        and np.array_equal(entry["corners_px"], parent["corners_px"])
        and np.array_equal(entry["homography_working"], parent["homography_working"])
        and run_w5.values_equal_with_nan(entry["gates"], parent["gates"])
        and entry.get("line_template") == parent.get("line_template")
    )


def resolve_saved_path(record: str, root: Path = ROOT) -> Path:
    """Resolve a comparison record written in another checkout to this checkout."""
    marker = Path("scratch/court_det_fix")
    source = Path(record)
    parts = source.parts
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == marker.parts:
            path = root.parents[1] / Path(*parts[index:])
            if path.is_file():
                return path
            raise FileNotFoundError(path)
    raise ValueError(f"Comparison record has no {marker} suffix: {record}")


def saved_source_entries(saved: dict, source: str) -> list[dict]:
    """Recover a frozen W5 source pool from its measured parent occurrences."""
    entries = []
    for parent in saved["parents"]:
        for occurrence in parent["source_occurrences"]:
            if occurrence["source"] != source:
                continue
            entry = {
                "candidate_id": occurrence["candidate_id"],
                "corners_px": parent["corners_px"],
                "homography_working": parent["homography_working"],
                "gates": parent["gates"],
                "pair_id": occurrence.get("pair_id"),
                "rotated_180": occurrence.get("rotated_180"),
                "axis_ids": occurrence.get("axis_ids"),
            }
            entries.append((occurrence["origin_index"], entry))
    entries.sort(key=lambda row: row[0])
    assert [index for index, _ in entries] == list(range(len(entries)))
    return [entry for _, entry in entries]


def selected_geometry(candidate: dict | None, context) -> dict | None:
    if candidate is None:
        return None
    native = np.asarray(candidate["corners_px"], dtype=float)
    scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
    return {
        "origin_key": candidate["origin_key"],
        "parent_origin_key": candidate.get("parent_origin_key"),
        "source": candidate["source"],
        "native_size_wh": list(context.native_size),
        "working_size_wh": list(context.size),
        "corners_native_px": native.tolist(),
        "corners_working_px": (native / scale).tolist(),
        "homography_working": candidate["homography_working"],
        "camera_eligible": candidate["camera_eligible"],
        "historical_fullcourt": candidate["historical"]["historical_fullcourt"],
        "player_fractions": candidate["gates"].get("player_fractions"),
        "scores": {name: candidate.get("evidence", {}).get(name) for name in (
            "q_paint10", "q_paint10_span_weighted", "q_geom", "q_geom_span_weighted",
        )},
    }


def replay_case(case_id: str, comparator: dict, control_pack: Path | None = None) -> dict:
    """Regenerate and score one seeded pool against its frozen W5 record."""
    cv2.setNumThreads(1)
    start = monotonic()
    saved_path = resolve_saved_path(comparator["record"])
    saved = read(saved_path)
    assert saved["schema"] == "w5-case-evidence/2"
    assert saved["case_id"] == case_id == comparator["case_id"]
    assert (saved["min_visible_lengthwise"], saved["min_visible_cross_court"]) == (4, 3)
    assert comparator["population_counts"] == saved["population_counts"]
    assert compare.selections(saved) == comparator["selections"]
    loaded_w5, verifier, runtime = run_cases.load_runtime(ROOT, control_pack)
    assert loaded_w5 is run_w5
    runtime["verifier"] = run_w5.load_verifier(ROOT)
    context = verifier.prepare_view(ROOT, case_id)
    verifier = runtime["verifier"]
    assert context.frame_relative_path == saved["provenance"]["frame_path"]
    assert list(context.native_size) == saved["provenance"]["native_dimensions"]
    assert list(context.size) == saved["provenance"]["working_dimensions"]
    assert context.same_image_mask_available == saved["provenance"]["same_image_mask_available"]
    assert context.person_mask_unavailable_reason == saved["provenance"]["person_mask_unavailable_reason"]
    saved_candidates = saved["parents"] + saved["valid_children"]
    assert verifier["rank_candidates"](saved_candidates) == saved["rankings"]["C"]
    saved_by_key = {candidate["origin_key"]: candidate for candidate in saved_candidates}
    assert len(saved_by_key) == len(saved_candidates)

    g0 = saved_source_entries(saved, "G0")
    g1 = saved_source_entries(saved, "G1")
    g0_source = saved["population_sources"]["G0"]
    g1_source = saved["population_sources"]["G1"]
    assert (len(g0), len(g1)) == (saved["population_counts"]["G0"], saved["population_counts"]["G1"])
    for source, entries in (("G0", g0), ("G1", g1)):
        for index, entry in enumerate(entries):
            key = f"{source}:{entry['candidate_id']}"
            parent = saved_by_key[key]
            assert parent["origin_index"] == index
            assert np.array_equal(entry["corners_px"], parent["corners_px"])
            assert np.array_equal(entry["homography_working"], parent["homography_working"])
            assert run_w5.values_equal_with_nan(entry["gates"], parent["gates"])
    print(f"saved G0/G1 geometry checked: {len(g0)}/{len(g1)}", flush=True)

    generated, seeds = generate_seeded(context, runtime, run_w5.import_detector())
    templates = list(generated.entries)
    assert generated.metadata["settings"]["global_rectangle_cap"] == 4096
    assert generated.metadata["settings"]["proposal_cap"] == 256
    assert (generated.metadata["settings"]["min_visible_lengthwise"],
            generated.metadata["settings"]["min_visible_cross_court"]) == (4, 3)
    assert len(templates) <= line_template_source.PROPOSAL_CAP
    assert generated.metadata["ordering"]["selected_rectangle_ids_first_cap"]
    assert len(generated.metadata["ordering"]["selected_rectangle_ids_first_cap"]) <= line_template_source.RECTANGLE_CAP
    print(f"seeded templates: {len(templates)}; seed points: {len(seeds)}", flush=True)
    identities, resolution = run_w5.canonicalise_populations(g0, g1, templates)
    assert not resolution["duplicate_groups"]
    assert not saved["identity_resolution"]["duplicate_groups"]
    assert not any(run_w5.find_forbidden_keys(entry) for entry in g0 + g1 + templates)
    parents = []
    children = []
    changed = []
    retained = []
    cache = {}
    for identity in identities:
        entry = identity["entry"]
        key = identity["origin_key"]
        saved_parent = saved_by_key.get(key)
        if saved_parent is not None and entry_matches_parent(entry, saved_parent):
            parent = deepcopy(saved_parent)
            retained.append(key)
            parent["origin_index"] = identity["origin_index"]
            parent["source_occurrences"] = identity["source_occurrences"]
            if identity["source"] == "line_template":
                parent["line_template_provenance"] = identity["source_occurrences"]
            saved_child = saved_by_key.get(f"{key}/child")
            if saved_child is not None:
                child = deepcopy(saved_child)
                child["origin_index"] = identity["origin_index"]
                child["source_occurrences"] = identity["source_occurrences"]
                if identity["source"] == "line_template":
                    child["line_template_provenance"] = identity["source_occurrences"]
                children.append(child)
        else:
            changed.append(key)
            parent, _ = run_w5.make_parent_record(
                context, entry, identity["source"], identity["source_order"],
                identity["origin_index"], runtime, cache, identity=identity,
            )
            _, child, _ = run_w5.attempt_refit(context, parent, runtime, cache)
            if child is not None:
                children.append(run_w5.public_candidate(child))
        parents.append(run_w5.public_candidate(parent))
    assert len(parents) == len(identities)
    assert len(changed) + len(retained) == len(parents)
    candidates = parents + children
    eligible_parents = [parent for parent in parents if parent["hard_valid"] and "evidence" in parent]
    b_rank = verifier["rank_candidates"](eligible_parents)
    c_rank = verifier["rank_candidates"](candidates)
    trial_record = {"parents": parents, "valid_children": children, "rankings": {"B": b_rank, "C": c_rank}}
    return {
        "saved_path": saved_path, "saved": saved, "context": context, "verifier": verifier,
        "trial_record": trial_record, "saved_by_key": saved_by_key,
        "g0_source": g0_source, "g1_source": g1_source, "generated": generated,
        "seeds": seeds, "templates": templates, "g0": g0, "g1": g1,
        "changed": changed, "retained": retained, "started": start,
    }


def main(output: Path, pool_output: Path | None = None) -> None:
    comparator = next(case for case in read(COMPARISON)["cases"] if case["case_id"] == CASE)
    numeric = next(case for case in read(NUMERIC)["cases"] if case["case_id"] == CASE)
    replay = replay_case(CASE, comparator)
    saved = replay["saved"]
    context = replay["context"]
    verifier = replay["verifier"]
    trial_record = replay["trial_record"]
    saved_by_key = replay["saved_by_key"]
    g0_source, g1_source = replay["g0_source"], replay["g1_source"]
    generated, seeds = replay["generated"], replay["seeds"]
    templates, g0, g1 = replay["templates"], replay["g0"], replay["g1"]
    changed, retained, start = replay["changed"], replay["retained"], replay["started"]
    parents, children = trial_record["parents"], trial_record["valid_children"]
    c_rank = trial_record["rankings"]["C"]
    candidates = parents + children
    if pool_output is not None:
        pool_record = dict(trial_record)
        pool_record.update(
            schema="am1-seeded-w5-pool/1", case_id=CASE, provenance=saved["provenance"],
            min_visible_lengthwise=4, min_visible_cross_court=3,
        )
        pool_output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(pool_output, "wt") as destination:
            json.dump(verifier["jsonable"](pool_record), destination, allow_nan=False)
    choices = compare.selections(trial_record)
    candidate_map = {candidate["origin_key"]: candidate for candidate in candidates}
    saved_choices = compare.selections(saved)
    selected = {}
    frozen_selected = {}
    for arm in choices:
        selected[arm] = {}
        frozen_selected[arm] = {}
        for gate in ("ungated", "gated"):
            key = choices[arm][gate]
            selected[arm][gate] = selected_geometry(candidate_map.get(key), context)
            frozen_key = saved_choices[arm][gate]
            frozen_selected[arm][gate] = selected_geometry(saved_by_key.get(frozen_key), context)
    retrospective = {}
    reference = numeric["reference"]
    for arm in choices:
        retrospective[arm] = {}
        for gate in ("ungated", "gated"):
            candidate = selected[arm][gate]
            if candidate is not None and reference.get("corners_px"):
                retrospective[arm][gate] = verifier["reference_corner_error"](
                    np.asarray(candidate["corners_native_px"]), np.asarray(reference["corners_px"]),
                )
    frozen_retrospective = {}
    for arm, arm_candidates in frozen_selected.items():
        frozen_retrospective[arm] = {}
        for gate, candidate in arm_candidates.items():
            if candidate is not None and reference.get("corners_px"):
                frozen_retrospective[arm][gate] = verifier["reference_corner_error"](
                    np.asarray(candidate["corners_native_px"]), np.asarray(reference["corners_px"]),
                )
    reference_landmarks = run_w5.load_reference(ROOT, CASE, verifier).get("landmarks", [])
    retrospective_alternatives = []
    if reference_landmarks:
        for candidate in candidates:
            if candidate["source"] != "line_template":
                continue
            errors = verifier["projected_landmark_diagnostics"](
                np.asarray(candidate["homography_working"]), reference_landmarks, context,
            )
            if not errors:
                continue
            retrospective_alternatives.append((max(error["error_px"] for error in errors), candidate))
        retrospective_alternatives.sort(key=lambda pair: pair[0])
    retrospective_nearest = []
    for max_error, candidate in retrospective_alternatives[:3]:
        key = candidate["origin_key"]
        retrospective_nearest.append({
            "label": "retrospective_reference_selected_only",
            "maximum_landmark_error_native_px": max_error,
            "geometry": selected_geometry(candidate, context),
            "final_camera_rank": c_rank["provisional_rank"].index(key) + 1
            if key in c_rank["provisional_rank"] else None,
            "corner_reference_error": verifier["reference_corner_error"](
                np.asarray(candidate["corners_px"]), np.asarray(reference["corners_px"]),
            ),
        })
    result = {
        "schema": "am1-top3-w5-trial/1",
        "case_id": CASE,
        "frame_path": saved["provenance"]["frame_path"],
        "saved_record": str(SAVED.relative_to(REPO)),
        "comparison_record": str(COMPARISON.relative_to(REPO)),
        "numeric_record": str(NUMERIC.relative_to(REPO)),
        "settings": {"seed_lengthwise_lines": SEED_LINE_COUNT, "seed_points": seeds.tolist(),
                     "rectangle_cap": line_template_source.RECTANGLE_CAP,
                     "template_cap": line_template_source.PROPOSAL_CAP,
                     "min_visible_lengthwise": 4, "min_visible_cross_court": 3},
        "sources": {"G0": g0_source, "G1": g1_source},
        "generation": generated.metadata,
        "population_counts": {"G0": len(g0), "G1": len(g1), "templates": len(templates),
                              "canonical_parents": len(parents), "valid_children": len(children)},
        "candidate_changes": {"changed_parents": changed, "retained_parents": retained,
                              "dropped_parent_ids": sorted(
                                  {parent["origin_key"] for parent in saved["parents"]}
                                  - {parent["origin_key"] for parent in parents}
                              )},
        "frozen_source_choices": saved_choices,
        "seeded_comparator_choices": choices,
        "frozen_selected_geometry": frozen_selected,
        "selected_geometry": selected,
        "retrospective_frozen_selected_reference_error": frozen_retrospective,
        "retrospective_selected_reference_error": retrospective,
        "retrospective_nearest_templates": retrospective_nearest,
        "rank_status": c_rank["status"],
        "rank_criterion": c_rank["r2_criterion"],
        "timings_seconds": {"total": monotonic() - start},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt") as destination:
        json.dump(verifier["jsonable"](result), destination, allow_nan=False)
    print(json.dumps({"saved": {gate: saved_choices["full"][gate] for gate in ("ungated", "gated")},
                      "seeded": {gate: choices["full"][gate] for gate in ("ungated", "gated")},
                      "changed": len(changed), "retained": len(retained),
                      "seconds": round(monotonic() - start, 1)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--pool-output", type=Path, help="Preserve all scored candidates for subsequent evidence trials.",
    )
    arguments = parser.parse_args()
    main(arguments.output, arguments.pool_output)
