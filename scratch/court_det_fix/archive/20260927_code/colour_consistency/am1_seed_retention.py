"""Check whether three saved W5 template pools retain proposals after VP seeding."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from time import monotonic

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import am1_recovery_trial as recovery
import cv2
import numpy as np
import run_cases

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "wider_evaluation/runs/20260922/comparison.json.gz"
OUTPUT = Path(__file__).with_suffix(".json.gz")
CASE_IDS = (
    "gxBQ_window_00_frame_5",
    "am4_window_00_frame_319",
    "shuttleset_21_scene_0010",
)


def geometry_matches(left: dict, right: dict) -> bool:
    return (
        np.array_equal(left["corners_px"], right["corners_px"])
        and np.array_equal(left["homography_working"], right["homography_working"])
    )


def proposal_matches(left: dict, right: dict) -> bool:
    return geometry_matches(left, right) and recovery.run_w5.values_equal_with_nan(left["gates"], right["gates"])


def gate_differences(old: dict, new: dict | None) -> dict | None:
    if new is None:
        return None
    differences = {}
    for field in sorted(old["gates"].keys() | new["gates"].keys()):
        before, after = old["gates"].get(field), new["gates"].get(field)
        if not recovery.run_w5.values_equal_with_nan(before, after):
            differences[field] = {"old": before, "new": after}
    return differences


def saved_templates(saved: dict) -> list[dict]:
    templates = []
    for parent in saved["parents"]:
        for occurrence in parent["source_occurrences"]:
            if occurrence["source"] == "line_template":
                templates.append({
                    "origin_index": occurrence["origin_index"],
                    "candidate_id": occurrence["candidate_id"],
                    "corners_px": parent["corners_px"],
                    "homography_working": parent["homography_working"],
                    "gates": parent["gates"],
                })
    templates.sort(key=lambda entry: entry["origin_index"])
    assert [entry["origin_index"] for entry in templates] == list(range(len(templates)))
    assert len(templates) == saved["population_counts"]["line_template"]
    return templates


def selected_witnesses(comparator: dict, saved: dict, old: list[dict], new: list[dict]) -> dict:
    parents = {parent["origin_key"]: parent for parent in saved["parents"]}
    candidates = {candidate["origin_key"]: candidate for candidate in saved["parents"] + saved["valid_children"]}
    witnesses = {}
    for arm in ("full", "g1_templates"):
        selected_key = comparator["selections"][arm]["gated"]
        selected = candidates[selected_key] if selected_key is not None else None
        parent_key = selected["parent_origin_key"] or selected_key if selected is not None else None
        parent = parents[parent_key] if parent_key is not None else None
        template_occurrences = [] if parent is None else [
            occurrence for occurrence in parent["source_occurrences"]
            if occurrence["source"] == "line_template"
        ]
        template_matches = []
        for occurrence in template_occurrences:
            original = old[occurrence["origin_index"]]
            assert original["candidate_id"] == occurrence["candidate_id"]
            assert proposal_matches(original, parent)
            same_id = next((entry for entry in new if entry["candidate_id"] == original["candidate_id"]), None)
            template_matches.append({
                "candidate_id": original["candidate_id"],
                "same_identity_and_geometry_homography": same_id is not None and geometry_matches(original, same_id),
                "same_geometry_homography": any(geometry_matches(original, entry) for entry in new),
                "same_identity_gate_differences": gate_differences(original, same_id),
                "same_identity_and_geometry_gates": any(
                    entry["candidate_id"] == original["candidate_id"] and proposal_matches(original, entry)
                    for entry in new
                ),
                "same_geometry_homography_gates": any(proposal_matches(original, entry) for entry in new),
            })
        witnesses[arm] = {
            "selected_origin_key": selected_key,
            "parent_origin_key": parent_key,
            "parent_source_memberships": parent["source_memberships"] if parent is not None else [],
            "parent_candidate_id": parent["candidate_id"] if parent is not None else None,
            "selected_is_child": selected_key != parent_key,
            "template_parent_matches": template_matches,
            "frozen_g0_g1_parent_preserved": bool(
                parent is not None and set(parent["source_memberships"]) & {"G0", "G1"}
            ),
        }
    return witnesses


def compare_pools(old: list[dict], new: list[dict]) -> dict:
    old_by_id = {entry["candidate_id"]: entry for entry in old}
    new_by_id = {entry["candidate_id"]: entry for entry in new}
    assert len(old_by_id) == len(old) and len(new_by_id) == len(new)
    shared_ids = old_by_id.keys() & new_by_id.keys()
    exact_ids = [
        candidate_id for candidate_id in shared_ids
        if proposal_matches(old_by_id[candidate_id], new_by_id[candidate_id])
    ]
    geometry_retained = [
        entry["candidate_id"] for entry in old
        if any(proposal_matches(entry, proposed) for proposed in new)
    ]
    return {
        "saved_count": len(old),
        "seeded_count": len(new),
        "shared_identity_count": len(shared_ids),
        "shared_identity_geometry_homography_count": sum(
            geometry_matches(old_by_id[candidate_id], new_by_id[candidate_id]) for candidate_id in shared_ids
        ),
        "shared_identity_geometry_homography_gates_count": len(exact_ids),
        "saved_geometry_homography_gates_retained_count": len(geometry_retained),
        "new_identity_count": len(new_by_id.keys() - old_by_id.keys()),
        "dropped_identity_count": len(old_by_id.keys() - new_by_id.keys()),
        "shared_identity_changed_geometry_or_gates_ids": sorted(shared_ids - set(exact_ids)),
        "new_identity_ids": sorted(new_by_id.keys() - old_by_id.keys()),
        "dropped_identity_ids": sorted(old_by_id.keys() - new_by_id.keys()),
        "saved_geometry_or_gates_not_retained_ids": sorted(old_by_id.keys() - set(geometry_retained)),
    }


def main() -> None:
    cv2.setNumThreads(1)
    comparison = recovery.read(COMPARISON)
    rows = {row["case_id"]: row for row in comparison["cases"] if row["case_id"] in CASE_IDS}
    assert set(rows) == set(CASE_IDS)
    run_w5, _, runtime = run_cases.load_runtime(ROOT)
    assert run_w5 is recovery.run_w5
    runtime["verifier"] = recovery.run_w5.load_verifier(ROOT)
    detector = recovery.run_w5.import_detector()
    results = []
    for case_id in CASE_IDS:
        comparator = rows[case_id]
        saved_path = Path(comparator["record"])
        if not saved_path.is_absolute():
            saved_path = recovery.REPO / saved_path
        saved = recovery.read(saved_path)
        assert saved["case_id"] == case_id
        assert (saved["min_visible_lengthwise"], saved["min_visible_cross_court"]) == (4, 3)
        assert comparator["population_counts"] == saved["population_counts"]
        assert comparator["selections"] == recovery.compare.selections(saved)
        context = runtime["verifier"]["prepare_view"](ROOT, case_id)
        assert context.case_id == case_id
        assert context.frame_relative_path == saved["provenance"]["frame_path"]
        assert list(context.native_size) == saved["provenance"]["native_dimensions"]
        assert list(context.size) == saved["provenance"]["working_dimensions"]
        old = saved_templates(saved)
        started = monotonic()
        generated, seeds = recovery.generate_seeded(context, runtime, detector)
        elapsed = monotonic() - started
        new = list(generated.entries)
        settings = generated.metadata["settings"]
        assert settings["global_rectangle_cap"] == 4096
        assert settings["proposal_cap"] == 256
        assert (settings["min_visible_lengthwise"], settings["min_visible_cross_court"]) == (4, 3)
        assert len(new) <= 256 and len(seeds) <= 3
        assert len(generated.metadata["ordering"]["selected_rectangle_ids_first_cap"]) <= 4096
        pool = compare_pools(old, new)
        result = {
            "case_id": case_id,
            "saved_record": str(saved_path),
            "frame_path": context.frame_relative_path,
            "native_size_wh": list(context.native_size),
            "working_size_wh": list(context.size),
            "seed_count": len(seeds),
            "generation_seconds": elapsed,
            "generator_reported_seconds": generated.metadata["generation"].get("elapsed_seconds"),
            "generator_status": generated.metadata["status"],
            "caps_and_floors": {
                key: settings[key] for key in (
                    "global_rectangle_cap", "proposal_cap", "min_visible_lengthwise", "min_visible_cross_court"
                )
            },
            "pool": pool,
            "gated_witnesses": selected_witnesses(comparator, saved, old, new),
        }
        results.append(result)
        print(case_id, json.dumps({
            "seconds": round(elapsed, 1), "shared_ids": pool["shared_identity_count"],
            "exact": pool["shared_identity_geometry_homography_gates_count"],
            "geometry_retained": pool["saved_geometry_homography_gates_retained_count"],
        }), flush=True)
    output = {
        "schema": "am1-seed-retention/1",
        "comparison_record": str(COMPARISON),
        "scope": "generation only; saved G0/G1 unchanged; no scoring, refitting, ranking, or visual ruling",
        "cases": results,
    }
    with gzip.open(OUTPUT, "wt") as destination:
        json.dump(output, destination, allow_nan=False, separators=(",", ":"))
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
