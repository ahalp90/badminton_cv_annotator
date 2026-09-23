"""Apply the fixed Am1 net preference to every saved wider-evaluation pool."""

from __future__ import annotations

import argparse
import hashlib
import sys
import traceback
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / "wider_evaluation/runs/20260922"
sys.path[:0] = [str(ROOT / "colour_consistency"), str(ROOT / "wider_evaluation")]

import am1_net_selection_trial as net  # pyrefly: ignore[missing-import]
from am1_recovery_trial import resolve_saved_path  # pyrefly: ignore[missing-import]
from run_cases import load_runtime  # pyrefly: ignore[missing-import]


def source_path(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def verify_inventory(comparison: dict, manifest: dict, inventory: dict) -> tuple[dict, list[dict], list[dict]]:
    """Check the original case contract before loading any candidate pool."""
    comparison_rows = comparison["cases"]
    manifest_rows = manifest["cases"]
    inventory_rows = inventory["cases"]
    for label, rows in (("comparison", comparison_rows), ("manifest", manifest_rows), ("inventory", inventory_rows)):
        ids = [row["case_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{label} has duplicate case IDs")
    by_manifest = {row["case_id"]: row for row in manifest_rows}
    by_inventory = {row["case_id"]: row for row in inventory_rows}
    if set(by_manifest) != set(by_inventory) or set(by_manifest) != {row["case_id"] for row in comparison_rows}:
        raise ValueError("Comparison, manifest, and input inventory have different case IDs")
    if comparison["missing_cases"] or not comparison["complete"]:
        raise ValueError("Stored wider comparison is incomplete")
    if len(comparison_rows) != comparison["planned_case_count"]:
        raise ValueError("Stored wider comparison case count differs from plan")
    failures = []
    inventory_differences = []
    for row in comparison_rows:
        case_id = row["case_id"]
        item = by_manifest[case_id]
        try:
            original = by_inventory[case_id]
            for field in ("group", "arm", "previous_w5_case", "image", "source_pack"):
                if item.get(field) != original.get(field):
                    raise ValueError(f"{field} differs from original input inventory")
            differing = sorted(
                field for field in item.keys() | original.keys() if item.get(field) != original.get(field)
            )
            if differing:
                inventory_differences.append({"case_id": case_id, "fields": differing})
            for field in ("group", "arm", "previous_w5_case", "reference_status", "view_status"):
                if row.get(field) != item.get(field):
                    raise ValueError(f"{field} differs from manifest")
            image = ROOT / item["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            if hashlib.md5(image.read_bytes()).hexdigest() != item["image_md5"]:
                raise ValueError(f"{image}: image MD5 differs from manifest")
            record = resolve_saved_path(row["record"])
            source_path(record)
        except Exception as error:  # noqa: BLE001 - keep independent inventory failures visible
            failures.append({"case_id": case_id, "stage": "input_inventory", "error": repr(error)})
    return by_manifest, failures, inventory_differences


def add_scores(case: dict, record: dict) -> None:
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    criterion = record["rankings"]["C"]["r2_criterion"]
    by_key = {row["origin_key"]: row for row in case["candidates"]}
    for row in case["candidates"]:
        row["paint_score"] = candidates[row["origin_key"]]["evidence"][criterion]
    for role in ("baseline", "trial"):
        selected = case[role]
        if selected is not None:
            row = by_key[selected["origin_key"]]
            selected["original_rank"] = row["original_rank"]
            selected["full_court_rank"] = row["full_court_rank"]
            selected["paint_score"] = row["paint_score"]
            selected["coverage"] = dict(zip(net.PIECE_NAMES, row["coverage"]))
            selected["visible_samples"] = dict(zip(net.PIECE_NAMES, row["visible_samples"]))
    case["strong_eligible_candidates"] = [
        row for row in case["candidates"] if row["strong_net"] and row["historical_fullcourt"]
    ]
    case["strong_ungated_diagnostic"] = [
        row for row in case["candidates"] if row["strong_net"] and not row["historical_fullcourt"]
    ]
    if case["changed"]:
        case["rank_depth"] = case["trial"]["full_court_rank"]
        case["paint_score_gap_trial_minus_baseline"] = case["trial"]["paint_score"] - case["baseline"]["paint_score"]
    else:
        case["rank_depth"] = None
        case["paint_score_gap_trial_minus_baseline"] = None


def retrospective(case: dict, reference: dict, verifier) -> dict:
    """Measure references after every automatic choice is fixed."""
    corners = reference.get("corners_px")
    if not corners:
        return {"status": "no_reference_corners", "reference_status": reference.get("reference_status")}
    visible = reference.get("corner_visible")
    visible_indices = [index for index, flag in enumerate(visible) if str(flag) == "1"] if visible else None
    result = {
        "status": "measured",
        "reference_status": reference.get("reference_status"),
        "corner_source": reference.get("corner_source"),
        "visible_corner_indices": visible_indices,
    }
    expected = np.asarray(corners, dtype=float)
    landmarks = reference.get("landmarks", [])
    projection_context = SimpleNamespace(native_size=case["native_size_wh"], size=case["working_size_wh"])
    for role in ("baseline", "trial"):
        selected = case[role]
        if selected is None:
            result[role] = None
            continue
        predicted = np.asarray(selected["corners_native_px"])
        direct = np.linalg.norm(predicted - expected, axis=1)
        rotated = np.linalg.norm(predicted[[2, 3, 0, 1]] - expected, axis=1)
        relabelled = bool(rotated.max() < direct.max())
        distances = rotated if relabelled else direct
        landmark_rows = verifier.projected_landmark_diagnostics(
            np.asarray(selected["homography_working"], dtype=float), landmarks, projection_context,
        )
        landmark_errors = np.asarray([row["error_px"] for row in landmark_rows], dtype=float)
        result[role] = {
            "relabelled_180": relabelled,
            "all_corners_native_px": distances.tolist(),
            "all_max_native_px": float(distances.max()),
            "all_median_native_px": float(np.median(distances)),
            "visible_corner_count": len(visible_indices) if visible_indices is not None else None,
            "visible_corner_max_native_px": float(distances[visible_indices].max()) if visible_indices else None,
            "visible_corner_median_native_px": float(np.median(distances[visible_indices])) if visible_indices else None,
            "visible_landmark_count": len(landmark_errors),
            "visible_landmark_max_native_px": float(landmark_errors.max()) if len(landmark_errors) else None,
            "visible_landmark_median_native_px": float(np.median(landmark_errors)) if len(landmark_errors) else None,
        }
    return result


def add_references(cases: list[dict], manifest: dict, numeric: dict, verifier) -> list[str]:
    """Read original references only after every automatic choice is fixed."""
    paths = {row["source_pack"] for row in manifest["cases"] if row["arm"] == "frozen_detector"}
    packs = {path: verifier.read_json_gz(ROOT / path) for path in paths}
    by_manifest = {row["case_id"]: row for row in manifest["cases"]}
    numeric_references = {row["case_id"]: row["reference"] for row in numeric["cases"]}
    if set(numeric_references) != set(by_manifest):
        raise ValueError("Numeric reference case IDs differ from manifest")
    for case in cases:
        case_id = case["case_id"]
        item = by_manifest[case_id]
        reference = packs[item["source_pack"]]["references"].get(case_id, {}) if item["arm"] == "frozen_detector" else {}
        if reference.get("corners_px") != numeric_references[case_id].get("corners_px"):
            raise ValueError(f"{case_id}: original and compact numeric reference corners differ")
        case["retrospective_reference_diagnostic_only"] = retrospective(case, reference, verifier)
    return [source_path(ROOT / path) for path in sorted(paths)]


def main(output: Path) -> None:
    cv2.setNumThreads(1)
    started = perf_counter()
    comparison_path = RUN / "comparison.json.gz"
    manifest_path = RUN / "manifest.json.gz"
    inventory_path = RUN / "input_inventory.json.gz"
    control_path = RUN / "control_inputs.json.gz"
    numeric_path = RUN / "numeric_fit.json.gz"
    comparison = net.verifier.read_json_gz(comparison_path)
    manifest = net.verifier.read_json_gz(manifest_path)
    inventory = net.verifier.read_json_gz(inventory_path)
    by_manifest, failures, inventory_differences = verify_inventory(comparison, manifest, inventory)
    if not control_path.is_file():
        raise FileNotFoundError(control_path)
    controls = net.verifier.read_json_gz(control_path)
    control_ids = {source["id"] for source in controls["cases"]}
    expected_controls = {row["case_id"] for row in manifest["cases"] if row["arm"] == "rejection_review"}
    if control_ids != expected_controls:
        raise ValueError("Original control pack IDs differ from the manifest")
    _, verifier, _ = load_runtime(ROOT, control_path)
    failed_ids = {failure["case_id"] for failure in failures}
    cases = []
    for row in comparison["cases"]:
        case_id = row["case_id"]
        if case_id in failed_ids:
            continue
        tick = perf_counter()
        try:
            record_path = resolve_saved_path(row["record"])
            record = verifier.read_json_gz(record_path)
            context = verifier.prepare_view(ROOT, case_id)
            item = by_manifest[case_id]
            if context.frame_relative_path != item["image"]:
                raise ValueError(f"prepared frame {context.frame_relative_path} differs from manifest {item['image']}")
            if list(context.native_size) != record["provenance"]["native_dimensions"]:
                raise ValueError("prepared native dimensions differ from saved record")
            if list(context.size) != record["provenance"]["working_dimensions"]:
                raise ValueError("prepared working dimensions differ from saved record")
            if context.image_kind != item["image_kind"] and item["arm"] != "rejection_review":
                raise ValueError("prepared image kind differs from manifest")
            if context.same_image_mask_available != item["same_image_boxes"]:
                raise ValueError("same-image mask availability differs from manifest")
            net.check_record(case_id, case_id, record, record_path, context, row)
            timings = {"projection": 0.0, "scoring": 0.0}
            case = net.evaluate_case(case_id, case_id, record, context, timings)
            if case["original_comparator_full"] != row["selections"]["full"]:
                raise ValueError("full/gated baseline differs from stored wider comparator")
            add_scores(case, record)
            case.update({
                "source_record": source_path(record_path),
                "group": row["group"],
                "arm": row["arm"],
                "reference_status": row.get("reference_status"),
                "view_status": row.get("view_status"),
                "previous_w5_case": row["previous_w5_case"],
                "provenance": {
                    key: record["provenance"][key] for key in (
                        "frame_path", "image_kind", "native_dimensions", "working_dimensions",
                        "same_image_mask_available", "player_foot_source", "person_mask_unavailable_reason",
                    )
                },
                "cost_seconds": {**timings, "total": perf_counter() - tick},
            })
            cases.append(case)
            print(f"{len(cases) + len(failures)}/{len(comparison['cases'])} {case_id}: "
                  f"{case['baseline'] and case['baseline']['origin_key']} -> "
                  f"{case['trial'] and case['trial']['origin_key']}", flush=True)
        except Exception as error:  # noqa: BLE001 - one failed case must not hide the rest
            failure = {"case_id": case_id, "stage": "evaluation", "error": repr(error),
                       "traceback": traceback.format_exc()}
            failures.append(failure)
            print(f"FAILED {case_id}: {error!r}", flush=True)

    # The fixture and references enter only after selection for the whole pool has finished.
    old_trial = net.verifier.read_json_gz(ROOT / "colour_consistency/am1_net_selection_trial.json.gz")
    fixture = {case["case_id"]: case for case in old_trial["cases"] if case["label"] != net.POOL_LABEL}
    by_case = {case["case_id"]: case for case in cases}
    regressions = []
    for case_id, expected in fixture.items():
        actual = by_case.get(case_id)
        if actual is None:
            regressions.append({"case_id": case_id, "status": "missing"})
            continue
        for role in ("baseline", "trial"):
            old = expected[role]
            new = actual[role]
            if (old and old["origin_key"]) != (new and new["origin_key"]):
                regressions.append({"case_id": case_id, "role": role, "status": "selection_mismatch"})
            elif old and old["net"].get("coverage") != new["net"].get("coverage"):
                regressions.append({"case_id": case_id, "role": role, "status": "coverage_mismatch"})
    numeric = net.verifier.read_json_gz(numeric_path)
    reference_packs = add_references(cases, manifest, numeric, verifier)
    labelled = [case for case in cases if case["arm"] == "rejection_review" and case["reference_status"] == "non_court"]
    result = {
        "schema": "saved-net-scan/1",
        "settings": {
            "rule": net.RULE,
            "piece_names": net.PIECE_NAMES,
            "samples_per_piece": net.SAMPLES_PER_PIECE,
            "perpendicular_tolerance_working_px": net.PERPENDICULAR_TOLERANCE_WORKING_PX,
            "direction_tolerance_deg": net.DIRECTION_TOLERANCE_DEG,
            "extent_margin_working_px": net.EXTENT_MARGIN_WORKING_PX,
            "strong_support": net.STRONG_SUPPORT,
            "fragments": "all prepared Hough fragments; no occlusion mask",
            "coverage_denominator": "in-frame samples only; null when no sample is in frame",
            "selection_fallback": "original full/gated winner when no strong eligible candidate",
        },
        "sources": {
            "rule_code": source_path(Path(net.__file__)),
            "comparison": source_path(comparison_path),
            "manifest": source_path(manifest_path),
            "input_inventory": source_path(inventory_path),
            "control_pack": source_path(control_path),
            "reference_for_retrospective_only": source_path(numeric_path),
            "original_reference_packs_for_retrospective_only": reference_packs,
            "old_three_case_fixture": source_path(ROOT / "colour_consistency/am1_net_selection_trial.json.gz"),
        },
        "planned_case_count": len(comparison["cases"]),
        "completed_case_count": len(cases),
        "failed_case_count": len(failures),
        "complete": len(cases) == len(comparison["cases"]) and not failures and not regressions,
        "failures": failures,
        "pre_freeze_inventory_differences": inventory_differences,
        "old_three_case_regressions": regressions,
        "changed_selection_count": sum(case["changed"] for case in cases),
        "labelled_non_court_controls": {
            "completed": len(labelled),
            "planned": sum(row.get("reference_status") == "non_court" for row in comparison["cases"]),
            "baseline_accepted": sum(case["baseline"] is not None for case in labelled),
            "trial_accepted": sum(case["trial"] is not None for case in labelled),
        },
        "cases": cases,
        "reference_note": "Retrospective only. Mixed annotation conventions and off-image corners limit small drifts.",
        "elapsed_seconds": perf_counter() - started,
    }
    verifier.write_json_gz(output, result)
    print(f"wrote {output}: {len(cases)}/{len(comparison['cases'])} cases; "
          f"{len(failures)} failures; {len(regressions)} fixture regressions", flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "net_recovery/saved_net_scan.json.gz")
    arguments = parser.parse_args()
    main(arguments.output)
