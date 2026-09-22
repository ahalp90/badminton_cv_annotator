#!/usr/bin/env python3
"""Read-only baseline-B SVD pruning screen; no detector, matcher or optimiser.

Run with Python 3.10+ and NumPy:
    python run_screen.py --inputs ../task3_inputs --output results
The input root may also be a checkout of the pinned repository revision.
Only svd_direction/fit_groups are loaded from the reviewed frozen producer.
"""
from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path
from typing import Callable

import numpy as np

REPOSITORY = "ahalp90/badminton_cv_annotator"
REVISION = "b47eee0ff43ec7cbebcc2d84eb9e5e98075d4eea"
PREFIX = Path("scratch/court_det_fix")
RUN = PREFIX / "direction_agreement/runs/direction_agreement_20260915_144900"
SVD_SOURCE = PREFIX / "frozen_helpers_20260914/automatic_axes/svd_fixed/run_svd_fixed.py"
CASES = (
    "gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5",
    "am2_window_00_frame_150", "am2_window_01_frame_28019", "am3_window_00_frame_0",
    "shuttleset_03_scene_0017", "shuttleset_03_scene_0019",
    "shuttleset_03_scene_0016", "shuttleset_21_scene_0020",
)
BUDGETS = (8, 10, 12, 14, 16)
GROUP_COUNT = 16
DIAGNOSTIC_ATOL = 1e-12
DIAGNOSTIC_RTOL = 1e-10
# Comparison-only tolerances: neither changes residual ranking nor admissions.
PIXEL_ATOL = 1e-8  # The archived E3 producer's replay tolerance.
TIE_ATOL = 1e-12
DIAGNOSTIC_KEYS = ("singular_values", "normalised_nullspace_gap", "algebraic_rms")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"Expected object: {path}")
    return value


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def write_gzip(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))
    temporary.replace(path)


def write_json(path: Path, value: dict | list) -> None:
    write_gzip(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    write_gzip(path, buffer.getvalue())


def load_fit_groups(root: Path) -> Callable:
    """Reuse precisely two reviewed producer functions, not its optimiser imports."""
    path = root / SVD_SOURCE
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"svd_direction", "fit_groups"}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in functions} == names, "Missing frozen SVD functions")
    namespace = {"np": np}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["fit_groups"]


def fit_diagnostics(lines: np.ndarray, masks: np.ndarray, fit_groups: Callable) -> list[dict]:
    require(lines.ndim == 2 and lines.shape[1] == 3, "Expected Nx3 lines")
    require(masks.ndim == 2 and masks.shape[1] == len(lines), "Support-mask shape mismatch")
    require(np.isfinite(lines).all(), "Non-finite line coefficient")
    require((np.linalg.norm(lines[:, :2], axis=1) > 0).all(), "Zero line normal")
    require((masks.sum(axis=1) >= 2).all(), "SVD requires at least two support lines")
    _, groups = fit_groups(lines, masks)
    return groups


def rank_groups(groups: list[dict]) -> list[int]:
    require(len({g["group_index"] for g in groups}) == len(groups), "Duplicate group index")
    require(all(np.isfinite(g["algebraic_rms"]) and g["algebraic_rms"] >= 0 for g in groups),
            "Invalid SVD residual")
    return [g["group_index"] for g in sorted(
        groups, key=lambda g: (g["algebraic_rms"], -g["line_count"], g["group_index"]))]


def fit_status(record: dict) -> str:
    error = record.get("max_corner_working_px")
    if "failure" in record or error is None or not np.isfinite(error):
        return "failed"
    require(isinstance(record.get("converged"), bool), "Missing convergence flag")
    return "converged" if record["converged"] else "unconverged"


def best_record(records: list[dict], converged_only: bool = False) -> dict | None:
    allowed = {"converged"} if converged_only else {"converged", "unconverged"}
    valid = [r for r in records if fit_status(r) in allowed]
    return min(valid, key=lambda r: (r["max_corner_working_px"], r["pair_id"]), default=None)


def brief(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {key: record[key] for key in
            ("pair_id", "groups", "max_corner_working_px", "converged")}


def evaluate(order: list[int], budget: int, records: list[dict]) -> dict:
    require(2 <= budget <= len(order), "Invalid direction budget")
    selected = set(order[:budget])
    pool = [r for r in records if set(r["groups"]).issubset(selected)]
    counts = {status: sum(fit_status(r) == status for r in pool)
              for status in ("failed", "unconverged", "converged")}
    require(len(pool) == budget * (budget - 1), "Incomplete ordered-pair cache")
    baseline_finite = best_record(records)
    baseline_converged = best_record(records, True)
    finite = best_record(pool)
    converged = best_record(pool, True)
    result = {
        "budget": budget, "selected_groups_rank_order": order[:budget],
        "selected_groups_original_order": sorted(selected),
        "dropped_groups": sorted(set(order) - selected),
        "ordered_pairs": len(pool), "full_ordered_pairs": len(records),
        "ordered_pair_reduction": 1.0 - len(pool) / len(records),
        **counts, "finite": counts["converged"] + counts["unconverged"],
        "best_finite": brief(finite), "best_converged": brief(converged),
        "failed_pair_ids": [r["pair_id"] for r in pool if fit_status(r) == "failed"],
        "unconverged_pair_ids": [r["pair_id"] for r in pool if fit_status(r) == "unconverged"],
    }
    for name, baseline, current in (("finite", baseline_finite, finite),
                                    ("converged", baseline_converged, converged)):
        present = baseline is not None and set(baseline["groups"]).issubset(selected)
        delta = None if baseline is None or current is None else (
            current["max_corner_working_px"] - baseline["max_corner_working_px"])
        require(delta is None or delta >= 0, "A subset cannot improve the cached minimum")
        equivalent = [] if baseline is None else [
            r["pair_id"] for r in records
            if fit_status(r) in ({"converged"} if name == "converged" else {"converged", "unconverged"})
            and abs(r["max_corner_working_px"] - baseline["max_corner_working_px"]) <= PIXEL_ATOL]
        retained_ids = {r["pair_id"] for r in pool}
        result.update({f"baseline_best_{name}_pair_retained": present,
                       f"best_{name}_delta_px": delta,
                       f"preserves_best_{name}_error": delta is not None and delta <= PIXEL_ATOL,
                       f"baseline_equivalent_{name}_pairs": len(equivalent),
                       f"retained_equivalent_{name}_pairs": sum(p in retained_ids for p in equivalent)})
    return result


def compare_diagnostics(current: list[dict], reference: list[dict]) -> dict:
    maximum = {key: 0.0 for key in DIAGNOSTIC_KEYS}
    within = True
    exact = True
    require(len(current) == len(reference), "Diagnostic group count mismatch")
    for left, right in zip(current, reference):
        require(left["group_index"] == right["group_index"], "Diagnostic group identity mismatch")
        require(left["line_count"] == right["line_count"], "Support count changed")
        for key in DIAGNOSTIC_KEYS:
            a, b = np.asarray(left[key]), np.asarray(right[key])
            maximum[key] = max(maximum[key], float(np.max(np.abs(a - b))))
            exact &= np.array_equal(a, b)
            within &= np.allclose(a, b, rtol=DIAGNOSTIC_RTOL, atol=DIAGNOSTIC_ATOL)
    return {"max_abs_difference": maximum, "bit_exact": bool(exact),
            "within_tolerance": bool(within)}


def coordinate_checks(lines: np.ndarray, transform: np.ndarray, masks: np.ndarray,
                      reference: list[dict], records: list[dict], fit_groups: Callable) -> dict:
    """Distinguish equivalent descriptions from a changed homogeneous SVD metric."""
    canonical = lines @ transform
    order = rank_groups(reference)
    working_maps = {
        "working_scale_2": np.diag([2.0, 2.0, 1.0]),
        "working_translate": np.array([[1., 0., 320.], [0., 1., -180.], [0., 0., 1.]]),
        "working_affine": np.array([[1.25, .125, 64.], [-.25, .75, -32.], [0., 0., 1.]]),
    }
    equivalent = []
    for name, h in working_maps.items():
        # Points x'=H x imply row lines l'=l H^-1 and saved map T'=H T.
        described = (lines @ np.linalg.inv(h)) @ (h @ transform)
        groups = fit_diagnostics(described, masks, fit_groups)
        equivalent.append({"test": name, "point_map": h.tolist(),
                           "canonical_line_max_abs_difference": float(np.max(abs(described - canonical))),
                           **compare_diagnostics(groups, reference),
                           "rank_identical": rank_groups(groups) == order})
    scales = np.geomspace(1e-3, 1e3, len(lines)) * np.where(np.arange(len(lines)) % 2, -1., 1.)
    angle = np.pi / 6
    rotate = np.array([[np.cos(angle), -np.sin(angle), 0.],
                       [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    descriptions = (
        ("signed_homogeneous_row_scales", (lines * scales[:, None]) @ transform, masks),
        ("reverse_line_rows_and_masks", canonical[::-1], masks[:, ::-1]),
        ("orthogonal_normalised_frame_rotation_30deg", canonical @ rotate.T, masks),
    )
    for name, described, described_masks in descriptions:
        groups = fit_diagnostics(described, described_masks, fit_groups)
        equivalent.append({"test": name, **compare_diagnostics(groups, reference),
                           "rank_identical": rank_groups(groups) == order})
    metric_maps = {
        "normalised_translate_0.5_minus0.25": np.array([[1., 0., .5], [0., 1., -.25], [0., 0., 1.]]),
        "normalised_scale_2": np.diag([2., 2., 1.]),
        "normalised_scale_0.5": np.diag([.5, .5, 1.]),
    }
    sensitivity = []
    matrices = [(name, canonical @ np.linalg.inv(h), h.tolist()) for name, h in metric_maps.items()]
    matrices.append(("omit_saved_transform_use_working_pixels", lines, None))
    for name, described, h in matrices:
        groups = fit_diagnostics(described, masks, fit_groups)
        changed_order = rank_groups(groups)
        sensitivity.append({"test": name, "point_map": h, "rank_identical": changed_order == order,
                            "rank_order": changed_order, "groups": groups,
                            "budgets": [evaluate(changed_order, k, records) for k in BUDGETS]})
    return {"equivalent_descriptions": equivalent, "changed_metric_negative_controls": sensitivity}


def validate_inputs(case_id: str, paths: dict[str, Path], data: dict[str, dict]) -> None:
    e2, e3, saved = data["e2"], data["e3"], data["baseline"]
    require(all(d["case_id"] == case_id for d in data.values()), "Case identity mismatch")
    require(e2["controls_loaded"] is False, "Admission inputs read control geometry")
    require(e3["automatic_detection"] is False and e3["label_guided_evaluation"] is True,
            "Unexpected cached fit semantics")
    require(e2["working_size"] == e3["working_size"] == saved["working_size"] == [960, 540],
            "Working-size mismatch")
    require(md5(paths["e2"]) == e3["e2_record_md5"], "E3/E2 MD5 mismatch")
    digest = md5(paths["baseline"])
    require(digest == e3["saved_estimator_md5"] == e2["code_md5"]["saved_estimator"],
            "Frozen estimator MD5 mismatch")
    b, est = e2["arms"]["B"], saved["estimator"]
    require(b["direction_count"] == GROUP_COUNT and b["matcher_eligible"], "Unexpected B population")
    require(e3["sets"]["B"]["status"] == "fitted", "Missing cached B fits")
    require(b["representative_candidate_ids"] == est["retained_candidate_ids"] ==
            e3["sets"]["B"]["representative_candidate_ids"], "Representative identities changed")
    require(b["support_masks"] == est["retained_support_masks"], "B membership changed")
    require(b["points_working"] == est["points_working"] == e3["sets"]["B"]["points_working"],
            "B directions changed")
    masks = np.asarray(b["support_masks"])
    require(masks.dtype == np.dtype(bool), "Support masks must contain booleans")
    require(masks.sum(axis=1).tolist() == b["support_counts"] == e3["sets"]["B"]["support_counts"],
            "Support counts inconsistent")
    fits = e3["sets"]["B"]["fits"]
    expected = [[i, j] for i in range(GROUP_COUNT) for j in range(GROUP_COUNT) if i != j]
    require([r["groups"] for r in fits["records"]] == expected, "Wrong ordered-pair identities/order")
    require([r["pair_id"] for r in fits["records"]] == list(range(len(expected))), "Wrong pair IDs")
    require(fits["best_finite"] == best_record(fits["records"]), "Inconsistent cached best-finite fit")
    require(fits["best_converged"] == best_record(fits["records"], True), "Inconsistent converged best")
    require(fits["attempted"] == len(expected), "Cached attempted count mismatch")
    for status in ("failed", "converged"):
        require(fits[status] == sum(fit_status(r) == status for r in fits["records"]),
                f"Cached {status} count mismatch")


def analyse_case(root: Path, case_id: str, fit_groups: Callable) -> dict:
    paths = {"e2": root / RUN / "e2" / f"{case_id}.json.gz",
             "e3": root / RUN / "e3" / f"{case_id}.json.gz",
             "baseline": root / PREFIX / "frozen_views/baseline_directions" / f"{case_id}.json.gz"}
    data = {name: read_json(path) for name, path in paths.items()}
    validate_inputs(case_id, paths, data)
    e2, e3, est = data["e2"], data["e3"], data["baseline"]["estimator"]
    b = e2["arms"]["B"]
    lines = np.asarray(est["direction_lines"], dtype=float)
    transform = np.asarray(est["normalised_to_working"], dtype=float)
    masks = np.asarray(b["support_masks"], dtype=bool)
    groups = fit_diagnostics(lines @ transform, masks, fit_groups)
    cached_groups = e3["sets"]["B_svd"]["groups"]  # Diagnostics only, never B_svd fits/directions.
    replay = compare_diagnostics(groups, cached_groups)
    require(replay["within_tolerance"], f"SVD replay failed: {case_id}")
    for actual, cached in zip(groups, cached_groups):
        require(actual["support_line_ids"] == cached["support_line_ids"], "SVD support IDs changed")
    order = rank_groups(groups)
    require(order == rank_groups(cached_groups), "Replayed/cached admission ranking mismatch")
    for g in groups:
        g.update({"representative_candidate_id": b["representative_candidate_ids"][g["group_index"]],
                  "residual_rank_1based": order.index(g["group_index"]) + 1,
                  "original_point_working": b["points_working"][g["group_index"]]})
    differences = [{"groups": [left, right],
                    "difference": groups[right]["algebraic_rms"] - groups[left]["algebraic_rms"]}
                   for left, right in zip(order, order[1:])]
    records = e3["sets"]["B"]["fits"]["records"]
    budgets = [evaluate(order, k, records) for k in BUDGETS]
    coordinates = coordinate_checks(lines, transform, masks, groups, records, fit_groups)
    require(all(test["within_tolerance"] and test["rank_identical"]
                for test in coordinates["equivalent_descriptions"]), "Equivalent-description check failed")
    witnesses = []
    original_best = best_record(records)
    for result in budgets:
        if not result["baseline_best_finite_pair_retained"]:
            admitted = set(result["selected_groups_rank_order"])
            dropped_best = [g for g in original_best["groups"] if g not in admitted]
            witnesses.append({"budget": result["budget"], "discarded_baseline_best_record": original_best,
                              "discarded_best_groups": [groups[g] for g in dropped_best],
                              "retained_best_record": next(r for r in records
                                  if r["pair_id"] == result["best_finite"]["pair_id"])})
    return {"case_id": case_id, "control": e3["control"], "working_size": e3["working_size"],
            "sources": {name: {"path": str(path.relative_to(root)), "md5": md5(path)}
                        for name, path in paths.items()},
            "input_identity_and_crosslink_checks_passed": True, "svd_replay": replay,
            "normalised_to_working": transform.tolist(), "groups": groups, "rank_order": order,
            "minimum_adjacent_residual_gap": min(differences, key=lambda d: d["difference"]),
            "exact_ties": [d for d in differences if d["difference"] == 0],
            "numerical_ties_at_1e_minus_12": [d for d in differences if d["difference"] <= TIE_ATOL],
            "baseline_best_finite": brief(original_best),
            "baseline_best_converged": brief(best_record(records, True)),
            "nonconverged_or_failed_cached_records": [r for r in records if fit_status(r) != "converged"],
            "budgets": budgets, "coordinate_checks": coordinates, "regression_witnesses": witnesses}


def tables(cases: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summary, groups, aggregate = [], [], []
    for case in cases:
        for result in case["budgets"]:
            best, converged = result["best_finite"], result["best_converged"]
            summary.append({"case_id": case["case_id"], "visually_approved": case["control"]["visually_approved"],
                            "control_source": case["control"]["control_source"], "budget": result["budget"],
                            "ordered_pairs": result["ordered_pairs"], "finite": result["finite"],
                            "failed": result["failed"], "unconverged": result["unconverged"],
                            "converged": result["converged"], "best_finite_px": None if best is None else best["max_corner_working_px"],
                            "best_converged_px": None if converged is None else converged["max_corner_working_px"],
                            "best_finite_pair_id": None if best is None else best["pair_id"],
                            "best_finite_groups": None if best is None else best["groups"],
                            "baseline_best_finite_px": case["baseline_best_finite"]["max_corner_working_px"],
                            "baseline_best_pair_retained": result["baseline_best_finite_pair_retained"],
                            "best_finite_delta_px": result["best_finite_delta_px"],
                            "selected_groups_rank_order": result["selected_groups_rank_order"],
                            "dropped_groups": result["dropped_groups"]})
        for group in case["groups"]:
            groups.append({"case_id": case["case_id"], "group_index": group["group_index"],
                           "representative_candidate_id": group["representative_candidate_id"],
                           "residual_rank_1based": group["residual_rank_1based"], "support_count": group["line_count"],
                           **dict(zip(("sigma_1", "sigma_2", "sigma_3"), group["singular_values"])),
                           "normalised_nullspace_gap": group["normalised_nullspace_gap"],
                           "algebraic_rms": group["algebraic_rms"], "support_line_ids": group["support_line_ids"]})
    for k in BUDGETS:
        selected = [r for r in summary if r["budget"] == k]
        aggregate.append({"budget": k, "ordered_pairs_per_view": k * (k - 1),
                          "pair_reduction_fraction": 1.0 - k * (k - 1) / (GROUP_COUNT * (GROUP_COUNT - 1)),
                          "best_pairs_preserved_all": sum(r["baseline_best_pair_retained"] for r in selected),
                          "best_pairs_preserved_approved": sum(r["baseline_best_pair_retained"] for r in selected if r["visually_approved"]),
                          "best_pairs_preserved_manual": sum(r["baseline_best_pair_retained"] for r in selected if not r["visually_approved"]),
                          "case_count": len(selected), "max_delta_px": max(r["best_finite_delta_px"] for r in selected),
                          "failed": sum(r["failed"] for r in selected),
                          "unconverged": sum(r["unconverged"] for r in selected),
                          "converged": sum(r["converged"] for r in selected)})
    return summary, groups, aggregate


def run(root: Path, output: Path) -> dict:
    require(root.is_dir(), f"Input root does not exist: {root}")
    marker = root / "PINNED_REVISION.txt"
    if marker.exists():
        require(marker.read_text().splitlines() == [REPOSITORY, REVISION], "Wrong revision marker")
    require(not output.exists() or not any(output.iterdir()), f"Output directory is not empty: {output}")
    fit_groups = load_fit_groups(root)
    cases = [analyse_case(root, case, fit_groups) for case in CASES]
    summary, groups, aggregate = tables(cases)
    result = {"schema": "baseline-B-fixed-SVD-budget-screen/1", "repository": REPOSITORY,
              "expected_evidence_revision": REVISION,
              "input_revision_marker": REVISION if marker.exists() else None,
              "revision_independently_verified_online": False,
              "run": "direction_agreement_20260915_144900", "automatic_detection": False,
              "control_used_for_admission": False, "fits_rerun": False, "directions_replaced": False,
              "ranking": ["algebraic_rms ascending", "support count descending", "original group index ascending"],
              "normalisation": "direction_lines @ normalised_to_working, then unit 2D line normals",
              "budgets": list(BUDGETS), "group_count": GROUP_COUNT,
              "tolerances": {"diagnostic_atol": DIAGNOSTIC_ATOL, "diagnostic_rtol": DIAGNOSTIC_RTOL,
                             "pixel_comparison_atol": PIXEL_ATOL, "tie_reporting_atol": TIE_ATOL,
                             "tolerances_used_in_admission": False},
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "producer": {"path": str(SVD_SOURCE), "md5": md5(root / SVD_SOURCE),
                           "functions_loaded": ["svd_direction", "fit_groups"]},
              "analysis_script_md5": md5(Path(__file__)), "aggregate": aggregate, "cases": cases}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "results.json.gz", result)
    write_csv(output / "summary.csv.gz", summary)
    write_csv(output / "groups.csv.gz", groups)
    write_csv(output / "aggregate.csv.gz", aggregate)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Input-bundle root or pinned repo checkout")
    parser.add_argument("--output", type=Path, required=True, help="New or empty results directory")
    args = parser.parse_args()
    try:
        result = run(args.inputs.resolve(), args.output.resolve())
    except (OSError, ValueError, KeyError, AssertionError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Screen failed: {type(error).__name__}: {error}\n")
    for row in result["aggregate"]:
        print(f"K={row['budget']}: {row['best_pairs_preserved_all']}/{row['case_count']} best pairs; "
              f"{row['ordered_pairs_per_view']} pairs/view; max delta={row['max_delta_px']:.6f} px")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
