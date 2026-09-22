"""Summarise completed expanded G0/G1 checkpoints without rerunning L2.

The expanded runner writes one case checkpoint and one geometry population
checkpoint per case.  This module only reads those checkpoints, keeps the
runner's manual-control cells intact, and adds separately labelled W5 frozen-
reference diagnostics.  It does not regenerate candidates or recompute
scores.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(os.environ.get("L2_REPO", Path(__file__).resolve().parents[5])).resolve()
COURT_DET_FIX = Path(
    os.environ.get("L2_COURT_DET_FIX", REPO / "scratch/court_det_fix")
).resolve()
OUTPUT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = COURT_DET_FIX / "worklog/remote_records_20260921/preserved_data"
CASE_ROOT = OUTPUT_ROOT / "cases"
POPULATION_ROOT = OUTPUT_ROOT / "populations"
CAMERA_LIMIT = 0.1
POPULATIONS = ("G0", "G1", "U")
SCORERS = ("S0", "S1")
KINDS = ("line", "paint")


def read_checkpoint(path: Path, verifier: dict[str, Any]) -> dict[str, Any]:
    return verifier["read_json_gz"](path)


def case_checkpoint_path(case_id: str) -> Path:
    return CASE_ROOT / f"{case_id}.json.gz"


def population_checkpoint_path(case_id: str) -> Path:
    return POPULATION_ROOT / f"{case_id}.json.gz"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def camera_eligible(entry: dict[str, Any]) -> bool:
    camera_error = entry.get("gates", {}).get("camera_error")
    return camera_error is not None and float(camera_error) <= CAMERA_LIMIT


def useful_eligible(entry: dict[str, Any]) -> bool:
    return camera_eligible(entry) and entry.get("profile", {}).get("score") is not None


def entry_origin(population: str, entry: dict[str, Any], entry_index: int) -> str:
    candidate_id = entry.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError(f"{population}[{entry_index}]: missing candidate_id")
    return f"{population}:{candidate_id}"


def population_entries(population_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for population in ("G0", "G1"):
        entries = population_payload.get(population)
        if not isinstance(entries, list):
            raise TypeError(f"population checkpoint has no {population} list")
        seen: set[str] = set()
        copied: list[dict[str, Any]] = []
        for entry_index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise TypeError(f"{population}[{entry_index}]: entry is not an object")
            origin_key = entry_origin(population, entry, entry_index)
            if origin_key in seen:
                raise ValueError(f"duplicate geometry origin {origin_key}")
            seen.add(origin_key)
            if "corners_px" not in entry or "homography_working" not in entry:
                raise ValueError(f"{origin_key}: saved geometry is incomplete")
            copied_entry = dict(entry)
            copied_entry["origin_key"] = origin_key
            copied_entry["_origin_index"] = entry_index
            copied.append(copied_entry)
        result[population] = copied
    result["U"] = result["G0"] + result["G1"]
    return result


def native_to_working(
    points: np.ndarray, native_size: tuple[int, int], working_size: tuple[int, int]
) -> np.ndarray:
    native_scale = np.asarray(native_size, dtype=float) / np.asarray(working_size, dtype=float)
    return np.asarray(points, dtype=float) / native_scale


def frozen_reference_error(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    native_size: tuple[int, int],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
) -> dict[str, Any]:
    candidate_corners = np.asarray(candidate["corners_px"], dtype=float)
    reference_corners = np.asarray(reference["corners_px"], dtype=float)
    if candidate_corners.shape != (4, 2):
        raise ValueError(f"{candidate['origin_key']}: corners_px has shape {candidate_corners.shape}")
    if reference_corners.shape != (4, 2):
        raise ValueError(f"frozen reference: corners_px has shape {reference_corners.shape}")
    if not np.isfinite(candidate_corners).all() or not np.isfinite(reference_corners).all():
        raise ValueError(f"{candidate['origin_key']}: non-finite frozen-reference geometry")
    candidate_working = native_to_working(candidate_corners, native_size, working_size)
    reference_working = native_to_working(reference_corners, native_size, working_size)
    error = verifier["reference_corner_error"](candidate_working, reference_working)
    native_scale = np.asarray(native_size, dtype=float) / np.asarray(working_size, dtype=float)
    direct = np.linalg.norm(candidate_working - reference_working, axis=1)
    rotated = np.linalg.norm(candidate_working - reference_working[[2, 3, 0, 1]], axis=1)
    return {
        "maximum_working_px": float(error["maximum"]),
        "direct_maximum_working_px": float(direct.max()),
        "rotated_180_maximum_working_px": float(rotated.max()),
        "relabelled_180": bool(error["relabelled_180"]),
        "per_corner_working_px": error["per_corner"],
        "native_to_working_scale": native_scale.tolist(),
    }


def reference_metadata(
    reference: dict[str, Any], native_size: tuple[int, int], working_size: tuple[int, int]
) -> dict[str, Any]:
    corners = reference.get("corners_px")
    return {
        "available": bool(corners),
        "diagnostic_only": True,
        "unverified": True,
        "interpretation": "Unverified frozen-reference geometry diagnostic; not an accuracy estimate.",
        "reference_status": reference.get("reference_status"),
        "corner_source": reference.get("corner_source"),
        "image_kind": reference.get("image_kind"),
        "corner_visible": reference.get("corner_visible"),
        "native_to_working_scale": (
            np.asarray(native_size, dtype=float) / np.asarray(working_size, dtype=float)
        ).tolist(),
    }


def geometry_witness(
    entry: dict[str, Any] | None,
    reference: dict[str, Any],
    native_size: tuple[int, int],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
) -> dict[str, Any] | None:
    if entry is None:
        return None
    return {
        "origin_key": entry["origin_key"],
        "candidate_id": entry["candidate_id"],
        "entry_index": entry["_origin_index"],
        "frozen_reference_error": (
            frozen_reference_error(entry, reference, native_size, working_size, verifier)
            if reference.get("corners_px")
            else None
        ),
    }


def nearest_witness(
    entries: list[dict[str, Any]],
    reference: dict[str, Any],
    native_size: tuple[int, int],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
    camera_only: bool,
) -> dict[str, Any] | None:
    if not reference.get("corners_px"):
        return None
    available = [entry for entry in entries if not camera_only or camera_eligible(entry)]
    if not available:
        return None
    ranked = [
        (
            frozen_reference_error(entry, reference, native_size, working_size, verifier)[
                "maximum_working_px"
            ],
            entry_index,
            entry,
        )
        for entry_index, entry in enumerate(available)
    ]
    _, _, nearest = min(ranked, key=lambda item: (item[0], item[1]))
    return geometry_witness(nearest, reference, native_size, working_size, verifier)


def pool_summary(
    population: str,
    entries: list[dict[str, Any]],
    reference: dict[str, Any],
    native_size: tuple[int, int],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
) -> dict[str, Any]:
    camera_count = sum(camera_eligible(entry) for entry in entries)
    useful_count = sum(useful_eligible(entry) for entry in entries)
    return {
        "population": population,
        "candidate_count": len(entries),
        "camera_eligible_count": camera_count,
        "eligible_count": useful_count,
        "zero_camera_eligible": camera_count == 0,
        "zero_eligible": useful_count == 0,
        "frozen_reference_diagnostic": {
            "reference_available": bool(reference.get("corners_px")),
            "nearest_all": nearest_witness(
                entries, reference, native_size, working_size, verifier, camera_only=False
            ),
            "nearest_camera_eligible": nearest_witness(
                entries, reference, native_size, working_size, verifier, camera_only=True
            ),
        },
    }


def selected_entry(
    cell: dict[str, Any],
    kind: str,
    population_entries_by_origin: dict[str, dict[str, Any]],
    case_id: str,
    cell_key: str,
) -> dict[str, Any] | None:
    selection = cell.get(kind)
    if selection is None:
        return None
    origin_key = selection.get("origin_key")
    if not isinstance(origin_key, str) or origin_key not in population_entries_by_origin:
        raise KeyError(f"{case_id} {cell_key} {kind}: geometry ID not found: {origin_key!r}")
    entry = population_entries_by_origin[origin_key]
    if selection.get("candidate_id") != entry["candidate_id"]:
        raise ValueError(f"{case_id} {cell_key} {kind}: candidate ID disagrees with origin key")
    if selection.get("entry_index") != entry["_origin_index"]:
        raise ValueError(f"{case_id} {cell_key} {kind}: entry index disagrees with saved population")
    return entry


def selection_diagnostics(
    case_id: str,
    cells: dict[str, Any],
    entries_by_population: dict[str, list[dict[str, Any]]],
    reference: dict[str, Any],
    native_size: tuple[int, int],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
    s1_available: bool,
) -> tuple[dict[str, Any], dict[str, list[str]], list[dict[str, Any]]]:
    by_origin = {
        entry["origin_key"]: entry
        for entries in entries_by_population.values()
        for entry in entries
    }
    diagnostics: dict[str, Any] = {}
    selected_roles: dict[str, list[str]] = {}
    table: list[dict[str, Any]] = []
    for cell_key, cell in cells.items():
        if not isinstance(cell, dict):
            raise TypeError(f"{case_id} {cell_key}: cell is not an object")
        population, scorer = cell_key.split(",", maxsplit=1)
        if population not in POPULATIONS or scorer not in SCORERS:
            raise ValueError(f"{case_id}: unexpected cell key {cell_key!r}")
        diagnostics[cell_key] = {
            "population": population,
            "scorer": scorer,
            "observation_available": scorer == "S0" or s1_available,
            "checkpoint_cell": cell,
            "selections": {},
        }
        for kind in KINDS:
            entry = selected_entry(cell, kind, by_origin, case_id, cell_key)
            if entry is None:
                diagnostics[cell_key]["selections"][kind] = None
                continue
            selected_roles.setdefault(entry["origin_key"], []).append(f"{cell_key}:{kind}")
            winner = dict(cell[kind])
            winner["frozen_reference_error"] = (
                frozen_reference_error(entry, reference, native_size, working_size, verifier)
                if reference.get("corners_px")
                else None
            )
            diagnostics[cell_key]["selections"][kind] = winner
            table.append({
                "case_id": case_id,
                "population": population,
                "scorer": scorer,
                "kind": kind,
                "origin_key": entry["origin_key"],
                "candidate_id": entry["candidate_id"],
                "entry_index": entry["_origin_index"],
                "frozen_reference_error": winner["frozen_reference_error"],
                "checkpoint_nearest_all_distance_working_px": cell.get(
                    "nearest_all_distance_working_px"
                ),
                "checkpoint_nearest_camera_distance_working_px": cell.get(
                    "nearest_camera_distance_working_px"
                ),
            })
    return diagnostics, selected_roles, table


def load_frame(
    source: dict[str, Any], provenance: object, verifier: dict[str, Any]
) -> tuple[np.ndarray, Path]:
    for root in (COURT_DET_FIX, REMOTE_ROOT):
        path = verifier["frame_path"](root, source, provenance)
        if not path.exists():
            continue
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(path)
        expected = (source["dimensions"]["height"], source["dimensions"]["width"])
        if frame.shape[:2] != expected:
            raise ValueError(f"{source['id']}: frame shape {frame.shape[:2]} != {expected}")
        return frame, path
    raise FileNotFoundError(source["id"])


def render_selected(
    case_id: str,
    selected_roles: dict[str, list[str]],
    entries_by_origin: dict[str, dict[str, Any]],
    source: dict[str, Any],
    provenance: object,
    reference: dict[str, Any],
    working_size: tuple[int, int],
    verifier: dict[str, Any],
    render_prediction: Any,
) -> dict[str, Any]:
    frame, frame_path = load_frame(source, provenance, verifier)
    gallery_root = OUTPUT_ROOT / "gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    context = {
        "id": case_id,
        "dimensions": source["dimensions"],
        "working_dimensions": list(working_size),
    }
    rendered: dict[str, Any] = {}
    for origin_key in sorted(selected_roles):
        entry = entries_by_origin[origin_key]
        stem = gallery_root / f"{safe_name(case_id)}__{safe_name(origin_key)}"
        links = render_prediction(
            frame,
            entry,
            context,
            selected_roles[origin_key],
            stem,
            reference,
        )
        rendered[origin_key] = {
            "candidate_id": entry["candidate_id"],
            "roles": selected_roles[origin_key],
            "files": links,
        }
    try:
        rendered_frame_path = str(frame_path.relative_to(COURT_DET_FIX))
    except ValueError:
        rendered_frame_path = str(frame_path)
    return {"frame_path": rendered_frame_path, "rendered": rendered}


def validate_checkpoint_counts(
    case_id: str,
    cells: dict[str, Any],
    pool_rows: dict[str, dict[str, Any]],
) -> None:
    for cell_key, cell in cells.items():
        population = cell_key.split(",", maxsplit=1)[0]
        if population not in pool_rows:
            raise ValueError(f"{case_id} {cell_key}: unknown population")
        summary = pool_rows[population]
        for field in ("candidate_count", "camera_eligible_count", "eligible_count"):
            if cell.get(field) != summary[field]:
                raise ValueError(
                    f"{case_id} {cell_key}: checkpoint {field}={cell.get(field)!r} "
                    f"does not match population={summary[field]!r}"
                )


def summarise_case(
    case_id: str,
    checkpoint: dict[str, Any],
    population_payload: dict[str, Any],
    verifier: dict[str, Any],
    load_reference: Any,
    render_prediction: Any | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if checkpoint.get("case_id") != case_id or population_payload.get("case_id") != case_id:
        raise ValueError(f"{case_id}: checkpoint case IDs do not match")
    if checkpoint.get("status") != "complete":
        raise ValueError(f"{case_id}: checkpoint status is {checkpoint.get('status')!r}")
    source = verifier["load_source"](COURT_DET_FIX, case_id)
    provenance = verifier["load_case_provenance"](COURT_DET_FIX, case_id)
    native_size = tuple(int(value) for value in checkpoint["native_size"])
    working_size = tuple(int(value) for value in checkpoint["working_size"])
    source_size = (int(source["dimensions"]["width"]), int(source["dimensions"]["height"]))
    if native_size != source_size:
        raise ValueError(f"{case_id}: checkpoint native size {native_size} != source {source_size}")
    s1_available = bool(checkpoint.get("observation_sets", {}).get("S1", {}).get("available"))
    reference = load_reference(COURT_DET_FIX, case_id, verifier)
    entries_by_population = population_entries(population_payload)
    pool_rows = {
        population: pool_summary(
            population,
            entries_by_population[population],
            reference,
            native_size,
            working_size,
            verifier,
        )
        for population in POPULATIONS
    }
    cells = checkpoint.get("cells")
    if not isinstance(cells, dict):
        raise TypeError(f"{case_id}: checkpoint has no cells object")
    validate_checkpoint_counts(case_id, cells, pool_rows)
    manual_control_present = any(
        "nearest_all_distance_working_px" in cell for cell in cells.values()
    )
    selected, selected_roles, selection_table = selection_diagnostics(
        case_id,
        cells,
        entries_by_population,
        reference,
        native_size,
        working_size,
        verifier,
        s1_available,
    )
    entries_by_origin = {
        entry["origin_key"]: entry
        for entries in entries_by_population.values()
        for entry in entries
    }
    gallery = {"rendered": {}, "frame_path": None}
    if render_prediction is not None:
        gallery = render_selected(
            case_id,
            selected_roles,
            entries_by_origin,
            source,
            provenance,
            reference,
            working_size,
            verifier,
            render_prediction,
        )
    s1_cells = sorted(key for key in cells if key.endswith(",S1"))
    case_result = {
        "case_id": case_id,
        "label": verifier["CASE_LABELS"].get(case_id, case_id),
        "status": "available",
        "observation_sets": checkpoint.get("observation_sets", {}),
        "s1_available": s1_available,
        "missing_s1": not s1_available or not s1_cells,
        "primary_diagnostic": (
            "checkpoint_manual_control_cells"
            if manual_control_present
            else "frozen_reference_diagnostic_only"
        ),
        "manual_control_present": manual_control_present,
        "reference": reference_metadata(reference, native_size, working_size),
        "population_sources": checkpoint.get("population_sources", {}),
        "population_counts": checkpoint.get("population_counts", {}),
        "pool_diagnostics": pool_rows,
        "cells": cells,
        "selection_diagnostics": selected,
        "selection_table": selection_table,
        "gallery": gallery,
    }
    return case_result, {
        "case_id": case_id,
        "label": case_result["label"],
        "s1_available": s1_available,
        "missing_s1": case_result["missing_s1"],
        "primary_diagnostic": case_result["primary_diagnostic"],
        "manual_control_present": manual_control_present,
        "G0_candidate_count": pool_rows["G0"]["candidate_count"],
        "G0_camera_eligible_count": pool_rows["G0"]["camera_eligible_count"],
        "G0_eligible_count": pool_rows["G0"]["eligible_count"],
        "G1_candidate_count": pool_rows["G1"]["candidate_count"],
        "G1_camera_eligible_count": pool_rows["G1"]["camera_eligible_count"],
        "G1_eligible_count": pool_rows["G1"]["eligible_count"],
        "U_candidate_count": pool_rows["U"]["candidate_count"],
        "U_camera_eligible_count": pool_rows["U"]["camera_eligible_count"],
        "U_eligible_count": pool_rows["U"]["eligible_count"],
        "selected_origin_count": len(selected_roles),
    }


def expected_cases(verifier: dict[str, Any], requested: list[str] | None) -> list[str]:
    available = list(verifier["ALL_CASE_IDS"])
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError("case IDs must be unique")
    return requested


def write_gallery_index(case_results: list[dict[str, Any]]) -> None:
    lines = [
        "# Expanded G0/G1 selected-origin gallery",
        "",
        "Each source-qualified selected origin is rendered once per case. Roles show every crossed cell that selected it.",
        "",
        "| case | origin | roles | crop | full | reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in case_results:
        for origin, rendered in sorted(case["gallery"]["rendered"].items()):
            links = rendered["files"]
            crop = next((link for link in links if link.endswith("__crop.png")), None)
            full = next((link for link in links if link.endswith("__full.png")), None)
            reference = next((link for link in links if link.endswith("__reference.png")), None)
            link = lambda name: f"[{name}]({name})" if name else "—"
            lines.append(
                f"| {case['case_id']} | {origin} | {', '.join(rendered['roles'])} | "
                f"{link(crop)} | {link(full)} | {link(reference)} |"
            )
    (OUTPUT_ROOT / "gallery" / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(
    requested: list[str] | None,
    verifier: dict[str, Any],
    load_reference: Any,
    render_prediction: Any | None,
) -> dict[str, Any]:
    render = render_prediction is not None
    case_ids = expected_cases(verifier, requested)
    case_results: list[dict[str, Any]] = []
    case_table: list[dict[str, Any]] = []
    missing_cases: list[str] = []
    missing_details: dict[str, list[str]] = {}
    selection_table: list[dict[str, Any]] = []
    population_table: list[dict[str, Any]] = []
    for case_id in case_ids:
        case_path = case_checkpoint_path(case_id)
        population_path = population_checkpoint_path(case_id)
        missing = []
        if not case_path.exists():
            missing.append("case_checkpoint")
        if not population_path.exists():
            missing.append("population_checkpoint")
        if missing:
            missing_cases.append(case_id)
            missing_details[case_id] = missing
            continue
        checkpoint = read_checkpoint(case_path, verifier)
        population_payload = read_checkpoint(population_path, verifier)
        case_result, overview = summarise_case(
            case_id,
            checkpoint,
            population_payload,
            verifier,
            load_reference,
            render_prediction,
        )
        case_results.append(case_result)
        case_table.append(overview)
        selection_table.extend(case_result["selection_table"])
        for population in POPULATIONS:
            row = case_result["pool_diagnostics"][population]
            population_table.append({
                "case_id": case_id,
                "population": population,
                "candidate_count": row["candidate_count"],
                "camera_eligible_count": row["camera_eligible_count"],
                "eligible_count": row["eligible_count"],
                "zero_camera_eligible": row["zero_camera_eligible"],
                "zero_eligible": row["zero_eligible"],
            })
    population_counts: dict[str, dict[str, int]] = {}
    for population in POPULATIONS:
        rows = [row for row in population_table if row["population"] == population]
        population_counts[population] = {
            "case_count": len(rows),
            "total_candidate_count": sum(row["candidate_count"] for row in rows),
            "zero_camera_eligible_case_count": sum(row["zero_camera_eligible"] for row in rows),
            "zero_eligible_case_count": sum(row["zero_eligible"] for row in rows),
        }
    summary = {
        "schema": "expanded-l2-comparison-summary/1",
        "scope": {
            "expected_case_count": len(case_ids),
            "available_case_count": len(case_results),
            "missing_case_count": len(missing_cases),
            "requested_cases": case_ids,
            "available_cases": [case["case_id"] for case in case_results],
            "missing_cases": missing_cases,
            "missing_case_details": missing_details,
            "development_set": True,
            "frozen_reference_errors_are_diagnostic_only": True,
            "accuracy_claim": "No all-case accuracy estimate is produced.",
        },
        "counts": {
            "s1_available_case_count": sum(case["s1_available"] for case in case_table),
            "missing_s1_case_count": sum(case["missing_s1"] for case in case_table),
            "population_counts": population_counts,
        },
        "case_table": case_table,
        "population_table": population_table,
        "selection_table": selection_table,
        "cases": case_results,
    }
    if render:
        write_gallery_index(case_results)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="case IDs; default is every expected case")
    parser.add_argument("--no-render", action="store_true", help="skip selected-geometry gallery rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path[:0] = [str(REPO / "src"), str(COURT_DET_FIX / "w5_holistic"), str(REPO)]
    import run_w5

    verifier = run_w5.load_verifier(COURT_DET_FIX)
    render_prediction = None
    if not args.no_render:
        from render_gallery import render_prediction as render_prediction_function

        render_prediction = render_prediction_function
    summary = build_summary(
        args.cases,
        verifier,
        run_w5.load_reference,
        render_prediction,
    )
    verifier["write_json_gz"](OUTPUT_ROOT / "comparison_summary.json.gz", summary)
    print(
        f"summarised {summary['scope']['available_case_count']} available cases; "
        f"{summary['scope']['missing_case_count']} missing",
        flush=True,
    )


if __name__ == "__main__":
    main()
