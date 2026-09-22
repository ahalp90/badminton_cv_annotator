"""Cross saved G0/G1 generation populations under fixed S0/S1 observations.

This is a small evidence wrapper around the frozen L2 scorer.  It deliberately
keeps the W5 source loader and G0 reconstruction separate from the historical
four-view L2 driver so that expanded results do not overwrite old evidence.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np

REPO = Path(os.environ.get("L2_REPO", Path(__file__).resolve().parents[5])).resolve()
COURT_DET_FIX = REPO / "scratch/court_det_fix"
OUTPUT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = COURT_DET_FIX / "worklog/remote_records_20260921/preserved_data"
G0_ROOT = REMOTE_ROOT / "automatic_axes_20260914/all_camera"
G1_ROOT = REMOTE_ROOT / "line_identity/runs/line_identity_20260915_222437/matcher/paint_observations"
S1_CASE_ROOT = REMOTE_ROOT / "line_identity/inputs/paint_observations/cases"
G1_STAGE_NAMES = ("results", "camera_first", "all_camera")
CAMERA_LIMIT = 0.1


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_runtime():
    l2 = load_module(
        COURT_DET_FIX / "next_steps_20260916/L2_scoring/run_l2_scoring.py",
        "expanded_l2_frozen",
    )
    l2.cv2.setNumThreads(1)
    sys.path.insert(0, str(COURT_DET_FIX / "w5_holistic"))
    import run_w5

    runtime = run_w5.load_runtime(COURT_DET_FIX)
    return l2, run_w5, runtime


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def frozen_frame_path(verifier, source: dict, provenance) -> Path:
    """Resolve legacy local frames first, then the recovered W5 packet frames."""
    for root in (COURT_DET_FIX, REMOTE_ROOT):
        path = verifier["frame_path"](root, source, provenance)
        if path.exists():
            return path
    raise FileNotFoundError(source["id"])


def prepare_context(verifier, case_id: str):
    """Use W5's context preparation with the recovered packet as a frame fallback."""
    import verifier as verifier_module

    original_frame_path = verifier_module.frame_path
    verifier_module.frame_path = lambda root, source, provenance: frozen_frame_path(
        verifier, source, provenance
    )
    try:
        return verifier["prepare_view"](COURT_DET_FIX, case_id)
    finally:
        verifier_module.frame_path = original_frame_path


def case_path(root: Path, case_id: str) -> Path:
    return root / f"{case_id}.json.gz"


def load_g0_entries(context, run_w5, runtime) -> tuple[list[dict], str, dict]:
    """Load direct G0 entries or replay global retention, then evaluate replayed gates."""
    verifier = runtime["verifier"]
    direct = COURT_DET_FIX / "frozen_views/baseline_generation" / f"{context.case_id}.json.gz"
    if direct.exists():
        record = run_w5.validate_generation_record(
            verifier["read_json_gz"](direct), context.case_id, "direct G0"
        )
        return record["entries"], f"direct:{verifier['relative_path'](direct, COURT_DET_FIX)}", record

    record_path = case_path(G0_ROOT, context.case_id)
    record = run_w5.validate_generation_record(
        verifier["read_json_gz"](record_path),
        context.case_id,
        "automatic G0 replay input",
        validate_entries=False,
    )
    shortlist = run_w5.reconstruct_generation_entries(
        record,
        context.native_size,
        runtime["select_pool"],
        run_w5.import_detector(),
    )
    saved_entries = {entry["candidate_id"]: entry for entry in record.get("entries", [])}
    reconstructed_ids = {entry["candidate_id"] for entry in shortlist}
    if len(reconstructed_ids) != len(shortlist):
        raise ValueError(f"{context.case_id}: reconstructed G0 IDs are not unique")
    for reconstructed in shortlist:
        saved = saved_entries.get(reconstructed["candidate_id"])
        if saved is None:
            continue
        if saved.get("pair_id") != reconstructed.get("pair_id"):
            raise ValueError(f"{context.case_id}: saved/reconstructed pair identity differs for {reconstructed['candidate_id']}")
        for key in ("corners_px", "homography_working"):
            if not np.allclose(
                np.asarray(saved[key], dtype=float),
                np.asarray(reconstructed[key], dtype=float),
                rtol=0,
                atol=1e-9,
            ):
                raise ValueError(f"{context.case_id}: saved/reconstructed {key} differs for {reconstructed['candidate_id']}")
    missing = [
        entry for entry in shortlist
        if entry["candidate_id"] not in saved_entries
        or "gates" not in saved_entries[entry["candidate_id"]]
        or "profile" not in saved_entries[entry["candidate_id"]]
    ]
    evaluated_by_id = {}
    run_automatic = run_w5.import_run_automatic()
    original_frame_path = run_automatic.frame_path
    run_automatic.frame_path = lambda source, _root: frozen_frame_path(
        verifier, source, context.provenance
    )
    try:
        if missing:
            evaluated = runtime["evaluate_pool"](
                context.source,
                missing,
                context.observations,
                context.size,
                context.segments,
                context.families,
                runtime["zone"],
                COURT_DET_FIX,
            )
            evaluated_by_id = {entry["candidate_id"]: entry for entry in evaluated}
    finally:
        run_automatic.frame_path = original_frame_path
    entries = []
    for entry in shortlist:
        candidate_id = entry["candidate_id"]
        if candidate_id in evaluated_by_id:
            entries.append(evaluated_by_id[candidate_id])
        elif candidate_id in saved_entries:
            entries.append(saved_entries[candidate_id])
        else:
            raise KeyError(f"{context.case_id}: no evaluated or saved G0 entry for {candidate_id}")
    run_w5.validate_population_entries(entries, context.case_id, "replayed G0")
    return entries, f"replayed:{verifier['relative_path'](record_path, COURT_DET_FIX)}", record


def load_g1_record(case_id: str, stage: str) -> tuple[dict, Path]:
    path = case_path(G1_ROOT / stage, case_id)
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json_gz(path), path


def eligible(entry: dict) -> bool:
    camera_error = entry.get("gates", {}).get("camera_error")
    return camera_error is not None and camera_error <= CAMERA_LIMIT and entry.get("profile", {}).get("score") is not None


def stage_summary(record: dict, stage: str, l2, control: np.ndarray | None, native_size, working_size) -> dict:
    entries = record.get("entries", [])
    camera_count = sum(
        entry.get("gates", {}).get("camera_error") is not None
        and entry["gates"]["camera_error"] <= CAMERA_LIMIT
        for entry in entries
    )
    eligible_count = sum(eligible(entry) for entry in entries)
    summary = {
        "stage": stage,
        "schema": record.get("schema"),
        "candidate_count": len(entries),
        "pooled_candidates": record.get("pooled_candidates"),
        "keep_per_pair": record.get("keep_per_pair"),
        "keep_global": record.get("keep_global"),
        "camera_eligible_count": camera_count,
        "eligible_count": eligible_count,
        "selection_stage": record.get("selection_stage"),
        "saved_line_winner_id": record.get("line_winner_id"),
        "saved_paint_winner_id": record.get("paint_winner_id"),
    }
    if control is not None:
        distances = [l2.corner_error(entry, control, native_size, working_size) for entry in entries]
        summary["nearest_all_distance_working_px"] = min(distances) if distances else None
    return summary


def winner_summary(entry: dict | None, kind: str, l2, control, native_size, working_size) -> dict | None:
    if entry is None:
        return None
    result = {
        "origin_key": entry["_origin_key"],
        "candidate_id": entry["candidate_id"],
        "entry_index": entry["_origin_index"],
        "stripe_exclusive_score": entry["stripe"]["exclusive"]["score"],
        "profile_score": entry["profile"].get("score"),
    }
    if control is not None:
        result["distance_working_px"] = l2.corner_error(entry, control, native_size, working_size)
    return result


def cell_summary(entries: list[dict], population: str, scorer: str, l2, control, native_size, working_size) -> dict:
    line = l2.winner_entry(entries, "line")
    paint = l2.winner_entry(entries, "paint")
    eligible_entries = [entry for entry in entries if eligible(entry)]
    result = {
        "population": population,
        "scorer": scorer,
        "candidate_count": len(entries),
        "camera_eligible_count": sum(
            entry.get("gates", {}).get("camera_error") is not None
            and entry["gates"]["camera_error"] <= CAMERA_LIMIT
            for entry in entries
        ),
        "eligible_count": len(eligible_entries),
        "line": winner_summary(line, "line", l2, control, native_size, working_size),
        "paint": winner_summary(paint, "paint", l2, control, native_size, working_size),
    }
    if control is not None:
        result["nearest_all_distance_working_px"] = min(
            (l2.corner_error(entry, control, native_size, working_size) for entry in entries),
            default=None,
        )
        result["nearest_camera_distance_working_px"] = min(
            (
                l2.corner_error(entry, control, native_size, working_size)
                for entry in entries
                if entry.get("gates", {}).get("camera_error") is not None
                and entry["gates"]["camera_error"] <= CAMERA_LIMIT
            ),
            default=None,
        )
    return result


def compact_entry(
    entry: dict,
    stripe_s0: dict,
    stripe_s1: dict | None,
    *,
    saved_stripe: dict | None = None,
) -> dict:
    """Keep the geometry/gates/profile contract and both crossed scores."""
    return {
        "candidate_id": entry["candidate_id"],
        "pair_id": entry.get("pair_id"),
        "axis_ids": entry.get("axis_ids"),
        "rotated_180": entry.get("rotated_180"),
        "axis_score": entry.get("axis_score"),
        "shortlist_score": entry.get("shortlist_score"),
        "corners_px": entry["corners_px"],
        "homography_working": entry["homography_working"],
        "gates": entry["gates"],
        "profile": entry["profile"],
        "stripe": saved_stripe if saved_stripe is not None else stripe_s0,
        "stripe_s0": stripe_s0,
        "stripe_s1": stripe_s1,
    }


def load_control(l2, case_id: str) -> np.ndarray | None:
    try:
        return np.asarray(l2.load_control(case_id)["corners_working_px"], dtype=float)
    except FileNotFoundError:
        return None


def reference_lookup(case_id: str, origins: list[str]) -> dict[str, dict]:
    path = COURT_DET_FIX / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_33/reference_diagnostics.json.gz"
    if not path.exists():
        return {}
    record = read_json_gz(path).get(case_id, {}).get("candidates", {})
    return {origin: record[origin].get("frozen_case_reference") for origin in origins if origin in record}


def run_case(case_id: str, l2, run_w5, runtime) -> dict:
    started = monotonic()
    verifier = runtime["verifier"]
    context = prepare_context(verifier, case_id)
    g0_entries, g0_source, g0_record = load_g0_entries(context, run_w5, runtime)
    g1_record, g1_path = load_g1_record(case_id, "results")
    run_w5.validate_generation_record(g1_record, case_id, "saved G1", expected_stage="results")
    g1_entries = g1_record["entries"]
    source_s1 = None
    s1_path = case_path(S1_CASE_ROOT, case_id)
    if s1_path.exists():
        source_s1 = read_json_gz(s1_path)

    observations0, weights0, size0, _, _ = l2.scorer_inputs(context.source)
    observations1 = weights1 = size1 = None
    if source_s1 is not None:
        observations1, weights1, size1, _, _ = l2.scorer_inputs(source_s1)
        if tuple(size1) != tuple(size0):
            raise ValueError(f"{case_id}: S0/S1 working sizes differ: {size0} vs {size1}")

    cache_s0: dict[bytes, dict] = {}
    scored_g0_s0 = l2.score_population(g0_entries, "G0", observations0, weights0, size0, cache_s0)
    scored_g1_s0 = l2.score_population(g1_entries, "G1", observations0, weights0, size0, cache_s0)
    scored_g0_s1 = scored_g1_s1 = None
    if source_s1 is not None:
        cache_s1: dict[bytes, dict] = {}
        scored_g0_s1 = l2.score_population(g0_entries, "G0", observations1, weights1, size1, cache_s1)
        scored_g1_s1 = l2.score_population(g1_entries, "G1", observations1, weights1, size1, cache_s1)

    control = load_control(l2, case_id)
    populations_s0 = {
        "G0": scored_g0_s0,
        "G1": scored_g1_s0,
        "U": scored_g0_s0 + scored_g1_s0,
    }
    cells = {
        f"{population},S0": cell_summary(entries, population, "S0", l2, control, context.native_size, size0)
        for population, entries in populations_s0.items()
    }
    populations_s1 = {}
    if scored_g0_s1 is not None and scored_g1_s1 is not None:
        populations_s1 = {
            "G0": scored_g0_s1,
            "G1": scored_g1_s1,
            "U": scored_g0_s1 + scored_g1_s1,
        }
        cells.update({
            f"{population},S1": cell_summary(entries, population, "S1", l2, control, context.native_size, size1)
            for population, entries in populations_s1.items()
        })

    g0_all_camera_path = case_path(G0_ROOT, case_id)
    g0_all_camera_record = read_json_gz(g0_all_camera_path)
    stage_records = {"G0_all_camera_saved": (g0_all_camera_record, "all_camera")}
    if g0_source.startswith("direct:"):
        stage_records["G0_generation_direct"] = (g0_record, "generation")
    else:
        stage_records["G0_generation_replayed"] = (
            {"entries": g0_entries, "pooled_candidates": len(g0_entries), "keep_global": len(g0_entries)},
            "generation_replayed",
        )
    for stage in G1_STAGE_NAMES:
        try:
            g1_stage_record, _ = load_g1_record(case_id, stage)
            stage_records[f"G1_{stage}"] = (g1_stage_record, stage)
        except FileNotFoundError:
            continue
    saved_stages = {
        name: stage_summary(record, stage, l2, control, context.native_size, size0)
        for name, (record, stage) in stage_records.items()
    }

    winners = []
    for cell in cells.values():
        for kind in ("line", "paint"):
            winner = cell[kind]
            if winner is not None:
                winners.append(winner["origin_key"])
    population_payload = {
        "case_id": case_id,
        "G0": [
            compact_entry(s0, s0["stripe"], s1["stripe"] if s1 is not None else None)
            for s0, s1 in zip(
                scored_g0_s0, scored_g0_s1 or [None] * len(scored_g0_s0), strict=True
            )
        ],
        "G1": [
            compact_entry(
                original,
                s0["stripe"],
                s1["stripe"] if s1 is not None else None,
                saved_stripe=original["stripe"],
            )
            for original, s0, s1 in zip(
                g1_entries, scored_g1_s0, scored_g1_s1 or [None] * len(scored_g1_s0), strict=True
            )
        ],
        "g0_source": g0_source,
    }
    references = reference_lookup(case_id, winners)
    return {
        "schema": "expanded-l2-crossed-evaluation/1",
        "case_id": case_id,
        "native_size": list(context.native_size),
        "working_size": list(size0),
        "observation_sets": {
            "S0": {"available": True, "source": "W5 frozen source pack", "segment_count": len(context.source["segments_px"])},
            "S1": {"available": source_s1 is not None, "source": verifier["relative_path"](s1_path, COURT_DET_FIX) if source_s1 is not None else None, "segment_count": len(source_s1["segments_px"]) if source_s1 is not None else None},
        },
        "population_sources": {"G0": g0_source, "G1": verifier["relative_path"](g1_path, COURT_DET_FIX)},
        "population_counts": {"G0_generation": len(g0_entries), "G1_generation": len(g1_entries)},
        "saved_stages": saved_stages,
        "cells": cells,
        "reference_errors_diagnostic": references,
        "population_checkpoint": f"populations/{case_id}.json.gz",
        "elapsed_s": monotonic() - started,
        "status": "complete",
        "population_payload": population_payload,
    }


def write_json_gz(path: Path, value: Any, verifier) -> None:
    verifier["write_json_gz"](path, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="case IDs; default is all 27")
    parser.add_argument("--force", action="store_true", help="recompute existing case checkpoints")
    parser.add_argument("--no-summary", action="store_true", help="write case checkpoints without shared summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    l2, run_w5, runtime = load_runtime()
    verifier = runtime["verifier"]
    requested = args.cases or list(verifier["ALL_CASE_IDS"])
    unknown = sorted(set(requested) - set(verifier["ALL_CASE_IDS"]))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "populations").mkdir(exist_ok=True)
    summary = {"schema": "expanded-l2-crossed-evaluation-summary/1", "cases": []}
    for case_id in requested:
        case_output = OUTPUT_ROOT / "cases" / f"{case_id}.json.gz"
        population_output = OUTPUT_ROOT / "populations" / f"{case_id}.json.gz"
        if case_output.exists() and population_output.exists() and not args.force:
            print(case_id, "skip existing", flush=True)
            summary["cases"].append({"case_id": case_id, "status": "existing"})
            continue
        result = run_case(case_id, l2, run_w5, runtime)
        population_payload = result.pop("population_payload")
        write_json_gz(population_output, population_payload, verifier)
        write_json_gz(case_output, result, verifier)
        summary["cases"].append({"case_id": case_id, "status": result["status"], "elapsed_s": result["elapsed_s"]})
        print(case_id, f"{result['elapsed_s']:.2f}s", flush=True)
    if not args.no_summary:
        write_json_gz(OUTPUT_ROOT / "summary.json.gz", summary, verifier)


if __name__ == "__main__":
    main()
