"""Renew evidence on fixed-assignment refits while keeping all starting courts."""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from types import ModuleType

import cv2
import numpy as np

from scratch.court_det_fix.court_detector import junctions
from scratch.court_det_fix.court_detector import stripe_measurements as stripes
from scratch.court_det_fix.court_detector.geometry import CORNER_COURT_M
from scratch.court_det_fix.court_detector.image_sources import (
    CaseProvenance,
    load_frozen_case_provenance,
    require_same_image_boxes,
)
from scratch.court_det_fix.court_detector.line_observations import prepare_observations

from .run_assignment import (
    ACCURATE_PX,
    attach_metrics,
    frozen_entries,
    read_replay_bytes,
)
from .run_fixed_refit import MODELS
from .run_junction_selection import rank
from .run_junctions import (
    bytes_md5,
    provenance_binding,
    require_replay_pack,
    validate_stripe_provenance,
)

POOLS = ("starts", *MODELS)
SELECTORS = ("stripe_exclusive", "complete_agreements_first")
OUTPUT_SCHEMA = "renewed-fixed-refit-selection/2"


def eligible(evidence: dict) -> bool:
    return bool(evidence["eligible"] and evidence["scheme_eligible"]["original"])


def rank_pools(entries: list[dict]) -> dict:
    """Retain parents in each comparison without allowing one model into the other."""
    orders = {}
    for pool in POOLS:
        active = [entry for entry in entries if entry["eligible"] and entry["model"] in ("start", pool)]
        ranked = rank(active)
        orders[pool] = {scheme: ranked[scheme] for scheme in SELECTORS}
    return orders


def verify_starts(
    case: dict,
    frozen: dict,
    saved: dict,
    legacy: ModuleType,
    provenance: CaseProvenance,
) -> tuple[dict, dict]:
    """Verify all archived gates and net scores before expensive refinement scoring."""
    if provenance.case_id != case["id"]:
        raise ValueError(f"provenance case {provenance.case_id!r} does not match {case['id']!r}")
    require_same_image_boxes(provenance)
    prepared = legacy.prepare_case(case)
    old = {entry["id"]: entry for entry in saved["entries"]}
    renewed = {}
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        construction = case["candidates"][source["source_index"]]
        observed = legacy.evidence(np.asarray(source["corners_px"]), case, prepared, construction)
        previous = old[source["id"]]
        if eligible(observed) != previous["eligible"] or observed["net_score"] != previous["evidence"]["net_score"]:
            raise ValueError(f"Renewed starting evidence differs: {case['id']}/{source['id']}")
        renewed[source["id"]] = observed
    return prepared, renewed


def preflight_cases(
    inputs: dict,
    stripe_results: dict,
    provenance: Mapping[str, CaseProvenance],
    binding: dict[str, str],
) -> None:
    """Validate the stripe/replay join and every box source before any refit work."""
    validate_stripe_provenance(stripe_results, binding)
    input_ids = [case["id"] for case in inputs["cases"]]
    stripe_ids = [record["id"] for record in stripe_results["records"]]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError("Replay contains duplicate case IDs")
    if len(stripe_ids) != len(set(stripe_ids)):
        raise ValueError("Stripe results contain duplicate case IDs")
    if not set(stripe_ids).issubset(input_ids):
        unknown = sorted(set(stripe_ids) - set(input_ids))
        raise ValueError(f"Stripe results contain cases absent from the replay: {unknown!r}")
    for case_id in stripe_ids:
        try:
            case = provenance[case_id]
        except KeyError as error:
            raise ValueError(f"Stripe case {case_id!r} is absent from the validated provenance pack") from error
        require_same_image_boxes(case)


def run_case(
    case: dict,
    frozen: dict,
    fits: dict,
    legacy: ModuleType,
    prepared: dict,
    starting_evidence: dict,
    provenance: CaseProvenance,
) -> dict:
    """Remeasure changed geometries while reusing the verified parent control."""
    if provenance.case_id != case["id"]:
        raise ValueError(f"provenance case {provenance.case_id!r} does not match {case['id']!r}")
    require_same_image_boxes(provenance)
    started = perf_counter()
    size, scale = prepared["size"], prepared["native_scale"]
    observations = prepare_observations(prepared["segments"], size)
    weights = stripes.fragment_weights(observations)
    fitted = {entry["id"]: entry for entry in fits["entries"]}
    entries = []
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        construction = case["candidates"][source["source_index"]]
        assigned = source["stripe_evidence"]["stripe"]["assignments"]
        variants = [("start", source["corners_px"])]
        for model in MODELS:
            result = fitted[source["id"]]["fits"][model]
            if result["successful"]:
                variants.append((model, result["corners_px"]))
        for model, coordinates in variants:
            corners = np.asarray(coordinates)
            observed = (starting_evidence[source["id"]] if model == "start"
                        else legacy.evidence(corners, case, prepared, construction))
            is_eligible = eligible(observed)
            entry = {"id": f"{source['id']}/{model}", "parent_id": source["id"], "model": model,
                     "corners_px": coordinates, "eligible": is_eligible, "gate_evidence": observed}
            if is_eligible:
                homography = cv2.getPerspectiveTransform(CORNER_COURT_M, (corners / scale).astype(np.float32))
                measured = stripes.measure(homography, observations, size)
                stripe = stripes.score_model(measured, weights, 3, assigned)
                sites = junctions.measure(homography, observations, prepared["boxes"], size)
                complete = sum(site["usable"] and len(site["agreements"]) == 2 for site in sites["sites"])
                score = (3 * stripe["exclusive"]["score"] + observed["net_score"]) / 4
                if model == "start" and score != source["scores"]["stripe_exclusive"]:
                    raise ValueError(f"Renewed starting stripe score differs: {case['id']}/{source['id']}")
                entry.update({"stripe_score": score, "stripe": stripe, "junctions": sites,
                              "complete_agreements": complete, "disagreements": sites["disagreements"],
                              "usable_sites": sites["usable_sites"]})
            entries.append(entry)
    return {"id": case["id"], "dimensions": case["dimensions"], "entries": entries,
            "orders": rank_pools(entries), "elapsed_seconds": perf_counter() - started}


def summarise(records: list[dict]) -> dict:
    accurate = {pool: dict.fromkeys(SELECTORS, 0) for pool in POOLS}
    available = dict.fromkeys(POOLS, 0)
    eligible_counts = dict.fromkeys(("start", *MODELS), 0)
    cases = []
    for record in records:
        entries = {entry["id"]: entry for entry in record["entries"]}
        for entry in entries.values():
            eligible_counts[entry["model"]] += entry["eligible"]
        picks = {}
        for pool in POOLS:
            active = [entry for entry in entries.values() if entry["eligible"] and entry["model"] in ("start", pool)]
            available[pool] += any(entry["metrics"]["corner_max_error_px"] <= ACCURATE_PX for entry in active)
            picks[pool] = {}
            for scheme in SELECTORS:
                order = record["orders"][pool][scheme]
                winner = entries[order[0]] if order else None
                picks[pool][scheme] = None if winner is None else {"id": winner["id"], **winner["metrics"]}
                if winner is not None:
                    accurate[pool][scheme] += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
        cases.append({"id": record["id"], "picks": picks})
    return {"frames": len(records), "accurate_picks": accurate, "useful_eligible_pools": available,
            "eligible_geometries": eligible_counts, "cases": cases}


def python_artefact_md5s(directory: Path) -> dict[str, str]:
    """Hash every Python source that can participate in the archived import tree."""
    return {
        str(path.relative_to(directory)): bytes_md5(path.read_bytes())
        for path in sorted(directory.rglob("*.py"))
    }


def output_provenance(
    binding: dict[str, str],
    replay_bytes: bytes,
    stripe_bytes: bytes,
    refit_bytes: bytes,
    legacy_artefacts: dict[str, str],
) -> dict:
    """Bind renewed output to every replay, result and archived-code input."""
    return {
        **binding,
        "replay_artefact_md5": bytes_md5(replay_bytes),
        "stripe_artefact_md5": bytes_md5(stripe_bytes),
        "refits_artefact_md5": bytes_md5(refit_bytes),
        "legacy_artefacts_md5": legacy_artefacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--provenance-pack", required=True, type=Path)
    parser.add_argument("--legacy-dir", required=True, type=Path,
                        help="Directory containing the unchanged scripts extracted from marking_refit_replay.zip")
    parser.add_argument("--stripes", required=True, type=Path)
    parser.add_argument("--refits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    replay_bytes = args.replay.read_bytes()
    provenance = load_frozen_case_provenance(args.provenance_pack)
    require_replay_pack(args.replay, args.provenance_pack)
    inputs, saved = read_replay_bytes(replay_bytes)
    stripe_bytes = args.stripes.read_bytes()
    stripes_saved = json.loads(gzip.decompress(stripe_bytes))
    binding = provenance_binding(args.provenance_pack)
    preflight_cases(inputs, stripes_saved, provenance, binding)
    legacy_artefacts = python_artefact_md5s(args.legacy_dir)
    sys.path.insert(0, str(args.legacy_dir.resolve()))
    legacy = importlib.import_module("run_alignment")
    # The archived replay fixes OpenCV threads; distance-map rounding depends on it.
    cv2.setNumThreads(1)
    refit_bytes = args.refits.read_bytes()
    refits = json.loads(gzip.decompress(refit_bytes))
    cases = {case["id"]: case for case in inputs["cases"]}
    original = {record["id"]: {"entries": frozen_entries(record)} for record in saved["records"]}
    fitted = {record["id"]: record for record in refits["records"]}
    controls = {}
    for frozen in stripes_saved["records"]:
        identifier = frozen["id"]
        controls[identifier] = verify_starts(
            cases[identifier], frozen, original[identifier], legacy, provenance[identifier]
        )
    print("All starting eligibility flags and net scores reproduce exactly.", flush=True)
    records = []
    for frozen in stripes_saved["records"]:
        identifier = frozen["id"]
        prepared, starting_evidence = controls.pop(identifier)
        result = run_case(
            cases[identifier], frozen, fitted[identifier], legacy, prepared, starting_evidence,
            provenance[identifier],
        )
        records.append(result)
        print(f"{identifier}: {len(result['entries'])} geometries, {result['elapsed_seconds']:.2f}s", flush=True)
    attach_metrics(records, inputs["references"])
    summary = summarise(records)
    output = {"schema": OUTPUT_SCHEMA, "development_data": True,
              "acceptance_evaluated": False, "assignment_held_fixed": True,
              "provenance": output_provenance(
                  binding, replay_bytes, stripe_bytes, refit_bytes, legacy_artefacts,
              ),
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
