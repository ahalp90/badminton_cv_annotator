"""Compare fixed centre/edge-position fits without re-ranking changed geometries."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import fixed_stripe_refit as fitting
from .assignment import prepare_observations
from .detector import CORNER_COURT_M
from .run_assignment import ACCURATE_PX, attach_metrics, read_replay

MODELS = ("nominal_centre", "fixed_position")


def run_case(case: dict, frozen: dict) -> dict:
    """Optimise unique fixed states while preserving every parent candidate ID."""
    started = perf_counter()
    size = tuple(frozen["working_size"])
    width, height = case["dimensions"]["width"], case["dimensions"]["height"]
    scale = np.array([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"]).reshape(-1, 4) / np.tile(scale, 2)
    observations = prepare_observations(segments, size)
    if observations.fragment_ids.tolist() != frozen["fragment_ids"]:
        raise ValueError(f"Observation IDs differ for {case['id']}")
    weights = np.asarray(frozen["fragment_weights"])
    records, cache = [], {}
    statuses: Counter = Counter()
    duplicates, attempted = 0, 0
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        corners = np.asarray(source["corners_px"])
        homography = cv2.getPerspectiveTransform(CORNER_COURT_M, (corners / scale).astype(np.float32))
        assignment = source["stripe_evidence"]["stripe"]["assignments"]
        constraints = fitting.prepare(homography, observations, assignment, weights)
        initial = fitting.initial_parameters(corners / scale, size)
        fits = {}
        for model in MODELS:
            key = (model, initial.tobytes(), constraints.fragment_ids.tobytes(), constraints.sample_ids.tobytes(),
                   constraints.intervals.tobytes(), constraints.positions.tobytes())
            attempted += 1
            if key in cache:
                duplicates += 1
            else:
                fitted = fitting.refine(corners / scale, constraints, size, model == "fixed_position", initial)
                if fitted["corners_px"] is not None:
                    fitted["corners_px"] = (np.asarray(fitted["corners_px"]) * scale).tolist()
                cache[key] = fitted
                statuses[fitted["status"]] += 1
            fits[model] = copy.deepcopy(cache[key])
        provenance = {}
        for field in ("fragment_ids", "sample_ids", "intervals", "positions", "weights"):
            provenance[field] = getattr(constraints, field).tolist()
        records.append({"id": source["id"], "corners_px": source["corners_px"],
                        "fit_samples": provenance, "fits": fits})
    return {"id": case["id"], "dimensions": case["dimensions"], "entries": records,
            "states": {"attempted": attempted, "unique": len(cache), "duplicates": duplicates,
                       "unique_statuses": dict(statuses)}, "elapsed_seconds": perf_counter() - started}


def attach_fit_metrics(records: list[dict], references: dict) -> None:
    """Evaluate all attempts only after fitting, with solver failures kept explicit."""
    attach_metrics(records, references)
    for record in records:
        for model in MODELS:
            entries = [entry["fits"][model] for entry in record["entries"]
                       if entry["fits"][model]["corners_px"] is not None]
            attach_metrics([{**record, "entries": entries}], references)


def summarise(records: list[dict], selections: dict) -> dict:
    totals = {model: {"successful": 0, "accurate_after": 0, "accurate_starts_lost": 0,
                      "newly_accurate": 0, "useful_pools_with_starts": 0} for model in MODELS}
    traced = {scheme: dict.fromkeys(("start", *MODELS), 0)
              for scheme in ("stripe_exclusive", "complete_agreements_first")}
    states: Counter = Counter()
    cases = []
    for record in records:
        for name in ("attempted", "unique", "duplicates"):
            states[name] += record["states"][name]
        entries = {entry["id"]: entry for entry in record["entries"]}
        useful = any(entry["metrics"]["corner_max_error_px"] <= ACCURATE_PX for entry in entries.values())
        for model in MODELS:
            refined_useful = False
            for entry in entries.values():
                initial_accurate = entry["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                fit = entry["fits"][model]
                final_accurate = fit["successful"] and fit["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                totals[model]["successful"] += fit["successful"]
                totals[model]["accurate_after"] += final_accurate
                totals[model]["newly_accurate"] += final_accurate and not initial_accurate
                totals[model]["accurate_starts_lost"] += initial_accurate and not final_accurate
                refined_useful |= final_accurate
            totals[model]["useful_pools_with_starts"] += useful or refined_useful
        picks = {}
        for scheme, counts in traced.items():
            order = selections[record["id"]]["orders"][scheme]
            if not order:
                picks[scheme] = None
                continue
            winner = entries[order[0]]
            counts["start"] += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
            picks[scheme] = {"id": winner["id"], "start": winner["metrics"], "fits": winner["fits"]}
            for model in MODELS:
                fit = winner["fits"][model]
                counts[model] += fit["successful"] and fit["metrics"]["corner_max_error_px"] <= ACCURATE_PX
        cases.append({"id": record["id"], "picks": picks})
    return {"frames": len(records), "states": dict(states), "refinement": totals,
            "fixed_winner_traces": traced, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--stripes", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, _ = read_replay(args.replay)
    stripes = json.loads(gzip.decompress(args.stripes.read_bytes()))
    selection = json.loads(gzip.decompress(args.selection.read_bytes()))
    cases = {case["id"]: case for case in inputs["cases"]}
    orders = {record["id"]: record for record in selection["records"]}
    records = []
    for frozen in stripes["records"]:
        record = run_case(cases[frozen["id"]], frozen)
        records.append(record)
        print(f"{record['id']}: {record['states']}, {record['elapsed_seconds']:.2f}s", flush=True)
    attach_fit_metrics(records, inputs["references"])
    summary = summarise(records, orders)
    output = {"schema": "fixed-assignment-stripe-refit/1", "development_data": True,
              "diagnostic_only": True, "refined_eligibility_evaluated": False,
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
