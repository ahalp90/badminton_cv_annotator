"""Replay frozen courts under the prespecified stripe-position/membership comparison."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from . import stripe_observations as stripes
from .assignment import MARKINGS, prepare_observations
from .detector import CORNER_COURT_M
from .run_assignment import ACCURATE_PX, attach_metrics, frozen_entries, read_replay

SCHEMES = ("recorded_original", "recorded_bidirectional", "group_assignment",
           "centre_independent", "centre_exclusive", "stripe_independent", "stripe_exclusive")


def rank(entries: list[dict]) -> dict[str, list[str]]:
    eligible = [entry for entry in entries if entry["eligible"]]
    orders = {}
    for scheme in SCHEMES:
        ordered = sorted(eligible, key=lambda entry: (-entry["scores"][scheme], entry["id"]))
        orders[scheme] = [entry["id"] for entry in ordered]
    return orders


def run_case(case: dict, entries: list[dict], baseline: dict) -> dict:
    """Keep geometry and saved gates fixed while changing the observation measurement."""
    started = perf_counter()
    width, height = case["dimensions"]["width"], case["dimensions"]["height"]
    factor = min(1.0, 960 / max(width, height))
    size = (round(width * factor), round(height * factor))
    scale = np.asarray([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    observations = prepare_observations(segments, size)
    weights = stripes.fragment_weights(observations)
    saved = {entry["id"]: entry for entry in baseline["entries"]}
    for entry in entries:
        previous = saved[entry["id"]]
        if entry["corners_px"] != previous["corners_px"] or entry["eligible"] != previous["eligible"]:
            raise ValueError(f"Baseline geometry or eligibility differs for {case['id']}/{entry['id']}")
        evidence = entry.pop("evidence")
        if not entry["eligible"]:
            continue
        homography = cv2.getPerspectiveTransform(
            CORNER_COURT_M, (np.asarray(entry["corners_px"]) / scale).astype(np.float32),
        )
        measured = stripes.measure(homography, observations, size)
        entry["stripe_evidence"] = stripes.compare(measured, weights)
        entry["net_score"] = evidence["net_score"]
        entry["scores"] = {"recorded_original": evidence["scores"]["original"],
                           "recorded_bidirectional": evidence["scores"]["bidirectional"],
                           "group_assignment": previous["scores"]["assignment"]}
        for position_model in ("centre", "stripe"):
            for membership in ("independent", "exclusive"):
                floor = entry["stripe_evidence"][position_model][membership]["score"]
                entry["scores"][f"{position_model}_{membership}"] = (3 * floor + evidence["net_score"]) / 4
    return {"id": case["id"], "dimensions": case["dimensions"], "working_size": size,
            "fragment_ids": observations.fragment_ids.tolist(), "fragment_weights": weights.tolist(),
            "entries": entries, "orders": rank(entries), "elapsed_seconds": perf_counter() - started}


def summarise(records: list[dict]) -> dict:
    counts = dict.fromkeys(SCHEMES, 0)
    available, eligible_available = 0, 0
    cases = []
    for record in records:
        entries = {entry["id"]: entry for entry in record["entries"]}
        accurate = [entry for entry in entries.values() if entry["metrics"]["corner_max_error_px"] <= ACCURATE_PX]
        available += bool(accurate)
        eligible_available += any(entry["eligible"] for entry in accurate)
        picks = {}
        for scheme, order in record["orders"].items():
            if order:
                selected = entries[order[0]]
                counts[scheme] += selected["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                picks[scheme] = {"id": selected["id"], **selected["metrics"]}
            else:
                picks[scheme] = None
        cases.append({"id": record["id"], "picks": picks})
    return {"frames": len(records), "accurate_picks": counts,
            "useful_pools": {"all": available, "eligible": eligible_available}, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    packed, saved = read_replay(args.replay)
    baseline = json.loads(gzip.decompress(args.baseline.read_bytes()))
    by_id = {record["id"]: record for record in baseline["records"]}
    frozen = {record["id"]: frozen_entries(record) for record in saved["records"]}
    cases = [case for case in packed["cases"] if not args.ids or case["id"] in args.ids]
    if args.ids and set(args.ids) != {case["id"] for case in cases}:
        raise ValueError("Requested case IDs must exist in the replay")
    records = []
    for case in cases:
        result = run_case(case, frozen[case["id"]], by_id[case["id"]])
        records.append(result)
        print(f"{case['id']}: {len(result['entries'])} geometries, "
              f"{len(result['fragment_ids'])} fragments, {result['elapsed_seconds']:.2f}s", flush=True)
    attach_metrics(records, packed["references"])
    summary = summarise(records)
    output = {"schema": "frozen-stripe-observations/1", "development_data": True,
              "acceptance_evaluated": False, "paired_support_is_diagnostic": True,
              "markings": MARKINGS, "positions": stripes.POSITION_NAMES,
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
