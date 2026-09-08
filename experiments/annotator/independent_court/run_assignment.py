"""Compare assignment and controls on saved courts without changing their geometry.

Run from the repository root with PYTHONPATH=src:. and --replay pointing to the
existing marking_refit_replay.zip. The output is compressed JSON; it includes
every retained alternative and attaches reference measurements after ranking.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import cv2
import numpy as np

from . import assignment, evaluate
from .detector import CORNER_COURT_M

SCHEMES = ("recorded_original", "recorded_bidirectional", "independent", "assignment", "ledger")
ACCURATE_PX = 15.0


def frozen_entries(record: dict) -> list[dict]:
    """Copy only geometry, identity and saved evidence; strip labels at the boundary."""
    entries = []
    for entry in record["entries"]:
        if entry["extra_gallery_probe"]:
            continue
        evidence = entry["evidence"]
        eligible = evidence["eligible"] and evidence["scheme_eligible"]["original"]
        entries.append({"id": f"{entry['source_index']:04d}:{entry['stage']}",
                        "source_index": entry["source_index"], "stage": entry["stage"],
                        "corners_px": entry["corners_px"], "eligible": bool(eligible),
                        "evidence": evidence})
    return entries


def score_entry(entry: dict, observations: assignment.Observations, size: tuple[int, int], scale: np.ndarray) -> dict:
    """Score one frozen geometry without labels or candidate-specific observations."""
    corners = np.asarray(entry["corners_px"]) / scale
    homography = cv2.getPerspectiveTransform(CORNER_COURT_M, corners.astype(np.float32))
    support = assignment.measure_support(homography, observations, size)
    matched = assignment.choose_assignment(support, observations.group_lengths)
    independent = assignment.independent_support(support, observations.group_lengths)
    evidence = entry["evidence"]
    scores = {"recorded_original": evidence["scores"]["original"],
              "recorded_bidirectional": evidence["scores"]["bidirectional"],
              "independent": (3 * independent["score"] + evidence["net_score"]) / 4,
              "assignment": (3 * matched["score"] + evidence["net_score"]) / 4}
    pairs = []
    used_groups = set()
    for marking, group in matched["pairs"]:
        used_groups.add(group)
        pairs.append({"marking": assignment.MARKINGS[marking], "group": group,
                      "forward": float(support.forward[marking, group]),
                      "reverse": float(support.reverse[marking, group])})
    matched = {key: value for key, value in matched.items() if key != "pairs"}
    return {"scores": scores, "assignment": matched, "independent": independent,
            "net_score": evidence["net_score"], "pairs": pairs,
            "visible_markings": np.flatnonzero(support.visible).tolist(),
            "unmatched_groups": sorted(set(range(len(observations.groups))) - used_groups),
            "ledger": assignment.ledger_summary(support, observations.group_lengths)}


def rank_entries(entries: list[dict], lengths: np.ndarray) -> tuple[dict, dict]:
    """Retain all scored alternatives; resolve exact score ties by stable source ID."""
    eligible = [entry for entry in entries if entry["eligible"]]
    orders = {}
    for scheme in SCHEMES[:-1]:
        ordered = sorted(eligible, key=lambda entry: (-entry["scores"][scheme], entry["id"]))
        orders[scheme] = [entry["id"] for entry in ordered]
    orders["ledger"], diagnostic = assignment.ledger_order(eligible, lengths)
    return orders, diagnostic


def run_case(case: dict, entries: list[dict]) -> dict:
    """Run all selectors against one shared observation set and frozen eligible pool."""
    started = perf_counter()
    width, height = case["dimensions"]["width"], case["dimensions"]["height"]
    factor = min(1.0, 960 / max(width, height))
    size = (round(width * factor), round(height * factor))
    scale = np.asarray([width / size[0], height / size[1]])
    segments = np.asarray(case["segments_px"], dtype=float).reshape(-1, 4) / np.tile(scale, 2)
    observations = assignment.prepare_observations(segments, size)
    for entry in entries:
        if entry["eligible"]:
            entry.update(score_entry(entry, observations, size, scale))
        entry.pop("evidence")
    orders, ledger_diagnostic = rank_entries(entries, observations.group_lengths)
    groups = []
    for group, members in enumerate(observations.groups):
        groups.append({"id": group, "fragment_ids": observations.fragment_ids[members].tolist(),
                       "segments_working_px": observations.segments[members].tolist(),
                       "union_length_px": float(observations.group_lengths[group])})
    return {"id": case["id"], "dimensions": case["dimensions"], "working_size": size,
            "raw_fragment_count": len(segments), "visible_fragment_count": len(observations.segments),
            "groups": groups, "entries": entries, "orders": orders,
            "ledger_diagnostic": ledger_diagnostic, "elapsed_seconds": perf_counter() - started}


def attach_metrics(records: list[dict], references: dict) -> None:
    """Access manual references only after every case has finished selection."""
    for record in records:
        width, height = record["dimensions"]["width"], record["dimensions"]["height"]
        for entry in record["entries"]:
            entry["metrics"] = evaluate._metrics(
                np.asarray(entry["corners_px"]), references[record["id"]], width, height,
            )


def summarise(records: list[dict]) -> dict:
    counts = dict.fromkeys(SCHEMES, 0)
    availability = {"all": 0, "eligible": 0}
    rows = []
    for record in records:
        entries = {entry["id"]: entry for entry in record["entries"]}
        accurate = [entry for entry in entries.values() if entry["metrics"]["corner_max_error_px"] <= ACCURATE_PX]
        availability["all"] += bool(accurate)
        availability["eligible"] += any(entry["eligible"] for entry in accurate)
        picks = {}
        for scheme, order in record["orders"].items():
            if not order:
                picks[scheme] = None
                continue
            selected = entries[order[0]]
            metrics = selected["metrics"]
            counts[scheme] += metrics["corner_max_error_px"] <= ACCURATE_PX
            picks[scheme] = {"id": selected["id"], **metrics}
        rows.append({"id": record["id"], "geometries": len(entries),
                     "eligible": sum(entry["eligible"] for entry in entries.values()),
                     "groups": len(record["groups"]), "picks": picks})
    return {"frames": len(records), "accurate_picks": counts, "useful_pools": availability, "cases": rows}


def read_replay(path: Path) -> tuple[dict, dict]:
    with ZipFile(path) as archive:
        inputs = json.loads(gzip.decompress(archive.read("marking_inputs.json.gz")))
        results = json.loads(gzip.decompress(archive.read("reverse_results.json.gz")))
    return inputs, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    arguments = parser.parse_args()
    inputs, saved = read_replay(arguments.replay)
    frozen = {record["id"]: frozen_entries(record) for record in saved["records"]}
    cases = [case for case in inputs["cases"] if not arguments.ids or case["id"] in arguments.ids]
    if arguments.ids and set(arguments.ids) != {case["id"] for case in cases}:
        raise ValueError("Requested case IDs must exist in the replay")
    records = []
    for case in cases:
        record = run_case(case, frozen[case["id"]])
        records.append(record)
        print(f"{case['id']}: {len(record['entries'])} geometries, {len(record['groups'])} groups, "
              f"{record['elapsed_seconds']:.2f}s", flush=True)
    attach_metrics(records, inputs["references"])
    summary = summarise(records)
    output = {"schema": "frozen-marking-assignment/1", "development_data": True,
              "acceptance_evaluated": False, "markings": assignment.MARKINGS,
              "summary": summary, "records": records}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
