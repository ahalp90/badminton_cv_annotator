"""Compare two prespecified junction rankings on unchanged stripe candidates."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from .run_assignment import ACCURATE_PX

SCHEMES = ("stripe_exclusive", "contradictions_first", "complete_agreements_first")


def rank(entries: list[dict]) -> dict[str, list[str]]:
    """Keep missing observations neutral and retain every eligible alternative."""
    stripe = sorted(entries, key=lambda entry: (-entry["stripe_score"], entry["id"]))
    contradictions = sorted(entries, key=lambda entry: (
        entry["disagreements"], -entry["stripe_score"], entry["id"],
    ))
    agreements = sorted(entries, key=lambda entry: (
        -entry["complete_agreements"], entry["disagreements"], -entry["stripe_score"], entry["id"],
    ))
    return dict(zip(SCHEMES, ([entry["id"] for entry in order] for order in (stripe, contradictions, agreements))))


def run_case(frozen: dict, diagnostic: dict) -> dict:
    """Rank from observations and scores before attaching saved reference metrics."""
    measured = {entry["id"]: entry for entry in diagnostic["entries"]}
    entries = []
    for source in frozen["entries"]:
        if not source["eligible"]:
            continue
        junctions = measured[source["id"]]
        complete = sum(site["usable"] and len(site["agreements"]) == 2 for site in junctions["sites"])
        entries.append({"id": source["id"], "stripe_score": source["scores"]["stripe_exclusive"],
                        "disagreements": junctions["disagreements"], "usable_sites": junctions["usable_sites"],
                        "complete_agreements": complete})
    orders = rank(entries)
    metrics = {entry["id"]: entry["metrics"] for entry in frozen["entries"]}
    for entry in entries:
        entry["metrics"] = metrics[entry["id"]]
    return {"id": frozen["id"], "entries": entries, "orders": orders}


def summarise(records: list[dict]) -> dict:
    accurate, unobserved = dict.fromkeys(SCHEMES, 0), dict.fromkeys(SCHEMES, 0)
    cases = []
    for record in records:
        entries = {entry["id"]: entry for entry in record["entries"]}
        picks = {}
        for scheme, order in record["orders"].items():
            winner = entries[order[0]] if order else None
            picks[scheme] = winner
            if winner is not None:
                accurate[scheme] += winner["metrics"]["corner_max_error_px"] <= ACCURATE_PX
                unobserved[scheme] += winner["usable_sites"] == 0
        cases.append({"id": record["id"], "picks": picks})
    return {"frames": len(records), "accurate_picks": accurate,
            "winners_without_usable_sites": unobserved, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stripes", required=True, type=Path)
    parser.add_argument("--junctions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    stripes = json.loads(gzip.decompress(args.stripes.read_bytes()))
    junctions = json.loads(gzip.decompress(args.junctions.read_bytes()))
    measured = {record["id"]: record for record in junctions["records"]}
    records = [run_case(record, measured[record["id"]]) for record in stripes["records"]]
    summary = summarise(records)
    output = {"schema": "frozen-junction-selection/1", "development_data": True,
              "acceptance_evaluated": False, "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(output, allow_nan=False).encode(), mtime=0))
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
