#!/usr/bin/env python3
"""Summarise actual axis/combine work from an automatic-generation JSON record."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def read(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--pretty", action="store_true", help="Print human-readable text after JSON")
    args = parser.parse_args()

    record = read(args.record)
    pairs = record.get("pairs")
    if not isinstance(pairs, list):
        raise SystemExit("record has no list-valued 'pairs' field")

    settings = record.get("settings", {})
    batch = int(settings.get("batch", 256))
    matched = [pair for pair in pairs if pair.get("status") == "matched"]
    enumerated = []
    score_batches = 0
    axis_cap_excluded = 0
    combined = geometry = players = raw_parents = 0
    elapsed: list[float] = []

    for pair in matched:
        role = pair.get("role", {})
        axes = role.get("axes", [])
        for axis in axes:
            diagnostics = axis.get("diagnostics", {})
            count = int(diagnostics.get("enumerated", 0))
            enumerated.append(count)
            score_batches += math.ceil(count / batch) if count else 0
            axis_cap_excluded += int(diagnostics.get("axis_cap_excluded", 0))
        combined += int(role.get("combined", 0))
        geometry += int(role.get("geometry_valid", 0))
        players += int(role.get("geometry_players", 0))
        raw_parents += int(pair.get("raw_parent_count", len(pair.get("shortlist", []))))
        if pair.get("elapsed_s") is not None:
            elapsed.append(float(pair["elapsed_s"]))

    status_counts: dict[str, int] = {}
    for pair in pairs:
        status = str(pair.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "case_id": record.get("case_id"),
        "direction_budget": record.get("direction_screen", {}).get("budget"),
        "pair_status_counts": status_counts,
        "matched_pairs": len(matched),
        "axis_count": len(enumerated),
        "axis_parameters_enumerated": sum(enumerated),
        "axis_parameters_median": median(enumerated) if enumerated else None,
        "score_batch_size": batch,
        "score_axes_calls_estimate": score_batches,
        "redundant_basis_inversions_in_score_axes_estimate": score_batches,
        "combined_courts": combined,
        "geometry_valid_courts": geometry,
        "geometry_and_player_valid_courts": players,
        "raw_parent_count": raw_parents,
        "per_pair_retained": record.get("pooled_candidates"),
        "global_entries": len(record.get("entries", [])),
        "axis_cap_exclusions": axis_cap_excluded,
        "pair_elapsed_sum_s": sum(elapsed),
        "pair_elapsed_median_s": median(elapsed) if elapsed else None,
        "pair_elapsed_p90_s": quantile(elapsed, 0.90),
        "record_elapsed_s": record.get("elapsed_s"),
        "record_cpu_s": record.get("cpu_s"),
        "notes": [
            "score_axes calls assume the current loop scores every enumerated row in fixed-size batches",
            "player_fractions currently performs a generic inverse for each combined court",
            "counts are record-derived; profile native work to confirm cost",
        ],
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.pretty:
        print()
        print(f"matched pairs: {summary['matched_pairs']}")
        print(f"axis hypotheses: {summary['axis_parameters_enumerated']:,}")
        print(f"score batches / repeated basis inversions: {score_batches:,}")
        print(f"combined courts / generic inversions: {combined:,}")
        print(f"geometry valid: {geometry:,}")
        print(f"geometry + players: {players:,}")
        print(f"global entries: {summary['global_entries']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
