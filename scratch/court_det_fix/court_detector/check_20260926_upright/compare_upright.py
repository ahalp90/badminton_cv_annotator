"""The upright-camera filter against the unfiltered detector: chosen courts, search ranks and time.

For each view: the 25 September chosen court and the filtered one, how far their corners are
apart in native px (allowing the court's 180-degree relabelling), and the search ranks of
the filtered chosen court's parent. Then total seconds per step for the two same-day timing
runs.

Usage: compare_upright.py OLD_RESULTS OUT
  OLD_RESULTS: check_20260925/correctness/results; OUT: the folder run_upright.sh wrote
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np


def corner_gap(first: list | None, second: list | None) -> str:
    if first is None or second is None:
        return "-"
    old, new = np.asarray(first), np.asarray(second)
    direct = np.linalg.norm(old - new, axis=1).max()
    rotated = np.linalg.norm(old - new[[2, 3, 0, 1]], axis=1).max()
    return f"{min(direct, rotated):.1f}"


def search_ranks(artefact_path: Path, chosen: str | None) -> str:
    """Where the chosen court's parent sat in the G0 and G1 shortlists (1 = best)."""
    if chosen is None:
        return "-"
    with gzip.open(artefact_path) as handle:
        record = json.load(handle)["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    parent_key = {child["origin_key"]: child["parent_origin_key"] for child in record["valid_children"]}
    occurrences = parents[parent_key.get(chosen, chosen)]["source_occurrences"]
    return " ".join(f"{occurrence['source']}#{occurrence['origin_index'] + 1}" for occurrence in occurrences)


old_results, out = Path(sys.argv[1]), Path(sys.argv[2])
print("view\tsame chosen key\tcorner gap px\told outcome\tnew outcome\tnew chosen from")
for path in sorted(old_results.glob("*.json")):
    old = json.loads(path.read_text())
    new = json.loads((out / "upright/results" / path.name).read_text())
    if new["error"] is not None:
        raise RuntimeError(f"{path.stem}: {new['error']}")
    outcomes = [result["no_court_reason"] or "court" for result in (old, new)]
    ranks = search_ranks(out / "upright/artefacts" / f"{path.stem}.json.gz", new["chosen_key"])
    print(f"{path.stem}\t{old['chosen_key'] == new['chosen_key']}\t"
          f"{corner_gap(old['corners_native_px'], new['corners_native_px'])}\t{outcomes[0]}\t{outcomes[1]}\t{ranks}")

print("\nseconds summed over views, self-checks off")
totals = {}
for arm in ("timing_any_roll", "timing_upright"):
    stages: dict[str, float] = {"detect": 0.0}
    for path in sorted((out / arm / "results").glob("*.json")):
        result = json.loads(path.read_text())
        if result["error"] is not None:
            raise RuntimeError(f"{arm} {path.stem}: {result['error']}")
        stages["detect"] += result["detect_seconds"]
        for stage, seconds in result["stage_seconds"].items():
            stages[stage] = stages.get(stage, 0.0) + seconds
    totals[arm] = stages
print(f"{'step':28s} {'any roll':>9s} {'upright':>9s} {'saved':>7s}")
for stage, before in totals["timing_any_roll"].items():
    after = totals["timing_upright"][stage]
    saved = f"{1 - after / before:.0%}" if before > 1 else "-"
    print(f"{stage:28s} {before:9.0f} {after:9.0f} {saved:>7s}")
