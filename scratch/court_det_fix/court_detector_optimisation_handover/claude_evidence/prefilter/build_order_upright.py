"""Build positions, within each pair, of the courts that matter, from a joined-detector run with the upright filter.

A pair builds its courts from two lists of line guesses, each sorted by score, horizontal first. A
court's candidate id is "pair:position", its place in that order among the courts that go on to full
scoring (valid geometry, below the horizon, players inside). Capping full scoring at K with no cheap
pass would keep positions 1 to K. A court found by several pairs or both searches counts at its
shallowest occurrence. Line-template courts have no build position and are left out.

Usage: python build_order_upright.py <artefact dir>
"""

import gzip
import json
import sys
from pathlib import Path

LIMITS = (256, 512, 1024, 2048, 4096, 8192)


def position(candidate_id: str) -> int:
    return int(candidate_id.split(":")[1]) + 1


def shallowest(parent: dict) -> int | None:
    positions = [position(occurrence["candidate_id"]) for occurrence in parent["source_occurrences"]
                 if occurrence["source"] in ("G0", "G1")]
    return min(positions) if positions else None


print("view\tchosen court\tdeepest of the top five\tdeepest in an overall shortlist")
deepest = {"chosen": [], "top five": [], "overall shortlist": []}
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    chosen = artefact["net_choice"]["chosen"]
    if chosen is None:
        continue
    record = artefact["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    parent_of = {child["origin_key"]: child["parent_origin_key"] for child in record["valid_children"]}
    rows = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])
    chosen_position = shallowest(parents[parent_of.get(chosen, chosen)])
    top_positions = [shallowest(parents[parent_of.get(row["origin_key"], row["origin_key"])]) for row in rows[:5]]
    top_positions = [value for value in top_positions if value is not None]
    shortlist_positions = [position(entry["candidate_id"]) for name in ("G0", "G1")
                           for entry in artefact["populations"][name]]
    cells = [chosen_position, max(top_positions, default=None), max(shortlist_positions, default=None)]
    for key, value in zip(deepest, cells, strict=True):
        if value is not None:
            deepest[key].append(value)
    print(path.name.removesuffix(".json.gz") + "\t" + "\t".join("line template" if value is None else str(value)
                                                             for value in cells))
for key, values in deepest.items():
    within = ", ".join(f"K {limit}: {sum(value <= limit for value in values)}/{len(values)}" for limit in LIMITS)
    print(f"# {key}: deepest {max(values)}; views fully within {within}")
