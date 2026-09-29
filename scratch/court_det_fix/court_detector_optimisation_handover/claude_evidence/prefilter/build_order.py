"""Where does each chosen court sit in its pair's build order, the order the pair constructs courts?

A pair builds its courts from two lists of line guesses, each sorted by score, horizontal first:
every court from the best horizontal guess, then every court from the second best, and so on.
Capping full scoring at K with no cheap pass would score only the first K built. A merged parent
counts at its shallowest occurrence.

Inputs as for k_depth.py, without the prefilter rows.
Usage: python build_order.py <populations dir> <artefact dir> [K, default 2048]
"""

import gzip
import json
import sys
from pathlib import Path

populations_dir, artefact_dir = Path(sys.argv[1]), Path(sys.argv[2])
k = int(sys.argv[3]) if len(sys.argv) > 3 else 2048


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


print("view\tsource\tbuild position\tcourts in pair\twithin K")
for path in sorted(artefact_dir.glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    artefact = read_gz(path)
    chosen = artefact["net_choice"]["chosen"]
    if chosen is None:
        continue
    record = artefact["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    children = {child["origin_key"]: child for child in record["valid_children"]}
    parent = parents[children[chosen]["parent_origin_key"]] if chosen in children else parents[chosen]
    placings = []
    for occurrence in parent["source_occurrences"]:
        if occurrence["source"] not in ("G0", "G1"):
            continue
        pair_id, position = map(int, occurrence["candidate_id"].split(":"))
        pairs = {pair["pair_id"]: pair for pair in read_gz(populations_dir / occurrence["source"] / f"{view}.json.gz")["pairs"]}
        placings.append((position + 1, pairs[pair_id]["raw_parent_count"], occurrence["source"]))
    if not placings:
        print(f"{view}\tline template\t-\t-\t-")
        continue
    position, courts, source = min(placings)
    print(f"{view}\t{source}\t{position}\t{courts}\t{position <= k}")
