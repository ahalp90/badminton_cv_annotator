"""One view's best courts at the net choice: where each came from and how far it sits from the winner.

Prints the net choice's top courts by combined score, each with its parent's rank in the
G0, G1 or line-template shortlist, its score gap to the winner and its largest corner
distance from the winner in native pixels. Then the search scores at a few shortlist ranks
of the winning parent's source.

Usage: python top_courts.py <artefact dir> <view ID> [how many courts, default 6]
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

artefact_dir, view = Path(sys.argv[1]), sys.argv[2]
count = int(sys.argv[3]) if len(sys.argv) > 3 else 6

with gzip.open(artefact_dir / f"{view}.json.gz") as handle:
    artefact = json.load(handle)
record = artefact["w5"]["record"]
parents = {parent["origin_key"]: parent for parent in record["parents"]}
candidates = {**parents, **{child["origin_key"]: child for child in record["valid_children"]}}
parent_key_of = {key: candidate.get("parent_origin_key") or key for key, candidate in candidates.items()}
# Python's sort is stable, so equal scores keep the net choice's row order.
rows = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])
winner = rows[0]
winner_corners = np.asarray(candidates[winner["origin_key"]]["corners_px"])

print("place\tcourt\tparent ranks\tbehind winner\tpx from winner")
for place, row in enumerate(rows[:count], start=1):
    parent = parents[parent_key_of[row["origin_key"]]]
    ranks = " ".join(f"{occurrence['source']}#{occurrence['origin_index'] + 1}"
                     for occurrence in parent["source_occurrences"])
    corner_gap = np.linalg.norm(np.asarray(candidates[row["origin_key"]]["corners_px"]) - winner_corners, axis=1).max()
    print(f"{place}\t{row['origin_key']}\t{ranks}\t{winner['combined_score'] - row['combined_score']:.3f}\t{corner_gap:.1f}")

first_occurrence = parents[parent_key_of[winner["origin_key"]]]["source_occurrences"][0]
if first_occurrence["source"] in ("G0", "G1"):
    scores = [entry["shortlist_score"] for entry in artefact["populations"][first_occurrence["source"]]]
    shown = sorted({1, first_occurrence["origin_index"] + 1, 128, len(scores)})
    print(f"{first_occurrence['source']} search scores by rank:",
          ", ".join(f"#{rank} {scores[rank - 1]:.3f}" for rank in shown))
