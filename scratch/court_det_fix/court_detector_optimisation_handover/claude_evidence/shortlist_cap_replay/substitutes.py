"""Under a cap, how much worse are the courts that move up into places 2-5 of the net choice?

A cap only deletes courts, so the capped list is the old list minus the cut courts. For each
place, compare the old court's score with the one that now holds that place, and say how far
the new one sits from the chosen court.

The artefacts are the joined detector's from its 25 September check, kept on Carmack
(court_detector/check_20260925/left_on_carmack.tsv).

Usage: python substitutes.py <artefact dir> <G0 cap> <G1 cap>
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

artefact_dir = Path(sys.argv[1])
caps = {"G0": int(sys.argv[2]), "G1": int(sys.argv[3])}
PLACES = 5


def kept(occurrence: dict) -> bool:
    return occurrence["source"] == "line_template" or occurrence["origin_index"] < caps[occurrence["source"]]


def corner_gap_px(first: dict, second: dict) -> float:
    return float(np.linalg.norm(np.asarray(first["corners_px"]) - np.asarray(second["corners_px"]), axis=1).max())


print("view\tplace\told_gap\tnew_gap\tsame_court\tnew_px_from_chosen")
for path in sorted(artefact_dir.glob("*.json.gz")):
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    record = artefact["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    candidates = {**parents, **{child["origin_key"]: child for child in record["valid_children"]}}
    parent_key_of = {key: candidate.get("parent_origin_key") or key for key, candidate in candidates.items()}
    surviving = {key for key in candidates if any(map(kept, parents[parent_key_of[key]]["source_occurrences"]))}
    # Python's sort is stable, so equal scores keep the net choice's row order.
    before = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])
    after = [row for row in before if row["origin_key"] in surviving]
    if not before or before[0]["origin_key"] != after[0]["origin_key"]:
        print(f"{path.name}\tchosen court changed or no court")
        continue
    top = before[0]
    for place in range(2, min(PLACES, len(before)) + 1):
        old = before[place - 1]
        new = after[place - 1] if place <= len(after) else None
        if new is None:
            print(f"{path.name.removesuffix('.json.gz')}\t{place}\t{top['combined_score'] - old['combined_score']:.4f}\t-\t-\t-")
            continue
        print(f"{path.name.removesuffix('.json.gz')}\t{place}\t{top['combined_score'] - old['combined_score']:.4f}\t"
              f"{top['combined_score'] - new['combined_score']:.4f}\t{new['origin_key'] == old['origin_key']}\t"
              f"{corner_gap_px(candidates[new['origin_key']], candidates[top['origin_key']]):.1f}")
