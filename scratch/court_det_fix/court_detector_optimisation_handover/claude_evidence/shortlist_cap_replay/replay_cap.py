"""Replay a cap on the G0 and G1 shortlists from the saved 25 September artefacts.

Each court is scored on its own, and the net choice takes the best combined score,
so a cap can only remove courts. This re-ranks the surviving courts with the live
ranker, filters the saved net-choice rows, and reports what the cap would change.

The artefacts are the joined detector's from its 25 September check, kept on Carmack
(court_detector/check_20260925/left_on_carmack.tsv).

Usage: python replay_cap.py <checkout> <artefact dir> [G0 cap] [G1 cap]
"""

import gzip
import json
import sys
from pathlib import Path

checkout, artefact_dir = Path(sys.argv[1]), Path(sys.argv[2])
caps = {"G0": int(sys.argv[3]) if len(sys.argv) > 3 else 128, "G1": int(sys.argv[4]) if len(sys.argv) > 4 else 128}
root = checkout / "scratch/court_det_fix"
sys.path[:0] = [str(checkout), str(checkout / "src"), str(root / "w5_holistic"), str(root / "wider_evaluation")]
import numpy as np
import verifier  # pyrefly: ignore[missing-import]


def kept(occurrence: dict) -> bool:
    return occurrence["source"] == "line_template" or occurrence["origin_index"] < caps[occurrence["source"]]


def parent_ranks(parent: dict) -> str:
    ranks = sorted((occurrence["origin_index"] + 1, occurrence["source"]) for occurrence in parent["source_occurrences"])
    return " ".join(f"{source}#{rank}" for rank, source in ranks)


def first_winner(rows: list[dict]) -> str | None:
    """The net choice's rule: the highest combined score, first row on exact ties."""
    best_key, best_score = None, -float("inf")
    for row in rows:
        if row["combined_score"] > best_score:
            best_key, best_score = row["origin_key"], row["combined_score"]
    return best_key


def corner_gap_px(first: dict, second: dict) -> float:
    """Largest corner movement between two courts, in native pixels."""
    return float(np.linalg.norm(np.asarray(first["corners_px"]) - np.asarray(second["corners_px"]), axis=1).max())


def replay_view(artefact: dict) -> tuple[dict, list[str]]:
    record = artefact["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    candidates = {**parents, **{child["origin_key"]: child for child in record["valid_children"]}}
    parent_key_of = {key: key for key in parents}
    parent_key_of.update({child["origin_key"]: child["parent_origin_key"] for child in record["valid_children"]})
    surviving_parents = {key for key, parent in parents.items() if any(map(kept, parent["source_occurrences"]))}
    surviving = {key for key, parent_key in parent_key_of.items() if parent_key in surviving_parents}
    notes = []

    # Re-rank the survivors with the live ranker; the order should be the old order with the cut courts removed.
    b_candidates = [parent for parent in record["parents"]
                    if parent.get("hard_valid") and "evidence" in parent and parent["origin_key"] in surviving]
    children = [child for child in record["valid_children"] if child["origin_key"] in surviving]
    ranking = verifier.rank_candidates(b_candidates + children)
    old = record["rankings"]["C"]
    if ranking["r2_criterion"] != old["r2_criterion"]:
        notes.append(f"criterion {old['r2_criterion']}->{ranking['r2_criterion']}")
    if ranking["provisional_rank"] != [key for key in old["provisional_rank"] if key in surviving]:
        notes.append("provisional order changed beyond removal")

    rows = artefact["net_choice"]["rows"]
    saved_chosen = artefact["net_choice"]["chosen"]
    if first_winner(rows) != saved_chosen:
        notes.append("saved rows do not reproduce the saved choice")
    new_chosen = first_winner([row for row in rows if row["origin_key"] in surviving])
    # A merged parent whose first occurrence is cut keeps its court but gets a new key and tie-break order.
    if new_chosen is not None and not kept(parents[parent_key_of[new_chosen]]["source_occurrences"][0]):
        notes.append("chosen court's parent is re-keyed; check exact ties")

    # How close the cut courts came: the best-placed net-choice row whose parent the cap removes.
    ordered = sorted(rows, key=lambda row: -row["combined_score"])
    cut = [(place, row) for place, row in enumerate(ordered, start=1) if row["origin_key"] not in surviving]
    summary = {
        "chosen": saved_chosen or "-",
        "chosen_parent_ranks": parent_ranks(parents[parent_key_of[saved_chosen]]) if saved_chosen else "-",
        "same": new_chosen == saved_chosen,
        "parents_before": len(parents),
        "parents_after": len(surviving_parents),
        "best_cut_place": "-", "best_cut_gap": "-", "best_cut_ranks": "-", "best_cut_px": "-",
        "without_winner_parent_px": "-",
    }
    if cut:
        place, row = cut[0]
        summary.update({"best_cut_place": place,
                        "best_cut_gap": f"{ordered[0]['combined_score'] - row['combined_score']:.4f}",
                        "best_cut_ranks": parent_ranks(parents[parent_key_of[row["origin_key"]]]),
                        "best_cut_px": f"{corner_gap_px(candidates[row['origin_key']], candidates[saved_chosen]):.1f}"})
    if saved_chosen is not None:
        # If the winner's own parent had been cut too, how far would the next pick sit from it?
        winner_parent = parent_key_of[saved_chosen]
        fallback = first_winner([row for row in rows
                                 if row["origin_key"] in surviving and parent_key_of[row["origin_key"]] != winner_parent])
        if fallback is not None:
            summary["without_winner_parent_px"] = f"{corner_gap_px(candidates[fallback], candidates[saved_chosen]):.1f}"
    return summary, notes


columns = ["chosen", "chosen_parent_ranks", "same", "parents_before", "parents_after",
           "best_cut_place", "best_cut_gap", "best_cut_ranks", "best_cut_px", "without_winner_parent_px"]
print("\t".join(["view", *columns, "notes"]))
before = after = 0
for path in sorted(artefact_dir.glob("*.json.gz")):
    with gzip.open(path) as handle:
        summary, notes = replay_view(json.load(handle))
    before += summary["parents_before"]
    after += summary["parents_after"]
    print("\t".join([path.name.removesuffix(".json.gz"), *(str(summary[column]) for column in columns), "; ".join(notes)]))
print(f"# caps {caps}: parents {before} -> {after} ({after / before:.1%} kept)")
