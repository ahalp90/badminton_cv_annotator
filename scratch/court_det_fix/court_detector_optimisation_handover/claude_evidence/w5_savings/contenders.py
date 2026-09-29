"""Which W5 courts can still change D17's choice, and how early could the rest be skipped (throwaway).

D17's net choice (bounded_trial.choose) takes the highest paint score plus a net bonus of at most
NET_WEIGHT, over courts that are hard-valid, camera-eligible and full-court. So a court whose
paint score is more than NET_WEIGHT below the best eligible court can never be chosen. Those
within NET_WEIGHT are the contenders.

W5 measures every parent before refitting any of them. So the best eligible parent score is known
before the first refit, and it can only be beaten, never lowered. The question per view: how far
below that best parent score do the parents of contender children sit? A refit rule of "only refit
parents within M of the best parent" is safe on these views if M covers the worst such gap.

Usage: contenders.py RUN_DIR   (with case_records/ and d17/ inside)
"""

import gzip
import json
import sys
from pathlib import Path

NET_WEIGHT = 0.04
root = Path(sys.argv[1])


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def eligible(candidate: dict, criterion: str) -> bool:
    return (candidate["hard_valid"] and candidate["camera_eligible"]
            and candidate["historical"]["historical_fullcourt"]
            and candidate.get("evidence", {}).get(criterion) is not None)


worst_gaps = []
print(f"{'view':38s} {'parents':>7s} {'measured':>8s} {'children':>8s} {'eligible':>8s} {'contend':>7s} "
      f"{'from children':>13s} {'worst parent gap':>16s} {'chosen':>24s}")
for record_path in sorted((root / "case_records").glob("*.json.gz")):
    record = read(record_path)
    summary = read(root / "d17" / record_path.name)
    criterion = record["rankings"]["C"]["r2_criterion"]
    chosen = summary["selection"]["bounded"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    children = record["valid_children"]
    measured = [parent for parent in parents.values() if "evidence" in parent]
    if criterion is None:
        print(f"{record['case_id']:38s} {len(parents):7d} {len(measured):8d} {len(children):8d}  no ranking criterion; chosen {chosen}")
        continue
    pool = [candidate for candidate in list(parents.values()) + children if eligible(candidate, criterion)]
    if not pool:
        print(f"{record['case_id']:38s} {len(parents):7d} {len(measured):8d} {len(children):8d}  no eligible court; chosen {chosen}")
        continue
    best = max(candidate["evidence"][criterion] for candidate in pool)
    contenders = [candidate for candidate in pool if candidate["evidence"][criterion] >= best - NET_WEIGHT]
    assert chosen is None or chosen in {candidate["origin_key"] for candidate in contenders}, chosen
    eligible_parents = [parent for parent in parents.values() if eligible(parent, criterion)]
    best_parent = max(parent["evidence"][criterion] for parent in eligible_parents) if eligible_parents else None
    # For each contender child: how far its parent's own paint score sits below the best eligible parent.
    gaps = []
    for child in contenders:
        if child["kind"] != "child":
            continue
        parent_score = parents[child["parent_origin_key"]]["evidence"].get(criterion)
        gaps.append(best_parent - parent_score if best_parent is not None and parent_score is not None else float("inf"))
    worst = max(gaps) if gaps else None
    worst_gaps.append((record["case_id"], worst, best_parent, record))
    from_children = sum(candidate["kind"] == "child" for candidate in contenders)
    worst_text = "n/a" if worst is None else f"{worst:.3f}"
    print(f"{record['case_id']:38s} {len(parents):7d} {len(measured):8d} {len(children):8d} {len(pool):8d} "
          f"{len(contenders):7d} {from_children:13d} {worst_text:>16s} {chosen!s:>24s}")

finite = [gap for _, gap, _, _ in worst_gaps if gap is not None]
margin = max(finite)
print(f"\nworst parent gap over all views: {margin:.3f}")
for multiple in (1.0, 1.5, 2.0):
    cut = margin * multiple
    refits = skipped = 0
    for _, _, best_parent, record in worst_gaps:
        criterion = record["rankings"]["C"]["r2_criterion"]
        for parent in record["parents"]:
            if "evidence" not in parent:
                continue
            refits += 1
            score = parent["evidence"].get(criterion)
            skipped += best_parent is not None and score is not None and score < best_parent - cut
    print(f"refit only parents within {cut:.3f} of the best eligible parent: skips {skipped} of {refits} refits "
          f"({skipped / refits:.0%})")
