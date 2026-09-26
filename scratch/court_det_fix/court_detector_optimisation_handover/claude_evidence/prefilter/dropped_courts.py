"""Would a cascade at K change any search's overall shortlist on the saved views?

A pair's shortlist loses a court when that court's coarse rank is deeper than K. The pair may
then admit other courts, but each of those scores below a court it replaces. The overall pick
walks every pair's courts in score order and stops once 256 are kept, so a full overall list
is unchanged when every lost court scores below its last court. An overall list with fewer
than 256 courts takes every pair court not within 2 px of a better one, so it must lose none.

The rows and populations are the same inputs k_depth.py takes.
Usage: python dropped_courts.py <prefilter rows dir> <populations dir> <samples per marking> <K>
"""

import gzip
import json
import sys
from pathlib import Path

rows_dir, populations_dir = Path(sys.argv[1]), Path(sys.argv[2])
samples, k = int(sys.argv[3]), int(sys.argv[4])

dropped_total = 0
problems = []
for path in sorted(rows_dir.glob("*.jsonl")):
    view = path.stem
    with open(path) as handle:
        rows = [json.loads(line) for line in handle]
    next_row = 0
    # Rows follow call order: G0's matched pairs, then G1's, skipping pairs with no scoring call.
    for source in ("G0", "G1"):
        with gzip.open(populations_dir / source / f"{view}.json.gz") as handle:
            population = json.load(handle)
        overall_ids = {entry["candidate_id"] for entry in population["entries"]}
        full = len(population["entries"]) == population["keep_global"]
        cutoff = population["entries"][-1]["shortlist_score"] if population["entries"] else None
        for pair in population["pairs"]:
            if pair["status"] != "matched" or next_row == len(rows):
                continue
            row = rows[next_row]
            positions = [int(entry["candidate_id"].split(":")[1]) for entry in pair["shortlist"]]
            if row["courts"] != pair["raw_parent_count"] or row["retained_positions"] != positions:
                continue
            next_row += 1
            for entry, rank in zip(pair["shortlist"], row[f"coarse{samples}_retained_ranks"], strict=True):
                if rank + 1 <= k:
                    continue
                dropped_total += 1
                if entry["candidate_id"] in overall_ids or not full or entry["shortlist_score"] >= cutoff:
                    problems.append(f"{view} {source} {entry['candidate_id']} coarse rank {rank + 1}")
    assert next_row == len(rows), f"{view}: matched {next_row} of {len(rows)} rows"

print(f"{samples} samples, K {k}: {dropped_total} courts dropped from pair shortlists; "
      f"{len(problems)} could change an overall shortlist")
for problem in problems:
    print(" ", problem)
