"""Summarise the Carmack prefilter rows: how deep the coarse order must go, and what two passes would cost (throwaway).

Each row is one court-scoring call (one generator pair). For coarse sample counts 4, 8 and 16:
- needed_k: coarse-rank depth that holds every court the full search kept (the exact shortlist)
- subset_same_shortlist: the shortlist rebuilt from those needed_k courts matched the full one
- two-pass cost: one distance-map build, the coarse pass over all courts and exact scoring of the
  top K, against exact scoring of all courts (which includes its own maps). The hook's timed
  exact rescoring of the needed_k courts builds the maps again, so that build (approximated by
  map_seconds) is taken out before scaling it linearly to K. Coarse ranking and retention are not
  counted. An estimate, not an end-to-end measurement.

Usage: summarise_prefilter.py ROWS_DIR   (the rows are kept in the Carmack run folder d17_prefilter_20260924)
"""

import json
import statistics
import sys
from pathlib import Path

COUNTS = (4, 8, 16)
FIXED_K = (1024, 2048, 4096, 8192, 16384, 32768)

rows = []
for path in sorted(Path(sys.argv[1]).glob("*.jsonl")):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        row["case"] = path.stem
        rows.append(row)
full_rows = [row for row in rows if row["shortlist_full"]]
print(f"rows (scoring calls): {len(rows)} over {len({row['case'] for row in rows})} views; "
      f"full shortlists (256 kept): {len(full_rows)}")
full_seconds = sum(row["finite_seconds"] for row in rows)
print(f"exact scoring, all calls: {full_seconds:.0f} s")

for count in COUNTS:
    key = f"coarse{count}"
    same = [row[f"{key}_subset_same_shortlist"] for row in rows]
    needed = [row[f"{key}_needed_k"] for row in rows]
    fraction = [row[f"{key}_needed_k"] / row["courts"] for row in rows]
    worst = max(rows, key=lambda row: row[f"{key}_needed_k"])
    worst_share = max(rows, key=lambda row: row[f"{key}_needed_k"] / row["courts"])
    coarse = sum(row["map_seconds"] + row[f"{key}_seconds"] for row in rows)
    topk_needed = sum(max(row[f"{key}_topk_full_seconds"] - row["map_seconds"], 0.0) for row in rows)
    print(f"\n{count} samples per marking: shortlist rebuilt identically from needed_k in {sum(same)} of {len(same)} calls")
    print(f"  needed_k: median {statistics.median(needed):.0f}, max {max(needed)} "
          f"({worst['case']} call {worst['sequence']}, {worst['courts']} courts)")
    print(f"  needed_k / courts: median {statistics.median(fraction):.1%}, max {max(fraction):.1%} "
          f"({worst_share['case']} call {worst_share['sequence']})")
    print(f"  maps and coarse pass {coarse:.0f} s ({coarse / full_seconds:.0%} of exact); with oracle K = needed_k, "
          f"two passes {coarse + topk_needed:.0f} s ({(coarse + topk_needed) / full_seconds:.0%})")
    for fixed in FIXED_K:
        covered = sum(row[f"{key}_needed_k"] <= fixed or row["courts"] <= fixed for row in rows)
        # Calls with at most K courts are scored exactly in one pass, as now.
        exact = sum(max(row[f"{key}_topk_full_seconds"] - row["map_seconds"], 0.0) / max(row[f"{key}_needed_k"], 1) * fixed
                    if row["courts"] > fixed else row["finite_seconds"] for row in rows)
        coarse_fixed = sum(row["map_seconds"] + row[f"{key}_seconds"] for row in rows if row["courts"] > fixed)
        print(f"  fixed K {fixed:6d}: exact in {covered}/{len(rows)} calls; two passes about "
              f"{coarse_fixed + exact:.0f} s ({(coarse_fixed + exact) / full_seconds:.0%} of exact scoring)")
