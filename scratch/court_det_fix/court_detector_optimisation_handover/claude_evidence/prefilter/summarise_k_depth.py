"""Sum k_depth.py's per-view tables: at each K, how many views keep each kind of court.

Usage (from this folder): python summarise_k_depth.py > k_depth_summary.txt
"""

import csv

KS = (256, 512, 1024, 2048, 4096, 8192)
COLUMNS = {
    "chosen_depth": "chosen court",
    "deepest_top5_picks": "top five",
    "deepest_top256": "overall shortlist",
    "deepest_kept": "every kept court",
}

for samples in (4, 8, 16):
    with open(f"k_depth_{samples}.tsv") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    # Template-won and no-court views have no search depth for some columns; they are left out.
    depths = {column: [int(row[column]) for row in rows if row[column].isdigit()] for column in COLUMNS}
    deepest = ", ".join(f"{label} {max(depths[column])}" for column, label in COLUMNS.items())
    print(f"{samples} samples per marking. Deepest: {deepest}")
    for k in KS:
        kept = ", ".join(f"{label} {sum(depth <= k for depth in depths[column])}/{len(depths[column])}"
                         for column, label in COLUMNS.items())
        print(f"  K {k:5}: views fully within K: {kept}")
