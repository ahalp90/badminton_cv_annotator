"""Old against new full-search D17 timings, by stage bucket and per view (throwaway).

Usage: d17_timing_compare.py OLD_RUN_DIR NEW_RUN_DIR
The non-court labels come from OLD_RUN_DIR/comparison.json.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path

from d17_report import BUCKETS, buckets_for, group_of

old_root, new_root = Path(sys.argv[1]), Path(sys.argv[2])
comparison = json.loads((old_root / "comparison.json").read_text())
non_court = {row["case_id"]: row["non_court"] for row in comparison["cases"]}
# Generator steps the two speed-ups target, summed over G0 and G1, and the whole per-pair loop.
PAIR_STEPS = ("pair/player_fractions", "pair/match_axis", "pair")


def read_run(root: Path) -> dict[str, dict]:
    views = {}
    for path in sorted((root / "budget16/d17").glob("*.json.gz")):
        with gzip.open(path, "rt") as stream:
            summary = json.load(stream)
        views[summary["case_id"]] = {
            "buckets": buckets_for(summary),
            "pair_steps": {step: sum(summary["stages"].get(f"populations/{label}/{step}", {}).get("seconds", 0.0)
                                     for label in ("G0_generation", "G1_generation"))
                           for step in PAIR_STEPS},
            "wall": summary["run_wall_s"] + summary["startup_s"].get("interpreter_start", 0.0),
            "cpu": summary["run_cpu_s"],
            "rss": summary["peak_rss_mb"],
        }
    return views


old, new = read_run(old_root), read_run(new_root)
assert set(old) == set(new), sorted(set(old) ^ set(new))
court_views = [case_id for case_id in old if not non_court[case_id]]

print("## Court views: targeted generator steps (G0 + G1) in seconds, old / new")
for step in PAIR_STEPS:
    old_total = sum(old[case_id]["pair_steps"][step] for case_id in court_views)
    new_total = sum(new[case_id]["pair_steps"][step] for case_id in court_views)
    print(f"{step}: {old_total:.0f} / {new_total:.0f} ({old_total / new_total:.1f}x faster)")
print()

print("## Court views: stage totals in seconds, old / new / change")
for name, description, _, _ in BUCKETS:
    old_total = sum(old[case_id]["buckets"][name] for case_id in court_views)
    new_total = sum(new[case_id]["buckets"][name] for case_id in court_views)
    print(f"{name} ({description}): {old_total:.0f} / {new_total:.0f} / {new_total - old_total:+.0f}")
for measure in ("wall", "cpu"):
    old_total = sum(old[case_id][measure] for case_id in court_views)
    new_total = sum(new[case_id][measure] for case_id in court_views)
    print(f"total {measure}: {old_total:.0f} / {new_total:.0f} ({new_total / old_total:.2f}x)")

print("\n## Wall seconds per view by group: old median / new median, old total / new total")
for group in sorted({group_of(case_id, non_court[case_id]) for case_id in old}):
    cases = [case_id for case_id in old if group_of(case_id, non_court[case_id]) == group]
    old_walls = [old[case_id]["wall"] for case_id in cases]
    new_walls = [new[case_id]["wall"] for case_id in cases]
    print(f"{group}: {statistics.median(old_walls):.0f} / {statistics.median(new_walls):.0f}, "
          f"{sum(old_walls):.0f} / {sum(new_walls):.0f}")

print("\n## Per view: wall old / new, G0+G1 search old / new, peak RSS MB old / new")
for case_id in sorted(old, key=lambda key: (group_of(key, non_court[key]), key)):
    searches = [run[case_id]["buckets"]["G0_search"] + run[case_id]["buckets"]["G1_search"] for run in (old, new)]
    print(f"{case_id} | {old[case_id]['wall']:.0f} / {new[case_id]['wall']:.0f} | "
          f"{searches[0]:.0f} / {searches[1]:.0f} | {old[case_id]['rss']:.0f} / {new[case_id]['rss']:.0f}")
