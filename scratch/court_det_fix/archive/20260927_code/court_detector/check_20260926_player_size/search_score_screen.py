"""Screen: would the search stage's score help the final choice on the hand-marked views?

For each eligible court: floor error before the final refit (median and largest, metres), paint,
fragment support, net bonus and its search score. The search score is the court's (or its parent's)
shortlist score over the best shortlist score in the same search on that view, so G0 and G1 compare.
Line-template courts have none; two stand-ins are tried: 0, and the view's lowest shortlisted value.
Usage, from the worktree root:
  python scratch/court_det_fix/court_detector/check_20260926_player_size/search_score_screen.py ARTEFACTS_DIR
    ARTEFACTS_DIR: the default arm's artefacts/ from run_carmack.sh (left on Carmack, see left_on_carmack.tsv)
"""
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path("scratch/court_det_fix")
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

manifest = statistics.read(statistics.MANIFEST)
rows_by_view = {row["case_id"]: row for row in manifest["cases"]}
references = statistics.load_references(manifest)
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3)

views = {}
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    landmarks = references.get(view, {}).get("landmarks")
    if not landmarks:
        continue
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    record = artefact["w5"]["record"]
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    best_per_search = {s: max(e["shortlist_score"] for e in pop) for s, pop in artefact["populations"].items()}
    relative = {f"{s}:{e['candidate_id']}": e["shortlist_score"] / best_per_search[s]
                for s, pop in artefact["populations"].items() for e in pop}
    floor_of_view = min(relative.values())
    height, width = cv2.imread(str(COURT_ROOT / rows_by_view[view]["image"])).shape[:2]
    marked = np.array([m["image_px"] for m in landmarks]) / max(1.0, max(height, width) / 960)
    court_m = np.array([m["court_m"] for m in landmarks])
    rows = []
    for row in artefact["net_choice"]["rows"]:
        key = row["origin_key"]
        parent_key = candidates[key]["parent_origin_key"] or key
        homography = np.asarray(candidates[key]["homography_working"])
        floor = np.column_stack((marked, np.ones(len(marked)))) @ np.linalg.inv(homography).T
        error = np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1)
        geometry = (row["combined_score"] - row["bonus"] - 0.9 * row["paint_score"]) / 0.1
        rows.append({"key": key, "median": float(np.median(error)), "largest": float(error.max()),
                     "paint": row["paint_score"], "geometry": geometry, "bonus": row["bonus"],
                     "combined": row["combined_score"], "search": relative.get(parent_key, np.nan)})
    views[view] = (rows, artefact["net_choice"]["chosen"], floor_of_view)


def choose(rows, weight, stand_in):
    def score(r):
        search = stand_in if np.isnan(r["search"]) else r["search"]
        return (0.9 - weight) * r["paint"] + 0.1 * r["geometry"] + weight * search + r["bonus"]
    return max(rows, key=score)  # first on ties, as the net choice keeps the first row


def show(r):
    search = "  none" if np.isnan(r["search"]) else f"{r['search']:.3f}"
    return (f"{r['key']:<44} err {r['median']:.2f}/{r['largest']:.2f}  paint {r['paint']:.4f}  "
            f"frag {r['geometry']:.3f}  bonus {r['bonus']:.2f}  search {search}")


print("Per view: the chosen court, then the eligible court with the lowest median floor error\n")
for view, (rows, chosen, _) in views.items():
    by_key = {r["key"]: r for r in rows}
    best = min(rows, key=lambda r: r["median"])
    searched = sorted((r for r in rows if not np.isnan(r["search"])), key=lambda r: -r["search"])
    rank = next((i + 1 for i, r in enumerate(searched) if r["key"] == best["key"]), None)
    print(f"{view}  ({len(rows)} eligible, {len(searched)} with a search score)")
    print(f"  chosen  {show(by_key[chosen])}")
    print(f"  best    {show(best)}   search rank {rank}")

for label, stand_in_of in (("templates' search score = 0", lambda f: 0.0),
                           ("templates' search score = view's lowest shortlisted", lambda f: f)):
    print(f"\nChosen court's floor error (median/largest) by search weight; {label}")
    print("view\t" + "\t".join(f"w={w}" for w in WEIGHTS))
    totals = np.zeros((len(WEIGHTS), 2))
    for view, (rows, chosen, floor_of_view) in views.items():
        cells = []
        for i, weight in enumerate(WEIGHTS):
            pick = choose(rows, weight, stand_in_of(floor_of_view))
            totals[i] += pick["median"], pick["largest"]
            cells.append(f"{pick['median']:.2f}/{pick['largest']:.2f}{'*' if pick['key'] != chosen else ''}")
        print(view + "\t" + "\t".join(cells))
    print("total\t" + "\t".join(f"{m:.2f}/{x:.2f}" for m, x in totals))
print("\n* the pick differs from today's choice")
