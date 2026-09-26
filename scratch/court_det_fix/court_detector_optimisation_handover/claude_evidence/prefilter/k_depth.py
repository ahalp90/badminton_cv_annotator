"""How deep in each pair's coarse ranking did the courts that matter sit?

The 24 September prefilter rows (run_prefilter.sh) give, for every per-pair shortlist, each
kept court's rank under the 4-, 8- and 16-sample coarse scores. The cascade keeps a court
only if that rank is within K. This maps rows to pairs, then looks up the coarse rank of the
courts that reached the overall shortlist, the net choice's top five, and the chosen court.

The populations are the research chain's G0 and G1 records from the 25 September check, and
the artefacts are the joined detector's from the same check; both stay on Carmack
(court_detector/check_20260925/left_on_carmack.tsv). summarise_k_depth.py sums the output.

Usage: python k_depth.py <prefilter rows dir> <populations dir> <artefact dir> [samples per marking: 4, 8 or 16] > k_depth_<samples>.tsv
"""

import gzip
import json
import sys
from pathlib import Path

rows_dir, populations_dir, artefact_dir = map(Path, sys.argv[1:4])
SAMPLES = int(sys.argv[4]) if len(sys.argv) > 4 else 16
TOP_PICKS = 5


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def coarse_ranks(view: str) -> dict[tuple[str, str], int]:
    """(source, 'pair:position') -> coarse rank (1 = best) within its pair, at SAMPLES samples per marking."""
    with open(rows_dir / f"{view}.jsonl") as handle:
        rows = [json.loads(line) for line in handle]
    ranks = {}
    next_row = 0
    # Rows follow call order: G0's matched pairs, then G1's, skipping pairs with no scoring call.
    for source in ("G0", "G1"):
        for pair in read_gz(populations_dir / source / f"{view}.json.gz")["pairs"]:
            if pair["status"] != "matched" or next_row == len(rows):
                continue
            row = rows[next_row]
            positions = [int(entry["candidate_id"].split(":")[1]) for entry in pair["shortlist"]]
            if row["courts"] != pair["raw_parent_count"] or row["retained_positions"] != positions:
                continue
            next_row += 1
            for position, rank in zip(row["retained_positions"], row[f"coarse{SAMPLES}_retained_ranks"], strict=True):
                ranks[(source, f"{pair['pair_id']}:{position}")] = rank + 1
    assert next_row == len(rows), f"{view}: matched {next_row} of {len(rows)} rows"
    return ranks


def parent_depth(parent: dict, ranks: dict) -> int | None:
    """Shallowest coarse rank over the parent's search occurrences; None for a template-only parent."""
    depths = [ranks[(occurrence["source"], occurrence["candidate_id"])]
              for occurrence in parent["source_occurrences"] if occurrence["source"] in ("G0", "G1")]
    return min(depths) if depths else None


print("view\tdeepest_kept\tdeepest_top256\tdeepest_top128\tdeepest_top5_picks\tchosen_depth")
for path in sorted(rows_dir.glob("*.jsonl")):
    view = path.stem
    ranks = coarse_ranks(view)
    artefact = read_gz(artefact_dir / f"{view}.json.gz")
    entry_depths = [(index, ranks[(source, entry["candidate_id"])])
                    for source, entries in artefact["populations"].items() for index, entry in enumerate(entries)]
    deepest_top = {cap: max((depth for index, depth in entry_depths if index < cap), default=0) for cap in (256, 128)}
    record = artefact["w5"]["record"]
    parents = {parent["origin_key"]: parent for parent in record["parents"]}
    parent_key_of = {key: key for key in parents}
    parent_key_of.update({child["origin_key"]: child["parent_origin_key"] for child in record["valid_children"]})
    ordered = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])[:TOP_PICKS]
    pick_depths = [parent_depth(parents[parent_key_of[row["origin_key"]]], ranks) for row in ordered]
    top_picks = max((depth for depth in pick_depths if depth is not None), default="-")
    chosen = artefact["net_choice"]["chosen"]
    chosen_depth = parent_depth(parents[parent_key_of[chosen]], ranks) if chosen else None
    print(f"{view}\t{max(ranks.values())}\t{deepest_top[256]}\t{deepest_top[128]}\t{top_picks}\t"
          f"{'template' if chosen and chosen_depth is None else chosen_depth or '-'}")
