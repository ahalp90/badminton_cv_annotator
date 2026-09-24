"""Measure shortlist_score drift and order changes between two generation record trees (throwaway)."""
import gzip, json, sys
from pathlib import Path
baseline_root, patched_root = Path(sys.argv[1]), Path(sys.argv[2])
for patched_path in sorted((patched_root / "generation" / "baseline").glob("*.json.gz")):
    baseline = json.load(gzip.open(baseline_root / "generation" / "baseline" / patched_path.name, "rt"))
    patched = json.load(gzip.open(patched_path, "rt"))
    deltas = []
    for old_pair, new_pair in zip(baseline["pairs"], patched["pairs"], strict=True):
        for old, new in zip(old_pair.get("shortlist", []), new_pair.get("shortlist", []), strict=True):
            deltas.append(abs(old["shortlist_score"] - new["shortlist_score"]))
    same_entry_order = [e["candidate_id"] for e in baseline["entries"]] == [e["candidate_id"] for e in patched["entries"]]
    drifted = sum(delta > 0 for delta in deltas)
    print(patched_path.name, f"scores {len(deltas)}, drifted {drifted}, max delta {max(deltas):.2e}, same pool order {same_entry_order}",
          "winners", baseline["line_winner_id"] == patched["line_winner_id"], baseline["paint_winner_id"] == patched["paint_winner_id"])
