"""Summarise per-pair generation time in the baseline and patched runs (throwaway)."""
import gzip, json, sys
from pathlib import Path
baseline_root, patched_root = Path(sys.argv[1]), Path(sys.argv[2])
for name in ("gxBQ_window_00_frame_0", "gxBQ_window_00_frame_5", "am1_window_00_frame_54", "shuttleset_03_scene_0019"):
    records = [json.load(gzip.open(root / "generation" / "baseline" / f"{name}.json.gz", "rt")) for root in (baseline_root, patched_root)]
    old, new = ({pair["pair_id"]: pair.get("elapsed_s", 0.0) for pair in record["pairs"]} for record in records)
    top = sorted(new, key=new.get, reverse=True)[:5]
    print(name, f"pairs {len(new)}, patched pair total {sum(new.values()):.0f} s (baseline {sum(old.values()):.0f} s), record elapsed {records[1]['elapsed_s']:.0f} s")
    print("   slowest patched pairs (pair: baseline -> patched s):", ", ".join(f"{p}: {old[p]:.1f} -> {new[p]:.1f}" for p in top))
