"""Print per-case stage wall times for two runs (throwaway)."""
import gzip, json, sys
from pathlib import Path
roots = [Path(p) for p in sys.argv[1:]]
for path in sorted((roots[1] / "cases" / "baseline").glob("*.json.gz")):
    row = [path.name.removesuffix(".json.gz")]
    for root in roots:
        timing = json.load(gzip.open(root / "cases" / "baseline" / path.name, "rt"))["timing"]
        row.append(" / ".join(f"{timing[stage]['wall_s']:.1f}" for stage in ("preparation", "generation", "refit_and_scoring")) + f" = {timing['total_wall_s']:.1f}")
    print(" | ".join(row))
