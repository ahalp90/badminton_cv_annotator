"""Compare D17 stage times between two run folders with the same views (throwaway).

Sums each view's run wall time, top-level phases, and self time per stage name (the last part of
a stage path, so nested stages are not double counted). Prints old, new and the change.

Usage: compare_stage_times.py OLD_RUN_DIR NEW_RUN_DIR
"""

import gzip
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def summaries(root: Path) -> dict[str, dict]:
    views = {}
    for path in sorted((root / "d17").glob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            views[path.name.removesuffix(".json.gz")] = json.load(stream)
    return views


def totals(views: dict[str, dict]) -> tuple[float, dict[str, float], dict[str, float]]:
    wall = sum(view["run_wall_s"] for view in views.values())
    phases, stage_self = defaultdict(float), defaultdict(float)
    for view in views.values():
        for phase, seconds in view["phases_s"].items():
            phases[phase] += seconds
        for path, values in view["stages"].items():
            stage_self[path.rsplit("/", 1)[-1]] += values["self_seconds"]
    return wall, phases, stage_self


def line(name: str, old: float, new: float, reference: float) -> str:
    change = f"{new / old - 1:+.0%}" if old else "n/a"
    return f"{name:32s} {old:9.0f} {new:9.0f} {change:>6s}   saved {100 * (old - new) / reference:5.1f}% of old run"


old_views, new_views = summaries(Path(sys.argv[1])), summaries(Path(sys.argv[2]))
assert old_views.keys() == new_views.keys(), sorted(old_views.keys() ^ new_views.keys())
old_wall, old_phases, old_stages = totals(old_views)
new_wall, new_phases, new_stages = totals(new_views)
print(f"views: {len(old_views)}   columns: old s, new s, change")
print(line("run wall (summed over views)", old_wall, new_wall, old_wall))
print("phases:")
for phase in sorted(old_phases, key=old_phases.get, reverse=True):
    print(line("  " + phase, old_phases[phase], new_phases.get(phase, 0.0), old_wall))
print("stage self time, largest 20 in the old run:")
for stage in sorted(old_stages, key=old_stages.get, reverse=True)[:20]:
    print(line("  " + stage, old_stages[stage], new_stages.get(stage, 0.0), old_wall))
ratios = sorted(new_views[case]["run_wall_s"] / old_views[case]["run_wall_s"] for case in old_views)
print(f"per-view wall ratio new/old: min {ratios[0]:.2f}, median {statistics.median(ratios):.2f}, max {ratios[-1]:.2f}")
slowest = max(old_views, key=lambda case: old_views[case]["run_wall_s"])
print(f"slowest view {slowest}: {old_views[slowest]['run_wall_s']:.0f} s -> {new_views[slowest]['run_wall_s']:.0f} s")
