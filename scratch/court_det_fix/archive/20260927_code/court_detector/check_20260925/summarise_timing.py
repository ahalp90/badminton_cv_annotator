"""Per-view seconds: run_d17.py (baseline and rerun) against the in-memory detector (checks on and off).

run_d17.py's run_wall_s includes imports and runtime loading; its phases exclude them, as the
detector's detect_seconds does. Compare "d17 phases" with the detector columns.

Usage: summarise_timing.py BASELINE_ARM OUT
  BASELINE_ARM holds the 24 Sept d17/ summaries; OUT is the court_detector_20260925 run folder.
"""

import gzip
import json
import sys
from pathlib import Path


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def process_line(log: Path) -> dict:
    """run_views.py's closing line: startup and wall seconds, peak memory."""
    for line in reversed(log.read_text().splitlines()):
        if line.startswith('{"views"'):
            return json.loads(line)
    raise ValueError(f"{log}: no process line")


def main() -> int:
    baseline, out = Path(sys.argv[1]), Path(sys.argv[2])
    rerun = out / "run_d17/standing/budget16"
    arms = {"checks_on": out / "timing", "checks_off": out / "timing_no_checks"}
    views = sorted(path.name[:-8] for path in (baseline / "d17").glob("*.json.gz"))
    header = (f"{'view':38s} {'d17 base':>9s} {'d17 rerun':>9s} {'d17 phases':>10s} {'det on':>8s} {'det off':>8s} "
              f"{'rss d17':>8s} {'rss det':>8s}")
    print(header)
    totals = {"base": 0.0, "rerun": 0.0, "phases": 0.0, "checks_on": 0.0, "checks_off": 0.0}
    stage_totals = {name: {} for name in arms}
    phase_totals: dict[str, float] = {}
    for view in views:
        base = read_json_gz(baseline / "d17" / f"{view}.json.gz")
        again = read_json_gz(rerun / "d17" / f"{view}.json.gz")
        for phase, seconds in again["phases_s"].items():
            phase_totals[phase] = phase_totals.get(phase, 0.0) + seconds
        row = {"base": base["run_wall_s"], "rerun": again["run_wall_s"], "phases": sum(again["phases_s"].values())}
        rss = None
        for name, folder in arms.items():
            result = json.loads((folder / "results" / f"{view}.json").read_text())
            if result["error"] is not None:
                raise RuntimeError(f"{name} {view}: {result['error']}")
            row[name] = result["detect_seconds"]
            for stage, seconds in result["stage_seconds"].items():
                stage_totals[name][stage] = stage_totals[name].get(stage, 0.0) + seconds
            if name == "checks_on":
                rss = process_line(folder / "logs" / f"{view}.log")["peak_rss_mb"]
        for key in totals:
            totals[key] += row[key]
        print(f"{view:38s} {row['base']:9.0f} {row['rerun']:9.0f} {row['phases']:10.0f} {row['checks_on']:8.0f} "
              f"{row['checks_off']:8.0f} {again['peak_rss_mb']:8.0f} {rss:8.0f}")
    print(f"{'total':38s} {totals['base']:9.0f} {totals['rerun']:9.0f} {totals['phases']:10.0f} "
          f"{totals['checks_on']:8.0f} {totals['checks_off']:8.0f}")
    print("\nrun_d17.py rerun phases, summed over views (s):")
    for phase, seconds in phase_totals.items():
        print(f"  {phase:28s} {seconds:8.0f}")
    for name, stages in stage_totals.items():
        print(f"\ndetector {name} stages, summed over views (s):")
        for stage, seconds in stages.items():
            print(f"  {stage:28s} {seconds:8.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
