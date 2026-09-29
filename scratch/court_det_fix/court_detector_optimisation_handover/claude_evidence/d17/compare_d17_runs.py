"""Compare two D17 runs' saved outputs for one search setting, exactly (throwaway).

Every JSON value and array must be equal, apart from timings and the run directory inside
saved paths. The only expected difference: `pattern_supported` is None in the new run where
the old run counted it. Floats are compared exactly; any drift is reported with its size.

Usage: compare_d17_runs.py OLD_RUN_DIR NEW_RUN_DIR [BUDGET]
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

TIMING_KEYS = {"cpu_s", "elapsed_s", "wall_s", "elapsed_seconds", "process_s", "timings_seconds",
               "stages", "startup_s", "run_wall_s", "run_cpu_s", "peak_rss_mb", "phases_s"}
old_root, new_root = Path(sys.argv[1]), Path(sys.argv[2])
budget = sys.argv[3] if len(sys.argv) > 3 else "16"


def compare(old: object, new: object, path: str, report: dict) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        if set(old) != set(new):
            report["structural"].append(f"{path}: keys {sorted(set(old) ^ set(new))}")
        for key in sorted(old.keys() & new.keys()):
            if key in TIMING_KEYS:
                continue
            if key == "pattern_supported" and isinstance(old[key], int) and new[key] is None:
                report["expected"] += 1
                continue
            compare(old[key], new[key], f"{path}.{key}", report)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            report["structural"].append(f"{path}: lengths {len(old)} != {len(new)}")
        for index, (left, right) in enumerate(zip(old, new)):
            compare(left, right, f"{path}[{index}]", report)
        return
    if isinstance(old, str) and isinstance(new, str):
        old, new = old.replace(str(old_root), "<run>"), new.replace(str(new_root), "<run>")
    if isinstance(old, float) and isinstance(new, float):
        if (math.isnan(old) and math.isnan(new)) or old == new:
            return
        report["drifted"] += 1
        if abs(old - new) > report["max_delta"]:
            report["max_delta"], report["max_path"] = abs(old - new), path
        return
    if old != new:
        report["structural"].append(f"{path}: {old!r} != {new!r}")


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


failures = 0
new_budget = new_root / f"budget{budget}"
for new_path in sorted(new_budget.glob("*/*.json.gz")) + sorted(new_budget.glob("*/*/*.json.gz")):
    relative = new_path.relative_to(new_root)
    report = {"structural": [], "expected": 0, "drifted": 0, "max_delta": 0.0, "max_path": None}
    compare(read(old_root / relative), read(new_path), "$", report)
    clean = not report["structural"] and not report["drifted"]
    failures += not clean
    print(f"{relative}: {'SAME' if clean else 'DIFFERS'}; pattern_supported None {report['expected']}, "
          f"structural {len(report['structural'])}, float drift {report['drifted']} "
          f"(max {report['max_delta']:.2e} at {report['max_path']})")
    for difference in report["structural"][:6]:
        print("   ", difference[:300])

for new_path in sorted(new_budget.glob("arrays/*.npz")):
    relative = new_path.relative_to(new_root)
    with np.load(old_root / relative) as old_arrays, np.load(new_path) as new_arrays:
        names_match = set(old_arrays.files) == set(new_arrays.files)
        unequal = [name for name in sorted(set(old_arrays.files) & set(new_arrays.files))
                   if not np.array_equal(old_arrays[name], new_arrays[name],
                                         equal_nan=old_arrays[name].dtype.kind in "fc")]
    clean = names_match and not unequal
    failures += not clean
    print(f"{relative}: {'SAME' if clean else 'DIFFERS'}; array names match {names_match}, unequal {unequal[:6]}")

print(f"\nfiles differing: {failures}")
sys.exit(1 if failures else 0)
