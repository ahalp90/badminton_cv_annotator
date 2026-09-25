"""Compare a run_d17.py rerun with the baseline arm, file by file.

Equal means the same keys in the same order, the same JSON types and the same values, with
-0.0 and 0.0 counted as different. Durations and peak memory, measurements of the run itself,
are skipped; stage names and call counts are compared. Arrays must match in name, dtype, shape
and raw bytes. A missing file or an empty folder counts as a difference.

Usage: compare_rerun.py BASELINE_ARM RERUN_ARM   (each holds d17/, populations/, case_records/, ...)
"""

import gzip
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Every value under these keys is a duration in seconds, apart from peak_rss_mb (memory).
RUN_MEASUREMENT_KEYS = {"peak_rss_mb", "run_cpu_s", "run_wall_s", "startup_s", "phases_s", "cpu_s", "elapsed_s",
                        "elapsed_seconds", "timings_seconds", "seconds", "self_seconds"}
JSON_FOLDERS = ("d17", "populations/G0", "populations/G1", "case_records", "inputs")


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def json_differences(left, right, path: str, found: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if list(left) != list(right):
            found.append(f"{path}: keys or key order differ")
        for key in left:
            if key in right and key not in RUN_MEASUREMENT_KEYS:
                json_differences(left[key], right[key], f"{path}.{key}", found)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append(f"{path}: lengths {len(left)} and {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            json_differences(left_item, right_item, f"{path}[{index}]", found)
        return
    signs_differ = isinstance(left, float) and math.copysign(1.0, left) != math.copysign(1.0, right)
    if type(left) is not type(right) or left != right or signs_differ:
        found.append(f"{path}: {str(left)[:80]} != {str(right)[:80]}")


def array_differences(left_path: Path, right_path: Path) -> list[str]:
    with np.load(left_path, allow_pickle=False) as left, np.load(right_path, allow_pickle=False) as right:
        if left.files != right.files:
            return [f"array names differ: {left.files} vs {right.files}"]
        found = []
        for name in left.files:
            left_array, right_array = left[name], right[name]
            if left_array.dtype != right_array.dtype or left_array.shape != right_array.shape:
                found.append(f"{name}: {left_array.dtype}{left_array.shape} vs {right_array.dtype}{right_array.shape}")
            elif left_array.tobytes() != right_array.tobytes():
                found.append(f"{name}: values differ")
        return found


def main() -> int:
    baseline, rerun = Path(sys.argv[1]), Path(sys.argv[2])
    patterns: Counter[str] = Counter()
    unequal_files = []
    for folder, suffix in [(name, ".json.gz") for name in JSON_FOLDERS] + [("arrays", ".npz")]:
        names = sorted(path.name for path in (baseline / folder).glob(f"*{suffix}"))
        rerun_names = sorted(path.name for path in (rerun / folder).glob(f"*{suffix}"))
        if names != rerun_names or not names:
            print(f"{folder}: file sets differ or are empty; only baseline {sorted(set(names) - set(rerun_names))}, "
                  f"only rerun {sorted(set(rerun_names) - set(names))}")
            unequal_files.append((folder, ["file sets differ or are empty"]))
        equal = 0
        for name in sorted(set(names) & set(rerun_names)):
            if suffix == ".npz":
                found = array_differences(baseline / folder / name, rerun / folder / name)
            else:
                found = []
                json_differences(read_json_gz(baseline / folder / name), read_json_gz(rerun / folder / name), "$",
                                 found)
            if found:
                unequal_files.append((f"{folder}/{name}", found))
                patterns.update(re.sub(r"\[\d+\]", "[]", line.split(":")[0]) for line in found)
            else:
                equal += 1
        print(f"{folder}: {equal} of {len(names)} files equal")
    for file_name, found in unequal_files[:12]:
        print(f"\n{file_name}: {len(found)} differences")
        for line in found[:5]:
            print("  " + line)
    if patterns:
        print("\nDifference paths, all files:")
        for pattern, count in patterns.most_common(25):
            print(f"  {count:7d}  {pattern}")
    return 1 if unequal_files else 0


if __name__ == "__main__":
    sys.exit(main())
