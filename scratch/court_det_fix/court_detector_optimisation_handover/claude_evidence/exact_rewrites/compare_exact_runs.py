"""Check that two D17 run folders hold the same results, bit for bit apart from timings (throwaway).

Compares every file under the two folders. JSON values in .json.gz files must match exactly: floats
by their bits, so -0.0 differs from 0.0, and types strictly, so True differs from 1. Only timing
values are skipped, and a saved path may differ only in its run-folder prefix. Arrays in .npz files
must match in dtype, shape and bytes. Any other file must match byte for byte. A file present in
only one folder counts as a difference.

Usage: compare_exact_runs.py OLD_RUN_DIR NEW_RUN_DIR
"""

from __future__ import annotations

import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np

# Values that are timings, skipped wherever they appear. "stages" is handled separately: its
# seconds are skipped and its call counts are listed for review.
TIMING_KEYS = {"cpu_s", "elapsed_s", "wall_s", "elapsed_seconds", "process_s", "timings_seconds",
               "startup_s", "run_wall_s", "run_cpu_s", "peak_rss_mb", "phases_s"}
# Absolute, because saved paths are absolute; no symlink resolution, so they match as written.
old_root, new_root = Path(sys.argv[1]).absolute(), Path(sys.argv[2]).absolute()


def without_run_prefix(text: str, root: Path) -> str:
    return "<run>" + text[len(str(root)):] if text.startswith(str(root)) else text


def compare(old: object, new: object, path: str, report: dict) -> None:
    if isinstance(old, dict) and isinstance(new, dict):
        old_keys, new_keys = set(old) - TIMING_KEYS, set(new) - TIMING_KEYS
        if old_keys != new_keys:
            report["structural"].append(f"{path}: keys {sorted(old_keys ^ new_keys)}")
        for key in sorted(old_keys & new_keys):
            if key == "stages":
                # Call counts describe the work done, not the results, and the shared-maps change moves
                # distance_maps on purpose. So they are listed for review rather than counted as failures.
                for stage in sorted(old[key].keys() | new[key].keys()):
                    old_calls = old[key].get(stage, {}).get("calls")
                    new_calls = new[key].get(stage, {}).get("calls")
                    if old_calls != new_calls:
                        report["stage_calls"].append(f"{stage}: calls {old_calls} -> {new_calls}")
            else:
                compare(old[key], new[key], f"{path}.{key}", report)
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            report["structural"].append(f"{path}: lengths {len(old)} != {len(new)}")
        for index, (left, right) in enumerate(zip(old, new, strict=False)):  # a length mismatch is already reported
            compare(left, right, f"{path}[{index}]", report)
        return
    if isinstance(old, str) and isinstance(new, str):
        old, new = without_run_prefix(old, old_root), without_run_prefix(new, new_root)
    if isinstance(old, float) and isinstance(new, float):
        if struct.pack("<d", old) == struct.pack("<d", new):
            return
        report["drifted"] += 1
        delta = abs(old - new)
        if delta > report["max_delta"] or report["max_path"] is None:
            report["max_delta"], report["max_path"] = delta, path
        return
    if type(old) is not type(new) or old != new:
        report["structural"].append(f"{path}: {old!r} != {new!r}")


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def same_arrays(old_path: Path, new_path: Path) -> tuple[bool, list[str]]:
    with np.load(old_path) as old_arrays, np.load(new_path) as new_arrays:
        names_match = set(old_arrays.files) == set(new_arrays.files)
        unequal = []
        for name in sorted(set(old_arrays.files) & set(new_arrays.files)):
            old_array, new_array = old_arrays[name], new_arrays[name]
            if (old_array.dtype, old_array.shape, old_array.tobytes()) != (new_array.dtype, new_array.shape, new_array.tobytes()):
                unequal.append(name)
    return names_match, unequal


def relative_files(root: Path) -> set[Path]:
    return {path.relative_to(root) for path in root.rglob("*") if path.is_file()}


old_files, new_files = relative_files(old_root), relative_files(new_root)
failures = len(old_files ^ new_files)
for relative in sorted(old_files ^ new_files):
    print(f"{relative}: ONLY IN {'old' if relative in old_files else 'new'}")
compared = 0
for relative in sorted(old_files & new_files):
    compared += 1
    old_path, new_path = old_root / relative, new_root / relative
    if relative.name.endswith(".json.gz"):
        report = {"structural": [], "drifted": 0, "max_delta": 0.0, "max_path": None, "stage_calls": []}
        compare(read(old_path), read(new_path), "$", report)
        for difference in report["stage_calls"]:
            print(f"{relative}: stage {difference} (review; not counted)")
        clean = not report["structural"] and not report["drifted"]
        if not clean:
            print(f"{relative}: DIFFERS; structural {len(report['structural'])}, float drift {report['drifted']} "
                  f"(max {report['max_delta']:.2e} at {report['max_path']})")
            for difference in report["structural"][:6]:
                print("   ", difference[:300])
    elif relative.suffix == ".npz":
        names_match, unequal = same_arrays(old_path, new_path)
        clean = names_match and not unequal
        if not clean:
            print(f"{relative}: DIFFERS; array names match {names_match}, unequal {unequal[:6]}")
    else:
        clean = old_path.read_bytes() == new_path.read_bytes()
        if not clean:
            print(f"{relative}: DIFFERS (raw bytes)")
    failures += not clean

print(f"files compared: {compared}; files differing or missing: {failures}")
sys.exit(1 if failures or not compared else 0)
