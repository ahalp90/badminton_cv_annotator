"""Exactly compare two SVD-search result trees, ignoring timing and output-path fields (throwaway)."""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

IGNORED_KEYS = {"timing", "cpu_s", "elapsed_s", "wall_s", "generation_log", "generation_record"}
baseline_root, patched_root = Path(sys.argv[1]), Path(sys.argv[2])


def compare(first: object, second: object, path: str, differences: list[str]) -> None:
    if len(differences) >= 10:
        return
    if isinstance(first, dict) and isinstance(second, dict):
        if set(first) != set(second):
            differences.append(f"{path}: keys {sorted(set(first) ^ set(second))}")
        for key in sorted(first.keys() & second.keys()):
            if key not in IGNORED_KEYS:
                compare(first[key], second[key], f"{path}.{key}", differences)
        return
    if isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            differences.append(f"{path}: lengths {len(first)} != {len(second)}")
        for index, (left, right) in enumerate(zip(first, second)):
            compare(left, right, f"{path}[{index}]", differences)
        return
    both_nan = isinstance(first, float) and isinstance(second, float) and math.isnan(first) and math.isnan(second)
    if first != second and not both_nan:
        differences.append(f"{path}: {first!r} != {second!r}")


for patched_path in sorted(patched_root.glob("*/baseline/*.json.gz")):
    relative = patched_path.relative_to(patched_root)
    with gzip.open(baseline_root / relative, "rt") as stream:
        baseline = json.load(stream)
    with gzip.open(patched_path, "rt") as stream:
        patched = json.load(stream)
    differences: list[str] = []
    compare(baseline, patched, "$", differences)
    print(relative, "identical" if not differences else f"{len(differences)}+ differences")
    for difference in differences:
        print("   ", difference[:300])
