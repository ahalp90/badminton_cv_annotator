"""Exactly compare two pickled harness outputs, reporting the first differing paths (throwaway)."""

from __future__ import annotations

import math
import pickle
import sys
from dataclasses import fields, is_dataclass

import numpy as np

differences: list[str] = []
largest_float_delta = 0.0


def compare(first: object, second: object, path: str) -> None:
    global largest_float_delta
    if len(differences) >= 20:
        return
    if is_dataclass(first) and not isinstance(first, type):
        for field in fields(first):
            compare(getattr(first, field.name), getattr(second, field.name), f"{path}.{field.name}")
        return
    if isinstance(first, dict):
        if set(first) != set(second):
            differences.append(f"{path}: keys {sorted(set(first) ^ set(second))}")
        for key in first.keys() & second.keys():
            compare(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)):
        if len(first) != len(second):
            differences.append(f"{path}: lengths {len(first)} != {len(second)}")
        for index, (left, right) in enumerate(zip(first, second)):
            compare(left, right, f"{path}[{index}]")
        return
    if isinstance(first, np.ndarray):
        if first.shape != second.shape or not np.array_equal(first, second, equal_nan=first.dtype.kind == "f"):
            differences.append(f"{path}: arrays differ")
        return
    if isinstance(first, float) and isinstance(second, float):
        if not (first == second or (math.isnan(first) and math.isnan(second))):
            largest_float_delta = max(largest_float_delta, abs(first - second))
            differences.append(f"{path}: {first!r} != {second!r}")
        return
    if first != second:
        differences.append(f"{path}: {first!r} != {second!r}")


with open(sys.argv[1], "rb") as stream:
    baseline = pickle.load(stream)
with open(sys.argv[2], "rb") as stream:
    candidate = pickle.load(stream)
compare(baseline, candidate, "$")
print("bit-identical" if not differences else f"{len(differences)}+ differences, largest float delta {largest_float_delta}")
for difference in differences:
    print(" ", difference[:300])
