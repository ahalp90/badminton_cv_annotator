#!/usr/bin/env python3
"""Recursively compare JSON or JSON.GZ records with explicit numeric tolerances."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def json_files(root: Path) -> dict[str, Path]:
    if root.is_file():
        # Two explicit files may legitimately have different basenames.
        return {"$file": root}
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith(".json") or name.endswith(".json.gz"):
            result[str(path.relative_to(root))] = path
    return result


class Comparator:
    def __init__(
        self,
        *,
        atol: float,
        rtol: float,
        ignore_key: re.Pattern[str] | None,
        max_diffs: int,
    ) -> None:
        self.atol = atol
        self.rtol = rtol
        self.ignore_key = ignore_key
        self.max_diffs = max_diffs
        self.diffs: list[str] = []
        self.compared = 0
        self.ignored = 0
        self.max_abs_delta = 0.0
        self.max_rel_delta = 0.0

    def add(self, path: str, message: str) -> None:
        if len(self.diffs) < self.max_diffs:
            self.diffs.append(f"{path}: {message}")

    def compare(self, first: Any, second: Any, path: str = "$") -> None:
        self.compared += 1
        if isinstance(first, dict) and isinstance(second, dict):
            first_keys = {
                key for key in first
                if not (self.ignore_key and self.ignore_key.search(str(key)))
            }
            second_keys = {
                key for key in second
                if not (self.ignore_key and self.ignore_key.search(str(key)))
            }
            self.ignored += (len(first) - len(first_keys)) + (len(second) - len(second_keys))
            if first_keys != second_keys:
                self.add(
                    path,
                    f"key sets differ: only-first={sorted(first_keys-second_keys)!r}, "
                    f"only-second={sorted(second_keys-first_keys)!r}",
                )
            for key in sorted(first_keys & second_keys, key=str):
                self.compare(first[key], second[key], f"{path}.{key}")
            return

        if isinstance(first, list) and isinstance(second, list):
            if len(first) != len(second):
                self.add(path, f"list lengths differ: {len(first)} != {len(second)}")
            for index, (left, right) in enumerate(zip(first, second)):
                self.compare(left, right, f"{path}[{index}]")
            return

        # bool is an int subclass; compare it as a strict scalar.
        if isinstance(first, bool) or isinstance(second, bool):
            if type(first) is not type(second) or first != second:
                self.add(path, f"values differ: {first!r} != {second!r}")
            return

        numeric = isinstance(first, (int, float)) and isinstance(second, (int, float))
        if numeric:
            left = float(first)
            right = float(second)
            if math.isnan(left) or math.isnan(right):
                if not (math.isnan(left) and math.isnan(right)):
                    self.add(path, f"NaN mismatch: {first!r} != {second!r}")
                return
            if math.isinf(left) or math.isinf(right):
                if left != right:
                    self.add(path, f"infinity mismatch: {first!r} != {second!r}")
                return
            delta = abs(left - right)
            scale = max(abs(left), abs(right))
            relative = delta / scale if scale else 0.0
            self.max_abs_delta = max(self.max_abs_delta, delta)
            self.max_rel_delta = max(self.max_rel_delta, relative)
            if delta > self.atol + self.rtol * scale:
                self.add(
                    path,
                    f"numeric mismatch: {first!r} != {second!r}; "
                    f"abs={delta:.17g}, rel={relative:.17g}",
                )
            return

        if type(first) is not type(second) or first != second:
            self.add(
                path,
                f"values differ ({type(first).__name__}/{type(second).__name__}): "
                f"{first!r} != {second!r}",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument(
        "--ignore-key-regex",
        help="Ignore dictionary keys whose name matches this regular expression",
    )
    parser.add_argument("--max-diffs", type=int, default=100)
    args = parser.parse_args()

    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    if args.max_diffs <= 0:
        parser.error("--max-diffs must be positive")

    baseline_files = json_files(args.baseline)
    candidate_files = json_files(args.candidate)
    if set(baseline_files) != set(candidate_files):
        print(
            "file sets differ\n"
            f"only baseline: {sorted(set(baseline_files)-set(candidate_files))}\n"
            f"only candidate: {sorted(set(candidate_files)-set(baseline_files))}",
            file=sys.stderr,
        )
        return 2

    ignore = re.compile(args.ignore_key_regex) if args.ignore_key_regex else None
    total_diffs: list[str] = []
    compared = ignored = 0
    max_abs = max_rel = 0.0

    for relative in sorted(baseline_files):
        comparator = Comparator(
            atol=args.atol,
            rtol=args.rtol,
            ignore_key=ignore,
            max_diffs=max(1, args.max_diffs - len(total_diffs)),
        )
        comparator.compare(
            read_json(baseline_files[relative]),
            read_json(candidate_files[relative]),
        )
        total_diffs.extend(f"{relative} {item}" for item in comparator.diffs)
        compared += comparator.compared
        ignored += comparator.ignored
        max_abs = max(max_abs, comparator.max_abs_delta)
        max_rel = max(max_rel, comparator.max_rel_delta)
        if len(total_diffs) >= args.max_diffs:
            break

    print(
        json.dumps(
            {
                "files": len(baseline_files),
                "nodes_compared": compared,
                "ignored_key_occurrences": ignored,
                "max_absolute_numeric_delta": max_abs,
                "max_relative_numeric_delta": max_rel,
                "differences_reported": len(total_diffs),
            },
            indent=2,
        )
    )
    for difference in total_diffs:
        print(difference, file=sys.stderr)

    return 1 if total_diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
