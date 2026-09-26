"""Check the rerun's path-valued differences against the baseline arm.

For the fields compare_rerun.py reports, every differing string must name its own arm folder,
with the same remainder after it, and both files must exist. Any other difference fails.

Usage: check_paths.py BASELINE_ARM RERUN_ARM
"""

import gzip
import json
import sys
from pathlib import Path

FIELDS = {
    "d17": ["selection.polarity_refit.source_record"],
    "case_records": ["population_sources.G0", "population_sources.G1", "provenance.g0_source"],
}


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def field(record: dict, dotted: str):
    for key in dotted.split("."):
        if record is None or key not in record:
            return None
        record = record[key]
    return record


def string_pairs(left, right, path: str, problems: list[str]):
    """Yield (path, left, right) for each differing string leaf; record any other difference."""
    if isinstance(left, dict) and isinstance(right, dict) and list(left) == list(right):
        for key in left:
            yield from string_pairs(left[key], right[key], f"{path}.{key}", problems)
    elif isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            yield from string_pairs(left_item, right_item, f"{path}[{index}]", problems)
    elif isinstance(left, str) and isinstance(right, str):
        if left != right:
            yield path, left, right
    elif type(left) is not type(right) or left != right:
        problems.append(f"{path}: non-path difference {str(left)[:80]} != {str(right)[:80]}")


def main() -> int:
    baseline, rerun = Path(sys.argv[1]), Path(sys.argv[2])
    problems: list[str] = []
    checked = 0
    for folder, names in FIELDS.items():
        for baseline_file in sorted((baseline / folder).glob("*.json.gz")):
            left_record = read_json_gz(baseline_file)
            right_record = read_json_gz(rerun / folder / baseline_file.name)
            for name in names:
                label = f"{folder}/{baseline_file.name}:{name}"
                for path, left, right in string_pairs(field(left_record, name), field(right_record, name), label,
                                                      problems):
                    checked += 1
                    left_rest = left.removeprefix(f"{baseline}/") if left.startswith(f"{baseline}/") else None
                    right_rest = right.removeprefix(f"{rerun}/") if right.startswith(f"{rerun}/") else None
                    exists = Path(left).exists() and Path(right).exists()
                    ok = left_rest is not None and left_rest == right_rest and exists
                    print(f"{'ok  ' if ok else 'FAIL'} {path} -> {right_rest} (both exist: {exists})")
                    if not ok:
                        problems.append(f"{path}: {left} vs {right}")
    print(f"\n{checked} differing path strings checked; {len(problems)} problems")
    for problem in problems:
        print("  " + problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
