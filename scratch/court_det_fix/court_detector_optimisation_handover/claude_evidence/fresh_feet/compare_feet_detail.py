"""Reference errors and pair-loop stage times for the feet-variant D17 runs (throwaway).

Usage (from the checkout's scratch/court_det_fix/d17_timing): compare_feet_detail.py NAME=ARM...
"""

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import compare_d17  # pyrefly: ignore[missing-import]

STAGE_ENDINGS = ("pair/match_axis", "pair/player_fractions", "pair/finite_scores", "pair")


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def main() -> None:
    arms = {name: Path(path) for name, path in (argument.split("=", 1) for argument in sys.argv[1:])}
    manifest = compare_d17.statistics.read(compare_d17.statistics.MANIFEST)
    manifest_rows = {row["case_id"]: row for row in manifest["cases"]}
    references = compare_d17.statistics.load_references(manifest)
    case_ids = sorted(path.name.removesuffix(".json.gz") for path in (next(iter(arms.values())) / "d17").glob("*.json.gz"))
    stage_totals = {name: defaultdict(float) for name in arms}
    print("case | reference median working px per arm: " + " / ".join(arms))
    for case_id in case_ids:
        reference = compare_d17.measurable_reference(case_id, manifest_rows, references)
        cells = []
        for name, arm in arms.items():
            summary = read(arm / f"d17/{case_id}.json.gz")
            for stage, value in summary["stages"].items():
                for ending in STAGE_ENDINGS:
                    if stage.endswith("/" + ending):
                        stage_totals[name][ending] += value["seconds"]
            sizes = {"native_size_wh": summary["native_size_wh"], "working_size_wh": summary["working_size_wh"]}
            geometry = compare_d17.corrected_geometry(summary["selection"]["polarity_refit"])
            result = compare_d17.reference_summary(sizes, geometry, reference)
            cells.append("-" if result is None else f"{result['median_working_px']:.2f}")
        print(f"{case_id} | " + " / ".join(cells))
    print()
    for name, totals in stage_totals.items():
        print(name, {ending: round(seconds) for ending, seconds in totals.items()})


if __name__ == "__main__":
    main()
