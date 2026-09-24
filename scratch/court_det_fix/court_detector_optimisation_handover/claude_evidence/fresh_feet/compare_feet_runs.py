"""Compare full-search D17 runs with different player feet against the baseline run.

Throwaway evaluation script for follow-up items 10 and 11. Per view: whether the bounded choice
and the corrected court match the baseline, whether a non-court control picked a court, run
time, and how much work reached scoring: axis hypotheses scored (the `scored` diagnostic; absent
in the baseline) and combined courts sent to distance-map scoring (`geometry_players`).

Usage (from the checkout's scratch/court_det_fix/d17_timing): compare_feet_runs.py BASELINE_ARM NAME=ARM...
An arm is a directory holding d17/ and populations/, such as RUN/budget16.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd()))
import compare_d17  # pyrefly: ignore[missing-import]


def read(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def work_counts(arm: Path, case_id: str) -> dict:
    """Summed over G0 and G1 matched pairs: axis hypotheses scored and combined courts scored."""
    scored, sent_to_scoring = 0, 0
    for population in ("G0", "G1"):
        for pair in read(arm / f"populations/{population}/{case_id}.json.gz")["pairs"]:
            if pair["status"] != "matched":
                continue
            role = pair["role"]
            axis_counts = [axis["diagnostics"].get("scored") for axis in role["axes"]]
            scored = None if scored is None or None in axis_counts else scored + sum(axis_counts)
            sent_to_scoring += role["geometry_players"]
    return {"axes_scored": scored, "courts_scored": sent_to_scoring}


def view_row(arm: Path, case_id: str) -> dict:
    summary = read(arm / f"d17/{case_id}.json.gz")
    selection = summary["selection"]
    corrected = compare_d17.corrected_geometry(selection["polarity_refit"])
    return {"wall_s": summary["run_wall_s"] + summary["startup_s"].get("interpreter_start", 0.0),
            "bounded": selection["bounded"],
            "corrected": None if corrected is None else corrected["corners_native_px"],
            "scale": np.asarray(summary["native_size_wh"], dtype=float) / np.asarray(summary["working_size_wh"], dtype=float),
            **work_counts(arm, case_id)}


def main() -> None:
    baseline_arm = Path(sys.argv[1])
    arms = {name: Path(path) for name, path in (argument.split("=", 1) for argument in sys.argv[2:])}
    manifest = compare_d17.statistics.read(compare_d17.statistics.MANIFEST)
    non_court = {row["case_id"] for row in manifest["cases"] if row.get("reference_status") == "non_court"}
    case_ids = sorted(path.name.removesuffix(".json.gz") for path in (baseline_arm / "d17").glob("*.json.gz"))
    totals = {name: {"views": 0, "wall_s": 0.0, "axes_scored": 0, "courts_scored": 0, "changed": []}
              for name in ["baseline", *arms]}
    print("case | wall baseline / " + " / ".join(arms) + " s | " + " | ".join(
        f"{name}: same pick, corrected diff px" for name in arms) + " | courts scored baseline / " + " / ".join(arms))
    for case_id in case_ids:
        baseline = view_row(baseline_arm, case_id)
        rows = {"baseline": baseline}
        for name, arm in arms.items():
            if (arm / f"d17/{case_id}.json.gz").exists():
                rows[name] = view_row(arm, case_id)
        cells = []
        for name in arms:
            row = rows.get(name)
            if row is None:
                cells.append(f"{name}: missing")
                continue
            same = row["bounded"] == baseline["bounded"]
            if row["corrected"] is not None and baseline["corrected"] is not None:
                difference = f"{compare_d17.corner_difference_working_px(row['corrected'], baseline['corrected'], baseline['scale']):.2f}"
            else:
                difference = f"picked {row['bounded'] is not None} vs {baseline['bounded'] is not None}"
            cells.append(f"{name}: {same}, {difference}")
            if not same:
                totals[name]["changed"].append(case_id)
        for name, row in rows.items():
            total = totals[name]
            total["views"] += 1
            total["wall_s"] += row["wall_s"]
            total["axes_scored"] = None if total["axes_scored"] is None or row["axes_scored"] is None \
                else total["axes_scored"] + row["axes_scored"]
            total["courts_scored"] += row["courts_scored"]
        label = f"{case_id} (non-court)" if case_id in non_court else case_id
        walls = " / ".join(f"{rows[name]['wall_s']:.0f}" if name in rows else "-" for name in ["baseline", *arms])
        courts = " / ".join(str(rows[name]["courts_scored"]) if name in rows else "-" for name in ["baseline", *arms])
        print(f"{label} | {walls} | " + " | ".join(cells) + f" | {courts}")
    print()
    for name, total in totals.items():
        print(f"{name}: {json.dumps(total)}")


if __name__ == "__main__":
    main()
