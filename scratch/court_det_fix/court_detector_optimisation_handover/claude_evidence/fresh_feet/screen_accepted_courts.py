"""Replay the player test on each view's accepted D17 court with every feet variant.

Throwaway screen for follow-up items 10 and 11. W5's full-court gate uses this same test, so a
variant that fails a view's accepted court here would lose that court in the full D17 run.

Usage (from the repo root): screen_accepted_courts.py ACCEPTED_COURTS_JSON VARIANTS_DIR
"""

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("scratch/court_det_fix")
sys.path[:0] = [".", "src", str(ROOT / "w5_holistic"), str(ROOT / "wider_evaluation")]
import run_cases  # pyrefly: ignore[missing-import]

from experiments.annotator.independent_court import detector

VARIANTS = tuple(sys.argv[3:]) or ("everyone", "standing", "movers", "standing_movers")


def fractions(zone, corners_native: np.ndarray, all_feet_px: list, scale: np.ndarray) -> tuple[float, float]:
    """As run_diagnosis.gate_evidence: court and feet in working pixels, then zone's player test."""
    homography = cv2.getPerspectiveTransform(detector.CORNER_COURT_M, (corners_native / scale).astype(np.float32))
    feet = np.asarray([[[np.nan, np.nan] if foot is None else foot for foot in frame]
                       for frame in all_feet_px], dtype=float) / scale
    anyone, both_halves = zone.player_fractions(homography[None], feet)
    return float(anyone[0]), float(both_halves[0])


def main() -> None:
    accepted = json.loads(Path(sys.argv[1]).read_text())
    variants_dir = Path(sys.argv[2])
    _, verifier, runtime = run_cases.load_runtime(ROOT, ROOT / "wider_evaluation/runs/20260922/control_inputs.json.gz")
    feet = {"frozen": None}
    for name in VARIANTS:
        with gzip.open(variants_dir / f"feet_{name}.json.gz", "rt") as stream:
            feet[name] = json.load(stream)
    failures = {name: [] for name in feet}
    for case_id, court in accepted.items():
        if court["bounded"] is None:
            continue
        scale = np.asarray(court["native_size"]) / np.asarray(court["working_size"])
        corners = np.asarray(court["bounded_corners_native_px"], dtype=float)
        row = {}
        for name, feet_by_case in feet.items():
            all_feet = verifier.load_source(ROOT, case_id)["all_feet_px"] if feet_by_case is None else feet_by_case[case_id]
            anyone, both_halves = fractions(runtime["zone"], corners, all_feet, scale)
            passed = anyone == 1 and both_halves >= .5
            row[name] = f"{'pass' if passed else 'FAIL'} {anyone:.2f}/{both_halves:.2f}"
            if not passed:
                failures[name].append(case_id)
        print(f"{case_id:36s} " + "  ".join(f"{name}: {value}" for name, value in row.items()))
    print(json.dumps({name: len(cases) for name, cases in failures.items()}))


if __name__ == "__main__":
    main()
