"""Reference errors of two runs' final courts, on the court views with hand-marked references.

Uses the D17 statistics' measure (paired_reference_analysis.reference_errors), in working px,
as compare_d17.py does. Prints the median and largest error per view for each run.

Usage: python reference_errors.py <old artefact dir> <new artefact dir>
"""

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

MAX_WORKING_DIMENSION = 960  # the detector's working-size limit (run_population.prepare)
old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
manifest = statistics.read(statistics.MANIFEST)
rows = {row["case_id"]: row for row in manifest["cases"]}
references = statistics.load_references(manifest)


def final_court(path: Path) -> dict | None:
    with gzip.open(path) as handle:
        refit = json.load(handle).get("stripe_refit")
    return refit["corrected"] if refit and refit["corrected"]["valid"] else None


def summary(sizes: dict, court: dict | None, reference: dict) -> str:
    if court is None:
        return "no court"
    _, distances = statistics.reference_errors(sizes, court, reference)
    return f"{np.median(distances):.2f} / {np.max(distances):.2f}"


print("view\told median / max px\tnew median / max px")
for path in sorted(old_dir.glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    reference = references.get(view, {})
    status = reference.get("reference_status", rows[view].get("reference_status"))
    if status == "view_unverified" or not (reference.get("landmarks") or reference.get("corners_px")):
        continue
    height, width = cv2.imread(str(COURT_ROOT / rows[view]["image"])).shape[:2]
    resize = min(1.0, MAX_WORKING_DIMENSION / max(width, height))
    sizes = {"native_size_wh": [width, height], "working_size_wh": [round(width * resize), round(height * resize)]}
    print(f"{view}\t{summary(sizes, final_court(path), reference)}\t"
          f"{summary(sizes, final_court(new_dir / path.name), reference)}")
