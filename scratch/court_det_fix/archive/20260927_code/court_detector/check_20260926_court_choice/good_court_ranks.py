"""Where the good courts sit in today's choice order, on the views with landmark hand marks.

Order: paint score plus net bonus, as the net choice ranks them. Errors are the largest hand-mark
error in floor metres before the final refit. A good court is within 0.1 m of the best eligible one.
Usage, from the worktree root: python good_court_ranks.py ARTEFACTS_DIR
"""
import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path("scratch/court_det_fix")
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

manifest = statistics.read(statistics.MANIFEST)
rows_by_view = {row["case_id"]: row for row in manifest["cases"]}
references = statistics.load_references(manifest)

print("view\teligible\tchosen error\tbest eligible error\tranks of good courts (error)")
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    landmarks = references.get(view, {}).get("landmarks")
    if not landmarks:
        continue
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    record = artefact["w5"]["record"]
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    ranked = sorted(artefact["net_choice"]["rows"], key=lambda row: -row["combined_score"])
    height, width = cv2.imread(str(COURT_ROOT / rows_by_view[view]["image"])).shape[:2]
    native_per_working = max(1.0, max(height, width) / 960)
    marked = np.array([landmark["image_px"] for landmark in landmarks]) / native_per_working
    court_m = np.array([landmark["court_m"] for landmark in landmarks])
    errors = []
    for row in ranked:
        homography = np.asarray(candidates[row["origin_key"]]["homography_working"])
        floor = np.column_stack((marked, np.ones(len(marked)))) @ np.linalg.inv(homography).T
        errors.append(float(np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1).max()))
    best = min(errors)
    good = [f"{rank}({error:.2f})" for rank, error in enumerate(errors, start=1) if error <= best + 0.1][:8]
    print(f"{view}\t{len(ranked)}\t{errors[0]:.2f}\t{best:.2f}\t{' '.join(good)}")
