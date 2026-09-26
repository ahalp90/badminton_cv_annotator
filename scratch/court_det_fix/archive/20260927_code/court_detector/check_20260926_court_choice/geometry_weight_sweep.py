"""Re-pick each view's court with (1 - w) x paint + w x geometry + net bonus, from the upright run's saved rows.

Errors are the largest hand-mark error in floor metres, before the final refit, as in Sol's check.
Usage, from the worktree root: python geometry_weight_sweep.py ARTEFACTS_DIR
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

WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

manifest = statistics.read(statistics.MANIFEST)
rows_by_view = {row["case_id"]: row for row in manifest["cases"]}
references = statistics.load_references(manifest)


def floor_error(homography: list, landmarks: list, native_per_working: float) -> float:
    marked = np.array([landmark["image_px"] for landmark in landmarks]) / native_per_working
    floor = np.column_stack((marked, np.ones(len(marked)))) @ np.linalg.inv(np.asarray(homography)).T
    court_m = np.array([landmark["court_m"] for landmark in landmarks])
    return float(np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1).max())


def pick(net_rows: list[dict], geometry: dict[str, float], weight: float) -> str | None:
    best_key, best_score = None, -np.inf
    for row in net_rows:
        score = (1 - weight) * row["paint_score"] + weight * geometry[row["origin_key"]] + row["bonus"]
        if score > best_score:
            best_key, best_score = row["origin_key"], score
    return best_key


changed_counts = {weight: 0 for weight in WEIGHTS}
table = []
for path in sorted(Path(sys.argv[1]).glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    with gzip.open(path) as handle:
        artefact = json.load(handle)
    record = artefact["w5"]["record"]
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    net_rows = artefact["net_choice"]["rows"]
    if not net_rows:
        continue
    geometry = {row["origin_key"]: candidates[row["origin_key"]]["evidence"]["q_geom_span_weighted"]
                for row in net_rows}
    assert pick(net_rows, geometry, 0.0) == artefact["net_choice"]["chosen"], view
    picks = {weight: pick(net_rows, geometry, weight) for weight in WEIGHTS}
    for weight in WEIGHTS:
        changed_counts[weight] += picks[weight] != picks[0.0]
    landmarks = references.get(view, {}).get("landmarks")
    if landmarks:
        height, width = cv2.imread(str(COURT_ROOT / rows_by_view[view]["image"])).shape[:2]
        native_per_working = max(1.0, max(height, width) / 960)
        best_available = min(floor_error(candidates[row["origin_key"]]["homography_working"], landmarks,
                                         native_per_working) for row in net_rows)
        errors = [floor_error(candidates[picks[weight]]["homography_working"], landmarks, native_per_working)
                  for weight in WEIGHTS]
        table.append((view, len(net_rows), best_available, errors))

print("largest hand-mark error before the final refit, floor metres, by geometry weight")
print("view\teligible\tbest eligible\t" + "\t".join(f"w={weight:g}" for weight in WEIGHTS))
for view, count, best_available, errors in table:
    print(f"{view}\t{count}\t{best_available:.2f}\t" + "\t".join(f"{error:.2f}" for error in errors))
print("worst\t\t\t" + "\t".join(f"{max(row[3][index] for row in table):.2f}" for index in range(len(WEIGHTS))))
print("sum\t\t\t" + "\t".join(f"{sum(row[3][index] for row in table):.2f}" for index in range(len(WEIGHTS))))
print("picks changed from w=0, all views with eligible courts\t\t\t"
      + "\t".join(str(changed_counts[weight]) for weight in WEIGHTS))
