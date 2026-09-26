"""Floor-metre hand-mark error of every candidate that reached the scoring stage, for one view.

Shows whether a better court than the winner was available. Candidates are compared before the
final stripe refit, which only the winner gets. The measure is floor_errors.py's.

Usage: python candidate_floor_errors.py <artefact dir> <view>
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

artefact_dir, view = Path(sys.argv[1]), sys.argv[2]
landmarks = statistics.load_references(statistics.read(statistics.MANIFEST))[view]["landmarks"]
court_m = np.array([landmark["court_m"] for landmark in landmarks])
with gzip.open(artefact_dir / f"{view}.json.gz") as handle:
    record = json.load(handle)
chosen = record["net_choice"]["chosen"]
winner_geometry = record["stripe_refit"]["corrected"]
native_per_working = (np.ptp(np.asarray(winner_geometry["corners_native_px"])[:, 0])
                      / np.ptp(np.asarray(winner_geometry["corners_working_px"])[:, 0]))
marked_working = np.array([landmark["image_px"] for landmark in landmarks]) / native_per_working
# Net choice picks the highest combined score among the candidates that passed the full-court checks.
net_rows = sorted(record["net_choice"]["rows"], key=lambda row: -row["combined_score"])
net_places = {row["origin_key"]: (place + 1, row) for place, row in enumerate(net_rows)}


def floor_errors(homography: list) -> np.ndarray:
    floor = np.column_stack((marked_working, np.ones(len(landmarks)))) @ np.linalg.inv(homography).T
    return np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1)


w5 = record["w5"]["record"]
rows = []
for candidate in w5["parents"] + w5["valid_children"]:
    errors = floor_errors(candidate["homography_working"])
    rows.append((errors.max(), float(np.median(errors)), candidate["origin_key"]))
rows.sort()
winner_max = next(maximum for maximum, _, key in rows if key == chosen)
better = [key for maximum, _, key in rows if maximum < winner_max]
print(f"{len(rows)} candidates scored, {len(net_rows)} passed the full-court checks")
print(f"winner {chosen}: largest error {winner_max:.2f} m")
print(f"{len(better)} candidates have a smaller largest error; {sum(key in net_places for key in better)} of them passed the checks")
print("\nlargest m\tmedian m\tcandidate\tnet-choice place\tpaint score\tnet bonus\tcombined score")
for maximum, median, key in rows[:10] + [row for row in rows if row[2] == chosen]:
    place, row = net_places.get(key, (None, None))
    if row is None:
        print(f"{maximum:.2f}\t{median:.2f}\t{key}\tdid not pass the checks")
        continue
    print(f"{maximum:.2f}\t{median:.2f}\t{key}\t{place}\t{row['paint_score']:.4f}\t{row['bonus']:.2f}\t{row['combined_score']:.4f}")
