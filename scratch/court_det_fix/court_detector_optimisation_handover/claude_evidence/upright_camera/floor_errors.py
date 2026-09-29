"""Hand-mark errors in floor metres for two runs' final courts, and why the chosen court changed.

The pixel measure (reference_errors.py) hides one-line slips at the far end of the court, where
neighbouring lines sit a few pixels apart. Here each hand mark is mapped back onto the floor
through the court's homography, and the error is how far it lands from where it should be.

For each view whose chosen court changed, it also prints where the new winner sat in the old
run's G0 and G1 shortlists, and both winners' final scores in the new run.

Usage: python floor_errors.py <old artefact dir> <new artefact dir> <old populations dir>
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

old_dir, new_dir, old_populations = (Path(argument) for argument in sys.argv[1:4])
references = statistics.load_references(statistics.read(statistics.MANIFEST))


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def floor_errors(court: dict, landmarks: list) -> np.ndarray:
    """Metres on the floor between each hand mark, mapped through the court, and its true position."""
    corners_native, corners_working = np.asarray(court["corners_native_px"]), np.asarray(court["corners_working_px"])
    native_per_working = np.ptp(corners_native[:, 0]) / np.ptp(corners_working[:, 0])
    marked_working = np.array([landmark["image_px"] for landmark in landmarks]) / native_per_working
    floor = np.column_stack((marked_working, np.ones(len(landmarks)))) @ np.linalg.inv(court["homography_working"]).T
    court_m = np.array([landmark["court_m"] for landmark in landmarks])
    return np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1)


def old_shortlist_places(view: str, occurrences: list) -> list[str]:
    """Where each G0 or G1 occurrence of a court sat in the old run: overall list, else its pair's list."""
    places = []
    for occurrence in occurrences:
        if occurrence["source"] not in ("G0", "G1"):
            continue
        population = read_gz(old_populations / occurrence["source"] / f"{view}.json.gz")
        overall_ids = [entry["candidate_id"] for entry in population["entries"]]
        candidate_id = occurrence["candidate_id"]
        if candidate_id in overall_ids:
            places.append(f"{occurrence['source']}: overall #{overall_ids.index(candidate_id) + 1}")
            continue
        pair_id = int(candidate_id.split(":")[0])
        pair = next(pair for pair in population["pairs"] if pair["pair_id"] == pair_id)
        pair_ids = [entry["candidate_id"] for entry in pair["shortlist"]]
        pair_place = f"#{pair_ids.index(candidate_id) + 1} in its pair" if candidate_id in pair_ids else "not in its pair's list"
        cutoff = population["entries"][-1]["shortlist_score"]
        entry_score = pair["shortlist"][pair_ids.index(candidate_id)]["shortlist_score"] if candidate_id in pair_ids else None
        score_text = f", shortlist score {entry_score:.3f} vs overall cut-off {cutoff:.3f}" if entry_score is not None else ""
        places.append(f"{occurrence['source']}: not in the overall top {len(overall_ids)}, {pair_place}{score_text}")
    return places


print("view\told median / max m\tnew median / max m")
for path in sorted(old_dir.glob("*.json.gz")):
    view = path.name.removesuffix(".json.gz")
    landmarks = references.get(view, {}).get("landmarks")
    if not landmarks:
        continue
    summaries = []
    for folder in (old_dir, new_dir):
        errors = floor_errors(read_gz(folder / path.name)["stripe_refit"]["corrected"], landmarks)
        summaries.append(f"{np.median(errors):.2f} / {errors.max():.2f}")
    print(f"{view}\t" + "\t".join(summaries))

print("\nviews whose chosen court changed")
for path in sorted(old_dir.glob("*.json.gz")):
    old, new = read_gz(path), read_gz(new_dir / path.name)
    old_key, new_key = old["net_choice"]["chosen"], new["net_choice"]["chosen"]
    if old_key == new_key or old_key is None or new_key is None:
        continue
    view = path.name.removesuffix(".json.gz")
    record = new["w5"]["record"]
    candidates = {candidate["origin_key"]: candidate for candidate in record["parents"] + record["valid_children"]}
    parent_key = candidates[new_key].get("parent_origin_key") or new_key
    scores = {row["origin_key"]: row["combined_score"] for row in new["net_choice"]["rows"]}
    old_score = f"{scores[old_key]:.4f}" if old_key in scores else "not a candidate"
    print(f"{view}: new winner {new_key} scores {scores[new_key]:.4f}, old winner {old_key} {old_score}")
    for place in old_shortlist_places(view, candidates[parent_key]["source_occurrences"]):
        print(f"  new winner in the old run, {place}")
