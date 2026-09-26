"""The gap-bounded paint test against the upright-camera check: chosen courts, floor-metre errors and time.

For each view: whether the chosen court or the outcome changed, and how far the final corners
moved in native px. For each view with landmark hand marks: the largest hand-mark error in floor
metres, before and after the final refit, for both runs. Each hand mark is mapped back onto the
floor through the court, and the error is how far it lands from where it should be. Then total
detect seconds; both runs had self-checks and artefacts on.

Usage: compare_paint_test.py OLD_RUN NEW_RUN
  each a folder with results/ and artefacts/: the upright check's upright/ run and this check's
  paint_test/ run
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def corner_gap(first: list | None, second: list | None) -> str:
    """Largest corner distance, allowing the court's 180-degree relabelling."""
    if first is None or second is None:
        return "-"
    old, new = np.asarray(first), np.asarray(second)
    direct = np.linalg.norm(old - new, axis=1).max()
    rotated = np.linalg.norm(old - new[[2, 3, 0, 1]], axis=1).max()
    return f"{min(direct, rotated):.1f}"


def largest_floor_error(homography_working: list, landmarks: list, native_per_working: float) -> float:
    marked_working = np.array([landmark["image_px"] for landmark in landmarks]) / native_per_working
    floor = np.column_stack((marked_working, np.ones(len(landmarks)))) @ np.linalg.inv(homography_working).T
    court_m = np.array([landmark["court_m"] for landmark in landmarks])
    return float(np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1).max())


old_run, new_run = Path(sys.argv[1]), Path(sys.argv[2])
references = statistics.load_references(statistics.read(statistics.MANIFEST))
results = {}
print("view\tsame chosen key\tcorner gap px\told outcome\tnew outcome")
for path in sorted((old_run / "results").glob("*.json")):
    old, new = json.loads(path.read_text()), json.loads((new_run / "results" / path.name).read_text())
    if new["error"] is not None:
        raise RuntimeError(f"{path.stem}: {new['error']}")
    results[path.stem] = (old, new)
    outcomes = [result["no_court_reason"] or "court" for result in (old, new)]
    print(f"{path.stem}\t{old['chosen_key'] == new['chosen_key']}\t"
          f"{corner_gap(old['corners_native_px'], new['corners_native_px'])}\t{outcomes[0]}\t{outcomes[1]}")

print("\nlargest hand-mark error, floor metres: before refit / after refit")
print("view\tupright check\tgap-bounded paint test")
for view in results:
    landmarks = references.get(view, {}).get("landmarks")
    if not landmarks:
        continue
    cells = []
    for run in (old_run, new_run):
        refit = read_gz(run / "artefacts" / f"{view}.json.gz")["stripe_refit"]
        final = refit["corrected"]
        # Only the refitted court records both corner sets; the scale is the same for the whole view.
        native_per_working = (np.ptp(np.asarray(final["corners_native_px"])[:, 0])
                              / np.ptp(np.asarray(final["corners_working_px"])[:, 0]))
        before, after = (largest_floor_error(court["homography_working"], landmarks, native_per_working)
                         for court in (refit["selected_geometry"], final))
        cells.append(f"{before:.2f} / {after:.2f}")
    print(f"{view}\t" + "\t".join(cells))

old_seconds = sum(old["detect_seconds"] for old, _ in results.values())
new_seconds = sum(new["detect_seconds"] for _, new in results.values())
print(f"\ndetect seconds over {len(results)} views: upright check {old_seconds:.0f}, "
      f"gap-bounded paint test {new_seconds:.0f} ({new_seconds / old_seconds - 1:+.0%})")
