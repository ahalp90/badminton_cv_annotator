"""Compare the court-choice arms with the baseline arm: picks, floor-metre errors, the rescored refits and time.

Errors are the largest hand-mark error in floor metres: each hand mark is mapped back onto the floor through
the court, and the error is how far it lands from where it should be. The keep rule is in README.md.

Usage, from the repository root: compare_court_choice.py REPLAY_OUT UPRIGHT_RUN
  REPLAY_OUT: replay_court_choice.py's output, one folder per arm
  UPRIGHT_RUN: the upright check's run folder, to check the replay covers all its views
"""

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(COURT_ROOT / "net_recovery/statistics"))
import paired_reference_analysis as statistics  # pyrefly: ignore[missing-import]

ARMS = ("blend", "refit", "both")
REFIT_ARMS = ("refit", "both")
KEEP_MARGIN_M = 0.2


def read_gz(path: Path) -> dict:
    with gzip.open(path) as handle:
        return json.load(handle)


def corner_gap(first: list | None, second: list | None) -> str:
    """Largest corner distance in native px, allowing the court's 180-degree relabelling."""
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


replay_out, upright_run = Path(sys.argv[1]), Path(sys.argv[2])
manifest = statistics.read(statistics.MANIFEST)
references = statistics.load_references(manifest)
images = {row["case_id"]: row["image"] for row in manifest["cases"]}
views = sorted(path.stem for path in (replay_out / "baseline" / "results").glob("*.json"))
upright_views = sorted(path.stem for path in (upright_run / "results").glob("*.json"))
if views != upright_views:
    raise ValueError("the replay's views differ from the upright run's")
results = {arm: {view: json.loads((replay_out / arm / "results" / f"{view}.json").read_text()) for view in views}
           for arm in ("baseline",) + ARMS}
# A view and arm whose replay check failed has no result; those views are listed, then left out.
failed = {view: [arm for arm in results if results[arm][view]["error"] is not None] for view in views}
for view, arms in failed.items():
    if arms:
        print(f"{view}: replay failed in {', '.join(arms)}: {results[arms[0]][view]['error']}")
views = [view for view in views if not failed[view]]
print()

print("picks against the baseline arm: same chosen court? / largest corner move, native px / outcome")
print("view\tbaseline outcome\t" + "\t".join(ARMS))
for view in views:
    base = results["baseline"][view]
    cells = []
    for arm in ARMS:
        new = results[arm][view]
        outcome = new["no_court_reason"] or "court"
        cells.append(f"{'same' if new['chosen_key'] == base['chosen_key'] else 'changed'} / "
                     f"{corner_gap(base['corners_native_px'], new['corners_native_px'])} / {outcome}")
    print(f"{view}\t{base['no_court_reason'] or 'court'}\t" + "\t".join(cells))

print("\nlargest hand-mark error, floor metres: before refit / after refit")
print("view\tbaseline\t" + "\t".join(ARMS))
after_refit = {arm: {} for arm in ("baseline",) + ARMS}
marked_views = [view for view in views if references.get(view, {}).get("landmarks")]
for view in marked_views:
    landmarks = references[view]["landmarks"]
    height, width = cv2.imread(str(COURT_ROOT / images[view])).shape[:2]
    native_per_working = max(1.0, max(height, width) / 960)
    cells = []
    for arm in ("baseline",) + ARMS:
        refit = read_gz(replay_out / arm / "artefacts" / f"{view}.json.gz")["stripe_refit"]
        before, after = (largest_floor_error(court["homography_working"], landmarks, native_per_working)
                         for court in (refit["selected_geometry"], refit["corrected"]))
        after_refit[arm][view] = after
        cells.append(f"{before:.2f} / {after:.2f}")
    print(f"{view}\t" + "\t".join(cells))

print(f"\nkeep rule, after refit, against the baseline arm (margin {KEEP_MARGIN_M} m)")
print("arm\tviews better\tviews worse\ttotal error over the marked views")
print(f"baseline\t-\t-\t{sum(after_refit['baseline'].values()):.2f}")
for arm in ARMS:
    changes = {view: after_refit[arm][view] - after_refit["baseline"][view] for view in marked_views}
    better = [view for view, change in changes.items() if change < -KEEP_MARGIN_M]
    worse = [view for view, change in changes.items() if change > KEEP_MARGIN_M]
    print(f"{arm}\t{', '.join(better) or '-'}\t{', '.join(worse) or '-'}\t{sum(after_refit[arm].values()):.2f}")

print("\nrefitted courts on the marked views: the winner's error after refit, the best error among the")
print("refitted courts, and where that best court placed after rescoring (1 = it won)")
print("view\t" + "\t".join(REFIT_ARMS))
for view in marked_views:
    landmarks = references[view]["landmarks"]
    height, width = cv2.imread(str(COURT_ROOT / images[view])).shape[:2]
    native_per_working = max(1.0, max(height, width) / 960)
    cells = []
    for arm in REFIT_ARMS:
        rows = read_gz(replay_out / arm / "artefacts" / f"{view}.json.gz")["refit_choice"]
        scored = [(row["refitted_score"], largest_floor_error(row["corrected"]["homography_working"], landmarks,
                                                              native_per_working))
                  for row in rows if row["refitted_score"] is not None]
        best_error = min(error for _, error in scored)
        best_score = max(score for score, error in scored if error == best_error)
        place = 1 + sum(score > best_score for score, _ in scored)
        cells.append(f"{after_refit[arm][view]:.2f} / {best_error:.2f} / {place} of {len(scored)}")
    print(f"{view}\t" + "\t".join(cells))

base_seconds = sum(results["baseline"][view]["choose_seconds"] for view in views)
print(f"\nseconds in choose_court over {len(views)} views, on the laptop")
print(f"baseline\t{base_seconds:.1f}")
for arm in ARMS:
    seconds = sum(results[arm][view]["choose_seconds"] for view in views)
    print(f"{arm}\t{seconds:.1f}\textra {seconds - base_seconds:+.1f} s")
