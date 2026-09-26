"""Compare the line-paint arm with the blend arm: picks, floor-metre errors, the good courts' places and time.

Errors are the largest hand-mark error in floor metres: each hand mark is mapped back onto the floor through
the court, and the error is how far it lands from where it should be. The keep rule is in README.md.

Usage, from the repository root: compare_line_paint.py REPLAY_OUT UPRIGHT_RUN
  REPLAY_OUT: replay_line_paint.py's output, one folder per arm
  UPRIGHT_RUN: the upright check's run folder, for its views and saved W5 records
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

ARMS = ("paint_only", "blend", "line_paint")
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
views = sorted(path.stem for path in (upright_run / "results").glob("*.json"))
for arm in ARMS:
    if sorted(path.stem for path in (replay_out / arm / "results").glob("*.json")) != views:
        raise ValueError(f"the {arm} arm's views differ from the upright run's")
results = {arm: {view: json.loads((replay_out / arm / "results" / f"{view}.json").read_text()) for view in views}
           for arm in ARMS}
# A view and arm whose replay check failed has no result; those views are listed, then left out.
failed = {view: [arm for arm in ARMS if results[arm][view]["error"] is not None] for view in views}
for view, arms in failed.items():
    if arms:
        print(f"{view}: replay failed in {', '.join(arms)}: {results[arms[0]][view]['error']}")
views = [view for view in views if not failed[view]]
print(f"{len(views)} views fully replayed\n")

print("picks against the blend arm: same chosen court? / largest corner move, native px / outcome")
print("view\tblend outcome\tline_paint")
for view in views:
    blend, line = results["blend"][view], results["line_paint"][view]
    print(f"{view}\t{blend['no_court_reason'] or 'court'}\t"
          f"{'same' if line['chosen_key'] == blend['chosen_key'] else 'changed'} / "
          f"{corner_gap(blend['corners_native_px'], line['corners_native_px'])} / {line['no_court_reason'] or 'court'}")

print("\nlargest hand-mark error, floor metres: before refit / after refit")
print("view\t" + "\t".join(ARMS))
after_refit = {arm: {} for arm in ARMS}
marked_views = [view for view in views if references.get(view, {}).get("landmarks")]
scales = {}
for view in marked_views:
    landmarks = references[view]["landmarks"]
    height, width = cv2.imread(str(COURT_ROOT / images[view])).shape[:2]
    scales[view] = max(1.0, max(height, width) / 960)
    cells = []
    for arm in ARMS:
        refit = read_gz(replay_out / arm / "artefacts" / f"{view}.json.gz")["stripe_refit"]
        before, after = (largest_floor_error(court["homography_working"], landmarks, scales[view])
                         for court in (refit["selected_geometry"], refit["corrected"]))
        after_refit[arm][view] = after
        cells.append(f"{before:.2f} / {after:.2f}")
    print(f"{view}\t" + "\t".join(cells))

print(f"\nkeep rule, after refit, line_paint against the blend arm (margin {KEEP_MARGIN_M} m)")
changes = {view: after_refit["line_paint"][view] - after_refit["blend"][view] for view in marked_views}
better = [view for view, change in changes.items() if change < -KEEP_MARGIN_M]
worse = [view for view, change in changes.items() if change > KEEP_MARGIN_M]
print(f"views better: {', '.join(better) or '-'}")
print(f"views worse: {', '.join(worse) or '-'}")
for arm in ARMS:
    print(f"total error over the marked views, {arm}: {sum(after_refit[arm].values()):.2f}")

print("\ngated courts on the marked views, before refit: the best court's error, and its place under each")
print("arm's net-choice score (1 = it won), out of the gated courts")
print("view\tbest error\t" + "\t".join(ARMS))
for view in marked_views:
    landmarks = references[view]["landmarks"]
    record = read_gz(upright_run / "artefacts" / f"{view}.json.gz")["w5"]["record"]
    candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
    cells = []
    best_error = None
    for arm in ARMS:
        rows = read_gz(replay_out / arm / "artefacts" / f"{view}.json.gz")["net_choice"]["rows"]
        scored = [(row["combined_score"], largest_floor_error(candidates[row["origin_key"]]["homography_working"],
                                                              landmarks, scales[view]))
                  for row in rows]
        best_error = min(error for _, error in scored)
        best_score = max(score for score, error in scored if error == best_error)
        cells.append(f"{1 + sum(score > best_score for score, _ in scored)} of {len(scored)}")
    print(f"{view}\t{best_error:.2f}\t" + "\t".join(cells))

print(f"\nseconds in choose_court over {len(views)} views, on the laptop")
blend_seconds = sum(results["blend"][view]["choose_seconds"] for view in views)
for arm in ARMS:
    seconds = sum(results[arm][view]["choose_seconds"] for view in views)
    print(f"{arm}\t{seconds:.1f}\tagainst blend {seconds - blend_seconds:+.1f} s")
