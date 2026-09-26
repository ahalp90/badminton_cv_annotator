"""Compare the player-size arm with the default on the 28 test views: picks, floor-metre errors, controls, work, time.

Errors are hand-mark errors in floor metres after the final refit: each view's median and largest
(check_20260926_line_paint/compare_line_paint.floor_errors). The keep rule is in README.md.

Usage, from the repository root: compare_player_size.py RUN FINAL_RUN
  RUN: run_carmack.sh's output, with blend_default/ and player_size/
  FINAL_RUN: the final 26 September Carmack run, whose blend_default arm this run's switch-off arm must match
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from scratch.court_det_fix.court_detector.check_20260926_line_paint.compare_line_paint import (
    COURT_ROOT,
    corner_gap,
    floor_errors,
    read_gz,
    statistics,
)

ARMS = ("blend_default", "player_size")
MARGIN_M = 0.1
TOLERANCE_NATIVE_PX = 1e-4
# The 8 views that show no court. The default detector already wrongly finds one on 14336 and 100347.
CONTROLS = tuple(f"sset_21_gloiZ_gTJaE_frame_{frame:08d}"
                 for frame in (300, 9558, 14336, 23893, 38228, 62120, 90790, 100347))
# One per matched direction pair, G0 and G1 alike: "<view> pair <id> [a, b] combined N players M retained K seconds S".
PAIR_LINE = re.compile(r"^(\S+) pair \d+ \[[^]]*\] combined \d+ players (\d+) retained \d+ seconds (\S+)$")


def read_results(folder: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}


def search_work(logs: Path) -> dict[str, tuple[int, float]]:
    """Per view: courts fully scored and seconds spent in matched pairs, over both searches."""
    courts, seconds = defaultdict(int), defaultdict(float)
    for log in sorted(logs.glob("*.log")):
        for line in log.read_text().splitlines():
            match = PAIR_LINE.match(line)
            if match:
                view, scored, pair_seconds = match.groups()
                courts[view] += int(scored)
                seconds[view] += float(pair_seconds)
    return {view: (courts[view], seconds[view]) for view in courts}


def main() -> None:
    run, final_run = Path(sys.argv[1]), Path(sys.argv[2])
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    cases = {row["case_id"]: row for row in manifest["cases"]}
    results = {arm: read_results(run / arm / "results") for arm in ARMS}
    views = sorted(results["blend_default"])
    for arm in ARMS:
        if sorted(results[arm]) != views:
            raise ValueError(f"the {arm} arm's views differ from the blend_default arm's")
        failed = [view for view, row in results[arm].items() if row["error"] is not None]
        print(f"{arm}: {len(results[arm])} views, errors in {failed or 'none'}")

    final = read_results(final_run / "blend_default" / "results")
    if views != sorted(final):
        raise ValueError(f"this run has {len(views)} views, the final run {len(final)}")
    differences = []
    for view in views:
        ours, theirs = results["blend_default"][view], final[view]
        same_pick = (ours["chosen_key"], ours["no_court_reason"]) == (theirs["chosen_key"], theirs["no_court_reason"])
        # The same code on the same host, so the corners should match exactly, in the same order.
        gap = (None if ours["corners_native_px"] is None or theirs["corners_native_px"] is None
               else float(np.abs(np.subtract(ours["corners_native_px"], theirs["corners_native_px"])).max()))
        if not same_pick or (gap is not None and gap > TOLERANCE_NATIVE_PX):
            differences.append(f"{view} (pick {'same' if same_pick else 'differs'}, corners {gap})")
    print(f"switch-off arm against the final run's default: {', '.join(differences) or 'identical picks and corners'}")

    print("\npicks: outcome per arm; player_size against blend_default: same chosen court? / largest corner move, native px")
    # The 8 controls and sset_21_gloiZ_gTJaE_frame_00000001, which shows a court, have no hand marks.
    print("view\thand marks\tblend_default\tplayer_size\tagainst the default")
    for view in views:
        blend, player = results["blend_default"][view], results["player_size"][view]
        marks = references.get(view) or {}
        marked = "landmarks" if marks.get("landmarks") else "corners" if marks else "none"
        print(f"{view}\t{marked}\t{blend['no_court_reason'] or 'court'}\t"
              f"{player['no_court_reason'] or 'court'}\t"
              f"{'same' if player['chosen_key'] == blend['chosen_key'] else 'changed'} / "
              f"{corner_gap(blend['corners_native_px'], player['corners_native_px'])}")

    changes = []
    for view in CONTROLS:
        blend_court, player_court = (results[arm][view]["no_court_reason"] is None for arm in ARMS)
        if player_court != blend_court:
            changes.append(f"{view} {'gains' if player_court else 'loses'} a court")
    print(f"\ncontrols with player_size against blend_default: {', '.join(changes) or 'no court gained or lost'}")

    print(f"\nhand-mark floor error after the refit, metres: median / largest; flagged when either moves by more than {MARGIN_M}")
    print("view\tblend_default\tplayer_size\tflag")
    for view in [view for view in views if references.get(view, {}).get("landmarks")]:
        landmarks = references[view]["landmarks"]
        height, width = cv2.imread(str(COURT_ROOT / cases[view]["image"])).shape[:2]
        native_per_working = max(1.0, max(height, width) / 960)
        errors = {}
        for arm in ARMS:
            refit = read_gz(run / arm / "artefacts" / f"{view}.json.gz").get("stripe_refit")
            errors[arm] = (None if refit is None or results[arm][view]["corners_native_px"] is None
                           else floor_errors(refit["corrected"]["homography_working"], landmarks, native_per_working))
        cells = ["no court" if errors[arm] is None else f"{np.median(errors[arm]):.2f} / {errors[arm].max():.2f}"
                 for arm in ARMS]
        if any(errors[arm] is None for arm in ARMS):
            flag = "outcome differs" if (errors["blend_default"] is None) != (errors["player_size"] is None) else ""
        else:
            changes = [np.median(errors["player_size"]) - np.median(errors["blend_default"]),
                       errors["player_size"].max() - errors["blend_default"].max()]
            flag = ("worse" if max(changes) > MARGIN_M else "better" if min(changes) < -MARGIN_M else "")
        print(f"{view}\t{cells[0]}\t{cells[1]}\t{flag}")

    print("\nwork and time per view: courts fully scored in the searches, seconds in matched pairs, detect seconds")
    work = {arm: search_work(run / arm / "logs") for arm in ARMS}
    print("view\t" + "\t".join(f"{arm}: courts, pair s, detect s" for arm in ARMS))
    totals = {arm: np.zeros(3) for arm in ARMS}
    for view in views:
        cells = []
        for arm in ARMS:
            courts, seconds = work[arm].get(view, (0, 0.0))
            detect = results[arm][view].get("detect_seconds") or 0.0
            totals[arm] += (courts, seconds, detect)
            cells.append(f"{courts}, {seconds:.0f}, {detect:.0f}")
        print(f"{view}\t" + "\t".join(cells))
    blend_totals, player_totals = totals["blend_default"], totals["player_size"]
    print("all\t" + "\t".join(f"{int(total[0])}, {total[1]:.0f}, {total[2]:.0f}" for total in totals.values()))
    print("player_size against blend_default: " + ", ".join(
        f"{label} {100 * (player / blend - 1):+.1f}%" for label, player, blend in
        zip(("courts", "pair seconds", "detect seconds"), player_totals, blend_totals, strict=True)))


if __name__ == "__main__":
    main()
