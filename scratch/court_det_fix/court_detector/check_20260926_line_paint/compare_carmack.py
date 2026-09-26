"""Compare the final Carmack run's two arms: floor-metre errors, the keep rule, agreement with the replay, and time.

Errors are the largest hand-mark error in floor metres, before and after the final refit. The keep rule is
in README.md: line paint against the blend default.

Usage, from the repository root: compare_carmack.py CARMACK_RUN REPLAY_OUT UPRIGHT_RUN
  CARMACK_RUN: run_carmack.sh's output, with blend_default/ and line_paint/
  REPLAY_OUT: replay_line_paint.py's output for the same code (v3/replay/)
  UPRIGHT_RUN: the upright check's run folder, for its detect time
"""

import json
import sys
from pathlib import Path

import cv2

from scratch.court_det_fix.court_detector.check_20260926_line_paint.compare_line_paint import (
    COURT_ROOT,
    KEEP_MARGIN_M,
    corner_gap,
    largest_floor_error,
    read_gz,
    statistics,
)

ARMS = {"blend_default": "blend", "line_paint": "line_paint"}  # Carmack arm: its replay arm


def read_results(folder: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(folder.glob("*.json"))}


def main() -> None:
    carmack, replay, upright = (Path(argument) for argument in sys.argv[1:4])
    manifest = statistics.read(statistics.MANIFEST)
    references = statistics.load_references(manifest)
    images = {row["case_id"]: row["image"] for row in manifest["cases"]}
    upright_results = read_results(upright / "results")
    results = {arm: read_results(carmack / arm / "results") for arm in ARMS}
    for arm in ARMS:
        if sorted(results[arm]) != sorted(upright_results):
            raise ValueError(f"the {arm} arm's views differ from the upright run's")
        failed = [view for view, row in results[arm].items() if row["error"] is not None]
        print(f"{arm}: {len(results[arm])} views, errors in {failed or 'none'}")
    views = sorted(upright_results)

    print("\npicks: outcome per arm; line paint against blend: same chosen court? / largest corner move, native px")
    print("view\tblend_default\tline_paint\tline paint against blend")
    for view in views:
        blend, line = results["blend_default"][view], results["line_paint"][view]
        print(f"{view}\t{blend['no_court_reason'] or 'court'}\t{line['no_court_reason'] or 'court'}\t"
              f"{'same' if line['chosen_key'] == blend['chosen_key'] else 'changed'} / "
              f"{corner_gap(blend['corners_native_px'], line['corners_native_px'])}")

    print("\nagreement with the local replay: views whose chosen court or outcome differs, or whose corners")
    print("differ by more than 0.0001 native px")
    for arm, replay_arm in ARMS.items():
        replayed = read_results(replay / replay_arm / "results")
        differences = []
        for view in views:
            local, remote = replayed[view], results[arm][view]
            if local["error"] is not None:
                differences.append(f"{view} (not replayable on the laptop)")
                continue
            same_pick = (local["chosen_key"], local["no_court_reason"]) == (remote["chosen_key"],
                                                                            remote["no_court_reason"])
            gap = corner_gap(local["corners_native_px"], remote["corners_native_px"])
            if not same_pick or (gap != "-" and float(gap) > 1e-4):
                differences.append(f"{view} (pick {'same' if same_pick else 'differs'}, corners {gap})")
        print(f"{arm}: {', '.join(differences) or 'none'}")

    print("\nlargest hand-mark error, floor metres: before refit / after refit")
    print("view\t" + "\t".join(ARMS))
    after_refit = {arm: {} for arm in ARMS}
    marked_views = [view for view in views if references.get(view, {}).get("landmarks")]
    for view in marked_views:
        landmarks = references[view]["landmarks"]
        height, width = cv2.imread(str(COURT_ROOT / images[view])).shape[:2]
        native_per_working = max(1.0, max(height, width) / 960)
        cells = []
        for arm in ARMS:
            refit = read_gz(carmack / arm / "artefacts" / f"{view}.json.gz")["stripe_refit"]
            before, after = (largest_floor_error(court["homography_working"], landmarks, native_per_working)
                             for court in (refit["selected_geometry"], refit["corrected"]))
            after_refit[arm][view] = after
            cells.append(f"{before:.2f} / {after:.2f}")
        print(f"{view}\t" + "\t".join(cells))

    print(f"\nkeep rule, after refit, line_paint against blend_default (margin {KEEP_MARGIN_M} m)")
    changes = {view: after_refit["line_paint"][view] - after_refit["blend_default"][view] for view in marked_views}
    print(f"views better: {', '.join(view for view, change in changes.items() if change < -KEEP_MARGIN_M) or '-'}")
    print(f"views worse: {', '.join(view for view, change in changes.items() if change > KEEP_MARGIN_M) or '-'}")
    for arm in ARMS:
        print(f"total error over the marked views, {arm}: {sum(after_refit[arm].values()):.2f}")

    print("\ndetect seconds over the 28 views, self-checks on, 8 views at a time on Carmack")
    upright_seconds = sum(row["detect_seconds"] for row in upright_results.values())
    print(f"upright check (W5's paint alone)\t{upright_seconds:.0f}")
    for arm in ARMS:
        seconds = sum(row["detect_seconds"] for row in results[arm].values())
        print(f"{arm}\t{seconds:.0f}\tagainst the upright check {100 * (seconds / upright_seconds - 1):+.1f}%")


if __name__ == "__main__":
    main()
