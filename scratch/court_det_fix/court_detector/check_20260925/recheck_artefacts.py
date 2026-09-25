"""Recheck the correctness run's saved artefacts against the baseline with run_views.baseline_checks.

The detector is not rerun: each view's result and artefacts come from the correctness run.
Run from the checkout root, at the commit whose baseline_checks should apply.

Usage: python recheck_artefacts.py CORRECTNESS_DIR BASELINE_ARM FEET_FILE
"""

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "src")]

from scratch.court_det_fix.court_detector.detect import CourtResult
from scratch.court_det_fix.court_detector.run_views import (
    SHOT_CHECK,
    baseline_checks,
    read_json_gz,
)


def main() -> int:
    correctness, baseline, feet_file = (Path(argument) for argument in sys.argv[1:4])
    feet_by_view = read_json_gz(feet_file)
    shot_rows = {row["case_id"]: row for row in map(json.loads, SHOT_CHECK.read_text().splitlines())}
    failed = []
    result_files = sorted((correctness / "results").glob("*.json"))
    for result_file in result_files:
        row = json.loads(result_file.read_text())
        view_id = row["view_id"]
        # baseline_checks reads only the chosen key from the result.
        result = CourtResult(view_id, None, row["no_court_reason"], row["chosen_key"], None)
        artefacts = read_json_gz(correctness / "artefacts" / f"{view_id}.json.gz")
        checks = baseline_checks(view_id, result, artefacts, baseline, feet_by_view, shot_rows)
        differing = {name: difference for name, difference in checks.items() if difference is not None}
        print(view_id, len(checks), "checks", "equal" if not differing else f"DIFFERENT {differing}")
        if differing:
            failed.append(view_id)
    print(f"{len(result_files)} views rechecked; failed: {failed}")
    return 1 if failed or len(result_files) != 28 else 0


if __name__ == "__main__":
    sys.exit(main())
