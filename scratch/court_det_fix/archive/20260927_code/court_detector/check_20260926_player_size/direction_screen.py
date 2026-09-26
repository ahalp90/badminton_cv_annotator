"""Screen: two other ways to combine the scoring stage's lengthwise and crosswise scores.

Today the final choice scores a court as 90% paint plus 10% fragment support plus the net-post
bonus, where paint and fragment support each take the weaker of the court's two directions.
E1 blends paint and fragment support within each direction, then takes the weaker direction.
E2 takes the harmonic mean of the two blended directions. A view whose ranking falls back to
fragment support alone keeps today's score in both.

Reads saved artefacts only, and checks that today's rule reproduces every saved score and pick.
Floor errors are before the final refit, median/largest in metres, on views with landmark hand
marks; "ends" gives the largest error in each half of the court, split at the net.
Usage, from the worktree root:
  python scratch/court_det_fix/court_detector/check_20260926_player_size/direction_screen.py \
    default=ARTEFACTS_DIR player_size=ARTEFACTS_DIR
    ARTEFACTS_DIR: an arm's artefacts/ from run_carmack.sh (left on Carmack, see left_on_carmack.tsv)
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

GEOMETRY_WEIGHT = 0.1
PAINT = "q_paint10_span_weighted"
GEOMETRY = "q_geom_span_weighted"
RULES = ("today", "E1", "E2")

manifest = statistics.read(statistics.MANIFEST)
rows_by_view = {row["case_id"]: row for row in manifest["cases"]}
references = statistics.load_references(manifest)


def rule_scores(evidence: dict, criterion: str, saved: dict) -> dict[str, float]:
    if criterion != PAINT:
        return dict.fromkeys(RULES, saved["combined_score"])
    directional = evidence["directional"]
    paint, geometry = (np.array([directional[f"{axis}_{name}"] for axis in ("lengthwise", "transverse")])
                       for name in (PAINT, GEOMETRY))
    blended = (1 - GEOMETRY_WEIGHT) * paint + GEOMETRY_WEIGHT * geometry  # (lengthwise, crosswise)
    today = (1 - GEOMETRY_WEIGHT) * paint.min() + GEOMETRY_WEIGHT * geometry.min() + saved["bonus"]
    harmonic = 0.0 if blended.sum() == 0 else 2 * blended.prod() / blended.sum()
    return {"today": today, "E1": blended.min() + saved["bonus"], "E2": harmonic + saved["bonus"]}


def floor_errors(view: str, homography: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    landmarks = references.get(view, {}).get("landmarks")
    if not landmarks:
        return None
    height, width = cv2.imread(str(COURT_ROOT / rows_by_view[view]["image"])).shape[:2]
    marked = np.array([m["image_px"] for m in landmarks]) / max(1.0, max(height, width) / 960)
    court_m = np.array([m["court_m"] for m in landmarks])
    floor = np.column_stack((marked, np.ones(len(marked)))) @ np.linalg.inv(homography).T
    return np.linalg.norm(floor[:, :2] / floor[:, 2:] - court_m, axis=1), court_m[:, 1]


def describe(view: str, homography: np.ndarray) -> tuple[str, np.ndarray | None]:
    measured = floor_errors(view, homography)
    if measured is None:
        return "no marks", None
    errors, along = measured
    ends = f"ends {errors[along < 6.7].max():.2f}, {errors[along >= 6.7].max():.2f}"
    return f"{np.median(errors):.2f}/{errors.max():.2f} ({ends})", errors


print("arm\tview\t" + "\t".join(RULES) + "\t(* = a different court from today's)")
for spec in sys.argv[1:]:
    arm, _, folder = spec.partition("=")
    totals = np.zeros((len(RULES), 2))
    for path in sorted(Path(folder).glob("*.json.gz")):
        view = path.name.removesuffix(".json.gz")
        with gzip.open(path) as handle:
            artefact = json.load(handle)
        record = artefact["w5"]["record"]
        candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
        criterion = record["rankings"]["C"]["r2_criterion"]
        rows = []
        for saved in artefact["net_choice"]["rows"]:
            scores = rule_scores(candidates[saved["origin_key"]]["evidence"], criterion, saved)
            assert abs(scores["today"] - saved["combined_score"]) < 1e-10, (view, saved["origin_key"])
            rows.append((saved["origin_key"], scores))
        # max keeps the first row on exact ties, as the net choice does.
        picks = [max(rows, key=lambda row: row[1][rule])[0] if rows else None for rule in RULES]
        assert picks[0] == artefact["net_choice"]["chosen"], (view, picks[0])
        if picks[0] is None:
            continue
        homographies = [np.asarray(candidates[key]["homography_working"]) for key in picks]
        cells = []
        for i, homography in enumerate(homographies):
            text, errors = describe(view, homography)
            if errors is not None:
                totals[i] += np.median(errors), errors.max()
            cells.append(text + ("*" if not np.array_equal(homography, homographies[0]) else ""))
        if any(cell.endswith("*") for cell in cells):
            print(f"{arm}\t{view}\t" + "\t".join(cells))
    print(f"{arm}\ttotal over the marked views\t" + "\t".join(f"{m:.2f}/{x:.2f}" for m, x in totals))
