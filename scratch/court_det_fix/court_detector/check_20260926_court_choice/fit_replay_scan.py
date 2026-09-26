"""Replay W5's saved fit for every court a refit arm could refit, on every view, and report the misses.

The courts are each view's 15 best by the net choice's score, with and without the geometry blend. Each
fit is replayed as the stripe refit's self-check replays it. A miss is a gap above 0.0001 native px from
the fit W5 saved on Carmack. For a miss, the fit is repeated from a start nudged by 1e-9 working px, to see
whether it is unstable on this machine too.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.check_20260926_court_choice.fit_replay_scan UPRIGHT_RUN
"""

import os
import sys
from pathlib import Path

# Before numpy loads: one thread per process, as run_views.py does.
for thread_variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[thread_variable] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import json

import cv2
import numpy as np

from experiments.annotator.independent_court import fixed_stripe_refit as fitting
from scratch.court_det_fix.court_detector import stripe_refit
from scratch.court_det_fix.court_detector.detect import (
    ROOT,
    freeze_arrays,
    load_live_modules,
    source_record,
)
from scratch.court_det_fix.court_detector.inputs import ViewInputs
from scratch.court_det_fix.court_detector.run_views import (
    VIEWS,
    pack_sources,
    read_json_gz,
)

TOLERANCE_NATIVE_PX = 1e-4
NUDGE_WORKING_PX = 1e-9


def shortlist_keys(rows: list[dict], candidates: dict[str, dict], geometry_weight: float) -> list[str]:
    """The 15 best courts by the net choice's score, with geometry_weight of the geometry score blended in."""
    def score(row: dict) -> float:
        geometry = candidates[row["origin_key"]]["evidence"]["q_geom_span_weighted"]
        return (1 - geometry_weight) * row["paint_score"] + geometry_weight * geometry + row["bonus"]
    return [row["origin_key"] for row in sorted(rows, key=lambda row: -score(row))[:15]]


def main() -> None:
    upright = Path(sys.argv[1])
    live = load_live_modules()
    verifier = live.verifier
    centres = verifier.paint_geometry.CENTRE_SEGMENTS_M
    sources, provenances, frame_paths = pack_sources(verifier.CASE_PACKS)
    anchors = {view["case_id"]: view["anchor"] for view in json.loads(VIEWS.read_text())}
    replayed, misses = 0, []
    for saved_path in sorted((upright / "results").glob("*.json")):
        view_id = saved_path.stem
        artefacts = read_json_gz(upright / "artefacts" / f"{view_id}.json.gz")
        rows = artefacts["net_choice"]["rows"]
        if not rows:
            continue
        record = artefacts["w5"]["record"]
        candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
        keys = dict.fromkeys(shortlist_keys(rows, candidates, 0.0) + shortlist_keys(rows, candidates, 0.1))
        source = sources[view_id]
        frame = cv2.imread(str(frame_paths.get(view_id) or verifier.frame_path(ROOT, source, provenances[view_id])))
        view = ViewInputs(view_id, frame, anchors[view_id], (anchors[view_id], anchors[view_id]),
                          np.asarray(source["segments_px"], dtype=float).reshape(-1, 4),
                          np.asarray(source["bbox_px"], dtype=float).reshape(-1, 4), provenances[view_id])
        context = verifier.view_context(view_id, source_record(view, artefacts["feet"]["all_feet_px"]),
                                        view.provenance, frame, view_id)
        freeze_arrays(context)
        scale = np.asarray(context.native_size, dtype=float) / np.asarray(context.size, dtype=float)
        with live.prepared_measurements(verifier):
            for key in keys:
                selected, parent, attempt = stripe_refit.selected_parent(record, key)
                homography = np.asarray(parent["homography_working"], dtype=float)
                constraints = fitting.prepare(homography, context.observations,
                                              parent["evidence"]["stripe_assignments"], context.weights,
                                              centres=centres)
                starting = verifier.detector.project(homography[None], verifier.detector.CORNER_COURT_M)[0][0]
                fit = fitting.refine(starting, constraints, context.size, use_positions=True, centres=centres)
                saved = np.asarray(attempt["attempted_corners_native"] if attempt else selected["corners_px"])
                gap = np.abs(np.asarray(fit["corners_px"]) * scale - saved).max()
                replayed += 1
                if gap <= TOLERANCE_NATIVE_PX:
                    continue
                nudged = fitting.refine(starting + NUDGE_WORKING_PX, constraints, context.size, use_positions=True,
                                        centres=centres)
                jump = (np.abs(np.asarray(nudged["corners_px"]) - np.asarray(fit["corners_px"])) * scale).max()
                misses.append(f"{view_id}\t{key}\tgap {gap:.3g} native px\tjump under the nudge {jump:.2g} px\t"
                              f"status {fit.get('status')}\trank {fit.get('jacobian_rank')}")
    print(f"{replayed} courts replayed, {len(misses)} missed by more than {TOLERANCE_NATIVE_PX} native px")
    print("\n".join(misses))


if __name__ == "__main__":
    main()
