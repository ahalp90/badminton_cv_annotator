"""Check that the rescore measures a court as W5 did: from its homography alone, W5's saved scores come back.

The refit's describe() measures a refitted court with verifier.measure_candidate, given only its homography.
This does the same for each view's 15 best unrefitted courts and compares with W5's saved paint and geometry
scores.

Usage, from the repository root:
  python -m scratch.court_det_fix.court_detector.check_20260926_court_choice.footing_check UPRIGHT_RUN VIEW [VIEW ...]
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


def main() -> None:
    upright, view_ids = Path(sys.argv[1]), sys.argv[2:]
    live = load_live_modules()
    verifier = live.verifier
    sources, provenances, frame_paths = pack_sources(verifier.CASE_PACKS)
    anchors = {view["case_id"]: view["anchor"] for view in json.loads(VIEWS.read_text())}
    for view_id in view_ids:
        artefacts = read_json_gz(upright / "artefacts" / f"{view_id}.json.gz")
        record = artefacts["w5"]["record"]
        candidates = {item["origin_key"]: item for item in record["parents"] + record["valid_children"]}
        source = sources[view_id]
        frame = cv2.imread(str(frame_paths.get(view_id) or verifier.frame_path(ROOT, source, provenances[view_id])))
        view = ViewInputs(view_id, frame, anchors[view_id], (anchors[view_id], anchors[view_id]),
                          np.asarray(source["segments_px"], dtype=float).reshape(-1, 4),
                          np.asarray(source["bbox_px"], dtype=float).reshape(-1, 4), provenances[view_id])
        context = verifier.view_context(view_id, source_record(view, artefacts["feet"]["all_feet_px"]),
                                        view.provenance, frame, view_id)
        freeze_arrays(context)
        ranked = sorted(artefacts["net_choice"]["rows"], key=lambda row: -row["combined_score"])[:15]
        paint_gaps, geometry_gaps = [], []
        with live.prepared_measurements(verifier):
            for row in ranked:
                saved = candidates[row["origin_key"]]
                homography = np.asarray(saved["homography_working"])
                evidence, _ = verifier.measure_candidate(context, {"homography_working": homography}, {})
                paint_gaps.append(abs(evidence["q_paint10_span_weighted"]
                                      - saved["evidence"]["q_paint10_span_weighted"]))
                geometry_gaps.append(abs(evidence["q_geom_span_weighted"] - saved["evidence"]["q_geom_span_weighted"]))
        print(f"{view_id}: largest paint-score gap {max(paint_gaps):.2e}, geometry-score gap "
              f"{max(geometry_gaps):.2e}, over {len(ranked)} courts")


if __name__ == "__main__":
    main()
