import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root / "wider_evaluation"))
from measurement import prepared_measurements
from run_cases import load_runtime

_, verifier, _ = load_runtime(root)
from experiments.annotator.independent_court import detector

context = verifier.prepare_view(root, "shuttleset_03_scene_0034")
out = root / "wider_evaluation/runs/20260922/corner_bias"
with gzip.open(out / "scene34_selected_pair.json.gz", "rt") as f:
    packet = json.load(f)
selected = next(c for c in packet["candidates"] if c["kind"] == "child")
base = np.asarray(selected["corners_px"], dtype=float)
rows = []
with prepared_measurements(verifier):
    for dy in [-2, 0, 2]:
        for dx in [-6, -4, -3, -2, 0, 2]:
            corners = base.copy()
            corners[0] += [dx, dy]
            h = cv2.getPerspectiveTransform(
                detector.CORNER_COURT_M.astype(np.float32), corners.astype(np.float32)
            )
            evidence, _ = verifier.measure_candidate(
                context, {"homography_working": h}, {}
            )
            rows.append(
                {
                    "dx": dx,
                    "dy": dy,
                    **{
                        k: evidence[k]
                        for k in [
                            "q_paint10_span_weighted",
                            "q_geom_span_weighted",
                            "q_paint10",
                            "q_geom",
                        ]
                    },
                }
            )
with gzip.open(out / "upper_left_score_probe.json.gz", "wt") as f:
    json.dump(
        {
            "diagnostic_only": True,
            "case_id": context.case_id,
            "selected_origin_key": selected["origin_key"],
            "changed": "upper-left corner only; other three fixed",
            "rows": rows,
        },
        f,
        indent=2,
    )
print("base", [r for r in rows if r["dx"] == 0 and r["dy"] == 0])
print("best paint", sorted(rows, key=lambda r: -r["q_paint10_span_weighted"])[:6])
print(
    "horizontal",
    [
        (
            r["dx"],
            round(r["q_paint10_span_weighted"], 5),
            round(r["q_geom_span_weighted"], 5),
        )
        for r in rows
        if r["dy"] == 0
    ],
)
