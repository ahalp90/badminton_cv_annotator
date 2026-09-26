"""Diagnostic transfer of a fixed corner movement; never changes selections."""

import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "wider_evaluation/runs/20260922"
sys.path.insert(0, str(ROOT / "wider_evaluation"))
from measurement import prepared_measurements
from run_cases import load_runtime


def check_case(case_id):
    cv2.setNumThreads(1)
    _, verifier, runtime = load_runtime(ROOT)
    from experiments.annotator.independent_court import detector, paint_geometry

    context = verifier.prepare_view(ROOT, case_id)
    review = json.loads(
        (
            ROOT
            / "evidence/holistic_admission/directional_20260921_r5/w5_directional_20260921_r5_43/review_candidates.json"
        ).read_text()
    )
    comparison = verifier.read_json_gz(BASE / "regression_comparison.json.gz")
    row = next(row for row in comparison["cases"] if row["case_id"] == case_id)
    key = row["selections"]["full"]["gated"]
    candidate = review[case_id]["candidates"][key]
    corners = np.asarray(candidate["corners_px"], dtype=float)
    assert context.size == (960, 540) and context.native_size == (960, 540)
    assert corners[0, 0] < corners[1, 0] and corners[0, 1] < corners[3, 1]
    maps = detector._distance_maps(
        detector._wide_line_families(context.segments), context.size
    )
    frame_path = next(
        row["image"]
        for row in verifier.read_json_gz(BASE / "manifest.json.gz")["cases"]
        if row["case_id"] == case_id
    )
    frame = cv2.imread(str(ROOT / frame_path))
    rows = []
    panels = []
    with prepared_measurements(verifier):
        for dx, dy in [(0, 0), (-3, -2)]:
            moved = corners.copy()
            moved[0] += [dx, dy]
            homography = cv2.getPerspectiveTransform(
                detector.CORNER_COURT_M.astype(np.float32), moved.astype(np.float32)
            )
            evidence, _ = verifier.measure_candidate(
                context, {"homography_working": homography}, {}
            )
            gates = runtime["gate_evidence"](
                moved,
                context.source,
                np.ones(2),
                context.size,
                context.families,
                maps,
                runtime["zone"],
            )
            rows.append(
                {
                    "dx": dx,
                    "dy": dy,
                    "q_paint10_span_weighted": evidence["q_paint10_span_weighted"],
                    "q_geom_span_weighted": evidence["q_geom_span_weighted"],
                    "camera_eligible": verifier.camera_eligible({"gates": gates}),
                    "historical": verifier.historical_predicates(gates),
                    "gates": gates,
                }
            )
            canvas = frame.copy()
            markings, _ = detector.project(
                homography[None], paint_geometry.CENTRE_SEGMENTS_M
            )
            for start, end in markings[0].reshape(-1, 2, 2):
                cv2.line(
                    canvas,
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(end).astype(int)),
                    (255, 190, 0),
                    1,
                    cv2.LINE_AA,
                )
            panels.append(canvas)
    destination = BASE / "corner_bias/transfer"
    destination.mkdir(exist_ok=True)
    x0, y0 = np.floor(corners[0] - [25, 20]).astype(int)
    x1, y1 = x0 + 100, y0 + 90
    crops = [
        cv2.resize(
            panel[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST
        )
        for panel in [frame, *panels]
    ]
    assert cv2.imwrite(str(destination / f"{case_id}.png"), cv2.hconcat(crops))
    return {"case_id": case_id, "selected_origin_key": key, "rows": rows}


if __name__ == "__main__":
    cases = [f"shuttleset_03_scene_{scene:04d}" for scene in [16, 17, 19, 29, 34, 38]]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(check_case, cases))
    result = {
        "diagnostic_only": True,
        "movement": "same fixed TL (-3,-2) native pixels; other corners unchanged",
        "workers": 6,
        "cases": rows,
    }
    with gzip.open(BASE / "corner_bias/transfer_probe.json.gz", "wt") as stream:
        json.dump(result, stream, indent=2)
    for row in rows:
        original, moved = row["rows"]
        print(
            row["case_id"],
            "paint",
            original["q_paint10_span_weighted"],
            moved["q_paint10_span_weighted"],
            "gates",
            moved["camera_eligible"],
            moved["historical"],
        )
