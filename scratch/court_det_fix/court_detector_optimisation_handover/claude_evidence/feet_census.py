"""Count foot samples per view and how many fall inside the reference court (throwaway)."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import cv2
import numpy as np

COURT_ROOT = Path(sys.argv[1]).resolve()
REPO = COURT_ROOT.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src"), str(COURT_ROOT / "w5_holistic"), str(COURT_ROOT / "wider_evaluation")]
import run_cases

run_w5, verifier_module, runtime = run_cases.load_runtime(COURT_ROOT)
cases_root = COURT_ROOT / "svd_search" / "run_20260923" / "cases" / "baseline"
for case_path in sorted(cases_root.glob("*.json.gz")):
    case_id = case_path.name.removesuffix(".json.gz")
    corners = np.asarray(json.load(gzip.open(case_path, "rt"))["reference_corners_native_px"], dtype=np.float32)
    source = verifier_module.prepare_view(COURT_ROOT, case_id).source
    per_frame = [[foot for foot in frame if foot is not None and None not in foot and np.isfinite(foot).all()] for frame in source["all_feet_px"]]
    inside = [sum(cv2.pointPolygonTest(corners.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0 for x, y in frame)
              for frame in per_frame]
    detected = [len(frame) for frame in per_frame]
    slots = sum(len(frame) for frame in source["all_feet_px"])
    print(f"{case_id:26s} frames {len(per_frame):3d} | foot slots {slots:5d} | detected {sum(detected):5d}"
          f" | inside court {sum(inside):4d} ({sum(inside) / max(sum(detected), 1):.0%})"
          f" | per frame: detected median {int(np.median(detected))}, inside median {int(np.median(inside))}")
