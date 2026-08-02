"""G2: rtmlib adapter contract test (CPU, reference-free).

Exercises the SHIPPED ``preparing_data.rtmlib_pose.RtmlibPoseExtractor`` (not the
scratchpad prototype) and asserts the per-frame *contract* both consumers depend
on: shape, dtype, COCO-17 count, box/score validity, the empty-frame guard,
and a reference-free order-sanity check. Keypoint *values* vs the committed
mmpose raw are gated separately by G1 (``gate_keypoint_value.py``).

The ``N_MAX`` truncation and NaN padding are deliberately NOT tested here: the
adapter returns only the real detections (``m`` may be 0); the cap + padding are
``raw_extract``'s contract, gated in Batch 2.

Checks (on one real clip + a synthetic black frame):

* COCO-17 constant: ``COCO_N_JOINTS == 17``.
* schema/dtype: ``keypoints (m,17,2) f32``, ``bboxes (m,4) f32``,
  ``bbox_scores (m,) f32``, ``kp_scores (m,17) f32``, ``m`` consistent.
* box validity: xyxy ordered (x2>x1, y2>y1) and finite.
* score ranges: ``bbox_scores`` in ``(det_score_thr, 1]``; ``kp_scores`` in
  ``[0, KP_SCORE_MAX]`` (RTMPose simcc confidence is a peak product, not a strict
  probability: the committed mmpose raw itself reaches ~1.16; a loose upper
  bound still catches NaN / negatives / un-sigmoided logits).
* empty-frame guard: an all-black frame yields ``m == 0`` with correctly-shaped,
  correctly-typed empty arrays (no fabricated whole-image "person").
* order-sanity: on the cleanest detection the head sits above the knees and ankles.

Run:
  PYTHONPATH=src:src/bst_x XDG_CACHE_HOME=<warm-cache> <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/adapter_contract_test.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from _common import ANKLES, HEAD, KNEES, find_clip

from pipeline.config import COCO_N_JOINTS
from preparing_data.rtmlib_pose import (
    DET_SCORE_THR,
    FrameDetections,
    RtmlibPoseExtractor,
)

STEM = os.environ.get("RTMLIB_GATE_STEM", "11_1_10_2")
SCORE_EPS = 1e-3     # sigmoid bbox scores can graze 1.0 by rounding
KP_SCORE_MAX = 1.2   # loose kp-confidence sanity bound (mmpose raw reaches ~1.16)


def _first_detected(ext: RtmlibPoseExtractor, mp4) -> FrameDetections | None:
    """First frame with >=1 detection (scan the whole clip, decode order)."""
    for det in ext.iter_video(mp4):
        if len(det.keypoints):
            return det
    return None


def main() -> int:
    mp4 = find_clip(STEM)
    if mp4 is None:
        print(f"FAIL: no mp4 for stem {STEM}")
        return 1

    ext = RtmlibPoseExtractor(device="cpu")
    checks: list[tuple[str, bool, str]] = []

    checks.append(("COCO-17 constant", COCO_N_JOINTS == 17, f"COCO_N_JOINTS={COCO_N_JOINTS}"))

    det = _first_detected(ext, mp4)
    if det is None:
        print(f"FAIL: no detections anywhere in {STEM}")
        return 1
    m = len(det.keypoints)

    schema_ok = (
        det.keypoints.shape == (m, COCO_N_JOINTS, 2) and det.keypoints.dtype == np.float32
        and det.bboxes.shape == (m, 4) and det.bboxes.dtype == np.float32
        and det.bbox_scores.shape == (m,) and det.bbox_scores.dtype == np.float32
        and det.kp_scores.shape == (m, COCO_N_JOINTS) and det.kp_scores.dtype == np.float32
    )
    checks.append(("schema/dtype", schema_ok,
                   f"m={m} kps{det.keypoints.shape}/{det.keypoints.dtype}"))

    b = det.bboxes
    box_ok = bool(np.isfinite(b).all() and (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all())
    checks.append(("box validity (xyxy ordered, finite)", box_ok,
                   f"x[{b[:,0].min():.0f},{b[:,2].max():.0f}] y[{b[:,1].min():.0f},{b[:,3].max():.0f}]"))

    bs, ks = det.bbox_scores, det.kp_scores
    score_ok = bool(
        (bs > DET_SCORE_THR).all() and (bs <= 1 + SCORE_EPS).all()
        and (ks >= -SCORE_EPS).all() and (ks <= KP_SCORE_MAX).all()
    )
    checks.append((f"score ranges (bbox in ({DET_SCORE_THR},1], kp in [0,{KP_SCORE_MAX}])", score_ok,
                   f"bbox[{bs.min():.2f},{bs.max():.2f}] kp[{ks.min():.2f},{ks.max():.2f}]"))

    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    empty = ext.detect_frame(black)
    empty_ok = (
        len(empty.keypoints) == 0 and empty.keypoints.shape == (0, COCO_N_JOINTS, 2)
        and empty.bboxes.shape == (0, 4) and empty.bbox_scores.shape == (0,)
        and empty.kp_scores.shape == (0, COCO_N_JOINTS) and empty.keypoints.dtype == np.float32
    )
    checks.append(("empty-frame guard", empty_ok, f"m={len(empty.keypoints)}"))

    clean = det.keypoints[int(np.argmax(det.kp_scores.mean(axis=1)))]  # (J, 2)
    head = clean[list(HEAD), 1].mean()
    knee = clean[list(KNEES), 1].mean()
    ankle = clean[list(ANKLES), 1].mean()
    order_ok = bool(head < knee and head < ankle)  # image y grows downward
    checks.append(("order-sanity (head above lower body)", order_ok,
                   f"head={head:.0f} knee={knee:.0f} ankle={ankle:.0f}"))

    print(f"clip {STEM}  ({mp4})\n")
    all_ok = True
    for name, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        all_ok &= ok
    print(f"\n{'PASS' if all_ok else 'FAIL'}: G2 adapter contract gate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
