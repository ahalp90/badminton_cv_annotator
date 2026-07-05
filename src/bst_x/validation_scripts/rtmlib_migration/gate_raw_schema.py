"""Batch-2 gate: raw_extract.extract_raw_frame assembly (synthetic, inference-free).

The adapter returns only real detections; the N_max cap + NaN padding live in
``raw_extract.extract_raw_frame``. This gate exercises that assembly directly with
synthetic ``FrameDetections``: the truncation/empty/partial edges a small real
clip never hits (G2). ``_common.assemble_raw_clip`` calls this
same function, so passing here validates the deployed-parity gates' assembly too.

Each synthetic detection ``i`` is tagged: all its keypoints equal ``i``, so the
marker read back from a padded slot says which detection landed there.

Checks:
* truncation: 20 dets with ascending scores, n_max=16 -> the 16 kept are the
  top-16 by score in DESCENDING order (markers 19..4); the 4 lowest dropped;
  slots 16.. are NaN; dtypes float32 / ndet int8.
* stable ties: scores [0.5, 0.9, 0.9], n_max=2 -> the two 0.9s kept in detector
  order (markers 1, 2), not reordered.
* empty: 0 dets -> all-NaN frame, ndet 0.
* partial: 1 det, n_max=16 -> slot 0 filled, slots 1.. NaN, ndet 1.

Run:
  PYTHONPATH=src/bst_x:src <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_raw_schema.py
"""
from __future__ import annotations

import sys

import numpy as np

from pipeline.config import COCO_N_JOINTS
from preparing_data.raw_extract import extract_raw_frame
from preparing_data.rtmlib_pose import FrameDetections


def _frame(scores: list[float]) -> FrameDetections:
    """Synthetic frame: detection i tagged with marker i on every field."""
    m = len(scores)
    markers = np.arange(m, dtype=np.float32)
    kps = np.tile(markers.reshape(m, 1, 1), (1, COCO_N_JOINTS, 2))  # (m, J, 2), all == i
    bboxes = np.stack([markers, markers, markers + 1, markers + 1], axis=1)  # valid xyxy
    kp_scores = np.full((m, COCO_N_JOINTS), 0.9, dtype=np.float32)
    return FrameDetections(
        keypoints=kps, bboxes=bboxes.astype(np.float32),
        bbox_scores=np.asarray(scores, dtype=np.float32), kp_scores=kp_scores,
    )


def _markers(kps: np.ndarray, n: int) -> list[int]:
    """Recover the detection marker from each of the first n padded slots."""
    return [int(kps[i, 0, 0]) for i in range(n)]


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    # --- truncation: 20 dets, ascending scores, keep top-16 descending ---
    scores = [(i + 1) / 20 for i in range(20)]  # det0=0.05 (low) .. det19=1.0 (high)
    kps, bboxes, sc, ksc, n = extract_raw_frame(_frame(scores), 16, "syn", 0, set())
    kept = _markers(kps, 16)
    expected = list(range(19, 3, -1))  # [19,18,...,4]
    dtype_ok = (kps.dtype == np.float32 and bboxes.dtype == np.float32
                and sc.dtype == np.float32 and ksc.dtype == np.float32)
    pad_ok = bool(np.isnan(kps[16:]).all() and np.isnan(bboxes[16:]).all()
                  and np.isnan(sc[16:]).all() and np.isnan(ksc[16:]).all())
    desc_ok = bool(np.all(np.diff(sc[:16]) <= 0))  # scores non-increasing
    trunc_ok = n == 16 and kept == expected and dtype_ok and pad_ok and desc_ok
    checks.append(("truncation (top-16 by score, descending, rest NaN)", trunc_ok,
                   f"n={n} kept[:4]={kept[:4]} rest_NaN={pad_ok} dtype={dtype_ok} desc={desc_ok}"))

    # --- stable ties: equal scores keep detector order ---
    kps2, _, _, _, n2 = extract_raw_frame(_frame([0.5, 0.9, 0.9]), 2, "syn", 0, set())
    tie_ok = n2 == 2 and _markers(kps2, 2) == [1, 2]
    checks.append(("stable ties (equal scores keep detector order)", tie_ok,
                   f"n={n2} kept={_markers(kps2, 2)} (want [1, 2])"))

    # --- empty frame ---
    kps0, bb0, sc0, ksc0, n0 = extract_raw_frame(_frame([]), 16, "syn", 0, set())
    empty_ok = (n0 == 0 and np.isnan(kps0).all() and np.isnan(bb0).all()
                and np.isnan(sc0).all() and np.isnan(ksc0).all())
    checks.append(("empty frame (all-NaN, ndet 0)", bool(empty_ok), f"n={n0}"))

    # --- partial frame (1 det, wide n_max) ---
    kps1, _, _, _, n1 = extract_raw_frame(_frame([0.7]), 16, "syn", 0, set())
    partial_ok = (n1 == 1 and _markers(kps1, 1) == [0] and bool(np.isnan(kps1[1:]).all()))
    checks.append(("partial frame (slot 0 filled, rest NaN)", partial_ok,
                   f"n={n1} slot0={_markers(kps1, 1)} rest_NaN={bool(np.isnan(kps1[1:]).all())}"))

    all_ok = True
    for name, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        all_ok &= ok
    print(f"\n{'PASS' if all_ok else 'FAIL'}: raw_extract schema/assembly gate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
