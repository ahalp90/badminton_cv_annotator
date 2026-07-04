"""HISTORICAL (nano era): diagnose the G-4 real fails of the rtmdet-nano@320 run.

Audit trail for the 0.3->0.15 recalibration decision (the retired
06_phase_a_decision.md, in git history).
The detector has since been restored to RTMDet-M@640 at the 0.3 cut
(07_detector_restoration.md), which removes the 320-input under-scoring this
script diagnosed; the hard-coded fail stems below are from the nano G-4 run
and are not expected to reproduce.

A. Dropped-player clips (video 4 + 16_1_10_6 + the 18.75pp 2_1_10_2): for every frame
   where rtmlib keeps fewer boxes than mmpose, re-detect at a low threshold and report
   whether the missing mm detection is GENUINELY ABSENT at 320 (only a 640-input
   detector recovers it) or merely UNDER-SCORED below 0.3 (present at 320, a scoring
   issue). This is the 320-vs-640 decider.
B. 5_1_10_1 keypoint tail: how much of the raw p90 an L/R swap explains. If most of
   it, the 35.6px p90 is the benign body7 L/R model drift G1 already corrects for
   (_confident_tail_lr) and G8 simply doesn't, i.e. a gate-metric gap not a defect.

Run on Bourbaki (same env as the gates):
  source gpu_env.sh
  PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \
    ~/venv-rtmlib-gpu/bin/python \
    src/bst_x/validation_scripts/rtmlib_migration/diag_g4_fails.py 2>&1 | tee diag_g4.out
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join("src", "bst_x", "validation_scripts", "rtmlib_migration"))
from _common import CONF_THR, find_clip, iou_xyxy, load_mmpose_raw, match_dets  # noqa: E402
from preparing_data.rtmlib_pose import DET_SCORE_THR, RtmlibPoseExtractor  # noqa: E402

DEVICE = os.environ.get("RTMLIB_GATE_DEVICE", "cuda")
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
DROPPED = ["4_1_10_2", "4_1_10_3", "4_1_10_4", "4_1_10_5", "16_1_10_6", "2_1_10_2"]
LR_TAIL = ["5_1_10_1"]


def lr_swap(kps):  # (J,2) -> L/R-swapped rows
    out = kps.copy()
    for a, b in LR_PAIRS:
        out[a], out[b] = kps[b], kps[a]
    return out


def diag_dropped(ext, stem):
    mp4 = find_clip(stem)
    if mp4 is None:
        print(f"  {stem}: no mp4")
        return
    mm = load_mmpose_raw(stem)
    print(f"\n=== DROPPED/FRAME-LOSS DIAG: {stem}  (keep thr={DET_SCORE_THR}) ===")
    cap = cv2.VideoCapture(str(mp4))
    f, hits, genuine, underscored = 0, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_mm = int(mm.ndet[f]) if f < mm.ndet.shape[0] else 0
        boxes_hi, _ = ext.det(frame)
        if n_mm >= 2 and len(boxes_hi) < n_mm:
            ext.det.score_thr = 0.01
            boxes_lo, scores_lo = ext.det(frame)
            ext.det.score_thr = DET_SCORE_THR
            print(f"  frame {f}: rt(>={DET_SCORE_THR})={len(boxes_hi)}  mm={n_mm}")
            for i in range(n_mm):
                mb = mm.bboxes[f, i]
                best_j, best_iou = -1, 0.0
                for j, lb in enumerate(boxes_lo):
                    v = iou_xyxy(mb, lb)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_j >= 0 and best_iou > 0.3:
                    sc = float(scores_lo[best_j])
                    if sc < DET_SCORE_THR:
                        underscored += 1
                        print(f"     mm[{i}] <- rt {sc:.3f} @IoU {best_iou:.2f}  UNDER-SCORED (present at 320)")
                    else:
                        print(f"     mm[{i}] <- rt {sc:.3f} @IoU {best_iou:.2f}  kept")
                else:
                    genuine += 1
                    print(f"     mm[{i}] <- NO rt box even @0.01  GENUINE 320-RECALL MISS (640 needed)")
            hits += 1
        f += 1
    cap.release()
    if hits:
        print(f"  -> {stem}: {genuine} genuine-miss vs {underscored} under-scored "
              f"across {hits} short frame(s)")
    else:
        print("  (no rt<mm frame reproduced: check CUDA determinism / thresholds)")


def diag_lr(ext, stem):
    mp4 = find_clip(stem)
    if mp4 is None:
        print(f"  {stem}: no mp4")
        return
    mm = load_mmpose_raw(stem)
    frames = list(ext.iter_video(mp4))
    F = min(len(frames), mm.kps.shape[0])
    direct, best = [], []
    for f in range(F):
        rt = frames[f]
        nd = int(mm.ndet[f])
        for i, j in match_dets(mm.bboxes[f, :nd], rt.bboxes):
            cf = (mm.kp_scores[f, i] > CONF_THR) & (rt.kp_scores[j] > CONF_THR)
            if not cf.any():
                continue
            d = np.linalg.norm(mm.kps[f, i] - rt.keypoints[j], axis=-1)
            s = np.linalg.norm(mm.kps[f, i] - lr_swap(rt.keypoints[j]), axis=-1)
            direct.append(float(np.percentile(d[cf], 90)))
            best.append(float(min(np.percentile(d[cf], 90), np.percentile(s[cf], 90))))
    direct, best = np.array(direct), np.array(best)
    print(f"\n=== KEYPOINT-TAIL / L-R DIAG: {stem} ===")
    if len(direct):
        improved = float((best < direct - 1e-6).mean())
        tag = "L/R MODEL DRIFT (benign; G1 corrects, G8 doesn't)" if improved > 0.10 else "genuine pose tail"
        print(f"  matched people={len(direct)}  p90 direct={np.median(direct):.1f}px  "
              f"L/R-corrected={np.median(best):.1f}px  people improved by swap={improved:.1%} -> {tag}")
    else:
        print("  no confident matched people")


def main():
    ext = RtmlibPoseExtractor(device=DEVICE)
    for stem in DROPPED:
        diag_dropped(ext, stem)
    for stem in LR_TAIL:
        diag_lr(ext, stem)


if __name__ == "__main__":
    main()
