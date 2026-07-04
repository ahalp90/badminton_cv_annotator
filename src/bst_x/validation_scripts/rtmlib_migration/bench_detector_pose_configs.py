"""Timed benchmark of detector + pose config combinations over the same clips.

Answers "what does each config cost, and what does it buy" with one table:
per config, median detector ms/frame, median pose ms/frame (all people in the
frame), end-to-end fps, mean people kept per frame, frames with fewer than two
people (frame-loss risk), and keypoint agreement against the committed mmpose
raw (IoU-matched px L2, all joints and confident-only). Configs are built via
``RtmlibPoseExtractor`` kwargs, so every combination runs from the current
tree with no code or git changes.

Default configs:
  nano_L256@0.15   rtmdet-nano@320 + RTMPose-L body7 256x192 (nano-era shipped)
  nano_L384@0.15   rtmdet-nano@320 + RTMPose-L body7 384x288
  m_L256@0.30      RTMDet-M@640    + RTMPose-L body7 256x192 (shipped)
  m_L384@0.30      RTMDet-M@640    + RTMPose-L body7 384x288

Timings are wall-clock on whatever device is selected; absolute numbers are
only comparable within one run on one host. The keypoint column is agreement
with the deployed extraction (whose pose model differs), not ground truth.

Env:
  RTMLIB_GATE_CLIPS / RTMLIB_GATE_RAW  as the other gates (_common.py)
  RTMLIB_GATE_DEVICE                   cpu (default) | cuda
  RTMLIB_GATE_STEMS                    comma-separated stems
                                       (default 11_1_10_2,13_1_10_1,14_1_10_1)
  RTMLIB_GATE_MAXFR                    frames per clip (default 50)

Run from the repo root:
  PYTHONPATH=src/bst_x:src <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/bench_detector_pose_configs.py
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

from _common import find_clip, load_mmpose_raw, matched_kp_l2
from preparing_data.rtmlib_pose import _MODEL_BASE, RtmlibPoseExtractor

DEVICE = os.environ.get("RTMLIB_GATE_DEVICE", "cpu")
STEMS = os.environ.get(
    "RTMLIB_GATE_STEMS", "11_1_10_2,13_1_10_1,14_1_10_1").split(",")
MAX_FRAMES = int(os.environ.get("RTMLIB_GATE_MAXFR", "50"))
WARMUP = 3  # frames excluded from timing (session + provider warm-up)

NANO_URL = _MODEL_BASE + "rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip"
M_URL = _MODEL_BASE + "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.zip"
L256_URL = _MODEL_BASE + "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip"
L384_URL = _MODEL_BASE + "rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.zip"

# name -> (det_url, det_input, det_thr, pose_url, pose_input (W, H))
CONFIGS = {
    "nano_L256@0.15": (NANO_URL, (320, 320), 0.15, L256_URL, (192, 256)),
    "nano_L384@0.15": (NANO_URL, (320, 320), 0.15, L384_URL, (288, 384)),
    "m_L256@0.30":    (M_URL,    (640, 640), 0.30, L256_URL, (192, 256)),
    "m_L384@0.30":    (M_URL,    (640, 640), 0.30, L384_URL, (288, 384)),
}


class _Timed:
    """Per-frame det/pose wall-clock, replicating ``detect_frame``'s steps."""

    def __init__(self, ext: RtmlibPoseExtractor) -> None:
        self.ext = ext
        self.det_ms: list[float] = []
        self.pose_ms: list[float] = []

    def __call__(self, frame_bgr):
        t0 = time.perf_counter()
        boxes, scores = self.ext.det(frame_bgr)
        t1 = time.perf_counter()
        if len(boxes) == 0:
            kps = np.empty((0, 17, 2), np.float32)
            kp_scores = np.empty((0, 17), np.float32)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            kps, kp_scores = self.ext.pose(rgb, bboxes=boxes)
        t2 = time.perf_counter()
        self.det_ms.append((t1 - t0) * 1e3)
        self.pose_ms.append((t2 - t1) * 1e3)
        # Shape-compatible with FrameDetections for matched_kp_l2.
        from preparing_data.rtmlib_pose import FrameDetections
        return FrameDetections(
            keypoints=np.asarray(kps, np.float32),
            bboxes=np.asarray(boxes, np.float32),
            bbox_scores=np.asarray(scores, np.float32),
            kp_scores=np.asarray(kp_scores, np.float32),
        )


def bench_config(name: str, spec, clips) -> dict:
    det_url, det_in, thr, pose_url, pose_in = spec
    ext = RtmlibPoseExtractor(
        device=DEVICE, det_url=det_url, det_input_size=det_in,
        det_score_thr=thr, pose_url=pose_url, pose_input_size=pose_in,
    )
    timed = _Timed(ext)
    people, low, l2_all, l2_conf = [], 0, [], []
    for stem, mp4, mm in clips:
        frames = []
        cap = cv2.VideoCapture(str(mp4))
        try:
            for _ in range(MAX_FRAMES):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(timed(fr))
        finally:
            cap.release()
        people += [len(f.bboxes) for f in frames]
        low += sum(len(f.bboxes) < 2 for f in frames)
        l2, conf = matched_kp_l2(mm, frames)
        if l2.size:
            l2_all.append(l2.ravel())
            l2_conf.append(l2[conf])
    det = np.asarray(timed.det_ms[WARMUP:])
    pose = np.asarray(timed.pose_ms[WARMUP:])
    tot = det + pose
    l2_all = np.concatenate(l2_all)
    l2_conf = np.concatenate(l2_conf)
    return dict(
        name=name, det=float(np.median(det)), pose=float(np.median(pose)),
        total=float(np.median(tot)), fps=1e3 / float(np.median(tot)),
        ppl=float(np.mean(people)), low=low,
        kp=float(np.median(l2_all)), kpc=float(np.median(l2_conf)),
    )


def main() -> int:
    clips = []
    for stem in STEMS:
        mp4 = find_clip(stem)
        if mp4 is None:
            print(f"  SKIP {stem}: clip not found")
            continue
        clips.append((stem, mp4, load_mmpose_raw(stem)))
    if not clips:
        print("no clips found")
        return 1
    n_fr = sum(min(MAX_FRAMES, mm.ndet.shape[0]) for _, _, mm in clips)
    print(f"bench: device={DEVICE}  clips={[s for s, _, _ in clips]}  "
          f"~{n_fr} frames/config  (medians; warm-up excluded)\n")
    header = (f"  {'config':<16} {'det ms':>7} {'pose ms':>8} {'total ms':>9} "
              f"{'fps':>6} {'ppl/fr':>7} {'fr<2':>5} {'kp px':>6} {'kp-conf':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, spec in CONFIGS.items():
        r = bench_config(name, spec, clips)
        print(f"  {r['name']:<16} {r['det']:>7.1f} {r['pose']:>8.1f} "
              f"{r['total']:>9.1f} {r['fps']:>6.1f} {r['ppl']:>7.2f} "
              f"{r['low']:>5d} {r['kp']:>6.2f} {r['kpc']:>8.2f}")
    print("\n  kp px / kp-conf: IoU-matched px L2 vs the committed mmpose raw"
          " (median; all joints / joints both models score >0.5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
