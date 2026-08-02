"""G1: keypoint-value gate (CPU, committed mmpose reference).

The critical value gate the bbox-driven deployed parity (G6) does not cover:
sticky_anchor selects players by bounding box, so G6 can pass even if the
keypoint *values* regressed. G1 pins the values directly against the committed
mmpose raw.

Per clip, run the shipped adapter, IoU-match to the committed mmpose detections
(``_common.matched_kp_l2``) and assert:

* all-joint median L2 <= ``MEDIAN_MAX`` px: proves the coordinate system, COCO
  ordering and pixel units;
* both-confident joints (kp_score > 0.5 in both models) p90 <= ``CONF_P90_MAX``
  and a **mirror-labeling-robust** p95 <= ``CONF_P95_MAX``: the signal-bearing
  joints agree tightly. The shipped RTMPose-L and the mmpose-era RTMPose-M (both
  body7) legitimately assign left/right
  oppositely on some ambiguous rotational poses (a shoulder/hip-width L2 that is
  not a coordinate error), so the p95 is taken over the per-person minimum of the
  direct and L/R-swapped distance (``_confident_tail_lr``). A *systematic* adapter
  L/R swap can't hide behind that: it would need relabeling on ~every match, which
  ``SWAP_FRAC_MAX`` trips. The residual tail is model-noise the Phase-B retrain
  absorbs, not a coordinate error;
* order-sanity backstop: on the cleanest detection (highest mean joint
  confidence) the head sits above the knees and ankles, a reference-free
  flip/reorder trip.

An x/y swap, flip, reorder or unit error shows tens-hundreds of px on every
joint, including the confident ones, so it fails the px thresholds loudly.

A **reverted RGB fix does not**: RTMPose-L body7 is channel-robust, so a BGR
feed lands ~1px from RGB, inside MEDIAN_MAX (measured on 11_1_10_2: RGB-vs-BGR
per-joint gap median 1.16px). The px thresholds cannot separate them,
so G1 also runs ``_rgb_fix_counterfactual`` on every resolved stem: a byte-exact
structural check that the shipped adapter feeds RTMPose an RGB crop (not BGR),
independent of accuracy.

Stems: ``RTMLIB_GATE_STEMS`` (comma-separated) or one present clip from each of
the first ``N_DEFAULT_VIDS`` distinct video-ids: breadth across courts/lighting,
not three clips of one match (where channel/exposure effects vanish).
Parity-at-scale against the mmpose baseline is the GPU gate (G8).

Run (CPU default, 5 clips):
  PYTHONPATH=src:src/bst_x XDG_CACHE_HOME=<warm-cache> <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_keypoint_value.py

Run over the 200-clip Phase-A sample: a GPU-box job with RTMLIB_GATE_DEVICE=cuda,
~20-30 min at the A100's ~81 ms/frame benchmark (on the default CPU device it
would be hours; the RGB counterfactual runs on every stem in the list, 12 frames
each, so a 200-stem list carries that cost too; the CPU default above stays at
5 clips). ``make_phase_a_sample.py`` writes one stem per line, so flatten its
output into the comma-separated env var:
  PYTHONPATH=src/bst_x:src <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/make_phase_a_sample.py \\
      --per-video 5 --out phase_a_stems.txt
  RTMLIB_GATE_STEMS=$(paste -sd, phase_a_stems.txt) RTMLIB_GATE_DEVICE=cuda \\
      PYTHONPATH=src:src/bst_x XDG_CACHE_HOME=<warm-cache> <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_keypoint_value.py

Env:
  RTMLIB_GATE_STEMS    comma-separated stems (overrides the default sample)
  RTMLIB_GATE_DEVICE   "cpu" (default: deterministic, matches the CPU gate
                       ladder) or "cuda" for the 200-clip run above
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from _common import (
    ANKLES,
    CLIPS,
    CONF_THR,
    HEAD,
    KNEES,
    RAW,
    find_clip,
    load_mmpose_raw,
    match_dets,
    matched_kp_l2,
)

from preparing_data.rtmlib_pose import RtmlibPoseExtractor

MEDIAN_MAX = 5.0     # all joints; coordinate system / COCO order / units
CONF_P90_MAX = 12.0  # both-confident joints agree tightly
CONF_P95_MAX = 18.0  # confident-joint tail, after removing L/R mirror-labeling
SWAP_FRAC_MAX = 0.20  # >20% of matches improved by L/R swap => systematic L/R bug
N_DEFAULT_VIDS = 5   # distinct video-ids in the default sample (courts/lighting)
# cpu default: this is a CPU-ladder gate and the CPU EP is deterministic; cuda
# exists for the 200-clip Phase-A invocation (see the docstring's Run block).
DEVICE = os.environ.get("RTMLIB_GATE_DEVICE", "cpu")
# COCO L/R joint pairs, for the mirror-labeling-robust tail (see _confident_tail_lr).
LR_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]


def _resolve_stems() -> list[str]:
    """Stems from ``RTMLIB_GATE_STEMS`` or one present clip per distinct video-id.

    The default spreads across the first ``N_DEFAULT_VIDS`` video-ids present in
    both the committed raw and the clip tree, rather than the first N clips of one
    match: colour/exposure differences are exactly what a single-match sample
    hides. The clip tree is indexed once (a recursive glob per stem over ~32k
    stems would be O(minutes)).
    """
    env = os.environ.get("RTMLIB_GATE_STEMS")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    raw_stems = {p.name[: -len("_raw_kps.npy")] for p in RAW.glob("*_raw_kps.npy")}
    clip_stems = {p.stem for p in CLIPS.glob("**/*.mp4")}
    present = sorted(raw_stems & clip_stems)
    picked: list[str] = []
    seen_vids: set[str] = set()
    for stem in present:
        vid = stem.split("_", 1)[0]
        if vid not in seen_vids:
            seen_vids.add(vid)
            picked.append(stem)
        if len(picked) >= N_DEFAULT_VIDS:
            break
    return picked


def _order_sanity(frames: list) -> tuple[bool, str]:
    """Head-above-lower-body on the single cleanest detection across the clip."""
    best_kps, best_conf = None, -1.0
    for fr in frames:
        for p in range(len(fr.keypoints)):
            c = float(fr.kp_scores[p].mean())
            if c > best_conf:
                best_conf, best_kps = c, fr.keypoints[p]
    if best_kps is None:
        return False, "no detections"
    head = best_kps[list(HEAD), 1].mean()
    knee = best_kps[list(KNEES), 1].mean()
    ankle = best_kps[list(ANKLES), 1].mean()
    return bool(head < knee and head < ankle), f"head={head:.0f} knee={knee:.0f} ankle={ankle:.0f}"


def _lr_swap(kps: np.ndarray) -> np.ndarray:
    """Exchange the COCO left/right joints of one (J, 2) pose."""
    out = kps.copy()
    for a, b in LR_PAIRS:
        out[[a, b]] = kps[[b, a]]
    return out


def _confident_tail_lr(mm, frames: list) -> tuple[float, float]:
    """Confident-joint L2 p95 with per-person L/R relabeling removed, plus the
    fraction of matched people an L/R swap explains.

    The shipped RTMPose-L and the mmpose-era RTMPose-M (both body7) legitimately
    disagree on left/right assignment in
    ambiguous rotational poses (measured on 16_1_10_1: 6 frames, correct IoU
    0.74-0.90, an L/R swap collapses ~29px -> ~1.4px). That mirror-labeling
    inflates the *raw* confident p95 without being an adapter defect, so the tail
    check uses the per-person minimum of the direct and L/R-swapped L2. A
    *systematic* adapter L/R swap instead needs relabeling on ~every match; that
    is what ``SWAP_FRAC_MAX`` catches, so the mirror-robustness can't hide a real
    L/R bug.

    :param mm: committed mmpose raw for the clip.
    :param frames: rtmlib ``FrameDetections`` per frame, decode order.
    :return: ``(p95_corrected, swap_fraction)``; p95 ``inf`` if nothing matched.
    """
    corrected: list[np.ndarray] = []  # per matched person: confident-joint L2 (min of dir/swap)
    swapped = total = 0
    F = min(len(frames), mm.kps.shape[0])
    for f in range(F):
        rt = frames[f]
        n = int(mm.ndet[f])
        for i, j in match_dets(mm.bboxes[f, :n], rt.bboxes):
            conf = (mm.kp_scores[f, i] > CONF_THR) & (rt.kp_scores[j] > CONF_THR)
            if not conf.any():
                continue
            d_dir = np.linalg.norm(mm.kps[f, i] - rt.keypoints[j], axis=-1)          # (J,)
            d_swap = np.linalg.norm(mm.kps[f, i] - _lr_swap(rt.keypoints[j]), axis=-1)
            total += 1
            if d_swap[conf].mean() < d_dir[conf].mean():
                swapped += 1
                corrected.append(d_swap[conf])
            else:
                corrected.append(d_dir[conf])
    if not corrected:
        return float("inf"), 0.0
    p95 = float(np.percentile(np.concatenate(corrected), 95))
    return p95, swapped / total


def _rgb_fix_counterfactual(
    ext: RtmlibPoseExtractor, mp4, max_frames: int = 12
) -> tuple[bool, str]:
    """Byte-exact proof the adapter feeds RTMPose an RGB crop, not BGR.

    The px thresholds above cannot catch a reverted RGB fix: RTMPose-L body7 is
    channel-robust, so a BGR feed sits ~1px from RGB (per-joint L2 median
    1.16px on 11_1_10_2, the same statistic as the module docstring), well
    inside MEDIAN_MAX. So this checks the fix *structurally*, not by accuracy margin. For
    each sampled frame with detections, recompute the pose independently under an
    RGB feed and a BGR feed on the detector's own boxes, then require:

    1. the shipped adapter's keypoints equal the RGB feed byte-for-byte: a
       reverted fix flips them onto the BGR feed and trips this;
    2. the RGB and BGR feeds actually differ, so the equality in (1) has teeth
       (RTMPose is not channel-invariant; if it were, a revert wouldn't matter).

    The median RGB<->BGR px gap is reported to make the channel-robustness that
    defeats the px thresholds visible in the log.

    :param ext: the shipped adapter under test.
    :param mp4: clip to sample frames from.
    :param max_frames: number of detection-bearing frames to check.
    :return: ``(ok, message)``.
    """
    cap = cv2.VideoCapture(str(mp4))
    checked = shipped_is_bgr = feeds_identical = 0
    gaps: list[float] = []
    try:
        while checked < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            boxes, _ = ext.det(frame)  # native-BGR detector, unchanged by the fix
            if len(boxes) == 0:
                continue
            rgb_kps, _ = ext.pose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), bboxes=boxes)
            bgr_kps, _ = ext.pose(frame, bboxes=boxes)
            shipped = ext.detect_frame(frame).keypoints
            if not np.array_equal(shipped, rgb_kps.astype(np.float32)):
                shipped_is_bgr += 1
            if np.array_equal(rgb_kps, bgr_kps):
                feeds_identical += 1
            gaps.append(float(np.median(np.abs(rgb_kps - bgr_kps))))
            checked += 1
    finally:
        cap.release()

    if checked == 0:
        return False, "no detection-bearing frames to check the RGB feed on"
    ok = shipped_is_bgr == 0 and feeds_identical < checked
    return ok, (
        f"{checked} frames | shipped==RGB-feed {checked - shipped_is_bgr}/{checked} "
        f"| RGB!=BGR {checked - feeds_identical}/{checked} "
        f"| RGB<->BGR gap median {float(np.median(gaps)):.2f}px"
    )


def _gate_clip(ext: RtmlibPoseExtractor, stem: str) -> tuple[bool, str]:
    mp4 = find_clip(stem)
    if mp4 is None:
        return False, "clip mp4 not found"
    mm = load_mmpose_raw(stem)
    frames = list(ext.iter_video(mp4))
    if len(frames) != mm.kps.shape[0]:
        return False, f"frame-count mismatch F_rtmlib={len(frames)} F_mmpose={mm.kps.shape[0]}"

    l2, conf = matched_kp_l2(mm, frames)
    if l2.size == 0:
        return False, "no IoU-matched detections"
    med = float(np.median(l2))
    c = l2[conf]
    c_p90 = float(np.percentile(c, 90)) if c.size else float("inf")
    c_p95_raw = float(np.percentile(c, 95)) if c.size else float("inf")
    c_p95, swap_frac = _confident_tail_lr(mm, frames)  # mirror-labeling removed
    order_ok, order_msg = _order_sanity(frames)

    ok = (med <= MEDIAN_MAX and c_p90 <= CONF_P90_MAX and c_p95 <= CONF_P95_MAX
          and swap_frac <= SWAP_FRAC_MAX and order_ok)
    msg = (f"median={med:.2f} | conf(both>{CONF_THR}) p90={c_p90:.2f} "
           f"p95={c_p95:.2f}(raw {c_p95_raw:.2f}, L/R-swap {100 * swap_frac:.0f}%) "
           f"({int(conf.sum())}/{conf.size} joints) | order[{order_msg}]")
    return ok, msg


def main() -> int:
    stems = _resolve_stems()
    if not stems:
        print(f"FAIL: no stems present in both {RAW} and {CLIPS}")
        return 1
    ext = RtmlibPoseExtractor(device=DEVICE)
    print(f"G1 keypoint-value gate over {len(stems)} clip(s): {', '.join(stems)}\n")
    all_ok = True
    for stem in stems:
        ok, msg = _gate_clip(ext, stem)
        print(f"  [{'PASS' if ok else 'FAIL'}] {stem}: {msg}")
        all_ok &= ok

    # RGB fix is a byte-exact structural check; the px thresholds above are
    # channel-robust and cannot catch a reverted fix on their own. A BGR/RGB swap
    # is per-model-input, and cheap coverage over every resolved stem (a couple of
    # CPU-minutes on the default sample) beats a single-stem spot check.
    print()
    for stem in stems:
        rgb_ok, rgb_msg = _rgb_fix_counterfactual(ext, find_clip(stem))
        print(f"  [{'PASS' if rgb_ok else 'FAIL'}] RGB-fix byte-exact ({stem}): {rgb_msg}")
        all_ok &= rgb_ok

    print(f"\n{'PASS' if all_ok else 'FAIL'}: G1 keypoint-value gate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
