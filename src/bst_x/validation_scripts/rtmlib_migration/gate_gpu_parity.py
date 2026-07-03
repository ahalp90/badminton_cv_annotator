"""G8 -- GPU extraction parity at scale (HALT-AND-HANDOFF, Bourbaki).

(Gate numbering follows 03_verification.md, the authoritative spec: G7 = CUDA
self-variance floor, G8 = this extraction parity, G9 = the Phase-A decision.)

The Tier-3 gate: run the shipped adapter on the actual deployment box (CUDA, or
CPU as a fallback) over the smoke50 sample and compare each clip to the committed
mmpose baseline on BOTH axes at once (one inference pass per clip):

* extraction: frame-count ``dF``, ``ndet`` gap (rtmlib mean vs mmpose mean), and
  the G1 keypoint value (all-joint median L2, both-confident p90);
* deployed: assemble the raw arrays, run the unchanged ``sticky_anchor``, and
  compare to the committed clean (``fmatch`` / ``pos_med`` / ``jnt_med``, the
  failed-rate delta, and the dropped-player check ``rt<2 <= mm<2``).

Never self-certified on the dev box: the CUDA EP is nondeterministic and differs
from CPU bit-for-bit, so this must run where the full re-extract will run. Run G7
(``gate_cuda_selfvariance.py``) first to get CUDA's run-to-run noise floor so a
keypoint delta here can be read against it, then feed this gate's JSON into G9
(``phase_a_decision.py``).

Thresholds are imported from G1 / G6 (single source). CUDA adds EP noise on top
of the model difference, so a keypoint p90 slightly above the CPU gate is
expected; G7 reports it and G9 makes the call against G8's measured noise floor.

Env:
  RTMLIB_GATE_DEVICE   "cuda" (default) or "cpu"
  RTMLIB_GATE_STEMFILE newline-separated stems (default: provenance _smoke50.txt)
  RTMLIB_GATE_STEMS    comma-separated stems (overrides the stemfile)
  RTMLIB_GATE_G8_JSON  per-clip results dump (default: ./g8_parity.json), read by G9

Run (on Bourbaki):
  PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \\
      <env>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_gpu_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from _common import (
    CONF_THR,
    assemble_raw_clip,
    find_clip,
    load_mmpose_raw,
    matched_kp_l2,
)
from gate_deployed_parity import (
    CLEAN,
    FMATCH_MIN,
    JNT_MED_MAX,
    POS_MED_MAX,
    _setup,
)
from gate_keypoint_value import CONF_P90_MAX, MEDIAN_MAX

from preparing_data.rtmlib_pose import DET_SCORE_THR, RtmlibPoseExtractor

DEVICE = os.environ.get("RTMLIB_GATE_DEVICE", "cuda")
# Detector keep-threshold override for the calibration sweep. This is a
# post-inference filter on the identical ONNX's output scores -- NOT a model
# change. Defaults to the shipped adapter's DET_SCORE_THR (0.15 after the G-4
# recalibration; the 0.3->0.15 sweep is recorded in 06_phase_a_decision.md).
DET_THR = float(os.environ.get("RTMLIB_GATE_DET_THR", DET_SCORE_THR))
STEMFILE = Path(os.environ.get(
    "RTMLIB_GATE_STEMFILE",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/"
    "ShuttleSet_keypoints_raw_provenance/_smoke50.txt",
))


def _stems() -> list[str]:
    env = os.environ.get("RTMLIB_GATE_STEMS")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    return [ln.strip() for ln in STEMFILE.read_text().splitlines() if ln.strip()]


def _gate_clip(ext, stem, setup) -> dict | None:
    res_df, court, params, RawClip, ClipContext, sticky_apply = setup
    mp4 = find_clip(stem)
    if mp4 is None:
        return None
    t0 = time.time()
    frames = list(ext.iter_video(mp4))          # single inference pass
    mm = load_mmpose_raw(stem)

    # --- extraction axis (G1) ---
    Frt, Fmm = len(frames), mm.kps.shape[0]
    l2, conf = matched_kp_l2(mm, frames)
    kp_med = float(np.median(l2)) if l2.size else float("nan")
    kp_cp90 = float(np.percentile(l2[conf], 90)) if conf.any() else float("nan")
    rt_ndet = float(np.mean([len(f.keypoints) for f in frames]))

    # --- deployed axis (G6) ---
    raw_arr = assemble_raw_clip(frames)
    raw = RawClip(kps=raw_arr.kps, bboxes=raw_arr.bboxes, scores=raw_arr.bbox_scores,
                  kp_scores=raw_arr.kp_scores, ndet=raw_arr.ndet)
    ctx = ClipContext(vid=int(stem.split("_", 1)[0]), all_court_info=court, res_df=res_df)
    out = sticky_apply(raw, ctx, **params)
    ref_pos = np.load(CLEAN / f"{stem}_pos.npy")
    ref_joints = np.load(CLEAN / f"{stem}_joints.npy")
    ref_failed = np.load(CLEAN / f"{stem}_failed.npy")

    F = min(Frt, Fmm)
    rt_f, mm_f = out.failed[:F], ref_failed[:F]
    fmatch = float((rt_f == mm_f).mean())
    rt_only_fail = int((rt_f & ~mm_f).sum())  # rtmlib zeroes a frame mmpose kept
    mm_only_fail = int((~rt_f & mm_f).sum())  # the reverse
    both = (~rt_f) & (~mm_f)
    nb = int(both.sum())
    if nb:
        pos_med = float(np.median(np.abs(out.pos[:F][both] - ref_pos[:F][both])))
        jnt_med = float(np.median(np.abs(out.joints[:F][both] - ref_joints[:F][both])))
    else:
        pos_med = jnt_med = float("nan")

    return dict(
        stem=stem, dF=Frt - Fmm, kp_med=kp_med, kp_cp90=kp_cp90,
        rt_ndet=rt_ndet, mm_ndet=float(mm.ndet.mean()),
        fmatch=fmatch, both=nb, pos_med=pos_med, jnt_med=jnt_med,
        rt_only_fail=rt_only_fail, mm_only_fail=mm_only_fail,
        rt_failrate=float(out.failed.mean()), mm_failrate=float(ref_failed.mean()),
        rt_lt2=int((raw_arr.ndet < 2).sum()), mm_lt2=int((mm.ndet < 2).sum()),
        secs=time.time() - t0,
    )


def _verdict(r: dict) -> bool:
    """Structural (all) + keypoint value + deployed value. Diagnostic clips still
    hard-gate on structure; value thresholds mirror G1/G6. A clip with no
    IoU-matched detection (``kp_med`` NaN) fails rather than passing vacuously."""
    kp_measured = not np.isnan(r["kp_med"])  # >=1 IoU-matched detection to gate on
    return (
        r["dF"] == 0 and r["rt_lt2"] <= r["mm_lt2"] and r["fmatch"] >= FMATCH_MIN
        and kp_measured and not (r["kp_med"] > MEDIAN_MAX)
        and not (r["kp_cp90"] > CONF_P90_MAX)
        and not (r["pos_med"] > POS_MED_MAX) and not (r["jnt_med"] > JNT_MED_MAX)
    )


def main() -> int:
    setup = _setup()
    ext = RtmlibPoseExtractor(device=DEVICE, det_score_thr=DET_THR)
    stems = _stems()
    print(f"G8 GPU extraction parity | device={DEVICE} | det_thr={DET_THR} | "
          f"{len(stems)} clip(s) | confident joint = both kp_score>{CONF_THR}\n")
    hdr = (f"  {'stem':13s} {'dF':>3} {'kpMed':>6} {'kpP90':>6} {'rtNd':>5} {'mmNd':>5} "
           f"{'fmatch':>6} {'posMed':>7} {'jntMed':>7} {'rt<2':>4} {'mm<2':>4}  verdict")
    print(hdr + "\n  " + "-" * (len(hdr) - 2))

    results, missing = [], []
    for stem in stems:
        r = _gate_clip(ext, stem, setup)
        if r is None:
            missing.append(stem)
            print(f"  {stem:13s}  -- no mp4 (skipped, logged)")
            continue
        r["ok"] = _verdict(r)
        results.append(r)
        flag = "  *DROPPED PLAYER?*" if r["rt_lt2"] > r["mm_lt2"] else ""
        print(f"  {r['stem']:13s} {r['dF']:3d} {r['kp_med']:6.2f} {r['kp_cp90']:6.2f} "
              f"{r['rt_ndet']:5.2f} {r['mm_ndet']:5.2f} {r['fmatch']:6.3f} "
              f"{r['pos_med']:7.4f} {r['jnt_med']:7.4f} {r['rt_lt2']:4d} {r['mm_lt2']:4d}  "
              f"{'PASS' if r['ok'] else 'FAIL'}{flag}")

    out_path = os.environ.get("RTMLIB_GATE_G8_JSON", "g8_parity.json")
    Path(out_path).write_text(json.dumps(results, indent=2))

    if not results:
        print("\nFAIL: no clips evaluated")
        return 1
    ok = all(r["ok"] for r in results)
    print(f"\n  clips={len(results)}  missing={len(missing)}  "
          f"mean fmatch={np.mean([r['fmatch'] for r in results]):.3f}  "
          f"kp_med(max)={np.nanmax([r['kp_med'] for r in results]):.2f}  "
          f"kp_p90(max)={np.nanmax([r['kp_cp90'] for r in results]):.2f}")
    if not ok:
        print(f"  per-clip failures: {[r['stem'] for r in results if not r['ok']]}")
    print(f"  results -> {out_path}  (feed to phase_a_decision.py)")
    print(f"\n{'PASS' if ok else 'FAIL'}: G8 GPU extraction parity")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
