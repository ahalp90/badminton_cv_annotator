"""G7: CUDA self-variance floor (HALT-AND-HANDOFF, Bourbaki; run FIRST).

(Gate numbering: G7 = this self-variance floor, G8 = extraction parity, G9 =
Phase-A decision.)

The CUDA EP is nondeterministic, so two runs of the same clip on the same GPU
differ slightly. This gate measures that run-to-run noise floor:

* ``eps_kp``: median / p90 / max per-keypoint L2 between two CUDA runs
  (IoU-matched detections, raw keypoints);
* ``eps_fail``: max per-clip |failed-rate(run A) - failed-rate(run B)| after
  the unchanged ``sticky_anchor``.

Every mmpose-vs-rtmlib threshold in G8/G9 must sit ABOVE these floors, else the
gate would be measuring CUDA jitter, not a real regression. Run this first and
carry the floors into G9.

It also sanity-fails if the floor is implausibly large (``eps_kp`` median >
KP_FLOOR_MAX px or ``eps_fail`` > FAIL_FLOOR_MAX): a large self-variance means the
extract itself is not reproducible and Phase B could not be trusted. On CPU the
floor is exactly 0 (deterministic, per G4), so a CPU run validates the code path.

Env:
  RTMLIB_GATE_DEVICE   "cuda" (default) or "cpu"
  RTMLIB_GATE_STEMFILE newline-separated stems (default: provenance _smoke50.txt)
  RTMLIB_GATE_STEMS    comma-separated stems (overrides the stemfile)
  RTMLIB_GATE_G7_JSON  per-clip floor dump (default: g7_selfvariance.json next to
                       this script, so the artifact lands in a predictable place
                       regardless of CWD), read by G9

Run (on Bourbaki, before G8):
  PYTHONUNBUFFERED=1 PYTHONPATH=src/bst_x:src RTMLIB_GATE_DEVICE=cuda \\
      <env>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_cuda_selfvariance.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from _common import assemble_raw_clip, find_clip, match_dets
from gate_deployed_parity import _setup

from preparing_data.rtmlib_pose import RtmlibPoseExtractor

DEVICE = os.environ.get("RTMLIB_GATE_DEVICE", "cuda")
STEMFILE = Path(os.environ.get(
    "RTMLIB_GATE_STEMFILE",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/"
    "ShuttleSet_keypoints_raw_provenance/_smoke50.txt",
))
KP_FLOOR_MAX = 3.0    # implausible-noise ceiling for eps_kp median (px)
FAIL_FLOOR_MAX = 0.02  # implausible-noise ceiling for eps_fail (fraction)
# Anchored to the script's own directory (not CWD) so the artifact lands in a
# predictable place no matter where the gate is invoked from.
G7_JSON_DEFAULT = Path(__file__).resolve().parent / "g7_selfvariance.json"


def _stems() -> list[str]:
    env = os.environ.get("RTMLIB_GATE_STEMS")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    return [ln.strip() for ln in STEMFILE.read_text().splitlines() if ln.strip()]


def _kp_selfvar(a: list, b: list) -> np.ndarray:
    """Per-keypoint L2 between two runs, IoU-matched per frame (flattened)."""
    out = []
    for fa, fb in zip(a, b):
        for i, j in match_dets(fa.bboxes, fb.bboxes):
            out.append(np.linalg.norm(fa.keypoints[i] - fb.keypoints[j], axis=-1))
    return np.concatenate(out) if out else np.array([0.0], dtype=np.float32)


def _fail_selfvar(a: list, b: list, stem: str, setup) -> float:
    res_df, court, params, RawClip, ClipContext, sticky_apply = setup
    ctx = ClipContext(vid=int(stem.split("_", 1)[0]), all_court_info=court, res_df=res_df)
    outs = []
    for frames in (a, b):
        r = assemble_raw_clip(frames)
        raw = RawClip(kps=r.kps, bboxes=r.bboxes, scores=r.bbox_scores,
                      kp_scores=r.kp_scores, ndet=r.ndet)
        outs.append(sticky_apply(raw, ctx, **params).failed.mean())
    return float(abs(outs[0] - outs[1]))


def main() -> int:
    setup = _setup()
    ext = RtmlibPoseExtractor(device=DEVICE)
    stems = _stems()
    print(f"G7 CUDA self-variance floor | device={DEVICE} | {len(stems)} clip(s)\n")

    rows, missing = [], []
    for stem in stems:
        mp4 = find_clip(stem)
        if mp4 is None:
            missing.append(stem)
            continue
        a = list(ext.iter_video(mp4))
        b = list(ext.iter_video(mp4))
        kp = _kp_selfvar(a, b)
        rows.append(dict(
            stem=stem, eps_kp_med=float(np.median(kp)), eps_kp_p90=float(np.percentile(kp, 90)),
            eps_kp_max=float(kp.max()), eps_fail=_fail_selfvar(a, b, stem, setup),
        ))

    if not rows:
        print("FAIL: no clips evaluated")
        return 1
    out_path = os.environ.get("RTMLIB_GATE_G7_JSON", str(G7_JSON_DEFAULT))
    Path(out_path).write_text(json.dumps(rows, indent=2))

    eps_kp_med = float(np.median([r["eps_kp_med"] for r in rows]))
    eps_kp_max = float(np.max([r["eps_kp_max"] for r in rows]))
    eps_fail_max = float(np.max([r["eps_fail"] for r in rows]))
    print(f"  clips={len(rows)}  missing={len(missing)}")
    print(f"  eps_kp:   median={eps_kp_med:.3f}px  max={eps_kp_max:.3f}px")
    print(f"  eps_fail: max per-clip={eps_fail_max:.4f}  ({eps_fail_max * 100:.2f} pp)")
    print(f"  floors -> {out_path}  (carry into phase_a_decision.py)")

    ok = eps_kp_med <= KP_FLOOR_MAX and eps_fail_max <= FAIL_FLOOR_MAX
    if not ok:
        print(f"  IMPLAUSIBLE NOISE: eps_kp_med>{KP_FLOOR_MAX} or eps_fail>{FAIL_FLOOR_MAX} "
              "(extract is not reproducible on this device)")
    print(f"\n{'PASS' if ok else 'FAIL'}: G7 CUDA self-variance floor")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
