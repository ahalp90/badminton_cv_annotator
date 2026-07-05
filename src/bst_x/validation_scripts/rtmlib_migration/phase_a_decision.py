"""G9: Phase-A decision gate (HALT-AND-HANDOFF, Bourbaki).

(Gate numbering: G7 = CUDA self-variance, G8 = extraction parity, G9 = this
decision.)

Consumes the G8 parity JSON (per-clip extraction + deployed metrics) and, if
present, the G7 self-variance floors, and applies the concrete Phase-A
acceptance thresholds:

* Frame count: 100% of clips ``dF == 0``; any mismatch is a hard NO-GO.
* Deployed failed-rate: aggregate ``|Δ|`` <= FAIL_DELTA_AGG_PP, per-clip
  ``|Δ|`` <= FAIL_DELTA_CLIP_PP; per-frame failed-agreement >= FMATCH_AGG_MIN.
* Keypoint agreement: aggregate median L2 <= KP_MED_MAX, confident p90 <=
  KP_P90_MAX (NaN clips dropped before aggregating, not NaN-poisoning the median),
  PLUS a per-clip bound so a median-of-medians can't hide one broken clip, PLUS
  every clip's own G8 verdict must PASS (G9 never overrides a G8 per-clip fail).
* No dropped players: ``rt<2 <= mm<2`` on every clip.
* Floor validity: the failed-rate and keypoint thresholds must sit ABOVE the G7
  noise floors, else the gate is measuring CUDA jitter (INVALID, not GO).
* ``joints`` delta: reported only (model-drift confounded).

Stratification: the decision is authoritative only when the sample covers every
distinct video-id in ``res_df`` (court homography is per-video). smoke50 alone is
mostly one match, so a smoke50-only run reports **GO (provisional)**; add
shard_00/01 + a per-video top-up for the authoritative call.

Env / args:
  argv[1] or RTMLIB_GATE_G8_JSON   G8 parity JSON (default: g8_parity.json)
  RTMLIB_GATE_G7_JSON              G7 floors JSON (default: g7_selfvariance.json;
                                   absent -> floors assumed 0 with a warning)

Run (on Bourbaki, after G7 + G8):
  PYTHONPATH=src/bst_x:src <env>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/phase_a_decision.py g8_parity.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

FAIL_DELTA_AGG_PP = 2.0   # aggregate deployed failed-rate |Δ| ceiling (pp)
FAIL_DELTA_CLIP_PP = 5.0  # per-clip failed-rate |Δ| ceiling (pp)
FMATCH_AGG_MIN = 0.95     # aggregate per-frame failed-agreement floor
KP_MED_MAX = 5.0          # aggregate keypoint median L2 (px)
KP_P90_MAX = 12.0         # aggregate confident keypoint p90 (px)


def _load(path: str) -> list | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def _coverage(stems: list[str]) -> tuple[int, int]:
    """Distinct video-ids covered by the sample vs total in res_df."""
    import pandas as pd
    from pipeline.config import RESOLUTION_CSV_PATH

    total = set(pd.read_csv(RESOLUTION_CSV_PATH).set_index("id").index.tolist())
    covered = {int(s.split("_", 1)[0]) for s in stems} & total
    return len(covered), len(total)


def main() -> int:
    g8_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "RTMLIB_GATE_G8_JSON", "g8_parity.json")
    rows = _load(g8_path)
    if not rows:
        print(f"FAIL: G8 parity JSON not found or empty: {g8_path}")
        return 1

    floors = _load(os.environ.get("RTMLIB_GATE_G7_JSON", "g7_selfvariance.json"))
    if floors:
        eps_kp = float(np.max([f["eps_kp_med"] for f in floors]))
        eps_fail_pp = float(np.max([f["eps_fail"] for f in floors])) * 100
    else:
        eps_kp = eps_fail_pp = 0.0
        print("WARNING: no G7 floor JSON, assuming 0 (valid only for a CPU/deterministic run)\n")

    # --- criteria ---
    frame_ok = all(r["dF"] == 0 for r in rows)
    clip_fail_pp = [abs(r["rt_failrate"] - r["mm_failrate"]) * 100 for r in rows]
    clip_fail_ok = max(clip_fail_pp) <= FAIL_DELTA_CLIP_PP
    agg_fail_pp = abs(np.mean([r["rt_failrate"] for r in rows])
                      - np.mean([r["mm_failrate"] for r in rows])) * 100
    agg_fail_ok = agg_fail_pp <= FAIL_DELTA_AGG_PP
    fmatch_agg = float(np.mean([r["fmatch"] for r in rows]))
    fmatch_ok = fmatch_agg >= FMATCH_AGG_MIN
    # NaN kp fields come from zero-match / no-confident-joint clips (G8 writes NaN
    # there and passes them per its own policy). Drop NaN before aggregating so a
    # single hard clip can't NaN-poison the median into a spurious NO-GO; a fully
    # unmatched sample leaves the list empty -> NaN -> fail (can't decide GO).
    kp_meds = [r["kp_med"] for r in rows if not np.isnan(r["kp_med"])]
    kp_p90s = [r["kp_cp90"] for r in rows if not np.isnan(r["kp_cp90"])]
    kp_med_agg = float(np.median(kp_meds)) if kp_meds else float("nan")
    kp_p90_agg = float(np.percentile(kp_p90s, 90)) if kp_p90s else float("nan")
    kp_med_ok = kp_med_agg <= KP_MED_MAX
    kp_p90_ok = kp_p90_agg <= KP_P90_MAX
    # A median-of-medians / p90-of-p90s hides one badly-broken clip; enforce the
    # per-clip keypoint bounds too (NaN-safe: NaN > x is False, matching G8).
    kp_clip_bad = [r["stem"] for r in rows
                   if r["kp_med"] > KP_MED_MAX or r["kp_cp90"] > KP_P90_MAX]
    kp_clip_ok = not kp_clip_bad
    # Ingest G8's own per-clip verdict so a clip G8 failed cannot pass here on
    # lenient aggregates (G8 writes "ok" per clip; absent -> treat as not-passed).
    g8_bad = [r["stem"] for r in rows if not r.get("ok", False)]
    g8_ok = not g8_bad
    dropped = [r["stem"] for r in rows if r["rt_lt2"] > r["mm_lt2"]]
    dropped_ok = not dropped
    # thresholds must sit above the noise floor to be meaningful
    floor_valid = eps_kp < KP_MED_MAX and eps_fail_pp < FAIL_DELTA_AGG_PP

    covered, total = _coverage([r["stem"] for r in rows])
    full_coverage = covered == total

    def line(name, ok, detail):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    print(f"Phase-A decision over {len(rows)} clip(s)  (G8={g8_path})\n")
    line("frame count (100% dF==0)", frame_ok,
         f"{sum(r['dF'] == 0 for r in rows)}/{len(rows)} clips")
    line(f"failed-rate aggregate |Δ|<={FAIL_DELTA_AGG_PP}pp", agg_fail_ok, f"{agg_fail_pp:.2f}pp")
    line(f"failed-rate per-clip |Δ|<={FAIL_DELTA_CLIP_PP}pp", clip_fail_ok,
         f"max {max(clip_fail_pp):.2f}pp")
    line(f"failed-agreement >={FMATCH_AGG_MIN}", fmatch_ok, f"{fmatch_agg:.3f}")
    line(f"keypoint median <={KP_MED_MAX}px", kp_med_ok, f"{kp_med_agg:.2f}px")
    line(f"keypoint p90 <={KP_P90_MAX}px", kp_p90_ok, f"{kp_p90_agg:.2f}px")
    line(f"per-clip keypoint bounds ({KP_MED_MAX}px/{KP_P90_MAX}px)", kp_clip_ok,
         kp_clip_bad or "every clip within bounds")
    line("per-clip G8 verdicts all PASS", g8_ok, g8_bad or "all PASS")
    line("no dropped players (rt<2<=mm<2)", dropped_ok, dropped or "none")
    line("floors below thresholds", floor_valid,
         f"eps_kp={eps_kp:.2f}px eps_fail={eps_fail_pp:.2f}pp")
    # Directional failed-frame loss: a one-directional excess means one
    # extractor is losing frames the other keeps, which the aggregate can hide.
    rt_only = sum(r.get("rt_only_fail", 0) for r in rows)
    mm_only = sum(r.get("mm_only_fail", 0) for r in rows)
    signed_fail_pp = (float(np.mean([r["rt_failrate"] for r in rows]))
                      - float(np.mean([r["mm_failrate"] for r in rows]))) * 100
    jnt_meds = [r["jnt_med"] for r in rows if not np.isnan(r["jnt_med"])]
    jnt_med_agg = float(np.median(jnt_meds)) if jnt_meds else float("nan")
    print(f"  joints delta (report only): median-of-clip-medians {jnt_med_agg:.4f}")
    print(f"  directional loss: rtmlib-only-fail={rt_only} mmpose-only-fail={mm_only} "
          f"(signed failed-rate delta {signed_fail_pp:+.2f}pp; expect near-symmetric "
          f"noise, a one-directional excess means systematic frame loss)")
    print(f"  video-id coverage: {covered}/{total} "
          f"{'(full)' if full_coverage else '(PARTIAL: add shards for authoritative)'}")

    hard_ok = (frame_ok and clip_fail_ok and agg_fail_ok and fmatch_ok
               and kp_med_ok and kp_p90_ok and kp_clip_ok and g8_ok
               and dropped_ok and floor_valid)
    if not hard_ok:
        decision = "NO-GO"
    elif not full_coverage:
        decision = "GO (provisional: sample does not cover all video-ids)"
    else:
        decision = "GO"
    print(f"\nPhase-A: {decision}")
    return 0 if hard_ok else 1


if __name__ == "__main__":
    sys.exit(main())
