"""G6: deployed-output parity gate (CPU).

The end-to-end check: run the shipped adapter over a clip, assemble the raw
arrays exactly as ``raw_extract`` would (``_common.assemble_raw_clip``), feed
them through the repo's UNCHANGED ``sticky_anchor``, and compare the deployed
2-player output (``pos`` / ``joints`` / ``failed``) against the committed clean
(``sticky_anchor`` over the old mmpose raw). Proves the migration preserves what
the model actually consumes, despite the model swap (the detector finds slightly
more boxes than mmpose's run did).

This gate is *bbox-driven*: ``sticky_anchor`` selects players by box, so it can
agree here even if keypoint values drifted. The keypoint values are gated
separately by G1 (``gate_keypoint_value.py``); run both.

Per-clip checks split by category, because the built-in set mixes representative
clips with deliberately-degenerate stress clips:

* Structural (ALL clips, hard): ``dF == 0`` (same frame count), ``fmatch`` >=
  FMATCH_MIN (failed-frame agreement floor), ``rt<2 <= mm<2`` (rtmlib does not
  drop a salient player mmpose kept).
* Value (``diverse`` clips only): ``pos_med`` <= POS_MED_MAX (normalised court
  coords) and ``jnt_med`` <= JNT_MED_MAX (bbox-diagonal-normalised).
* ``busted`` stress clips: value deltas are reported as *diagnostic*, not gated.
  The mmpose baseline on these was already bad, so matching it to a few
  percent is not the acceptance criterion; keeping the players and the frame
  count is.

Over the built-in representative set the aggregate also requires mean ``fmatch``
>= MEAN_FMATCH_MIN (a population-health signal, prototype 0.981). That mean check
is skipped when ``RTMLIB_GATE_STEMS`` overrides the set with a custom subset.

Stems: ``RTMLIB_GATE_STEMS`` (comma-separated, gated as ``diverse``) or the
built-in diverse + busted set. Absent clips are logged, not silently skipped.
Broad shard coverage is the GPU parity gate (G8).

Env:
  RTMLIB_GATE_CLEAN  committed clean (sticky_anchor) dir
  RTMLIB_GATE_STEMS  comma-separated stems (overrides the built-in set)
  RTMLIB_GATE_JSON   optional path to dump per-clip results as JSON

Run:
  PYTHONPATH=src/bst_x:src XDG_CACHE_HOME=<warm-cache> <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_deployed_parity.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from _common import RAW, court_setup, deployed_parity, find_clip

from preparing_data.rtmlib_pose import RtmlibPoseExtractor

FMATCH_MIN = 0.85       # per-clip failed-frame agreement floor (all categories)
MEAN_FMATCH_MIN = 0.95  # aggregate over the default representative set
POS_MED_MAX = 0.02      # normalised court coords (diverse clips)
JNT_MED_MAX = 0.03      # bbox-diagonal-normalised (diverse clips)

# Diverse (non-11) representative clips; tags note match / class / lighting.
DIVERSE: list[tuple[str, str]] = [
    ("44_1_12_12", "HSBC-DARK men Top_smash"),
    ("38_1_10_2", "HSBC-DARK women Bot_smash"),
    ("1_1_11_21", "Fuzhou men Bot_smash"),
    ("18_1_11_3", "AllEngland Top_smash"),
    ("32_1_10_2", "Thailand men Bot_net"),
    ("25_1_12_5", "Thailand women Bot_net"),
    ("44_1_11_5", "HSBC-DARK men Top_net"),
    ("1_1_12_8", "Fuzhou Bot_drive"),
    ("44_1_1_4", "HSBC-DARK Bot_drive"),
    ("32_1_13_9", "Thailand Top_drive"),
    ("38_1_13_4", "HSBC-DARK Bot_rush"),
    ("32_1_24_6", "Thailand Bot_rush"),
    ("40_1_23_6", "HSBC-DARK women Xnet"),
    ("21_1_12_8", "Thailand women Bot_smash"),
]
# Known-hard busted clips (mmpose baseline already degenerate): structural only.
BUSTED: list[tuple[str, str]] = [
    ("11_2_25_10", "busted g2"),
    ("11_2_29_12", "busted g2"),
    ("11_2_7_14", "busted g2"),
    ("11_2_21_18", "busted g2"),
    ("11_1_35_17", "busted g1"),
    ("11_1_4_3", "busted g1"),
]
DEFAULT_STEMS: list[tuple[str, str, str]] = (
    [(s, t, "diverse") for s, t in DIVERSE] + [(s, t, "busted") for s, t in BUSTED]
)


def _stems() -> list[tuple[str, str, str]]:
    env = os.environ.get("RTMLIB_GATE_STEMS")
    if env:
        # A hand-picked subset breaks the population-health assumption behind
        # MEAN_FMATCH_MIN, so main() skips that aggregate entirely for this run.
        print(f"WARNING: RTMLIB_GATE_STEMS override active; the aggregate mean "
              f"fmatch>={MEAN_FMATCH_MIN} check does not apply to this custom "
              f"stem subset (per-clip verdicts still gate).")
        return [(s.strip(), "env", "diverse") for s in env.split(",") if s.strip()]
    return DEFAULT_STEMS


def _gate_clip(ext, stem, category, setup) -> dict | None:
    mp4 = find_clip(stem)
    if mp4 is None:
        return None
    t0 = time.time()
    frames = list(ext.iter_video(mp4))
    p = deployed_parity(frames, stem, setup)
    mm_ndet = np.load(RAW / f"{stem}_raw_ndet.npy")
    Frt, Fmm = len(p.out_failed), len(p.ref_failed)

    return dict(
        stem=stem, category=category, Frt=Frt, Fmm=Fmm, dF=Frt - Fmm,
        fmatch=p.fmatch, both=p.nb, pos_med=p.pos_med, jnt_med=p.jnt_med,
        rt_only_fail=p.rt_only_fail, mm_only_fail=p.mm_only_fail,
        rt_lt2=int((p.raw_arr.ndet < 2).sum()), mm_lt2=int((mm_ndet < 2).sum()),
        secs=time.time() - t0,
    )


def _verdict(r: dict) -> bool:
    """Structural checks on all clips; value checks on diverse clips only."""
    structural = (r["dF"] == 0 and r["fmatch"] >= FMATCH_MIN
                  and r["rt_lt2"] <= r["mm_lt2"])
    if r["category"] == "busted":
        return structural
    # Diverse clips must be *evaluated* on the value axis, not pass vacuously when
    # zero both-success frames leave pos_med/jnt_med NaN (NaN > MAX is False).
    value_measured = (r["both"] > 0 and not np.isnan(r["pos_med"])
                      and not np.isnan(r["jnt_med"]))
    return (structural and value_measured
            and r["pos_med"] <= POS_MED_MAX and r["jnt_med"] <= JNT_MED_MAX)


def main() -> int:
    setup = court_setup()
    ext = RtmlibPoseExtractor(device="cpu")
    stems = _stems()

    hdr = (f"  {'stem':13s} {'cat':7s} {'dF':>3} {'fmatch':>6} {'both':>4} {'posMed':>7} "
           f"{'jntMed':>7} {'rtLoss':>6} {'mmLoss':>6} {'rt<2':>4} {'mm<2':>4}  verdict")
    print(f"G6 deployed-parity over {len(stems)} clip(s)  "
          f"(rtLoss/mmLoss = frames one side fails and the other keeps)"
          f"\n\n{hdr}\n  {'-' * (len(hdr) - 2)}")

    results, missing = [], []
    for stem, _tag, category in stems:
        r = _gate_clip(ext, stem, category, setup)
        if r is None:
            missing.append(stem)
            print(f"  {stem:13s}  no mp4 (skipped, logged)")
            continue
        r["ok"] = _verdict(r)
        results.append(r)
        flags = ""
        if r["rt_lt2"] > r["mm_lt2"]:
            flags += "  *DROPPED PLAYER?*"
        if category == "busted" and (r["pos_med"] > POS_MED_MAX or r["jnt_med"] > JNT_MED_MAX):
            flags += "  [busted: value diag over diverse-threshold]"
        print(f"  {r['stem']:13s} {category:7s} {r['dF']:3d} {r['fmatch']:6.3f} {r['both']:4d} "
              f"{r['pos_med']:7.4f} {r['jnt_med']:7.4f} {r['rt_only_fail']:6d} {r['mm_only_fail']:6d} "
              f"{r['rt_lt2']:4d} {r['mm_lt2']:4d}  {'PASS' if r['ok'] else 'FAIL'}{flags}")

    if (out_path := os.environ.get("RTMLIB_GATE_JSON")):
        Path(out_path).write_text(json.dumps(results, indent=2))

    if not results:
        print("\nFAIL: no clips evaluated (all missing?)")
        return 1
    mean_fmatch = float(np.mean([r["fmatch"] for r in results]))
    per_clip_ok = all(r["ok"] for r in results)
    # The population mean gates only the built-in representative set.
    default_set = not os.environ.get("RTMLIB_GATE_STEMS")
    mean_ok = mean_fmatch >= MEAN_FMATCH_MIN or not default_set
    agg_ok = per_clip_ok and mean_ok
    tot_rt_only = sum(r["rt_only_fail"] for r in results)
    tot_mm_only = sum(r["mm_only_fail"] for r in results)
    print(f"\n  clips={len(results)}  missing={len(missing)}  mean fmatch={mean_fmatch:.3f}"
          f"  pos_med(max)={np.nanmax([r['pos_med'] for r in results]):.4f}"
          f"  jnt_med(max)={np.nanmax([r['jnt_med'] for r in results]):.4f}")
    print(f"  directional frame loss (all clips): rtmlib-only-fail={tot_rt_only}  "
          f"mmpose-only-fail={tot_mm_only}  (expect near-symmetric noise; a "
          f"one-directional excess means one extractor systematically loses frames "
          f"the other keeps)")
    if missing:
        print(f"  missing stems (not evaluated): {', '.join(missing)}")
    if not per_clip_ok:
        print(f"  per-clip failures: {[r['stem'] for r in results if not r['ok']]}")
    if default_set and mean_fmatch < MEAN_FMATCH_MIN:
        print(f"  population mean fmatch {mean_fmatch:.3f} < {MEAN_FMATCH_MIN} on the default set")
    print(f"\n{'PASS' if agg_ok else 'FAIL'}: G6 deployed-output parity")
    return 0 if agg_ok else 1


if __name__ == "__main__":
    sys.exit(main())
