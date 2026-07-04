"""G5: CPU downstream byte-equality precondition (dual-invocation).

Proves the migration touches nothing downstream of the extractor, and that the
new numpy-2 / pandas-3 environment reproduces the committed pipeline output.

Mechanism (inference-free, pure heuristic on fixed input):

* Feed the *committed mmpose raw* (the fixed reference input, unchanged by this
  branch) through the repo's CURRENT ``sticky_anchor``.
* Determinism: run it twice; the two outputs must be bit-identical.
* Reproduction: compare to the committed clean (``sticky_anchor`` over the same
  mmpose raw, produced on main): ``_failed`` exact, ``_pos`` / ``_joints``
  within ``ATOL``.

If this passes, ``sticky_anchor`` + collate are provably untouched and env-stable,
so any G6 deployed-parity difference is attributable to the extractor swap alone,
not to downstream drift or a pandas-version change. If ``_pos`` / ``_joints``
drift above ``ATOL``, the new env perturbs the heuristic (e.g. a pandas-3
inplace/iloc behaviour change) and must be investigated before trusting G6.

The reference here is the committed clean, a durable production artifact, not a
capture this script writes, so it will not rot when paths flip.

Stems: ``RTMLIB_GATE_STEMS`` (comma-separated) or the first ``MAX_CLIPS`` (sorted)
present in both the committed raw and clean dirs.

Run:
  PYTHONPATH=src/bst_x:src <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_downstream_byteeq.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from _common import RAW, load_mmpose_raw

CLEAN = Path(os.environ.get(
    "RTMLIB_GATE_CLEAN",
    "/srv/mergerfs/main_pool/320_cosc594_data-bourbaki/ShuttleSet_keypoints_clean_sticky_anchor",
))
ATOL = 1e-5
MAX_CLIPS = int(os.environ.get("RTMLIB_GATE_MAXCLIPS", "50"))


def _resolve_stems() -> list[str]:
    env = os.environ.get("RTMLIB_GATE_STEMS")
    if env:
        return [s.strip() for s in env.split(",") if s.strip()]
    raw_stems = {p.name[: -len("_raw_kps.npy")] for p in RAW.glob("*_raw_kps.npy")}
    clean_stems = {p.name[: -len("_failed.npy")] for p in CLEAN.glob("*_failed.npy")}
    return sorted(raw_stems & clean_stems)[:MAX_CLIPS]


def _setup():
    import pandas as pd
    from pipeline.config import RESOLUTION_CSV_PATH, SET_INFO_DIR
    from pipeline.court_utils import get_court_info

    from preparing_data.heuristics.base import ClipContext, RawClip
    from preparing_data.heuristics.sticky_anchor import StickyAnchorParams
    from preparing_data.heuristics.sticky_anchor import apply as sticky_apply

    print(f"numpy {np.__version__} | pandas {pd.__version__}")
    res_df = pd.read_csv(RESOLUTION_CSV_PATH).set_index("id")
    homo_df = pd.read_csv(str(SET_INFO_DIR / "homography.csv")).set_index("id")
    court = {vid: get_court_info(homo_df, vid) for vid in res_df.index}
    params = {f.name: f.default for f in fields(StickyAnchorParams)}
    return res_df, court, params, RawClip, ClipContext, sticky_apply


def _equal_nan(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def _max_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a-b| ignoring positions that are NaN in both (failed frames)."""
    if a.shape != b.shape:
        return float("inf")
    m = ~(np.isnan(a) & np.isnan(b))
    return float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0


def _gate_clip(stem, setup) -> tuple[bool, str]:
    res_df, court, params, RawClip, ClipContext, sticky_apply = setup
    mm = load_mmpose_raw(stem)
    raw = RawClip(kps=mm.kps, bboxes=mm.bboxes, scores=mm.bbox_scores,
                  kp_scores=mm.kp_scores, ndet=mm.ndet)
    ctx = ClipContext(vid=int(stem.split("_", 1)[0]), all_court_info=court, res_df=res_df)
    out1 = sticky_apply(raw, ctx, **params)
    out2 = sticky_apply(raw, ctx, **params)

    det_ok = (_equal_nan(out1.failed, out2.failed) and _equal_nan(out1.pos, out2.pos)
              and _equal_nan(out1.joints, out2.joints))

    ref_failed = np.load(CLEAN / f"{stem}_failed.npy")
    ref_pos = np.load(CLEAN / f"{stem}_pos.npy")
    ref_joints = np.load(CLEAN / f"{stem}_joints.npy")

    failed_ok = _equal_nan(out1.failed, ref_failed)
    dpos = _max_delta(out1.pos, ref_pos)
    djnt = _max_delta(out1.joints, ref_joints)
    ok = det_ok and failed_ok and dpos <= ATOL and djnt <= ATOL
    msg = f"det={det_ok} failed_exact={failed_ok} max|Δpos|={dpos:.2e} max|Δjnt|={djnt:.2e}"
    return ok, msg


def main() -> int:
    setup = _setup()
    stems = _resolve_stems()
    if not stems:
        print(f"FAIL: no stems present in both {RAW} and {CLEAN}")
        return 1
    print(f"G5 downstream byte-equality over {len(stems)} clip(s)\n")
    all_ok = True
    fails = []
    for stem in stems:
        ok, msg = _gate_clip(stem, setup)
        all_ok &= ok
        if not ok:
            fails.append(stem)
            print(f"  [FAIL] {stem}: {msg}")
    # succinct summary; individual PASS lines suppressed for the large default set
    print(f"\n  {'all clips reproduce the committed clean' if all_ok else f'{len(fails)} clip(s) FAILED: ' + ', '.join(fails)}")
    print(f"\n{'PASS' if all_ok else 'FAIL'}: G5 downstream byte-equality")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
