"""G4: CPU determinism gate.

Two independent adapter instances (fresh onnxruntime sessions) over the same clip
must produce bit-identical detections. onnxruntime's CPU EP is deterministic for
a fixed input + model + thread count, so any run-to-run drift here means a
nondeterministic op or random initialisation slipped into the extract, which
would make the raw arrays (and every downstream stage) unreproducible.

This asserts *run-to-run on one machine*. Cross-machine determinism additionally
needs the same onnxruntime build and a pinned thread count (set
``OMP_NUM_THREADS=1`` / a single intra-op thread); the CUDA EP is not
deterministic at all. Those are properties of the GPU handoff, not of this gate.

Env:
  RTMLIB_GATE_STEM    clip stem (default 11_1_10_2)
  RTMLIB_GATE_MAXFR   frames to compare (default 20; determinism is per-op, so a
                      short prefix with several detections exercises it fully)

Run:
  PYTHONPATH=src:src/bst_x XDG_CACHE_HOME=<warm-cache> OMP_NUM_THREADS=1 <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_cpu_determinism.py
"""
from __future__ import annotations

import os
import sys
from itertools import islice

import numpy as np
from _common import find_clip

from preparing_data.rtmlib_pose import RtmlibPoseExtractor

STEM = os.environ.get("RTMLIB_GATE_STEM", "11_1_10_2")
MAX_FRAMES = int(os.environ.get("RTMLIB_GATE_MAXFR", "20"))


def _run(ext: RtmlibPoseExtractor, mp4, n: int) -> list:
    return list(islice(ext.iter_video(mp4), n))


def main() -> int:
    omp = os.environ.get("OMP_NUM_THREADS")
    if omp != "1":
        print(
            f"FAIL: OMP_NUM_THREADS={omp!r}; the determinism check is only "
            f"meaningful single-threaded. Rerun with OMP_NUM_THREADS=1."
        )
        return 1

    mp4 = find_clip(STEM)
    if mp4 is None:
        print(f"FAIL: no mp4 for stem {STEM}")
        return 1

    a = _run(RtmlibPoseExtractor(device="cpu"), mp4, MAX_FRAMES)
    b = _run(RtmlibPoseExtractor(device="cpu"), mp4, MAX_FRAMES)

    if len(a) != len(b):
        print(f"FAIL: frame count differs {len(a)} != {len(b)}")
        return 1

    mism = []
    for f, (fa, fb) in enumerate(zip(a, b)):
        if len(fa.keypoints) != len(fb.keypoints):
            mism.append(f"frame {f}: ndet {len(fa.keypoints)} != {len(fb.keypoints)}")
            continue
        for field in ("keypoints", "bboxes", "bbox_scores", "kp_scores"):
            if not np.array_equal(getattr(fa, field), getattr(fb, field)):
                d = float(np.abs(getattr(fa, field) - getattr(fb, field)).max())
                mism.append(f"frame {f}: {field} differs (max|Δ|={d:g})")

    ok = not mism
    print(f"clip {STEM}: compared {len(a)} frames across two fresh sessions")
    if ok:
        print(f"\nPASS: G4 CPU determinism (bit-identical over {len(a)} frames)")
        return 0
    for m in mism[:10]:
        print(f"  [FAIL] {m}")
    print(f"\nFAIL: G4 CPU determinism ({len(mism)} mismatches)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
