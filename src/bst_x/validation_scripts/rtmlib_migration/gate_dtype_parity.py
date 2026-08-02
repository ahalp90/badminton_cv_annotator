"""G3: dtype-parity on the detect_players_2d path (RUN IN BATCH 3).

Under mmpose, ``detect_players_2d`` built ``np.array([p["keypoints"] for p ...])``
from Python lists -> float64. rtmlib returns float32 arrays, so without a cast the
computation stays float32, shifting ``normalize_joints`` / court projection at the
atol boundary. Batch 3's fix casts the adapter's keypoints/bbox to float64. This
gate asserts the fix is in effect.

Decisiveness: output dtype alone is NOT enough: ``_order_two_on_court`` fails
some frames (no valid two-player pair), and ``np.stack`` of float32
success rows + float64 failed-zeros promotes the whole array to float64, masking a
missing cast. So the gate checks the success-frame joints carry float64
*precision*: a float64 computation yields values not exactly representable in
float32, whereas a missing-cast float32 computation is equal to its own float32
round-trip. This is decisive on any clip with >=1 success frame.

Targets the Batch-3 signature ``detect_players_2d(extractor, video_path, ...)``.
Pre-migration the first parameter is still ``inferencer``, so the gate SKIPs
(exit 2) by signature inspection; it never swallows exceptions, so a real
post-migration error surfaces as a failure.

Env:
  RTMLIB_GATE_STEM  clip stem to test (default 11_1_10_2)

Run (in Batch 3):
  PYTHONPATH=src/bst_x:src XDG_CACHE_HOME=<warm-cache> <venv>/bin/python \\
      src/bst_x/validation_scripts/rtmlib_migration/gate_dtype_parity.py
"""
from __future__ import annotations

import inspect
import os
import sys

import numpy as np
from _common import find_clip

from preparing_data.rtmlib_pose import RtmlibPoseExtractor

STEM = os.environ.get("RTMLIB_GATE_STEM", "11_1_10_2")
SKIP = 2


def _court_setup():
    import pandas as pd
    from pipeline.config import RESOLUTION_CSV_PATH, SET_INFO_DIR
    from shared.court import get_court_info

    res_df = pd.read_csv(RESOLUTION_CSV_PATH).set_index("id")
    homo_df = pd.read_csv(str(SET_INFO_DIR / "homography.csv")).set_index("id")
    court = {vid: get_court_info(homo_df, vid) for vid in res_df.index}
    return res_df, court


def main() -> int:
    from preparing_data.prepare_train_on_shuttleset import detect_players_2d

    # Pre/post-migration switch by signature, NOT by catching exceptions, so a
    # real post-migration error surfaces instead of masquerading as a skip.
    first_param = next(iter(inspect.signature(detect_players_2d).parameters))
    if first_param == "inferencer":
        print("SKIP: detect_players_2d still on the MMPoseInferencer signature "
              f"(first param '{first_param}'). Run this gate in Batch 3, after the migration.")
        return SKIP

    mp4 = find_clip(STEM)
    if mp4 is None:
        print(f"FAIL: no mp4 for stem {STEM}")
        return 1
    res_df, court = _court_setup()
    ext = RtmlibPoseExtractor(device="cpu")
    # detect_players_2d now returns a 4th element (per-frame doubles over-count);
    # this gate only checks the pose dtypes, so it discards it.
    failed_ls, positions, joints, _overcount = detect_players_2d(ext, mp4, court, res_df)

    failed = np.asarray(failed_ls, dtype=bool)
    n_success = int((~failed).sum())
    succ = joints[~failed]  # (n_success, 2, J, 2); the cast-carrying rows

    pos_ok = positions.dtype == np.float64
    jnt_ok = joints.dtype == np.float64
    exercised_ok = n_success >= 1
    # float64 precision: success joints must NOT equal their float32 round-trip.
    f64_ok = bool(exercised_ok
                  and not np.array_equal(succ, succ.astype(np.float32).astype(np.float64)))

    print(f"clip {STEM}: {n_success}/{len(failed_ls)} frames kept (success path)\n")
    checks = [
        ("success path exercised (>=1 frame kept)", exercised_ok, f"{n_success} frames"),
        ("players_positions dtype float64", pos_ok, str(positions.dtype)),
        ("players_joints dtype float64", jnt_ok, str(joints.dtype)),
        ("success joints carry float64 precision (cast in effect)", f64_ok,
         "not float32-representable" if f64_ok else "values are float32-valued (cast missing?)"),
    ]
    all_ok = True
    for name, ok, msg in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        all_ok &= ok
    print(f"\n{'PASS' if all_ok else 'FAIL'}: G3 dtype parity")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
