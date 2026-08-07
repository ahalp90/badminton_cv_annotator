"""Gate: stitched output feeds the production downstream path unchanged.

Runs a sharded fake extraction of a video, then pushes the published five
arrays through the *production* consumer surfaces, none of them modified:

- ``apply_heuristic._raw_files_present`` / ``_load_raw_clip`` (the raw
  reader contract);
- ``apply_heuristic._vid_from_stem`` (the numeric-stem requirement);
- both registered heuristics (``current`` and the cross-frame-stateful
  ``sticky_anchor``) via ``heuristics.REGISTRY``.

Court context uses the identity-homography pattern established by
``tests/test_sticky_anchor.py`` at the video's real resolution — the gate
demonstrates array-contract compatibility; a production run would supply real
court info through the existing ``apply_heuristic`` CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from preparing_data.apply_heuristic import (
    _load_raw_clip,
    _raw_files_present,
    _vid_from_stem,
)
from preparing_data.heuristics import REGISTRY
from preparing_data.heuristics.base import ClipContext

from shared.video_sharding.range_decode import iter_frame_range
from shared.video_sharding.run_sharded import extract_sharded


def identity_ctx(vid: int, width: int, height: int) -> ClipContext:
    """Synthetic ClipContext: pixel->normalised mapping is identity over the frame."""
    court_info = {
        "H": np.eye(3, dtype=np.float64),
        "border_L": 0.0,
        "border_R": float(width),
        "border_U": 0.0,
        "border_D": float(height),
    }
    res_df = pd.DataFrame({"width": [width], "height": [height]}, index=[vid])
    return ClipContext(vid=vid, all_court_info={vid: court_info}, res_df=res_df)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--stem", default="21_full_poc",
                        help="numeric-prefixed stem, per the production naming contract")
    parser.add_argument("--n-shards", type=int, default=4)
    parser.add_argument("--limit-frames", type=int, default=200)
    args = parser.parse_args()

    vid = _vid_from_stem(args.stem)
    if vid is None:
        print(f"GATE downstream: FAIL — stem {args.stem!r} has no numeric video-id prefix; "
              f"apply_heuristic would silently skip it")
        return 1

    publish = extract_sharded(
        video_path=args.video,
        out_root=args.workdir,
        stem=args.stem,
        n_shards=args.n_shards,
        extractor_spec="fake",
        limit_frames=args.limit_frames,
    )

    if not _raw_files_present(publish, args.stem):
        print("GATE downstream: FAIL — _raw_files_present rejected the published set")
        return 1
    raw = _load_raw_clip(publish, args.stem)
    n_frames = raw.kps.shape[0]

    first_frame = next(iter_frame_range(args.video, 0, 1))
    height, width = first_frame.shape[:2]
    ctx = identity_ctx(vid, width, height)

    for name, heuristic_fn in REGISTRY.items():
        output = heuristic_fn(raw, ctx)
        shapes = {
            "pos": output.pos.shape,
            "joints": output.joints.shape,
            "failed": output.failed.shape,
            "overcount": output.overcount.shape,
        }
        frame_counts_ok = all(shape[0] == n_frames for shape in shapes.values())
        print(f"HEURISTIC {name}: ran on {n_frames} stitched frames, "
              f"output shapes {shapes}, frame-counts-ok={int(frame_counts_ok)}")
        if not frame_counts_ok:
            print("GATE downstream: FAIL — heuristic output frame count mismatch")
            return 1

    print("GATE downstream: PASS — production loader and both heuristics consumed "
          "the stitched output unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
