"""Bounded worker-scaling probe for sharded extraction of one video.

Times ``extract_sharded`` at each requested worker count over the same bounded
input (workers == shards in this PoC). Reports wall time, throughput, and
whether the runs published successfully. Timing includes worker spawn, model
load, decode, inference, shard IO and stitch — the number that matters for a
real re-extract.

Intended for the CUDA host; run inside tmux. Example::

    OMP_NUM_THREADS=2 PYTHONPATH=src/bst_x python -m \
        shared.video_sharding.bench_worker_scaling \
        --video <match.mp4> --workdir /scratch/.../bench --extractor cuda \
        --limit-frames 12000 --worker-counts 1,2,4,8
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from shared.video_sharding.gate_parity import cut_first_frames
from shared.video_sharding.run_sharded import extract_sharded
from shared.video_sharding.shard_worker import EXTRACTOR_SPECS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--extractor", choices=EXTRACTOR_SPECS, default="cuda")
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--limit-frames", type=int, default=12000)
    parser.add_argument("--worker-counts", default="1,2,4,8",
                        help="comma-separated shard/worker counts to time")
    args = parser.parse_args()
    counts = [int(token) for token in args.worker_counts.split(",")]

    args.workdir.mkdir(parents=True, exist_ok=True)
    video = cut_first_frames(args.video, args.limit_frames, args.workdir)
    print(f"bounded input: {video}")

    print(f"{'workers':>7} {'frames':>7} {'wall_s':>8} {'ms/frame':>9} {'fps':>7}")
    for n_workers in counts:
        run_root = args.workdir / f"scale_{n_workers}"
        if run_root.exists():
            shutil.rmtree(run_root)  # fresh timing run, no stale shards
        started = time.perf_counter()
        publish = extract_sharded(
            video_path=video,
            out_root=run_root,
            stem="0_bench",
            n_shards=n_workers,
            n_max=args.n_max,
            extractor_spec=args.extractor,
        )
        wall = time.perf_counter() - started
        n_frames = int(np.load(publish / "0_bench_raw_ndet.npy").shape[0])
        print(f"{n_workers:>7} {n_frames:>7} {wall:>8.1f} "
              f"{1000 * wall / n_frames:>9.2f} {n_frames / wall:>7.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
