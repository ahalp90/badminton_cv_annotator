"""Orchestrate a frame-range-sharded extraction of one video.

Plan shards, record the run manifest, spawn one worker process per shard
(spawn context: each worker builds its own extractor/onnxruntime session),
join, then stitch and publish. Any worker with a nonzero exit code aborts the
run before stitching — exit status stays observable, and the stitcher would
refuse the manifest-less shard anyway.

Run from the repo root::

    PYTHONPATH=src:src/bst_x python -m shared.video_sharding.run_sharded \
        --video <match.mp4> --stem 21_full --out-root <dir> --n-shards 4 \
        --extractor fake

``--limit-frames`` bounds the run to the first N frames (for bounded real
inference experiments); the end-of-video probe only applies when the whole
video is planned.
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import uuid
from pathlib import Path

from shared.video_sharding.range_decode import (
    md5_file,
    metadata_frame_count,
)
from shared.video_sharding.shard_plan import (
    NDET_INT8_CAP,
    plan_frame_shards,
)
from shared.video_sharding.shard_worker import EXTRACTOR_SPECS, worker_entry
from shared.video_sharding.stitch import (
    stitch_and_publish,
    write_run_manifest,
)


def extract_sharded(
    *,
    video_path: Path,
    out_root: Path,
    stem: str,
    n_shards: int,
    n_max: int = 16,
    extractor_spec: str = "fake",
    decode_mode: str = "seek",
    limit_frames: int | None = None,
    run_id: str | None = None,
    expected_frame_count: int | None = None,
) -> Path:
    """Run the full plan->workers->stitch pipeline; return the publish dir."""
    if isinstance(n_max, bool) or not isinstance(n_max, int) or not 0 < n_max <= NDET_INT8_CAP:
        raise ValueError(f"n_max must be in [1, {NDET_INT8_CAP}], got {n_max!r}")
    if not stem or Path(stem).name != stem or stem in {".", ".."}:
        raise ValueError(f"stem must be a path-safe basename: {stem!r}")
    observed_frame_count = metadata_frame_count(video_path)
    if expected_frame_count is not None:
        if (
            isinstance(expected_frame_count, bool)
            or not isinstance(expected_frame_count, int)
            or expected_frame_count <= 0
        ):
            raise ValueError(
                f"expected_frame_count must be a positive integer, got {expected_frame_count!r}"
            )
        if observed_frame_count != expected_frame_count:
            raise ValueError(
                "OpenCV frame count differs from canonical metadata: "
                f"observed={observed_frame_count}, canonical={expected_frame_count}"
            )
    n_frames = observed_frame_count
    whole_video = limit_frames is None or limit_frames >= n_frames
    if not whole_video:
        n_frames = limit_frames
    plan = plan_frame_shards(n_frames, n_shards)
    source_md5 = md5_file(video_path)
    run_id = run_id or uuid.uuid4().hex[:12]
    if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
        raise ValueError(f"run_id must be a path-safe basename: {run_id!r}")

    run_dir = out_root / f"run_{run_id}"
    run_dir.mkdir(parents=True)
    write_run_manifest(run_dir, {
        "run_id": run_id,
        "source_name": Path(video_path).name,
        "source_md5": source_md5,
        "n_frames": n_frames,
        "n_max": n_max,
        "n_shards": n_shards,
        "extractor": extractor_spec,
        "decode_mode": decode_mode,
        "plan": [list(shard_range) for shard_range in plan],
    })

    ctx = multiprocessing.get_context("spawn")
    workers: list[tuple[tuple[int, int], multiprocessing.Process]] = []
    try:
        for start, end in plan:
            kwargs = {
                "video_path": str(video_path),
                "start": start,
                "end": end,
                "n_max": n_max,
                "run_dir": str(run_dir),
                "run_id": run_id,
                "source_md5": source_md5,
                "extractor_spec": extractor_spec,
                "decode_mode": decode_mode,
                # Only the true last shard of a whole-video run probes for extra
                # frames beyond the plan (guards a lying container frame count).
                "probe_past_end": whole_video and end == n_frames,
            }
            process = ctx.Process(
                target=worker_entry,
                args=(kwargs,),
                name=f"shard_{start}_{end}",
            )
            process.start()
            workers.append(((start, end), process))

        failed: list[str] = []
        for (start, end), process in workers:
            process.join()
            if process.exitcode != 0:
                failed.append(f"[{start}, {end}) exit={process.exitcode}")
    except BaseException:
        for _, process in workers:
            if process.is_alive():
                process.terminate()
        for _, process in workers:
            process.join()
        raise
    if failed:
        raise RuntimeError(f"{len(failed)} shard worker(s) failed: {'; '.join(failed)}")

    return stitch_and_publish(run_dir, out_root / f"publish_{run_id}", stem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--stem", required=True,
                        help="published stem; must start with the numeric video id "
                             "(e.g. 21_full) for apply_heuristic's stem parsing")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--n-max", type=int, default=16)
    parser.add_argument("--extractor", choices=EXTRACTOR_SPECS, default="fake")
    parser.add_argument("--decode-mode", choices=("seek", "scan"), default="seek")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--expected-frame-count", type=int, default=None)
    args = parser.parse_args()

    published = extract_sharded(
        video_path=args.video,
        out_root=args.out_root,
        stem=args.stem,
        n_shards=args.n_shards,
        n_max=args.n_max,
        extractor_spec=args.extractor,
        decode_mode=args.decode_mode,
        limit_frames=args.limit_frames,
        expected_frame_count=args.expected_frame_count,
    )
    print(f"published: {published}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
