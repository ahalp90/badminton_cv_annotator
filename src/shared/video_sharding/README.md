# video_sharding — map

PoC for extracting RTMLib pose from one long video with several worker
processes, each decoding its own frame range. Start with `HANDOFF.md` for
findings; this file is just a map.

Run anything here from the repo root with `PYTHONPATH=src:src/bst_x`, e.g.:

```bash
PYTHONPATH=src:src/bst_x python -m shared.video_sharding.run_sharded --help
```

## The pipeline

| File | What it does |
|---|---|
| `shard_plan.py` | Splits `[0, n_frames)` into contiguous ranges, one per worker. |
| `range_decode.py` | Reads an exact frame range from a video (seek, or slow decode-from-zero as a control). Also file/frame MD5 helpers. |
| `shard_worker.py` | One worker process: decode its range, run the extractor, save the five arrays compressed plus a manifest. Manifest written last = shard complete. |
| `stitch.py` | Checks every shard (coverage, run ID, source MD5, shapes, `n_max`), then concatenates and publishes the normal `{stem}_raw_*.npy` files. Refuses loudly on anything suspicious. |
| `run_sharded.py` | The entry point. Plans, launches workers, checks their exit codes, stitches. CLI and importable `extract_sharded()`. |
| `fake_pose.py` | Fake extractor whose output depends only on frame pixels. Lets tests prove sequential == sharded without models or a GPU. |

## The checks (gates)

| File | What it proves |
|---|---|
| `gate_decode_identity.py` | Seeked frames are identical to a full sequential decode. Run `baseline` once per video, then `check`. Run this first on any new source. |
| `gate_parity.py` | Sequential extraction and sharded extraction give identical arrays. `--extractor fake\|cpu\|cuda`; `--self-variance` measures run-to-run noise instead. |
| `gate_downstream.py` | The stitched output loads through the real `apply_heuristic` code and both heuristics unchanged. |
| `bench_worker_scaling.py` | Times 1/2/4/8 workers on the same input. |

## Everything else

| File | What it is |
|---|---|
| `HANDOFF.md` | The findings and the recommended production change. Read this one. |
| `INVESTIGATION.md` | The experiment ledger: what was uncertain before each run and what settled it. Historical; do not update. |
| `run_remote_bourbaki.sh` | The exact ladder of gates run on bourbaki (env setup, paths, tmux-friendly). Keep until the gates have a production home. |
| `tests/test_video_sharding.py` (repo root) | 18 deterministic tests: planning, parity, worker failure, stitch corruption, downstream. No GPU or rtmlib needed. |
