"""Frame-range-sharded RTMLib extraction core for one long video.

The dataset builder uses the planner/worker/stitch path through its isolated
pose-interpreter boundary. See INVESTIGATION.md for the historical ledger and
HANDOFF.md for evidence plus environment-specific acceptance gates.

Layout:

- ``shard_plan``            partition [0, n_frames) into contiguous shards
- ``range_decode``          seek/scan decode of an exact frame range, file md5
- ``fake_pose``             deterministic frame-content-derived fake extractor
- ``shard_worker``          one worker process: decode range -> five raw arrays
                            -> compressed shard artefacts + manifest
- ``stitch``                validate a shard set and publish the canonical five
                            ``{stem}_raw_*.npy`` files
- ``run_sharded``           orchestrator: plan, spawn workers, stitch (CLI)
- ``gate_decode_identity``  frame-range decode vs sequential-decode MD5 ledger
- ``gate_parity``           sequential vs sharded extraction comparison
- ``gate_downstream``       stitched output through the production raw loaders
                            and heuristics
- ``bench_worker_scaling``  bounded worker-count timing probe
"""
