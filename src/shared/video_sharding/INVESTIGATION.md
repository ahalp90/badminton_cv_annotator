# RTMLib long-video sharding — investigation ledger

Keep this short. It exists to structure the experiments, not to become a design essay.

## Context

- `BASE_SHA`: 95f812be8af0a05c364e810823ab085fbc113391
- PoC branch: `poc/rtmlib-video-sharding` (worktree `wt_rtmlib_sharding_poc`)
- Host/runtime: local laptop (no usable GPU, cv2 5.0 / numpy 2.4.4, **no onnxruntime/rtmlib**)
  plus bourbaki A100-PCIE-40GB (`~/.venvs/venv-rtmlib`: py3.11.13, cv2 5.0.0, numpy 2.4.6,
  onnxruntime-gpu 1.27.0 with CUDA provider)
- Relevant source entrypoints inspected: `preparing_data/rtmlib_pose.py`,
  `preparing_data/raw_extract.py`, `preparing_data/prepare_train_on_shuttleset.py`,
  `preparing_data/heuristics/base.py` (`RawClip`, `RAW_SUFFIXES`),
  `preparing_data/apply_heuristic.py`, `pipeline/config.py`,
  `docs/architecture_notes/rtmlib_migration/extraction_saturation_runbook.md`
- Prior design notes inspected: `shard_stitch_pose_spec.md`, `STALE.md`,
  `note_tracknet_efficiency_sharding.md` exist only under `local_scratch/`, which the user
  has ruled off-limits as grossly outdated. Treated as unavailable; all assumptions
  re-derived from current source.

## Current data flow

1. Full match videos live at `data/shuttleset/raw_video/`; `pipeline/clip_generator.py`
   cuts them into per-rally clips under `data/shuttleset/clips/`.
2. `raw_extract.py` iterates clip mp4s with `RtmlibPoseExtractor.iter_video`
   (cv2 read-to-EOF; no seek/range surface exists anywhere in the adapter).
3. Per frame, `RTMDetScored` (640x640) + `RTMPose` (192x256) return the real
   detections; `extract_raw_frame` NaN-pads to `n_max` (CLI default 16, int8 cap 127)
   with top-`n_max`-by-`bbox_score` truncation.
4. Five arrays are saved per clip stem: `_raw_kps` (F,N_max,J,2) f32, `_raw_bboxes`
   (F,N_max,4), `_raw_scores` (F,N_max), `_raw_kp_scores` (F,N_max,J), `_raw_ndet` (F,)
   int8 — `_raw_ndet` written last as the resume marker.
5. `apply_heuristic.py` loads all five per stem from one flat dir (`RAW_SUFFIXES`,
   `RawClip`) and runs `current` or `sticky_anchor` (stateful across frames: EMA anchors),
   emitting `_pos/_joints/_overcount/_failed` npys consumed by collation.
6. Existing HPC parallelism is clip-level: N processes, disjoint clip stem lists, one
   shared save dir (extraction saturation runbook; 8 workers saturate the A100).

## Questions and discriminating experiments

| Question | What current source/evidence suggests | What must actually be established | Cheapest useful experiment |
|---|---|---|---|
| Frame-range identity | `iter_video` never seeks; cv2 `CAP_PROP_POS_FRAMES` seek is frame-accurate for some codecs/builds, unproven here | That seek-then-read yields byte-identical frames to a full sequential decode at awkward boundaries, per host | MD5 every frame of a sequential decode of a 1080p h264 full match; seek-decode assorted ranges (mid-GOP, near-EOF, frame 0) and compare digests |
| Independent process extraction | Runbook already runs 8 disjoint rtmlib processes per GPU safely (clip-level); per-process ONNX sessions | Same holds for frame-range workers on ONE file: exit status + produced frame count observable, no silent short shard | Multiprocess run with a killed/short worker; assert stitch refuses |
| Stitch integrity | No existing stitcher; production resume marker is `_raw_ndet`-last convention | Gap/overlap/duplicate/stale/mixed-run/partial-write/`n_max`-mismatch all rejected before publication | Unit tests corrupting one shard artefact each way; assert loud failure, no final output |
| Sequential parity | CPU deterministic at fixed threads (module docstring + retired determinism gate); CUDA policy-nondeterministic though G7 measured zero self-variance on this stack | Sharded == sequential byte-exact for deterministic paths; observed difference distribution for CUDA with a self-variance control | Fake deterministic extractor end-to-end equality; then real CPU on a bounded segment; CUDA seq-twice vs seq-vs-sharded on bourbaki |
| Useful scaling | Runbook: clip-level scaling 1→8 workers cuts wall 209→95 s (A100); decode+Python binds, not GPU | That the same shape holds when workers decode ONE shared 1080p file (seek cost, page-cache and IO contention are new) | 1/2/4/8-worker probe over a fixed span of a full match on bourbaki |
| Downstream compatibility | `apply_heuristic` needs five flat `{stem}_raw_*.npy`, stem starting with numeric video id; `sticky_anchor` is cross-frame stateful | Stitched output loads through `RawClip` unchanged and heuristics run on it; stitch must precede heuristic | Run `apply_heuristic`'s load path + both heuristics on stitched fake-extractor output |

## Candidate approaches

| Approach | Main advantage | Main risk/unknown | What would decide |
|---|---|---|---|
| cv2 seek per worker (`CAP_PROP_POS_FRAMES`), each worker owns its own capture + ONNX session | Reuses the exact production decoder; no new deps | Seek frame-accuracy on this codec/build; cost of seek on long GOPs | Decode-identity experiment vs sequential MD5 ledger |
| Sequential-skip per worker (decode from 0, discard until start) | Trivially identical to sequential decode | O(n_shards * F) decode waste; late shards pay near-full decode cost | Only needed if seek fails identity; timing shows if waste is tolerable |
| Pre-split video into shard files (ffmpeg segment) then existing clip path | Reuses clip-level machinery end-to-end | Re-encode changes pixels (parity broken) or keyframe-aligned copy changes range boundaries; extra disk pass | Only if both in-process approaches fail; parity requirements likely rule out re-encode |

## Experiment results / decisions

Append rows as experiments finish. Do not rewrite earlier expectations to match results.

| Experiment | Result | Decision / implication |
|---|---|---|
| PoC test suite (18 tests: plan, fake parity seek+scan on synthetic 4-GOP video, worker short-read/overrun, 10 stitch corruption cases, downstream loader+heuristics) | all pass locally | Stitch guards and deterministic end-to-end behaviour demonstrated |
| Fake-extractor parity on real 1080p h264 match (sset_21 cut, 2401 frames, 6 shards, seek) | all five arrays byte-exact vs sequential control | Decode+assembly path is order/content-identical under multiprocess seek sharding on the real codec |
| Existing repo suite as regression in worktree | 1380 passed; 4 pre-existing `test_namespace_migration` failures (main fails 6 of same file) | No regression from PoC additions |
| ruff + pyrefly on new files | clean (repo-wide ruff noise is local 0.16.1 vs CI-pinned 0.15.12; pyrefly's 3 errors pre-exist on main) | — |
| sset_21 source properties | h264 1920x1080, CFR 30/1 (r == avg rate), metadata nb_frames=100349 | VFR seek hazard absent for this source; VFR inputs remain untested scope |
| Sequential full decode of sset_21 (local, cv2 5.0 headless) | 100,349 frames decoded == container metadata | Metadata frame count is a sound plan basis for this source |
| Seek identity vs sequential MD5 ledger (local): 5 default awkward probes + all 8 production shard boundaries + tail [100309,100349) + EOF-crossing | 14/14 frame-exact; EOF-crossing read detectably short | `CAP_PROP_POS_FRAMES` seek is frame-accurate on this codec/build; slow scan control also exact |
| SIGKILL live shard workers mid-run (real bounded video, 4 shards) | orchestrator raised "4 shard worker(s) failed ... exit=-9"; no publish dir created | Hard worker death is observable and cannot yield a silent short shard |
| Sequential decode on bourbaki (cv2 5.0.0, A100 node) vs local (cv2 5.0.0.93 headless) | per-frame MD5 ledgers identical for all 100,349 frames (ledger file MD5s equal; same source file) | Whole-video decode is bit-identical across the two hosts/builds in play |
| Seek identity on bourbaki: default probes + 5 extra unaligned ranges incl. [100000,100349) | PASS | Seek accuracy holds on the extraction host |
| Real RTMLib CPU self-variance (bourbaki, OMP_NUM_THREADS=2, ~600-frame bounded cut, run A vs B) | all five arrays byte-exact | CPU inference is deterministic here; exact equality is the right bar for CPU parity |
| Real RTMLib CPU sequential vs 4-shard sharded (bourbaki, 721-frame real cut) | all five arrays byte-exact | Frame-range sharding with real inference reproduces the production sequential path exactly on CPU |
| Real RTMLib CUDA self-variance (bourbaki A100, onnxruntime-gpu 1.27, 3,601-frame cut, seq A vs B) | byte-exact | Reproduces the G7 zero-self-variance finding on this stack; still not treated as a guarantee across builds/cards |
| Real RTMLib CUDA sequential vs 4-shard sharded (same cut) | all five arrays byte-exact | On this build+card, CUDA sharded parity is exact — no numeric-drift tolerance needed; runbook's disjoint-shard provenance rule stays |
| Worker scaling 1/2/4/8 (bourbaki A100, CUDA, 14,401-frame 1080p cut) | 711.5 / 508.8 / 387.9 / 299.3 s wall (49.4 -> 20.8 ms/frame) | 2.38x at 8 workers, matching the clip-level runbook's shape; one shared source file adds no new bottleneck at these counts |

## Decision

The cv2-seek-per-worker approach (candidate 1) is adopted; the scan path stays as a
correctness control and the ffmpeg pre-segmentation alternative stays unimplemented
(see HANDOFF.md section 9 for the graduation plan).

## Material surprises

- CUDA inference was byte-exact both run-to-run and sequential-vs-sharded on this
  stack, so parity needed no tolerance at all (the prompt anticipated a difference
  distribution; there was none to report).
- Whole-video cv2 decode is bit-identical across the two hosts and cv2 builds —
  cross-host shard provenance can be checked with plain frame MD5s.
- `apply_heuristic` silently skips stems without a numeric video-id prefix, so a
  sharded publication named `sset_21_*` would vanish downstream without error; the
  publish stem must be `21_...`.
- `np.save(path)` appends `.npy` to unknown extensions, which broke the atomic
  `.tmp`-then-rename publish; writing through a file handle fixed it (worth knowing
  for any future atomic-write of npys).
