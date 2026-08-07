# RTMLib long-video sharding PoC — handoff

## 1. Executive result

| Question | Verdict | Scope / condition |
|---|---|---|
| Frame-range decode correct? | ESTABLISHED | h264 1080p CFR match video, cv2 5.0, both hosts; VFR/other codecs untested |
| Independent shard execution sound? | ESTABLISHED | multiprocess spawn workers, per-process sessions; SIGKILL and short-read abort loudly |
| Stitch/final-publication sound? | ESTABLISHED | 10 corruption modes refused before publication; ndet-last marker semantics |
| Sequential-vs-sharded parity? | ESTABLISHED (CPU/fake), SUPPORTED (CUDA) | byte-exact everywhere tested; CUDA verdict scoped to onnxruntime-gpu 1.27 on the A100 |
| Useful multiprocess scaling? | ESTABLISHED | 2.38x wall at 8 workers on the A100, one shared 1080p file, 14.4k-frame probe |
| Clean downstream compatibility? | ESTABLISHED | production loader + both heuristics on stitched output, unmodified; numeric-prefix stem required |
| Production integration ready to plan? | SUPPORTED | plan in section 9; per-shard resume and multi-source identity remain unimplemented/untested |

- Most important positive finding: cv2 `CAP_PROP_POS_FRAMES` seek is frame-exact on the
  production codec, and whole-video decode is bit-identical across the laptop and the
  A100 node, so sharded extraction can be validated by exact equality end to end.
- Most important limitation: every identity/parity claim is scoped to h264 CFR
  ShuttleSet-style sources and cv2 5.0; VFR or other codecs re-open the seek question.
- Biggest remaining uncertainty: whether seek identity and CUDA exactness hold across
  other source videos (all evidence is one match) and future build/driver changes.
- Recommended next action: run the decode-identity gate over the other full matches
  (cheap, CPU-only), then plan the small production integration in section 9.

## 2. Environment

| Item | Value |
|---|---|
| BASE_SHA | 95f812be8af0a05c364e810823ab085fbc113391 |
| PoC branch | `poc/rtmlib-video-sharding` (worktree `wt_rtmlib_sharding_poc`) |
| Host / GPU | local laptop (CPU only; no onnxruntime) and bourbaki A100-PCIE-40GB, driver 610.57.04 |
| Python | 3.12 local (`badminton-cicd` venv), 3.11.13 bourbaki (`venv-rtmlib`) |
| RTMLib / ONNXRuntime | bourbaki: onnxruntime-gpu 1.27.0 (CUDA EP), rtmlib RTMDet-M@640 + RTMPose-L@256 |
| Tested video / codec | `sset_21_gloiZ_gTJaE.mp4`: h264 1920x1080, CFR 30 fps, 100,349 frames (file MD5 identical on both hosts) |

## 3. Evidence that changed the conclusion

| Question | Experiment | Result | Verdict |
|---|---|---|---|
| Frame count trustable? | full sequential decode vs container metadata, both hosts | 100,349 == metadata | ESTABLISHED |
| Seek identity | per-frame MD5 vs sequential ledger: 5 awkward probes, all 8 production shard boundaries, tail, EOF-crossing; repeated on bourbaki with 5 more unaligned ranges | all frame-exact; EOF-crossing reads short, loudly | ESTABLISHED |
| Cross-host decode | local vs bourbaki sequential MD5 ledgers | identical for all 100,349 frames | ESTABLISHED |
| Deterministic parity | fake extractor, sequential (production `extract_one_clip`) vs sharded, synthetic 4-GOP video + real 2,401-frame 1080p cut, seek and scan | all five arrays byte-exact | ESTABLISHED |
| CPU determinism | real RTMLib CPU sequential run A vs B, OMP_NUM_THREADS=2, ~600-frame cut | byte-exact | ESTABLISHED |
| CPU parity | real RTMLib CPU, sequential vs 4-shard, 721-frame real cut | byte-exact | ESTABLISHED |
| CUDA self-variance / parity | seq A vs B, then seq vs 4-shard, 3,601-frame real cut on the A100 | both byte-exact | SUPPORTED (build/card-scoped) |
| Worker scaling | 1/2/4/8 workers, CUDA, 14.4k-frame 1080p cut | 711 -> 299 s wall (2.38x) | ESTABLISHED |
| Failure containment | SIGKILL live workers; 10 stitch corruption tests | run aborts, no publication looks complete | ESTABLISHED |

## 4. Frame-range decoding

| Approach | Identity evidence | Cost / limitation | Disposition |
|---|---|---|---|
| cv2 seek (`CAP_PROP_POS_FRAMES`) then sequential read | frame-exact vs full-decode MD5 ledger at every probed boundary, both hosts | seek cost trivial vs inference; unproven off h264/CFR/cv2-5.0 | adopt for workers |
| scan (decode from 0, discard) | exact by construction; verified on a mid-video range | O(F) per shard — late shards pay near-full decode | keep as correctness control only |
| ffmpeg pre-segmentation | not exercised | extra disk pass + copies; keyframe-aligned boundaries only | fallback if seek ever fails on a source |

Known failing case: none observed. A range crossing EOF reads short and the worker
refuses to write; the last shard also probes one frame past its planned end so an
undercounting container header cannot silently truncate a run.

## 5. Sequential vs sharded parity

| Test | Exact? | Detection-count differences | Ordering differences | Numeric difference | Verdict |
|---|---:|---:|---:|---:|---|
| Deterministic fake (synthetic + real 1080p cut) | yes | 0 | 0 | 0 | ESTABLISHED |
| Decode identity (frame MD5, full match) | yes | — | — | — | ESTABLISHED |
| RTMLib CPU (self-variance control) | yes | 0 | 0 | 0 | ESTABLISHED |
| RTMLib CPU (seq vs sharded, 721-frame real cut, 4 shards) | yes | 0 | 0 | 0 | ESTABLISHED |
| RTMLib CUDA (self-variance, 3,601-frame real cut) | yes | 0 | 0 | 0 | ESTABLISHED |
| RTMLib CUDA (seq vs sharded, same cut, 4 shards) | yes | 0 | 0 | 0 | SUPPORTED* |

\* Byte-exact on this build+card (onnxruntime-gpu 1.27.0, A100), consistent with the
G7 zero-self-variance measurement. Kept at SUPPORTED, not ESTABLISHED: zero variance
on one build/card is not a guarantee, so CUDA parity should be re-gated after any
onnxruntime/driver/GPU change (command in section 11).

## 6. Performance

Bourbaki A100, CUDA, 14,401-frame 1080p cut of sset_21, workers == shards,
`OMP_NUM_THREADS=2`. Wall time includes worker spawn, per-worker model load, decode,
inference, compressed shard IO and stitch.

| Workers | Frames | Wall time | ms/frame | Throughput |
|---:|---:|---:|---:|---:|
| 1 | 14,401 | 711.5 s | 49.41 | 20.2 fps |
| 2 | 14,401 | 508.8 s | 35.33 | 28.3 fps |
| 4 | 14,401 | 387.9 s | 26.93 | 37.1 fps |
| 8 | 14,401 | 299.3 s | 20.78 | 48.1 fps |

Concurrency materially helps: 2.38x at 8 workers, the same scaling shape as the
clip-level extraction-saturation runbook measured on this card (2.2x), so decoding one
shared 1080p file instead of many clips introduces no new bottleneck at these counts.
Single-worker cost is lower than the runbook's clip workload (49.4 vs 83.7 ms/frame)
because there is no per-clip open/teardown. Extrapolated: a full 100k-frame match takes
roughly 35 min at 8 workers vs roughly 83 min sequentially. Next bottleneck was not
isolated here; the runbook's GPU-pinning evidence says past 8 workers gains stop.

## 7. Meaningful failure handling

All demonstrated by tests (`tests/test_video_sharding.py`) unless noted.

| Failure | Behaviour | Demonstrated? |
|---|---|---|
| Missing/failed shard | stitch refuses: "shard incomplete"; worker death observable via exit code (SIGKILL run: no publish) | yes |
| Gap/overlap/duplicate | plan validation refuses; duplicate range impossible under filename scheme, unplanned manifests refused | yes |
| Stale or mixed run | per-shard `run_id` + source MD5 checked against run manifest | yes |
| Partial write | all writes `.tmp`+rename; manifest written last; publication writes ndet last (production marker convention) | yes |
| Source mismatch | shard `source_md5` vs run manifest | yes |
| Schema / `n_max` mismatch | per-shard shape/dtype recomputed from expectation formula, not trusted from manifest | yes |
| Undercounting container metadata | last shard probes past planned end and aborts | yes |

## 8. PoC architecture

```
run_sharded (orchestrator)
  ├─ plan_frame_shards ── [0,F) -> contiguous [start,end) shards
  ├─ run manifest (run_id, source MD5, n_frames, n_max, plan)   [written first]
  ├─ spawn N x shard_worker (own cv2 capture + own extractor/session)
  │     range_decode.iter_frame_range (seek) -> raw_extract.extract_raw_frame
  │     -> five .npy.xz + shard manifest [manifest last = complete marker]
  └─ stitch: validate everything -> concat -> publish {stem}_raw_*.npy [ndet last]
```

| Added component | Responsibility |
|---|---|
| `shard_plan` | contiguous exact-cover partition of the frame index space |
| `range_decode` | exact-range seek/scan decode; file/frame MD5 |
| `fake_pose` | deterministic frame-content-derived extractor (parity control) |
| `shard_worker` | one process: decode range, reuse production array assembly, persist shard |
| `stitch` | refuse-then-publish; owns every integrity check |
| `run_sharded` | plan, spawn, join, stitch (CLI + library entry) |
| `gate_decode_identity` / `gate_parity` / `gate_downstream` / `bench_worker_scaling` | evidence CLIs |

Production array semantics (`extract_raw_frame`), the raw file contract
(`RAW_SUFFIXES`) and model configuration are imported from their existing owners, not
duplicated.

## 9. Recommended production integration

### Recommended flow

```
extract_sharded_video (new, preparing_data/)          raw_extract.py (unchanged)
  plan -> run manifest -> N worker processes                 |
     worker: seek-decode [s,e) -> RtmlibPoseExtractor        |  same five-array
             -> extract_raw_frame -> shard + manifest        |  contract
  stitch: validate -> publish {vid}_full_raw_*.npy  ---------+--> apply_heuristic /
                                                                  heuristics (unchanged)
```

### Responsibility boundaries

| Responsibility | Recommended owner |
|---|---|
| RTMLib inference | `preparing_data.rtmlib_pose` (unchanged) |
| Video decoding | `rtmlib_pose.iter_video` grows optional `start`/`end` (seek), default unchanged |
| Shard planning | new `preparing_data/extract_sharded_video.py` |
| Worker lifecycle | same new module (spawn, join, exit-code check) |
| Run/shard state | same new module (run manifest + per-shard manifest, as in the PoC) |
| Stitching | same new module (refuse-then-publish) |
| Canonical pose representation | `raw_extract.extract_raw_frame` + `heuristics.base.RAW_SUFFIXES` (unchanged) |
| Downstream consumption | `apply_heuristic` (unchanged; stems must be numeric-prefixed) |
| Mode/config orchestration | the new module's CLI: `--video --stem --n-shards --n-max --device --save-dir` |

### Proposed interfaces

```python
# rtmlib_pose.py — only production edit of substance
def iter_video(self, video_path, start: int = 0, end: int | None = None): ...

# preparing_data/extract_sharded_video.py (graduated from the PoC)
def extract_sharded(video_path, save_dir, stem, n_shards, n_max=16,
                    device="cuda") -> Path: ...   # publishes five {stem}_raw_*.npy
```

### Minimal existing-code changes

| Existing file/function | Recommended change | Must NOT own |
|---|---|---|
| `rtmlib_pose.iter_video` | optional frame-range args (seek) | shard state, padding, n_max |
| `preparing_data/` | add one module (above) | heuristics, court logic, clip pipeline, TrackNet |

Nothing else changes: `raw_extract`, `apply_heuristic`, heuristics and all consumers
stay byte-for-byte as they are — that is what the parity gates demonstrated.

### Sequential/sharded relationship

The sharded module is an additive front-end producing the identical artefact set
(proven byte-exact on CPU, and on CUDA for this build). `raw_extract` remains the
canonical clip path and the fallback: `--n-shards 1` degenerates to sequential
behaviour through the same code, and the old path is never removed during migration.

### Migration sequence

1. Land the `iter_video` range parameters + the decode-identity test on the tiny
   synthetic video (pure addition, CI-safe).
2. Graduate planner/worker/stitch into `preparing_data/extract_sharded_video.py`,
   porting the PoC tests.
3. Re-run the CPU parity gate + CUDA parity gate on the HPC once, record results.
4. Use it for the first real full-match extraction; keep `run_dir` artefacts until
   the publication is spot-checked.
5. Fold the gates into `validation_scripts/` maintenance use; delete PoC leftovers.

### Strongest alternative

ffmpeg keyframe-aligned pre-segmentation into chunk files, fed through the existing
clip-level machinery (the extraction-saturation runbook flow). Second choice because
it adds a disk copy pass, boundary imprecision (keyframe-aligned only), and a second
provenance layer for no measured benefit — in-process seek is frame-exact here. It
becomes first choice if a future source (VFR, other codec) fails the decode-identity
gate, which is exactly what the gate exists to detect cheaply.

### PoC disposition

| PoC piece | Graduate / rewrite / test-only / delete | Why |
|---|---|---|
| `shard_plan`, `stitch`, `shard_worker`, `run_sharded` | graduate (light rename/merge into one module) | proven logic, already reuses production owners |
| `range_decode` | graduate into `rtmlib_pose.iter_video` + a small helper | one decode surface, not two |
| `fake_pose`, `tests/test_video_sharding.py` | keep test-only | deterministic CI coverage |
| `gate_decode_identity`, `gate_parity`, `gate_downstream`, `bench_worker_scaling` | keep as validation_scripts gates | re-run after env changes / new sources |
| `run_remote_bourbaki.sh`, `INVESTIGATION.md`, this report | archive with the PoC branch | record only |

## 10. Remaining unknowns

- Seek identity on the other full matches and on any VFR/re-encoded source: run
  `gate_decode_identity baseline` + `check` per source (CPU-only, ~15 min each).
- CUDA exactness across onnxruntime/driver/GPU changes: re-run `gate_parity
  --extractor cuda` after any environment change; fall back to reporting the
  difference distribution it prints.
- Per-shard resume: manifests support skip-completed-shards, but the PoC re-runs a
  failed run from scratch; implement + test during graduation (section 9, step 2).
- Court-context wiring for full-match heuristic runs: the gate used a synthetic
  identity court; production needs the real `all_court_info`/resolution row for the
  match, an `apply_heuristic` input question, not an extraction one.
- Scaling on carmack (L40) and past 8 workers: same bench command, different host.
- Stitch memory at much larger `n_max` or multi-hour sources: ~0.5 GiB held for a
  56-min match at n_max=16; linear growth, unmeasured beyond that.

## 11. Commands actually run

All from the worktree root with `PYTHONPATH=src:src/bst_x`; `$V21` is the sset_21
full match, `$LEDGER` its sequential MD5 ledger.

```
# tests (local): 18 passed; full repo suite: 1380 passed, 4 pre-existing failures
python -m pytest tests/test_video_sharding.py -q
python -m pytest -q

# frame identity (run on both hosts; ledgers byte-identical, MD5 0a3e82e3...)
python -m validation_scripts.video_sharding.gate_decode_identity baseline $V21 $LEDGER
python -m validation_scripts.video_sharding.gate_decode_identity check $V21 $LEDGER --mode seek
python -m validation_scripts.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode seek --ranges 0:40,12544:12584,...,100309:100349        # all PASS
python -m validation_scripts.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode scan --ranges 50176:50216                               # PASS (control)

# parity (local fake; bourbaki cpu/cuda; all "PASS (exact)")
python -m validation_scripts.video_sharding.gate_parity --video $V21 --workdir W \
  --extractor fake --n-shards 6 --limit-frames 2000
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu  --limit-frames 600 --self-variance
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu  --limit-frames 600 --n-shards 4
... gate_parity --extractor cuda --limit-frames 3000 --self-variance
... gate_parity --extractor cuda --limit-frames 3000 --n-shards 4

# downstream + scaling
python -m validation_scripts.video_sharding.gate_downstream --video $V21 --workdir W \
  --stem 21_full_poc --limit-frames 300                          # PASS
... bench_worker_scaling --extractor cuda --limit-frames 12000 --worker-counts 1,2,4,8

# bourbaki ladder driver (tmux session shard_poc; ended itself, verified gone)
bash src/bst_x/validation_scripts/video_sharding/run_remote_bourbaki.sh
```

## 12. Diff hygiene

- BASE_SHA: 95f812be8af0a05c364e810823ab085fbc113391
- Branch: `poc/rtmlib-video-sharding`
- Added files: `src/bst_x/validation_scripts/video_sharding/` (14 files: 9 modules,
  2 gates docs — this report and INVESTIGATION.md — plus `__init__.py`,
  `run_remote_bourbaki.sh`) and `tests/test_video_sharding.py`; verified all `A`
  status against BASE_SHA
- Existing tracked files modified: **0**
- Existing tracked files deleted/renamed: **0**
- Pushed: **no**
- Off-repo state: bourbaki logs + frame ledger kept at
  `/scratch/comp320a/ahalperi/rtmlib_sharding_poc_out/` (3.3 MB); synced code tree
  and heavy artefacts removed; tmux session `shard_poc` ended. Bourbaki video
  renames done as instructed: `pilot*` -> `sset_01*`, `vid15*` -> `sset_15*`
  (MD5-verified against the local 288p copies before renaming).
