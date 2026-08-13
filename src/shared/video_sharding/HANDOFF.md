# RTMLib long-video sharding PoC — what the agent found

## Production-integration status (2026-08-10)

Issue 15 Batch 5B promotes the tested core into the dataset builder without
changing the five-array pose contract. `pose_shards = 1` retains the original
sequential child. Larger values run the direct-seek planner/workers/stitcher in
the configured pose interpreter, require OpenCV's planned frame count to equal
canonical strict metadata before workers start, validate the final arrays, and
publish only the existing compressed pose artefacts. Shard compression now
streams through atomic XZ files instead of buffering a whole `.npy` payload.

The deterministic local suite is part of the implementation gate. The two
selected Bourbaki sources still require full sequential-versus-seek frame
identity, and the moved production boundary still requires one-shard versus
eight-shard RTMLib parity on the fixed A100 clip before the replacement E2E
run. The historical investigation and measurements below remain unchanged.

## TL;DR

The agent trialled splitting one long match video into frame ranges, running those ranges in separate RTMLib processes, and stitching the results back together afterwards. We wanted to know whether this could speed up full-match pose extraction without changing the files the rest of the repo already expects.

On the setup the agent tested, it works. The sharded version read the right frames, produced the same five raw output arrays as the sequential version, rejected broken/incomplete shard sets, and was quite a bit faster on the A100.

The first thing we needed to know was whether OpenCV can seek to an exact frame reliably enough for this. If a worker is told to start at frame 37,632 and gets 37,631 instead, the whole idea is broken. The agent decoded the full test match sequentially, hashed every frame, then sought directly to awkward ranges and compared the results. The source was `sset_21_gloiZ_gTJaE.mp4`: H.264, 1920×1080, constant 30 fps, 100,349 frames. Every requested frame matched exactly, including shard boundaries, unaligned ranges, the tail and EOF cases. The agent ran the important checks both locally and on Bourbaki, and the full sequential decode also matched across the two machines.

The agent then checked whether splitting the work changed the RTMLib output. A deterministic fake extractor matched byte-for-byte first, which checked the range reading, ordering and stitching. Real RTMLib on CPU also matched byte-for-byte between sequential and sharded runs. CUDA on the A100 did the same: repeated sequential runs matched each other, and the four-shard run matched the sequential run exactly.

We should not assume that CUDA result is true on every future setup. It is what happened on this A100 with `onnxruntime-gpu 1.27.0` and the current driver/runtime. If we change GPU, driver, ONNX Runtime or something similar, rerun the parity check.

The speedup is enough that this looks worth keeping. On a 14,401-frame 1080p section, one worker took 711.5 seconds and eight workers took 299.3 seconds: about **2.38× faster**. That timing includes worker startup, model loading, decoding, inference, compressed shard writes and stitching. All eight workers were reading different parts of the same video, and that did not create an obvious new bottleneck at these worker counts.

A rough projection from that benchmark puts a 100,000-frame match at around 35 minutes with eight workers instead of roughly 83 minutes with one. That is only a projection from the shorter benchmark, not a measured full-match timing.

The failure handling also looks sane. The stitch step checks that the planned ranges are all present exactly once, that the run ID and source MD5 match, and that the arrays have the expected shapes, dtypes and `n_max`. The agent tested ten deliberately broken shard cases and they were all rejected. It also killed live workers with SIGKILL; the parent process noticed and did not publish final output that looked complete.

The stitched files work with the existing downstream loader and both heuristics the agent tried. The PoC reused `extract_raw_frame` and the existing five-file raw format, so the rest of the repo does not need a new pose format. One existing gotcha turned up: `apply_heuristic` expects the stem to start with a numeric video ID. `sset_21_*` can be silently skipped; `21_full_*` works.

So for the project: **this looks worth moving into the main code, but we should first run the cheap frame-identity check on the other full-match videos.** The main thing the trial has not proved is that exact seeking behaves the same on every source we care about. VFR video, other codecs and re-encoded files are still unknown. If the other normal full matches pass, the code change can stay small: let `RtmlibPoseExtractor.iter_video` take optional `start`/`end` frames, add one module that plans ranges/runs workers/stitches results, and leave the raw extraction and heuristic code alone.

We should keep the current sequential path as well. One shard should behave like the normal sequential case, and if a source ever fails the seek check we can either use that path or fall back to splitting the video with ffmpeg. Per-shard resume is also still missing: if one shard dies right now, the whole run starts again.

In short: the basic idea held up. The important correctness checks passed on the tested full match, the output stayed identical, bad shards did not get published, and eight workers gave a useful speedup. The next useful thing is not more design work; it is checking a few more real source videos, then moving the tested code into `preparing_data` and running one real full-match job end to end.

## Contents

1. [What the agent changed for the PoC](#what-the-agent-changed-for-the-poc)
2. [The frame-seeking bit](#the-frame-seeking-bit)
3. [Did sharding change the output?](#did-sharding-change-the-output)
4. [How much faster was it?](#how-much-faster-was-it)
5. [What happens when a worker or shard is bad?](#what-happens-when-a-worker-or-shard-is-bad)
6. [Does the rest of the repo still work with it?](#does-the-rest-of-the-repo-still-work-with-it)
7. [How the PoC is put together](#how-the-poc-is-put-together)
8. [How we could fold it into the main code](#how-we-could-fold-it-into-the-main-code)
9. [Things the PoC has not proved yet](#things-the-poc-has-not-proved-yet)
10. [What we should do next](#what-we-should-do-next)
11. [Fallback if OpenCV seeking breaks on another source](#fallback-if-opencv-seeking-breaks-on-another-source)
12. [Test setup](#test-setup)
13. [Tests and checks the agent ran](#tests-and-checks-the-agent-ran)
14. [Commands](#commands)
15. [What is worth keeping from the PoC](#what-is-worth-keeping-from-the-poc)

## What the agent changed for the PoC

The PoC adds another way to process a long video.

The current approach is basically: open a video, read it from start to finish, run RTMLib on every frame, and save the five raw arrays.

The PoC does this instead:

1. split the video into contiguous frame ranges;
2. start one process per range;
3. let each process open the same source video and seek to its own start frame;
4. run the normal per-frame RTMLib extraction for that range;
5. save one set of shard files plus a small manifest;
6. once every worker is done, check the shards and concatenate them in frame order; and
7. publish the same five `{stem}_raw_*.npy` files the rest of the code already expects.

The PoC reused the existing `raw_extract.extract_raw_frame` path and `RAW_SUFFIXES`. The PoC is changing how the frames are divided between processes, not inventing a new raw-pose format.

## The frame-seeking bit

This was the main thing the trial needed to establish first.

OpenCV has `CAP_PROP_POS_FRAMES`, but the fact that the API lets us ask for frame 50,000 does not automatically mean every codec/build/source will land on exactly frame 50,000. So the agent tested exact seeking rather than assuming it.

The main source was:

- `sset_21_gloiZ_gTJaE.mp4`
- H.264
- 1920×1080
- constant 30 fps
- 100,349 frames

The agent decoded the whole thing sequentially and stored an MD5 for every decoded frame. It then opened the video again, sought directly to selected ranges, read forward, and checked those frame hashes against the sequential ledger.

The agent checked frame 0, awkward mid-video positions, the normal shard boundaries, extra unaligned ranges, the tail, and a range that went past EOF. All of the real frames matched exactly.

The agent repeated the important checks on Bourbaki. The full sequential frame ledger from Bourbaki also matched the local ledger for all 100,349 frames.

The EOF behaviour was useful too. If a worker asks for more frames than exist, it gets a short read and refuses to write a complete shard. The final shard also probes one frame past the planned end, so if the container claims there are fewer frames than are really decodable, we do not silently drop the tail.

The agent kept a slow "decode from frame 0 and throw frames away until `start`" mode as a correctness control. It works, but it defeats most of the point of sharding for later ranges.

## Did sharding change the output?

Not in the tests the agent ran.

| Test | Result |
|---|---|
| Fake deterministic extractor on synthetic video | all five arrays byte-exact |
| Fake extractor on a real 2,401-frame 1080p cut | byte-exact |
| Full-video frame hashes | exact |
| Real RTMLib CPU, sequential run A vs B | byte-exact |
| Real RTMLib CPU, sequential vs four shards | byte-exact |
| Real RTMLib CUDA, sequential run A vs B | byte-exact |
| Real RTMLib CUDA, sequential vs four shards | byte-exact |

The CPU comparison used a 721-frame real cut for sequential-vs-sharded. The CUDA comparison used a 3,601-frame real cut on the A100.

The CUDA result is specific to the environment the agent tested. It was an A100 with `onnxruntime-gpu 1.27.0`. If we change the GPU/runtime stack, rerun the parity command rather than assuming the result carries over.

## How much faster was it?

This was the 1/2/4/8 worker test on Bourbaki's A100 using a 14,401-frame 1080p section.

| Workers | Wall time | ms/frame | Throughput |
|---:|---:|---:|---:|
| 1 | 711.5 s | 49.41 | 20.2 fps |
| 2 | 508.8 s | 35.33 | 28.3 fps |
| 4 | 387.9 s | 26.93 | 37.1 fps |
| 8 | 299.3 s | 20.78 | 48.1 fps |

So eight workers were about **2.38× faster** than one.

That timing includes worker startup, each worker loading its own model/session, decoding, inference, compressed shard writes, and the final stitch. In other words, it is not just model inference timing.

The useful thing here is that several processes reading different ranges from one 1080p file did not obviously kill the scaling. The shape is similar to the existing clip-level multi-process measurements on the same card.

If the same throughput held for a roughly 100,000-frame match, that would work out to around 35 minutes at eight workers versus about 83 minutes with one worker. Again: **that is a rough extrapolation from the 14,401-frame test, not a measured full-match result.**

The agent did not test more than eight workers here or try the L40/Carmack host.

## What happens when a worker or shard is bad?

The rule in the PoC is simple: if anything about the shard set looks wrong, do not create final output.

The stitcher checks things like:

- did every planned range finish?
- do the ranges cover the video exactly once, with no gaps or overlaps?
- does every shard belong to this run?
- does every shard come from the same source file?
- are the five arrays the expected shapes and dtypes?
- does `n_max` match?
- is the shard actually marked complete?

Writes use temporary files and rename them into place. The shard manifest is written last, so it acts as the shard's completion marker.

The agent made ten deliberately broken shard cases in the tests and they were all rejected before final publication.

The agent also killed live workers with SIGKILL. The parent process saw the non-zero exits and stopped. It did not leave a final directory that looked like a successful extraction.

One missing bit: **per-shard resume is not implemented yet.** If shard 7 dies after shards 1–6 already finished, the current PoC reruns the job. The manifests contain enough information that reusing completed shards should be fairly straightforward to add.

## Does the rest of the repo still work with it?

Yes for what the agent tested.

After stitching, the files are still:

- `_raw_kps`
- `_raw_bboxes`
- `_raw_scores`
- `_raw_kp_scores`
- `_raw_ndet`

The existing `RawClip` loader read them, and both `current` and `sticky_anchor` ran on them without changing those bits of code.

### One existing filename trap

`apply_heuristic` expects the stem to start with a numeric video ID. If the files are called something like `sset_21_full_*`, they can be silently skipped. A stem such as `21_full_*` works.

That is existing downstream behaviour, not something sharding introduced. It just matters when choosing the final output name.

The rule is also easy to change if we ever want to. It lives in one function: `apply_heuristic._vid_from_stem`, which runs `int(stem.split("_", 1)[0])` and uses that number to look up the video's court calibration and resolution. To allow stems like `sset_21`, change that one function, or pass the video ID in directly. If we do touch it, we should also make an unparseable stem raise an error — right now it is skipped silently.

The downstream test used a synthetic identity court. A real full-match heuristic run still needs the correct `all_court_info` / resolution row for that match.

## How the PoC is put together

Roughly:

```text
run_sharded
  |
  +-- split [0, total_frames) into ranges
  +-- write a run manifest
  +-- start N worker processes
  |     |
  |     +-- open source video
  |     +-- seek to start frame
  |     +-- run RTMLib over [start, end)
  |     +-- use extract_raw_frame
  |     +-- save shard arrays
  |     +-- write shard manifest last
  |
  +-- wait for workers
  +-- check all shards
  +-- concatenate them in order
  +-- write the normal {stem}_raw_*.npy files
```

The PoC files are split up like this:

| File/module | What it does |
|---|---|
| `shard_plan` | makes contiguous frame ranges |
| `range_decode` | reads exact ranges and has the frame/file hashing helpers |
| `fake_pose` | deterministic extractor for tests |
| `shard_worker` | handles one range in one process |
| `stitch` | checks and combines shards |
| `run_sharded` | ties planning, workers and stitching together |
| `gate_decode_identity` | compares seeked frames against a sequential hash ledger |
| `gate_parity` | compares sequential and sharded raw outputs |
| `gate_downstream` | checks the normal loader/heuristics on stitched output |
| `bench_worker_scaling` | times different worker counts |

## How we could fold it into the main code

We can keep this pretty boring.

First, change `RtmlibPoseExtractor.iter_video` so it can optionally read a range:

```python
def iter_video(self, video_path, start: int = 0, end: int | None = None):
    ...
```

With no arguments, it should behave exactly as it does now.

Then add something like:

```text
preparing_data/extract_sharded_video.py
```

That file can own:

- splitting the frame ranges;
- starting/joining worker processes;
- checking exit codes;
- run/shard manifests;
- shard validation; and
- stitching the normal five output files.

The public function could be as simple as:

```python
def extract_sharded(
    video_path,
    save_dir,
    stem,
    n_shards,
    n_max=16,
    device="cuda",
) -> Path:
    ...
```

We should **not** move the existing raw-pose assembly, heuristics, court logic, clip generation or TrackNet code into this. The nice thing about the PoC is that none of those need to know sharding exists.

We should also leave the current sequential route in place. At least initially, there is no reason to delete a working fallback. The new code should also behave sensibly with `n_shards=1`.

## Things the PoC has not proved yet

These are the bits we should still treat as open:

- **Other full matches.** The exact seek test has only been done on one full 1080p match.
- **VFR / other codecs / re-encoded files.** They may seek differently. Test them before assuming this path works.
- **Different CUDA stacks.** Exact parity is only shown on this A100 + current driver + ONNX Runtime 1.27 setup.
- **Per-shard resume.** Not implemented yet.
- **Real court context.** The downstream check did not use the real court information for a full match.
- **L40 performance.** Not tested.
- **More than eight workers.** Not tested here.
- **Bigger memory cases.** Stitching was estimated at about 0.5 GiB for a 56-minute match with `n_max=16`; much longer videos or bigger `n_max` were not measured.

The biggest one for us is source coverage. Everything else is secondary if seeking turns out not to be reliable on the rest of the videos.

## What we should do next

In order:

1. Run the frame-identity check on the other normal full-match videos. It is CPU-only and cheap compared with RTMLib inference.
2. If those are fine, add `start`/`end` to `iter_video`.
3. Move the tested planner/worker/stitch code into `preparing_data/extract_sharded_video.py` and move the tests with it.
4. Add per-shard resume while that code is being cleaned up.
5. Rerun CPU and CUDA sequential-vs-sharded parity on the moved code.
6. Run one actual full match through the new path.
7. Feed that output through the normal downstream path with the real court info.
8. If that all looks normal, start using sharding for the full-match jobs where it helps.

We should keep the frame-identity and parity commands around afterwards. They are cheap insurance when a video source, OpenCV build, GPU or ONNX Runtime changes.

## Fallback if OpenCV seeking breaks on another source

If another source fails the frame-identity test, we should not hand-wave the mismatch away.

The simple fallback is to pre-split that video with ffmpeg and process the chunks through the existing clip-style flow.

That is less attractive than direct seeking because it creates another disk pass and more intermediate files, and copy-based segmentation is constrained by keyframes. But it is a reasonable escape hatch if a particular source format does not seek exactly with OpenCV.

The slowest but simplest fallback is still the old sequential path.

## Test setup

| Item | Value |
|---|---|
| Base commit | `95f812be8af0a05c364e810823ab085fbc113391` |
| PoC branch | `poc/rtmlib-video-sharding` |
| Worktree | `wt_rtmlib_sharding_poc` |
| Local | CPU only, Python 3.12, OpenCV 5.0 |
| Bourbaki | A100-PCIE-40GB, driver 610.57.04, Python 3.11.13 |
| ONNX Runtime | `onnxruntime-gpu 1.27.0` |
| Models | RTMDet-M@640 + RTMPose-L@256 |
| Main video | `sset_21_gloiZ_gTJaE.mp4` |
| Video | H.264, 1920×1080, CFR 30 fps, 100,349 frames |

The source file MD5 matched between local and Bourbaki.

## Tests and checks the agent ran

The PoC-specific test file has 18 tests. They cover the range plan, deterministic parity, short reads, overruns, ten bad-stitch cases, and the downstream loader/heuristics.

The wider repo test run had 1,380 passing tests. Four `test_namespace_migration` failures were already present; main had six failures in the same file.

Other checks/results:

- full sequential decode produced exactly 100,349 frames;
- local seek checks passed 14/14 tested ranges;
- Bourbaki seek checks passed the normal probes plus five extra unaligned ranges;
- all 100,349 sequential frame hashes matched across local and Bourbaki;
- SIGKILLing workers caused a hard failure and no final publish;
- CPU repeated runs were byte-exact;
- CPU sequential vs four shards was byte-exact;
- CUDA repeated runs were byte-exact on the tested stack;
- CUDA sequential vs four shards was byte-exact on the tested stack; and
- the 1/2/4/8 worker timings were 711.5 / 508.8 / 387.9 / 299.3 seconds.

Two random repo gotchas the agent found during the trial:

- `apply_heuristic` can silently ignore stems without a numeric video ID prefix;
- `np.save(path)` may append `.npy` to an unfamiliar filename extension, so for atomic `.tmp` writes it is safer to pass an open file handle.

## Commands

The code ended up under `src/shared/video_sharding/`. These commands use that final module path.

Run from the worktree root with `PYTHONPATH=src:src/bst_x`. `$V21` is the full `sset_21` video and `$LEDGER` is the sequential frame-MD5 ledger.

```bash
# PoC tests + repo tests
python -m pytest tests/test_video_sharding.py -q
python -m pytest -q

# Build/check the frame hash ledger
python -m shared.video_sharding.gate_decode_identity baseline $V21 $LEDGER
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER --mode seek
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode seek --ranges 0:40,12544:12584,...,100309:100349
python -m shared.video_sharding.gate_decode_identity check $V21 $LEDGER \
  --mode scan --ranges 50176:50216

# Sequential-vs-sharded checks
python -m shared.video_sharding.gate_parity --video $V21 --workdir W \
  --extractor fake --n-shards 6 --limit-frames 2000
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu --limit-frames 600 --self-variance
OMP_NUM_THREADS=2 ... gate_parity --extractor cpu --limit-frames 600 --n-shards 4
... gate_parity --extractor cuda --limit-frames 3000 --self-variance
... gate_parity --extractor cuda --limit-frames 3000 --n-shards 4

# Downstream check + worker scaling
python -m shared.video_sharding.gate_downstream --video $V21 --workdir W \
  --stem 21_full_poc --limit-frames 300
... bench_worker_scaling --extractor cuda --limit-frames 12000 --worker-counts 1,2,4,8

# Bourbaki test script
bash src/shared/video_sharding/run_remote_bourbaki.sh
```

## What is worth keeping from the PoC

We should keep/move these:

- `shard_plan`, `shard_worker`, `stitch`, `run_sharded`: this is the useful core.
- `range_decode`: fold the actual range-reading part into `rtmlib_pose.iter_video`; keep a small helper if useful.
- `fake_pose`: tests only.
- `tests/test_video_sharding.py`: keep the useful cases when the code moves.
- `gate_decode_identity`: keep for checking new video sources / decoder changes.
- `gate_parity`: keep for RTMLib/ONNX/GPU changes.
- `gate_downstream`: useful sanity check after touching the output path.
- `bench_worker_scaling`: handy if we want to try the L40 or different worker counts.

We should archive the Bourbaki driver script and the original investigation/handoff notes with the PoC branch once the useful code and tests have moved.

Current repo state from the PoC worktree:

- new files only under `src/shared/video_sharding/` plus `tests/test_video_sharding.py`;
- existing tracked files modified: **0**;
- existing tracked files deleted/renamed: **0**;
- branch not pushed;
- Bourbaki logs/frame ledger are still under `/scratch/comp320a/ahalperi/rtmlib_sharding_poc_out/` (about 3.3 MB);
- temporary synced code/heavy artefacts were removed and the `shard_poc` tmux session ended.
